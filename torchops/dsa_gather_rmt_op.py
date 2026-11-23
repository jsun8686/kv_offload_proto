# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""npu_dsa_gather_rmt / npu_dsa_gather_rmt_v2 的 torch 层实现（13 参张量 → pybind 适配）。

语义/约束/错误码与 doc/dsa_gather_rmt.md §3 一致；原地写，返回 (kv, rope) alias
（对齐原版 dsa_gather 调用形态）。维度全推导：调用方只传 attr 两项
（block_size/num_requests），其余从张量 shape 取。

v1（npu_dsa_gather_rmt）：AIV + UBC_CTP，poll 至全部核完成（同步）。
v2（npu_dsa_gather_rmt_v2）：AICPU 搬运 kernel + Hcomm device-RoCE 数据面（A3 路径），
block_dim 恒 1；kvops 内同步（SynchronizeStream）并返回 status rc，非 0 抛错。
"""

import torch
import torch_npu

import kvops_ext._kvops as _kvops

_POLL_TIMEOUT_US = 30_000_000

_DTYPE_PAIRS = {(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16),
                (torch.int8, torch.bfloat16)}


def _elem_size(dt):
    return 1 if dt == torch.int8 else 2


def _validate_gather_rmt_args(resident_kv_cache, resident_rope_cache, resident_block_table,
                              host_block_table, request_pool_indices, topk_token_positions,
                              resident_slots, miss_mask, block_size, num_requests,
                              host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr):
    """13 参公共校验（v1/v2 同约束），返回推导维度字典。"""
    bs = int(block_size)
    nreq = int(num_requests)
    if bs <= 0 or nreq <= 0:
        raise ValueError("block_size/num_requests must be positive")

    # ── 池校验（4 维 [P, bs, 1, dim]，kv_heads 恒 1）──
    for t, name in ((resident_kv_cache, "resident_kv_cache"), (resident_rope_cache, "resident_rope_cache")):
        if t.dim() != 4 or t.shape[2] != 1:
            raise ValueError("%s: shape must be [P, block_size, 1, dim], got %s" % (name, tuple(t.shape)))
        if t.shape[1] != bs:
            raise ValueError("%s: shape[1]=%d != block_size %d" % (name, t.shape[1], bs))
        if not t.is_contiguous():
            raise ValueError("%s: must be contiguous" % name)
    kv_dt, rope_dt = resident_kv_cache.dtype, resident_rope_cache.dtype
    if (kv_dt, rope_dt) not in _DTYPE_PAIRS:
        raise TypeError("dtype pair must be fp16/fp16, bf16/bf16 or int8+bf16, got %s/%s" % (kv_dt, rope_dt))
    kv_dim = resident_kv_cache.shape[3]
    rope_dim = resident_rope_cache.shape[3]
    kv_elem, rope_elem = _elem_size(kv_dt), _elem_size(rope_dt)
    if kv_dim * kv_elem % 32 != 0 or rope_dim * rope_elem % 32 != 0:
        raise ValueError("kv/rope record bytes must be 32B aligned: %dx%d / %dx%d" % (
            kv_dim, kv_elem, rope_dim, rope_elem))
    resident_blk_num = resident_kv_cache.shape[0]
    if resident_rope_cache.shape[0] != resident_blk_num:
        raise ValueError("resident pools block num mismatch: %d vs %d" % (
            resident_blk_num, resident_rope_cache.shape[0]))

    # ── 表校验 ──
    for t, name in ((resident_block_table, "resident_block_table"), (host_block_table, "host_block_table"),
                    (request_pool_indices, "request_pool_indices"), (topk_token_positions,
                                                                    "topk_token_positions"),
                    (resident_slots, "resident_slots"), (miss_mask, "miss_mask")):
        if t.dtype != torch.int32:
            raise TypeError("%s: dtype must be int32, got %s" % (name, t.dtype))
        if not t.is_contiguous():
            raise ValueError("%s: must be contiguous" % name)
    for t, name in ((host_rmt_channel, "host_rmt_channel"), (host_rmt_kv_addr, "host_rmt_kv_addr"),
                    (host_rmt_rope_addr, "host_rmt_rope_addr")):
        if t.dtype != torch.int64:
            raise TypeError("%s: dtype must be int64, got %s" % (name, t.dtype))
        if not t.is_contiguous():
            raise ValueError("%s: must be contiguous" % name)
    if topk_token_positions.dim() != 2 or resident_slots.shape != topk_token_positions.shape or \
            miss_mask.shape != topk_token_positions.shape:
        raise ValueError("topk/slots/mask shape mismatch: %s/%s/%s" % (
            tuple(topk_token_positions.shape), tuple(resident_slots.shape), tuple(miss_mask.shape)))
    if topk_token_positions.shape[0] != nreq or request_pool_indices.shape != (nreq,) or \
            resident_block_table.shape[0] != nreq:
        raise ValueError("num_requests=%d inconsistent: pool_indices=%s, resident_table=%s, topk rows=%d" % (
            nreq, tuple(request_pool_indices.shape), tuple(resident_block_table.shape),
            topk_token_positions.shape[0]))
    host_blk_num = host_rmt_channel.shape[0]
    if host_rmt_kv_addr.shape[0] != host_blk_num or host_rmt_rope_addr.shape[0] != host_blk_num:
        raise ValueError("host_rmt_* tables length %d/%d/%d inconsistent" % (
            host_blk_num, host_rmt_kv_addr.shape[0], host_rmt_rope_addr.shape[0]))

    # ── 设备一致性 ──
    tensors = (resident_kv_cache, resident_rope_cache, resident_block_table, host_block_table,
               request_pool_indices, topk_token_positions, resident_slots, miss_mask,
               host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr)
    for t in tensors:
        if t.device != resident_kv_cache.device:
            raise ValueError("all tensors must be on %s, got %s" % (resident_kv_cache.device, t.device))
    return {"bs": bs, "nreq": nreq, "kv_dim": kv_dim, "rope_dim": rope_dim, "kv_elem": kv_elem,
            "rope_elem": rope_elem, "resident_blk_num": resident_blk_num, "host_blk_num": host_blk_num,
            "host_pool_capacity": host_block_table.shape[0], "host_table_width": host_block_table.shape[1],
            "resident_table_width": resident_block_table.shape[1],
            "query_count": topk_token_positions.shape[1]}


def npu_dsa_gather_rmt(resident_kv_cache, resident_rope_cache, resident_block_table,
                       host_block_table, request_pool_indices, topk_token_positions,
                       resident_slots, miss_mask, block_size, num_requests,
                       host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr, block_dim=1):
    """miss 记录按需回源（远端 EMS 池 → resident HBM 池，原地写）。

    dtype 配对：fp16/fp16、bf16/bf16、int8+bf16；record 字节须 32B 对齐；
    resident 池须在 cli mem_reg 注册范围（ReadNbi dst，由池管理器保证）。
    """
    if int(block_dim) <= 0:
        raise ValueError("block_dim must be positive")
    d = _validate_gather_rmt_args(resident_kv_cache, resident_rope_cache, resident_block_table,
                                  host_block_table, request_pool_indices, topk_token_positions,
                                  resident_slots, miss_mask, block_size, num_requests,
                                  host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr)

    # ── 下发（26 参 pybind；维度从 shape 推导；stream=当前流） ──
    st = _kvops.alloc_status(int(block_dim))
    stream = torch.npu.current_stream().npu_stream
    _kvops.dsa_gather_rmt(
        resident_kv_cache.data_ptr(), resident_rope_cache.data_ptr(), resident_block_table.data_ptr(),
        host_block_table.data_ptr(), request_pool_indices.data_ptr(), topk_token_positions.data_ptr(),
        resident_slots.data_ptr(), miss_mask.data_ptr(), host_rmt_channel.data_ptr(),
        host_rmt_kv_addr.data_ptr(), host_rmt_rope_addr.data_ptr(), st, d["nreq"],
        d["query_count"], d["bs"], d["kv_dim"], d["rope_dim"], d["kv_elem"], d["rope_elem"],
        d["host_pool_capacity"], d["host_table_width"], d["resident_table_width"],
        d["host_blk_num"], d["resident_blk_num"], stream, int(block_dim))
    for i in range(int(block_dim)):  # 每核独立 64B 槽，全部完成才返回（同步语义）
        rc = _kvops.poll_status(st + i * 64, _POLL_TIMEOUT_US)
        if rc != 0:
            raise RuntimeError("npu_dsa_gather_rmt kernel failed (core %d) rc=0x%x" % (i, rc & 0xFFFFFFFF))
    return resident_kv_cache, resident_rope_cache


def npu_dsa_gather_rmt_v2(resident_kv_cache, resident_rope_cache, resident_block_table,
                          host_block_table, request_pool_indices, topk_token_positions,
                          resident_slots, miss_mask, block_size, num_requests,
                          host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr, block_dim=1):
    """v2：AICPU 搬运 kernel（Hcomm device-RoCE 数据面，A3 路径），语义与 v1 完全一致。

    前置：EmsPool(proto="roce") 建链并完成 register_rmt_thread；fatbin 同目录须有
    libcann_kvops_aicpu.json/.so（kernel 构建产物，随 fatbin 一起部署）。
    block_dim 恒 1（单 AICPU 实例）。同步算子：kvops 内 SynchronizeStream，
    status rc 非 0 抛错（错误码语义同 v1：0x2xx/0x3xx=读失败，0x4xx/0x5xx=收尾）。
    """
    if int(block_dim) != 1:
        raise ValueError("v2: block_dim must be 1 (single AICPU instance)")
    d = _validate_gather_rmt_args(resident_kv_cache, resident_rope_cache, resident_block_table,
                                  host_block_table, request_pool_indices, topk_token_positions,
                                  resident_slots, miss_mask, block_size, num_requests,
                                  host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr)

    st = _kvops.alloc_status(1)
    stream = torch.npu.current_stream().npu_stream
    rc = _kvops.dsa_gather_rmt_v2(
        resident_kv_cache.data_ptr(), resident_rope_cache.data_ptr(), resident_block_table.data_ptr(),
        host_block_table.data_ptr(), request_pool_indices.data_ptr(), topk_token_positions.data_ptr(),
        resident_slots.data_ptr(), miss_mask.data_ptr(), host_rmt_channel.data_ptr(),
        host_rmt_kv_addr.data_ptr(), host_rmt_rope_addr.data_ptr(), st, d["nreq"],
        d["query_count"], d["bs"], d["kv_dim"], d["rope_dim"], d["kv_elem"], d["rope_elem"],
        d["host_pool_capacity"], d["host_table_width"], d["resident_table_width"],
        d["host_blk_num"], d["resident_blk_num"], stream)
    if rc != 0:
        full = _kvops.peek_status(st)  # 诊断：word1=搬运 record 数
        raise RuntimeError(
            "npu_dsa_gather_rmt_v2 kernel failed rc=0x%x status=0x%x (word1=copied records)" % (
                rc & 0xFFFFFFFF, full))
    return resident_kv_cache, resident_rope_cache


def load_dsa_gather_rmt_op():
    """幂等挂载到 torch_npu（对齐生产 custom_op_loader 属性挂载形态）。"""
    if not hasattr(torch_npu, "npu_dsa_gather_rmt"):
        torch_npu.npu_dsa_gather_rmt = npu_dsa_gather_rmt
    if not hasattr(torch_npu, "npu_dsa_gather_rmt_v2"):
        torch_npu.npu_dsa_gather_rmt_v2 = npu_dsa_gather_rmt_v2
    return torch_npu.npu_dsa_gather_rmt
