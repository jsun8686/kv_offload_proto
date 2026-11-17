# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""npu_dsa_gather_rmt 的 torch 层实现（13 参张量 → 26 参 pybind 适配）。

语义/约束/错误码与 doc/dsa_gather_rmt.md §3 一致；原地写，返回 (kv, rope) alias
（对齐原版 dsa_gather 调用形态）。同步算子（poll 至全部核完成）。
维度全推导：调用方只传 attr 两项（block_size/num_requests），其余从张量 shape 取。
"""

import torch
import torch_npu

import kvops_ext._kvops as _kvops

_POLL_TIMEOUT_US = 30_000_000

_DTYPE_PAIRS = {(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16),
                (torch.int8, torch.bfloat16)}


def _elem_size(dt):
    return 1 if dt == torch.int8 else 2


def npu_dsa_gather_rmt(resident_kv_cache, resident_rope_cache, resident_block_table,
                       host_block_table, request_pool_indices, topk_token_positions,
                       resident_slots, miss_mask, block_size, num_requests,
                       host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr, block_dim=1):
    """miss 记录按需回源（远端 EMS 池 → resident HBM 池，原地写）。

    dtype 配对：fp16/fp16、bf16/bf16、int8+bf16；record 字节须 32B 对齐；
    resident 池须在 cli mem_reg 注册范围（ReadNbi dst，由池管理器保证）。
    """
    bs = int(block_size)
    nreq = int(num_requests)
    if bs <= 0 or nreq <= 0 or int(block_dim) <= 0:
        raise ValueError("block_size/num_requests/block_dim must be positive")

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

    # ── 下发（26 参 pybind；维度从 shape 推导；stream=当前流） ──
    st = _kvops.alloc_status(int(block_dim))
    stream = torch.npu.current_stream().npu_stream
    _kvops.dsa_gather_rmt(
        resident_kv_cache.data_ptr(), resident_rope_cache.data_ptr(), resident_block_table.data_ptr(),
        host_block_table.data_ptr(), request_pool_indices.data_ptr(), topk_token_positions.data_ptr(),
        resident_slots.data_ptr(), miss_mask.data_ptr(), host_rmt_channel.data_ptr(),
        host_rmt_kv_addr.data_ptr(), host_rmt_rope_addr.data_ptr(), st, nreq,
        topk_token_positions.shape[1], bs, kv_dim, rope_dim, kv_elem, rope_elem,
        host_block_table.shape[0], host_block_table.shape[1], resident_block_table.shape[1],
        host_blk_num, resident_blk_num, stream, int(block_dim))
    for i in range(int(block_dim)):  # 每核独立 64B 槽，全部完成才返回（同步语义）
        rc = _kvops.poll_status(st + i * 64, _POLL_TIMEOUT_US)
        if rc != 0:
            raise RuntimeError("npu_dsa_gather_rmt kernel failed (core %d) rc=0x%x" % (i, rc & 0xFFFFFFFF))
    return resident_kv_cache, resident_rope_cache


def load_dsa_gather_rmt_op():
    """幂等挂载到 torch_npu（对齐生产 custom_op_loader 属性挂载形态）。"""
    if not hasattr(torch_npu, "npu_dsa_gather_rmt"):
        torch_npu.npu_dsa_gather_rmt = npu_dsa_gather_rmt
    return torch_npu.npu_dsa_gather_rmt
