# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""npu_dsa_gather_rmt 的 torch 层实现（13 参张量 → 26 参 pybind 适配）。

两层 API（图友好，对齐 MemFabric 提交/等待分离模式）：
  submit：纯提交（校验 → 池化 status 槽 → launch），立即返回——capture 天然安全（无 host
          阻塞/无 malloc），NPUGraph 捕获区间内调用本层即可（姿势：torch.npu.graph(
          auto_dispatch_capture=True)，Step1 实证 HostArgs launch 可被记录）。
  wait_and_check：逐核槽 poll + 错误码校验——图外任意时刻调用（回放后 synchronize 再查）。
  npu_dsa_gather_rmt（同步组合）：submit + wait 一行拼装，eager 调用方零感知。

图模式纪律（用例参照 tests/test_graph_probe.py）：
  ① warmup 图外 eager 首跑（内核 hcomm.Init 首执行不得在 capture 中）；
  ② 捕获区间只调 submit；③ 回放前重置输出 sentinel；④ 三表/池张量引用持有至最后回放。
"""

from typing import NamedTuple

import torch
import torch_npu

import kvops_ext._kvops as _kvops

_POLL_TIMEOUT_US = 30_000_000
_STATUS_POOL_SLOTS = 8  # 每 device 池化槽数（64B/槽，block_dim≤8 时单次最多全占）

_DTYPE_PAIRS = {(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16),
                (torch.int8, torch.bfloat16)}

# status 槽池：device index -> [base_addr, cursor]；图外首次分配后地址终身固定（复用不释放）
_STATUS_POOL = {}


def _acquire_status(device, block_dim):
    key = device.index if device.index is not None else torch.npu.current_device()
    pool = _STATUS_POOL.get(key)
    if pool is None:
        pool = [_kvops.alloc_status(_STATUS_POOL_SLOTS), 0]
        _STATUS_POOL[key] = pool
    if pool[1] + block_dim > _STATUS_POOL_SLOTS:
        pool[1] = 0  # 环形回卷（内核 KvcStatusInit 每次启动重置槽，回卷安全）
    addr = pool[0] + pool[1] * 64
    pool[1] = (pool[1] + block_dim) % _STATUS_POOL_SLOTS
    return addr


class DsaGatherRmtStatus(NamedTuple):
    """submit 返回的等待句柄：status 基址 + 核数（wait_and_check 入参）。"""
    status_addr: int
    block_dim: int


def _elem_size(dt):
    return 1 if dt == torch.int8 else 2


def _validate(resident_kv_cache, resident_rope_cache, resident_block_table, host_block_table,
              request_pool_indices, topk_token_positions, resident_slots, miss_mask, block_size,
              num_requests, host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr):
    """全量校验（submit 与同步组合共用）；返回 launch 所需推导量。"""
    bs = int(block_size)
    nreq = int(num_requests)
    if bs <= 0 or nreq <= 0:
        raise ValueError("block_size/num_requests must be positive")
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
    for t, name in ((resident_block_table, "resident_block_table"), (host_block_table, "host_block_table"),
                    (request_pool_indices, "request_pool_indices"),
                    (topk_token_positions, "topk_token_positions"), (resident_slots, "resident_slots"),
                    (miss_mask, "miss_mask")):
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
    tensors = (resident_kv_cache, resident_rope_cache, resident_block_table, host_block_table,
               request_pool_indices, topk_token_positions, resident_slots, miss_mask,
               host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr)
    for t in tensors:
        if t.device != resident_kv_cache.device:
            raise ValueError("all tensors must be on %s, got %s" % (resident_kv_cache.device, t.device))
    return bs, nreq, kv_dim, rope_dim, kv_elem, rope_elem, resident_blk_num, host_blk_num


def npu_dsa_gather_rmt_submit(resident_kv_cache, resident_rope_cache, resident_block_table,
                              host_block_table, request_pool_indices, topk_token_positions,
                              resident_slots, miss_mask, block_size, num_requests,
                              host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr, block_dim=1):
    """纯提交层：launch 到当前流（capture 中即记录到 capture stream），返回等待句柄。

    NPUGraph 捕获区间内调用本函数（不调 wait）；回放后由调用方 synchronize，
    错误检查用 wait_and_check（图外）或 golden 对拍兜底。
    """
    bs, nreq, kv_dim, rope_dim, kv_elem, rope_elem, res_blk, host_blk = _validate(
        resident_kv_cache, resident_rope_cache, resident_block_table, host_block_table,
        request_pool_indices, topk_token_positions, resident_slots, miss_mask, block_size, num_requests,
        host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr)
    if int(block_dim) <= 0:
        raise ValueError("block_dim must be positive")
    st = _acquire_status(resident_kv_cache.device, int(block_dim))
    stream = torch.npu.current_stream().npu_stream
    _kvops.dsa_gather_rmt(
        resident_kv_cache.data_ptr(), resident_rope_cache.data_ptr(), resident_block_table.data_ptr(),
        host_block_table.data_ptr(), request_pool_indices.data_ptr(), topk_token_positions.data_ptr(),
        resident_slots.data_ptr(), miss_mask.data_ptr(), host_rmt_channel.data_ptr(),
        host_rmt_kv_addr.data_ptr(), host_rmt_rope_addr.data_ptr(), st, nreq,
        topk_token_positions.shape[1], bs, kv_dim, rope_dim, kv_elem, rope_elem,
        host_block_table.shape[0], host_block_table.shape[1], resident_block_table.shape[1],
        host_blk, res_blk, stream, int(block_dim))
    return DsaGatherRmtStatus(st, int(block_dim))


def wait_and_check(handle, timeout_us=_POLL_TIMEOUT_US):
    """等待层：逐核槽 poll 至终态并校验错误码（图外任意时刻调用；eager/图回放后通用）。"""
    for i in range(handle.block_dim):
        rc = _kvops.poll_status(handle.status_addr + i * 64, timeout_us)
        if rc != 0:
            raise RuntimeError("npu_dsa_gather_rmt kernel failed (core %d) rc=0x%x" % (i, rc & 0xFFFFFFFF))


def npu_dsa_gather_rmt(resident_kv_cache, resident_rope_cache, resident_block_table,
                       host_block_table, request_pool_indices, topk_token_positions,
                       resident_slots, miss_mask, block_size, num_requests,
                       host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr, block_dim=1):
    """同步组合（eager 语义）：submit + wait；原地写，返回 (kv, rope) alias。"""
    handle = npu_dsa_gather_rmt_submit(
        resident_kv_cache, resident_rope_cache, resident_block_table, host_block_table,
        request_pool_indices, topk_token_positions, resident_slots, miss_mask, block_size, num_requests,
        host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr, block_dim=block_dim)
    wait_and_check(handle)
    return resident_kv_cache, resident_rope_cache


def load_dsa_gather_rmt_op():
    """幂等挂载到 torch_npu（对齐生产 custom_op_loader 属性挂载形态）。"""
    if not hasattr(torch_npu, "npu_dsa_gather_rmt"):
        torch_npu.npu_dsa_gather_rmt = npu_dsa_gather_rmt
    return torch_npu.npu_dsa_gather_rmt
