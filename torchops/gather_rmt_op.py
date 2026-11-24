# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""npu_gather_selection_kv_cache_rmt 的 torch 层实现（13 参张量 → 27 参 pybind 适配）。

两层 API（图友好，对齐 MemFabric 提交/等待分离模式；同 torchops/dsa_gather_rmt_op）：
  submit：纯提交（校验 → 池化 status 槽 → launch）。图模式额外约定：actual_seq_out 传入
          预分配张量（capture 中不新建内存，地址固定），None 则内部新建（eager 兼容）。
  wait_and_check：逐核槽 poll + 错误码校验（图外任意时刻）。
  npu_gather_selection_kv_cache_rmt（同步组合）：submit + wait，返回 actual_seq。

图模式纪律同 dsa_gather_rmt（warmup 图外 / 捕获区间只 submit / 回放前重置输出 /
三表与页表张量引用持有至最后回放），用例参照 tests/test_graph_probe.py。
"""

from typing import NamedTuple, Optional

import torch
import torch_npu

import kvops_ext._kvops as _kvops

_POLL_TIMEOUT_US = 30_000_000
_STATUS_POOL_SLOTS = 8

_STATUS_POOL = {}  # device index -> [base_addr, cursor]，图外首次分配后地址终身固定


def _acquire_status(device, block_dim):
    key = device.index if device.index is not None else torch.npu.current_device()
    pool = _STATUS_POOL.get(key)
    if pool is None:
        pool = [_kvops.alloc_status(_STATUS_POOL_SLOTS), 0]
        _STATUS_POOL[key] = pool
    if pool[1] + block_dim > _STATUS_POOL_SLOTS:
        pool[1] = 0
    addr = pool[0] + pool[1] * 64
    pool[1] = (pool[1] + block_dim) % _STATUS_POOL_SLOTS
    return addr


class GatherSelectionRmtStatus(NamedTuple):
    """submit 返回的等待句柄：status 基址 + 核数 + 输出张量引用（供同步组合返回）。"""
    status_addr: int
    block_dim: int
    actual_seq: torch.Tensor


def _chk_pool(t, name, bs):
    if t.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("%s: dtype must be fp16/bf16, got %s" % (name, t.dtype))
    if t.shape[1] != bs:
        raise ValueError("%s: shape[1]=%d != block_size %d" % (name, t.shape[1], bs))
    if not t.is_contiguous():
        raise ValueError("%s: must be contiguous" % name)


def _validate(selection_k_rope, selection_kv_cache, selection_kv_block_table, selection_kv_block_status,
              selection_topk_indices, full_k_rope, full_kv_cache, full_kv_block_table, full_kv_actual_seq,
              full_q_actual_seq, full_rmt_channel, full_rmt_kv_addr, full_rmt_rope_addr,
              selection_topk_block_size, block_size):
    """全量校验（submit 与同步组合共用）；返回推导量元组。"""
    B, S, H = (selection_kv_block_status.shape[0], selection_kv_block_status.shape[1],
               selection_kv_block_status.shape[2])
    if H != 1:
        raise ValueError("head must be 1 (MLA shared KV), got %d" % H)
    topk = selection_topk_indices.shape[-1]
    if selection_topk_indices.shape != (B, S, H, topk) or \
            selection_kv_block_status.shape != (B, S, H, topk + 1):
        raise ValueError("status/topk shape mismatch: %s vs %s" % (
            tuple(selection_kv_block_status.shape), tuple(selection_topk_indices.shape)))
    rows = B * S * H
    bs = int(block_size)
    if bs % int(selection_topk_block_size) != 0:
        raise ValueError("block_size %d not divisible by selection_topk_block_size %d" % (
            bs, selection_topk_block_size))
    kv_dim = selection_kv_cache.shape[2]
    rope_dim = selection_k_rope.shape[2]
    _chk_pool(selection_kv_cache, "selection_kv_cache", bs)
    _chk_pool(selection_k_rope, "selection_k_rope", bs)
    if selection_kv_cache.dtype != selection_k_rope.dtype:
        raise TypeError("selection pools dtype mismatch: %s vs %s" % (
            selection_kv_cache.dtype, selection_k_rope.dtype))
    full_empty = full_kv_cache.numel() == 0 or full_k_rope.numel() == 0
    f_blk_num = full_rmt_channel.shape[0]  # 物理块坐标系以三表长度为锚
    if not full_empty:
        _chk_pool(full_kv_cache, "full_kv_cache", bs)
        _chk_pool(full_k_rope, "full_k_rope", bs)
        if full_kv_cache.shape[0] != f_blk_num or full_k_rope.shape[0] != f_blk_num:
            raise ValueError("full pools block num %d/%d != full_rmt_* tables %d" % (
                full_kv_cache.shape[0], full_k_rope.shape[0], f_blk_num))
    for t, name in ((selection_kv_block_table, "selection_kv_block_table"),
                    (full_kv_block_table, "full_kv_block_table"),
                    (full_kv_actual_seq, "full_kv_actual_seq"), (full_q_actual_seq, "full_q_actual_seq"),
                    (selection_topk_indices, "selection_topk_indices"),
                    (selection_kv_block_status, "selection_kv_block_status")):
        if t.dtype != torch.int32:
            raise TypeError("%s: dtype must be int32, got %s" % (name, t.dtype))
        if not t.is_contiguous():
            raise ValueError("%s: must be contiguous" % name)
    for t, name in ((full_rmt_channel, "full_rmt_channel"), (full_rmt_kv_addr, "full_rmt_kv_addr"),
                    (full_rmt_rope_addr, "full_rmt_rope_addr")):
        if t.dtype != torch.int64:
            raise TypeError("%s: dtype must be int64, got %s" % (name, t.dtype))
        if not t.is_contiguous():
            raise ValueError("%s: must be contiguous" % name)
    if full_kv_actual_seq.shape != (B,) or full_q_actual_seq.shape != (B,):
        raise ValueError("actual_seq shape must be (%d,)" % B)
    sel_max_blk = selection_kv_block_table.shape[1]
    if selection_kv_block_table.shape[0] != rows:
        raise ValueError("selection_kv_block_table rows %d != B*S*H %d" % (
            selection_kv_block_table.shape[0], rows))
    f_max_blk = full_kv_block_table.shape[1]
    if full_kv_block_table.shape[0] != B:
        raise ValueError("full_kv_block_table batch %d != %d" % (full_kv_block_table.shape[0], B))
    if full_rmt_kv_addr.shape[0] != f_blk_num or full_rmt_rope_addr.shape[0] != f_blk_num:
        raise ValueError("full_rmt_* tables length %d/%d/%d != F_BLOCK_NUM %d" % (
            full_rmt_channel.shape[0], full_rmt_kv_addr.shape[0], full_rmt_rope_addr.shape[0], f_blk_num))
    # 注：不做值相关校验（如 channel 含 0 互斥）——.item() 是 D2H 同步，capture 区间非法（EE1016）；
    # 该检查挪到同步组合（eager）路径执行，图模式由调用方保证 channel 全非 0
    tensors = (selection_k_rope, selection_kv_cache, selection_kv_block_table,
               selection_kv_block_status, selection_topk_indices, full_kv_block_table,
               full_kv_actual_seq, full_q_actual_seq, full_rmt_channel, full_rmt_kv_addr,
               full_rmt_rope_addr)
    for t in tensors:
        if t.device != selection_kv_cache.device:
            raise ValueError("all tensors must be on %s, got %s" % (selection_kv_cache.device, t.device))
    return B, S, H, topk, rows, bs, kv_dim, rope_dim, full_empty, f_blk_num, f_max_blk, sel_max_blk


def npu_gather_selection_kv_cache_rmt_submit(selection_k_rope, selection_kv_cache,
                                             selection_kv_block_table, selection_kv_block_status,
                                             selection_topk_indices, full_k_rope, full_kv_cache,
                                             full_kv_block_table, full_kv_actual_seq, full_q_actual_seq,
                                             full_rmt_channel, full_rmt_kv_addr, full_rmt_rope_addr,
                                             selection_topk_block_size=1, block_size=8,
                                             actual_seq_out: Optional[torch.Tensor] = None,
                                             block_dim=1):
    """纯提交层：launch 到当前流（capture 中即被记录），返回等待句柄。

    图模式：actual_seq_out 须传入预分配 int32 [rows] 张量（地址固定复用）；
    None 则内部新建（eager 兼容，capture 中不要传 None）。
    """
    (B, S, H, topk, rows, bs, kv_dim, rope_dim, full_empty, f_blk_num, f_max_blk,
     sel_max_blk) = _validate(
        selection_k_rope, selection_kv_cache, selection_kv_block_table, selection_kv_block_status,
        selection_topk_indices, full_k_rope, full_kv_cache, full_kv_block_table, full_kv_actual_seq,
        full_q_actual_seq, full_rmt_channel, full_rmt_kv_addr, full_rmt_rope_addr,
        selection_topk_block_size, block_size)
    if int(block_dim) <= 0:
        raise ValueError("block_dim must be positive")
    if actual_seq_out is None:
        actual_seq = torch.zeros(rows, dtype=torch.int32, device=selection_kv_cache.device)
    else:
        if actual_seq_out.dtype != torch.int32 or not actual_seq_out.is_contiguous() or \
                tuple(actual_seq_out.shape) != (rows,) or \
                actual_seq_out.device != selection_kv_cache.device:
            raise ValueError("actual_seq_out must be contiguous int32 [%d] on %s" % (
                rows, selection_kv_cache.device))
        actual_seq = actual_seq_out
    st = _acquire_status(selection_kv_cache.device, int(block_dim))
    stream = torch.npu.current_stream().npu_stream
    _kvops.gather_selection_rmt(
        selection_k_rope.data_ptr(), selection_kv_cache.data_ptr(),
        selection_kv_block_table.data_ptr(), selection_kv_block_status.data_ptr(),
        selection_topk_indices.data_ptr(),
        full_k_rope.data_ptr() if not full_empty else 0,
        full_kv_cache.data_ptr() if not full_empty else 0,
        full_kv_block_table.data_ptr(), full_kv_actual_seq.data_ptr(), full_q_actual_seq.data_ptr(),
        full_rmt_channel.data_ptr(), full_rmt_kv_addr.data_ptr(), full_rmt_rope_addr.data_ptr(),
        st, actual_seq.data_ptr(), B, S, H, topk, bs, f_max_blk, sel_max_blk, f_blk_num,
        int(selection_topk_block_size), kv_dim, rope_dim, stream, int(block_dim))
    return GatherSelectionRmtStatus(st, int(block_dim), actual_seq)


def wait_and_check(handle, timeout_us=_POLL_TIMEOUT_US):
    """等待层：逐核槽 poll 至终态并校验错误码（图外任意时刻调用）。"""
    for i in range(handle.block_dim):
        rc = _kvops.poll_status(handle.status_addr + i * 64, timeout_us)
        if rc != 0:
            raise RuntimeError("npu_gather_selection_kv_cache_rmt kernel failed (core %d) rc=0x%x" % (
                i, rc & 0xFFFFFFFF))


def npu_gather_selection_kv_cache_rmt(selection_k_rope, selection_kv_cache, selection_kv_block_table,
                                      selection_kv_block_status, selection_topk_indices, full_k_rope,
                                      full_kv_cache, full_kv_block_table, full_kv_actual_seq,
                                      full_q_actual_seq, full_rmt_channel, full_rmt_kv_addr,
                                      full_rmt_rope_addr, selection_topk_block_size=1, block_size=8):
    """同步组合（eager 语义）：submit + wait，返回 selection_kv_actual_seq [B*S*H] int32。"""
    full_empty = full_kv_cache.numel() == 0 or full_k_rope.numel() == 0
    if full_empty and full_rmt_channel.numel() > 0 and bool((full_rmt_channel == 0).any().item()):
        raise ValueError("full pools empty but channel table has local(0) entries")  # 值校验仅 eager 路径
    handle = npu_gather_selection_kv_cache_rmt_submit(
        selection_k_rope, selection_kv_cache, selection_kv_block_table, selection_kv_block_status,
        selection_topk_indices, full_k_rope, full_kv_cache, full_kv_block_table, full_kv_actual_seq,
        full_q_actual_seq, full_rmt_channel, full_rmt_kv_addr, full_rmt_rope_addr,
        selection_topk_block_size=selection_topk_block_size, block_size=block_size)
    wait_and_check(handle)
    return handle.actual_seq


def load_gather_rmt_op():
    """幂等挂载到 torch_npu（对齐 GLM custom_op_loader.load_offload_gather_op 形态）。"""
    if not hasattr(torch_npu, "npu_gather_selection_kv_cache_rmt"):
        torch_npu.npu_gather_selection_kv_cache_rmt = npu_gather_selection_kv_cache_rmt
    return torch_npu.npu_gather_selection_kv_cache_rmt
