# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""npu_gather_selection_kv_cache_rmt 的 torch 层实现（13 参张量 → 27 参 pybind 适配）。

语义/约束/错误码与 doc/npu_gather_selection_kv_cache_rmt.md §3 一致；实现为同步算子
（poll 至内核完成再返回，对齐生产算子 eager 行为）。单核单流（blockDim=1，V2 实证稳态）。
"""

import torch
import torch_npu

import kvops_ext._kvops as _kvops

_POLL_TIMEOUT_US = 30_000_000


def _chk_pool(t, name, elem, bs):
    if t.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("%s: dtype must be fp16/bf16, got %s" % (name, t.dtype))
    if t.shape[1] != bs:
        raise ValueError("%s: shape[1]=%d != block_size %d" % (name, t.shape[1], bs))
    if not t.is_contiguous():
        raise ValueError("%s: must be contiguous" % name)


def npu_gather_selection_kv_cache_rmt(
    selection_k_rope,
    selection_kv_cache,
    selection_kv_block_table,
    selection_kv_block_status,
    selection_topk_indices,
    full_k_rope,
    full_kv_cache,
    full_kv_block_table,
    full_kv_actual_seq,
    full_q_actual_seq,
    full_rmt_channel,
    full_rmt_kv_addr,
    full_rmt_rope_addr,
    selection_topk_block_size=1,
    block_size=8,
):
    """账本式 topk 紧凑搬运（本地/远端/混布三态，返回 selection_kv_actual_seq [B*S*H]）。

    纯远端部署：full_k_rope/full_kv_cache 传空 tensor（int8 惯例），此时三表 channel 须全非 0。
    """
    # ── 形状/类型推导 ──
    B, S, H = selection_kv_block_status.shape[0], selection_kv_block_status.shape[1], \
        selection_kv_block_status.shape[2]
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
    _chk_pool(selection_kv_cache, "selection_kv_cache", kv_dim, bs)
    _chk_pool(selection_k_rope, "selection_k_rope", rope_dim, bs)
    if selection_kv_cache.dtype != selection_k_rope.dtype:
        raise TypeError("selection pools dtype mismatch: %s vs %s" % (
            selection_kv_cache.dtype, selection_k_rope.dtype))
    sel_blk_num = selection_kv_cache.shape[0]

    full_empty = full_kv_cache.numel() == 0 or full_k_rope.numel() == 0
    f_blk_num = full_rmt_channel.shape[0]  # 物理块坐标系以三表长度为锚
    if not full_empty:
        _chk_pool(full_kv_cache, "full_kv_cache", kv_dim, bs)
        _chk_pool(full_k_rope, "full_k_rope", rope_dim, bs)
        if full_kv_cache.shape[0] != f_blk_num or full_k_rope.shape[0] != f_blk_num:
            raise ValueError("full pools block num %d/%d != full_rmt_* tables %d" % (
                full_kv_cache.shape[0], full_k_rope.shape[0], f_blk_num))

    # ── 表校验 ──
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
    has_local = bool((full_rmt_channel == 0).any().item()) if f_blk_num > 0 else False
    if full_empty and has_local:
        raise ValueError("full pools empty but channel table has local(0) entries")

    # ── 设备一致性 ──
    tensors = (selection_k_rope, selection_kv_cache, selection_kv_block_table,
               selection_kv_block_status, selection_topk_indices, full_kv_block_table,
               full_kv_actual_seq, full_q_actual_seq, full_rmt_channel, full_rmt_kv_addr,
               full_rmt_rope_addr)
    for t in tensors:
        if t.device != selection_kv_cache.device:
            raise ValueError("all tensors must be on %s, got %s" % (selection_kv_cache.device, t.device))

    # ── 下发（27 参 pybind；stream=当前流；单核单流） ──
    actual_seq = torch.zeros(rows, dtype=torch.int32, device=selection_kv_cache.device)
    st = _kvops.alloc_status(1)
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
        int(selection_topk_block_size), kv_dim, rope_dim, stream, 1)
    rc = _kvops.poll_status(st, _POLL_TIMEOUT_US)
    if rc != 0:
        raise RuntimeError("npu_gather_selection_kv_cache_rmt kernel failed rc=0x%x" % (rc & 0xFFFFFFFF))
    return actual_seq


def load_gather_rmt_op():
    """幂等挂载到 torch_npu（对齐 GLM custom_op_loader.load_offload_gather_op 形态）。"""
    if not hasattr(torch_npu, "npu_gather_selection_kv_cache_rmt"):
        torch_npu.npu_gather_selection_kv_cache_rmt = npu_gather_selection_kv_cache_rmt
    return torch_npu.npu_gather_selection_kv_cache_rmt
