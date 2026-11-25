# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""npu_dsa_gather_rmt 的 torch 层实现（13 参张量 → 26 参 pybind 适配）。

三件套（torch_npu 属性挂载，地位对等）：
  submit：纯提交（校验 → 池化 status 槽 → launch），立即返回——capture 天然安全（无 host
          阻塞/无 malloc），NPUGraph 捕获区间内调用本层即可（姿势：torch.npu.graph(
          auto_dispatch_capture=True)，Step1 实证 HostArgs launch 可被记录）。
  result：纯读（零等待零同步——调用方须已 torch.npu.synchronize()）：逐核 peek status，
          完成位契约校验（未同步的误用在此显性报错）+ 错误码校验。
  npu_dsa_gather_rmt（eager 同步组合）：submit + synchronize + result 一行拼装。

图模式纪律：
  ① warmup 图外 eager 首跑（内核 hcomm.Init 首执行不得在 capture 中）；
  ② 捕获区间只调 submit；③ 回放后 synchronize → result 校验；④ 输出张量引用持有至最后回放。
"""

from typing import NamedTuple

import torch
import torch_npu

import kvops_ext._kvops as _kvops

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
    """submit 返回的等待句柄：status 基址 + 核数（npu_dsa_gather_rmt_result 入参）。"""
    status_addr: int
    block_dim: int


def _elem_size(dt):
    return 1 if dt == torch.int8 else 2


def _validate(resident_kv_cache, resident_rope_cache, resident_block_table, host_block_table,
              request_pool_indices, topk_token_positions, resident_slots, miss_mask, block_size,
              num_requests, host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr, block_dim=1):
    """全量校验（submit 与同步组合共用）；返回 launch 所需推导量。"""
    bs = int(block_size)
    nreq = int(num_requests)
    if bs <= 0 or nreq <= 0 or int(block_dim) <= 0:
        raise ValueError("block_size/num_requests/block_dim must be positive")
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
    # D2a 每核独立通道：addr 表长度 = host 块数（内核 hostBlkNum 锚），channel 表平铺 [host_blk_num × block_dim]
    host_blk_num = host_rmt_kv_addr.shape[0]
    if host_rmt_rope_addr.shape[0] != host_blk_num:
        raise ValueError("host_rmt_kv/rope_addr tables length %d/%d inconsistent" % (
            host_blk_num, host_rmt_rope_addr.shape[0]))
    if host_rmt_channel.shape[0] != host_blk_num * int(block_dim):
        raise ValueError("host_rmt_channel length %d != host_blk_num %d x block_dim %d (re-export pool with "
                         "n_core=block_dim)" % (host_rmt_channel.shape[0], host_blk_num, int(block_dim)))
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
    错误检查用 npu_dsa_gather_rmt_result（图外）或 golden 对拍兜底。
    """
    bs, nreq, kv_dim, rope_dim, kv_elem, rope_elem, res_blk, host_blk = _validate(
        resident_kv_cache, resident_rope_cache, resident_block_table, host_block_table,
        request_pool_indices, topk_token_positions, resident_slots, miss_mask, block_size, num_requests,
        host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr, block_dim=int(block_dim))
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


def npu_dsa_gather_rmt_result(handle):
    """结果校验（纯读，零等待零同步——调用方须已 torch.npu.synchronize()）。

    逐核 peek status 槽：完成位契约校验（外层未同步的误用在此显性报错，而非死等）
    + 错误码校验（图回放下是拿到内核错误码的唯一通道——回放不会因内核内部错误失败）。
    """
    for i in range(handle.block_dim):
        v = _kvops.peek_status(handle.status_addr + i * 64)
        if v == 0xFFFFFFFFFFFFFFFF:
            raise RuntimeError("npu_dsa_gather_rmt core %d peek status failed (memcpy error)" % i)
        if (v >> 32) != 1:
            raise RuntimeError("npu_dsa_gather_rmt core %d not finished (synchronize before result)" % i)
        rc = v & 0xFFFFFFFF
        if rc != 0:
            raise RuntimeError("npu_dsa_gather_rmt kernel failed (core %d) rc=0x%x" % (i, rc))


def npu_dsa_gather_rmt(resident_kv_cache, resident_rope_cache, resident_block_table,
                       host_block_table, request_pool_indices, topk_token_positions,
                       resident_slots, miss_mask, block_size, num_requests,
                       host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr, block_dim=1):
    """同步组合（eager 语义）：submit + synchronize + result；原地写，返回 (kv, rope) alias。"""
    h = npu_dsa_gather_rmt_submit(
        resident_kv_cache, resident_rope_cache, resident_block_table, host_block_table,
        request_pool_indices, topk_token_positions, resident_slots, miss_mask, block_size, num_requests,
        host_rmt_channel, host_rmt_kv_addr, host_rmt_rope_addr, block_dim=block_dim)
    torch.npu.synchronize()
    npu_dsa_gather_rmt_result(h)
    return resident_kv_cache, resident_rope_cache


def load_dsa_gather_rmt_op():
    """幂等挂载三件套到 torch_npu（对齐生产 custom_op_loader 属性挂载形态）。

    npu_dsa_gather_rmt（eager 同步）/ npu_dsa_gather_rmt_submit（图捕获区间纯提交）/
    npu_dsa_gather_rmt_result（图外错误码校验，调用方先 synchronize）地位对等——graph runner
    集成形态下 capture/replay 两侧调用方只 import torch_npu。返回 eager 主算子（兼容旧调用方）。
    """
    if not hasattr(torch_npu, "npu_dsa_gather_rmt"):
        torch_npu.npu_dsa_gather_rmt = npu_dsa_gather_rmt
    if not hasattr(torch_npu, "npu_dsa_gather_rmt_submit"):
        torch_npu.npu_dsa_gather_rmt_submit = npu_dsa_gather_rmt_submit
    if not hasattr(torch_npu, "npu_dsa_gather_rmt_result"):
        torch_npu.npu_dsa_gather_rmt_result = npu_dsa_gather_rmt_result
    return torch_npu.npu_dsa_gather_rmt
