# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# XdsOffloadCoordinator：XDS 侧 offload 协调器（参考 sglang-0730 hisparse_coordinator.py，
# 按本原型数据结构裁剪：SIEVE 哈希-only + visited 位图 + 热缓冲槽固定偏移契约）。
#
# 职责：sieve 状态张量（按层持久）与输出张量管理、请求 admission（恒等布局建表）、
# decode 每步序列（sieve_update → 由上层发起 batch_read）、miss 统计。
# 全程 device 张量操作，无 .cpu()/.item()（热路径零同步，图捕获安全）。

import torch

from .xds_ems_pool import align, compute_ht_size

try:
    import kvops_ext._kvops as _kvops
except ImportError:
    _kvops = None


def _npu_stream() -> int:
    """torch 当前流句柄（与 xds_ems_pool._npu_stream 同契约）：
    kvops 内核 launch 到 torch 流上，与 torch 写入/回读同流有序。"""
    try:
        return int(torch.npu.current_stream().npu_stream)
    except (AttributeError, RuntimeError):  # 离线单测：无 torch_npu
        return 0


class XdsOffloadCoordinator:
    def __init__(self, device: str, layer_num: int, pool_size: int, device_buffer_size: int, top_k: int,
                 max_num_reqs: int, max_context_len: int):
        if _kvops is None:
            raise RuntimeError("kvops_ext not installed (NPU host only)")
        self.device = device
        self.layer_num = layer_num
        self.pool_size = pool_size
        self.device_buffer_size = device_buffer_size
        self.top_k = top_k
        self.max_num_reqs = max_num_reqs
        self.max_context_len = max_context_len

        # padded：热缓冲行数上取 16 对齐（64B 行 + uint16 槽号上限 65534 由 kvops 校验）
        self.padded_buffer_size = align(device_buffer_size, 16)
        # visited 位图：ceil(padded/8) 上取 64B 对齐（cache line + DataCopy 粒度）
        self.visited_stride_b = align((self.padded_buffer_size + 7) // 8, 64)
        self.ht_size = compute_ht_size(self.padded_buffer_size, top_k)

        p = self.pool_size
        padded = self.padded_buffer_size
        L = self.layer_num
        # 按层持久状态（HBM 预算见设计文档 §4.1 表）
        self.device_buffer_tokens = torch.full((L, p, padded), -1, dtype=torch.int32, device=device)
        self.device_buffer_visited = torch.zeros((L, p, self.visited_stride_b), dtype=torch.uint8, device=device)
        self.device_buffer_ht = torch.zeros((L, p, self.ht_size), dtype=torch.int16, device=device)
        self.sieve_hand = torch.zeros((L, p, 16), dtype=torch.int32, device=device)
        # 请求级元数据
        self.prefill_lens = torch.zeros(p, dtype=torch.int32, device=device)
        self.req_host_base = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        # decode 步输入/输出（静态缓冲，图捕获形态）
        self.top_k_device_slots = torch.zeros((max_num_reqs, top_k), dtype=torch.int32, device=device)
        self.is_miss = torch.zeros((max_num_reqs, top_k), dtype=torch.int8, device=device)
        self.num_real = torch.zeros(1, dtype=torch.int32, device=device)

    # ── admission：请求进入 decode（prefill KV 已全层导出到远端） ────────────
    def admit_request(self, req_batch_idx: int, req_pool_idx: int, prefill_len: int, host_base_row: int) -> None:
        """恒等布局建热缓冲：槽 s = token s（sink/近期上下文开局即命中）；建全层哈希表。

        req_batch_idx：批内槽位（req_host_base 按批行索引，gather 内核契约 hostBasePtr[bid]）。
        """
        if req_pool_idx < 0 or req_pool_idx >= self.pool_size:
            raise ValueError("req_pool_idx out of range: {}".format(req_pool_idx))
        if req_batch_idx < 0 or req_batch_idx >= self.max_num_reqs:
            raise ValueError("req_batch_idx out of range: {}".format(req_batch_idx))
        init_n = min(prefill_len, self.device_buffer_size)
        # token 行种成恒等布局（剩余槽保持 -1）
        self.device_buffer_tokens[:, req_pool_idx, :init_n] = \
            torch.arange(init_n, dtype=torch.int32, device=self.device)
        self.device_buffer_visited[:, req_pool_idx, :].zero_()
        self.sieve_hand[:, req_pool_idx, :].zero_()
        self.prefill_lens[req_pool_idx] = prefill_len
        self.req_host_base[req_batch_idx] = host_base_row
        # 全层哈希表构建（blockDim = layerNum × 1）；launch 到 torch 当前流，
        # 与上方播种/prefill_lens 的 torch 写入同流有序
        _kvops.sieve_ht_init(self.device_buffer_tokens.data_ptr(), self.device_buffer_ht.data_ptr(),
                             self.prefill_lens.data_ptr(), req_pool_idx, 1, self.layer_num, self.pool_size,
                             self.device_buffer_size, self.padded_buffer_size, self.ht_size, _npu_stream(),
                             self.layer_num)

    def set_batch(self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor) -> None:
        """每 decode 步前设置批上下文（eager；图模式由调用方 fill_ num_real）。"""
        if req_pool_indices.dtype != torch.int64:
            raise ValueError("req_pool_indices must be int64")
        if seq_lens.dtype != torch.int32:
            seq_lens = seq_lens.to(torch.int32)
        self._req_pool_indices = req_pool_indices
        self._seq_lens = seq_lens
        self.num_real[0] = req_pool_indices.shape[0]

    # ── decode 每步：SIEVE miss 识别/淘汰更新（本层切片） ─────────────────────
    def sieve_update(self, layer_id: int, topk_indices: torch.Tensor, block_dim: int = 8) -> None:
        if topk_indices.dtype != torch.int32:
            raise ValueError("topk_indices must be int32")
        _kvops.sieve_update(topk_indices.data_ptr(), self._req_pool_indices.data_ptr(),
                            self._seq_lens.data_ptr(), self.prefill_lens.data_ptr(),
                            self.device_buffer_tokens[layer_id].data_ptr(),
                            self.device_buffer_visited[layer_id].data_ptr(),
                            self.device_buffer_ht[layer_id].data_ptr(),
                            self.sieve_hand[layer_id].data_ptr(), self.top_k_device_slots.data_ptr(),
                            self.is_miss.data_ptr(), self.num_real.data_ptr(), self.top_k,
                            self.device_buffer_size, self.padded_buffer_size, self.visited_stride_b,
                            self.max_num_reqs, self.pool_size, _npu_stream(), block_dim)

    def miss_count(self) -> torch.Tensor:
        """device 侧 miss 统计张量（padding 行 is_miss==0，无需 num_real 切片；
        上层按步累积、定期 flush 时才 host 同步）"""
        return self.is_miss.sum()
