# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# MemPoolHost：XDS 的 KV cache 卸载适配层（单文件，设计文档 4.2.4）。
# 行号分配（free-list）+ 批量导出/导入（AIV 直驱，图捕获安全：launch 全 device 参数）。
#
# kvops/_cli 扩展缺失时仅 launch 类方法不可用（协议与分配逻辑可离线单测）。

from typing import List, Optional

from . import kv_layout

try:
    import kvops_ext._kvops as _kvops
except ImportError:  # 本地离线单测（无 NPU 环境）
    _kvops = None
try:
    import ems_client._cli as _cli
except ImportError:
    _cli = None


def _ptr(x) -> int:
    """tensor（含 data_ptr）或已是指针 int → device 地址"""
    if isinstance(x, int):
        return x
    dp = getattr(x, "data_ptr", None)
    if dp is None:
        raise TypeError("expect tensor or int address, got {}".format(type(x)))
    return dp()


class RowAllocator:
    """框架侧行号分配器：free-list（区间），alloc 优先返回连续区间（利于导出条带聚合）。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._free = [(0, capacity)]  # [start, end) 区间表（升序）
        self._used = 0

    def alloc(self, n: int) -> List[int]:
        if n <= 0:
            raise ValueError("n must be > 0")
        if n > self.free_rows():
            raise MemoryError("allocator exhausted: free={} need={}".format(self.free_rows(), n))
        # 首次适应：找第一个能整放的区间
        for i, (s, e) in enumerate(self._free):
            if e - s >= n:
                del self._free[i]
                if e - s > n:
                    self._free.insert(i, (s + n, e))
                self._used += n
                return list(range(s, s + n))
        # 无整块连续：拼接离散行（碎片化后）
        rows: List[int] = []
        while len(rows) < n:
            s, e = self._free.pop(0)
            take = min(e - s, n - len(rows))
            rows.extend(range(s, s + take))
            if e - s > take:
                self._free.insert(0, (s + take, e))
        self._free.sort()
        self._used += n
        return rows

    def free(self, rows: List[int]) -> None:
        if not rows:
            return
        for r in rows:
            if r < 0 or r >= self.capacity:
                raise ValueError("row out of range: {}".format(r))
        self._free.extend([(r, r + 1) for r in rows])
        self._free.sort()
        # 相邻区间合并
        merged = []
        for s, e in self._free:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        self._free = merged
        self._used -= len(rows)

    def free_rows(self) -> int:
        return sum(e - s for s, e in self._free)

    def used_rows(self) -> int:
        return self._used


class MemPoolHost:
    """对 XDS 提供创建内存池/批量导出/批量导入（设计文档 4.2.4）。"""

    def __init__(self, ems, device_id: int, eid_hex: str, layer_num: int, layer_rows: int, row_bytes: int,
                 stripe_rows: int = 1, max_batch: int = 4096, acl_lib: str = "libascendcl.so",
                 hcomm_lib: str = "libhcomm.so"):
        kv_layout.validate(layer_num, layer_rows, row_bytes, stripe_rows, 64)  # 上限校验（N 由建池确定）
        self.ems = ems
        self.device_id = device_id
        self.eid_hex = eid_hex
        self.layer_num = layer_num
        self.layer_rows = layer_rows
        self.row_bytes = row_bytes
        self.stripe_rows = stripe_rows
        self.max_batch = max_batch
        self.acl_lib = acl_lib
        self.hcomm_lib = hcomm_lib
        self.pool = None
        self.node_num = 0
        self.per_layer_seg_rows = 0
        self.channels = []          # 每段 ChannelHandle（u64）
        self._hbm_reg = False
        self._status_base = 0
        self._status_count = 0
        self._status_cursor = 0

    # ── 生命周期 ────────────────────────────────────────────────────────────
    def create_pool(self, hbm_pool_base: int, hbm_pool_size: int, ems_size_bytes: Optional[int] = None) -> None:
        """EMS 建池（master 打散）→ NPU 侧注册 HBM 池 → 逐段 LINK_EXCHANGE + AIV client 通道。"""
        if _cli is None or _kvops is None:
            raise RuntimeError("kvops_ext / ems_client native ext not installed (NPU host only)")
        kv_layout.validate(self.layer_num, self.layer_rows, self.row_bytes, self.stripe_rows, 1)
        if ems_size_bytes is None:
            ems_size_bytes = self.layer_num * self.layer_rows * self.row_bytes
        self.pool = self.ems.pool_create(ems_size_bytes, self.layer_num, self.layer_rows, self.row_bytes,
                                         self.stripe_rows)
        self.node_num = self.pool.node_num
        self.per_layer_seg_rows = kv_layout.per_layer_seg_rows(self.layer_rows, self.node_num)

        _cli.init(self.device_id, self.eid_hex, self.acl_lib, self.hcomm_lib)
        _cli.mem_reg(hbm_pool_base, hbm_pool_size, "kv_hbm")
        self._hbm_reg = True
        local_desc = _cli.local_desc()
        mem_desc = _cli.mem_desc()

        self.channels = []
        for seg in self.pool.segments:
            status = self.ems.link_exchange(seg, local_desc, mem_desc)
            if status != 0:
                raise RuntimeError("link_exchange failed on node {} seg {}: status={}".format(
                    seg.node_id, seg.ctrl_ip, status))
            # 对端（远端段）内存导入本端（校验地址一致性）
            remote_addr, remote_size, _t = _cli.mem_import(seg.mem_desc)
            if remote_addr != seg.seg_base:
                raise RuntimeError("seg base mismatch, node {}: ems={} import={}".format(
                    seg.node_id, seg.seg_base, remote_addr))
            handle, _st = _cli.link_channel(local_desc, seg.hcomm_listen_port, seg.channel_tag, 60)
            self.channels.append(handle)

        self._status_count = 8 * self.node_num
        self._status_base = _kvops.alloc_status(self._status_count)

    def next_status(self) -> int:
        addr = self._status_base + self._status_cursor * 64
        self._status_cursor = (self._status_cursor + 1) % self._status_count
        return addr

    def destroy(self) -> None:
        if _cli is not None:
            for h in self.channels:
                _cli.channel_destroy(h)
            if self._hbm_reg:
                _cli.close()
        if self.pool is not None:
            self.ems.pool_destroy(self.pool.pool_id)
            self.pool = None

    # ── 行分配 ──────────────────────────────────────────────────────────────
    def make_allocator(self) -> RowAllocator:
        return RowAllocator(self.layer_rows)

    # ── 批量导入（decode miss 回取）：N 次 launch，共享 indices ─────────────
    def batch_import(self, layer_id: int, indices_host, indices_dst, num_real, dst_layer_base, stream: int = 0,
                     block_dim: int = 1) -> int:
        if self.pool is None:
            raise RuntimeError("pool not created")
        status = self.next_status()
        for s in range(self.node_num):
            seg = self.pool.segments[s]
            _kvops.import_seg(self.channels[s], seg.seg_base, layer_id, self.per_layer_seg_rows, self.row_bytes,
                              _ptr(indices_host), _ptr(indices_dst), _ptr(num_real), _ptr(dst_layer_base),
                              self.max_batch, self.stripe_rows, self.node_num, s, status + s * 64, stream, block_dim)
        return status

    # ── 批量导出（prefill 全层连续区间）──────────────────────────────────────
    def batch_export_alllayers(self, row_lo: int, row_cnt: int, src_layer0_base, src_layer_stride: int, stream: int = 0,
                               block_dim: int = 1) -> int:
        if self.pool is None:
            raise RuntimeError("pool not created")
        status = self.next_status()
        for s in range(self.node_num):
            seg = self.pool.segments[s]
            _kvops.export_seg_alllayer(self.channels[s], seg.seg_base, self.per_layer_seg_rows, self.row_bytes,
                                       row_lo, row_cnt, self.layer_num, _ptr(src_layer0_base), src_layer_stride,
                                       self.stripe_rows, self.node_num, s, status + s * 64, stream, block_dim)
        return status

    # ── 批量导出（decode 增量，单层索引）────────────────────────────────────
    def batch_export_rows(self, layer_id: int, indices_host, indices_dst, num_real, src_layer_base, stream: int = 0,
                          block_dim: int = 1) -> int:
        if self.pool is None:
            raise RuntimeError("pool not created")
        status = self.next_status()
        for s in range(self.node_num):
            seg = self.pool.segments[s]
            _kvops.export_rows(self.channels[s], seg.seg_base, layer_id, self.per_layer_seg_rows, self.row_bytes,
                               _ptr(indices_host), _ptr(indices_dst), _ptr(num_real), _ptr(src_layer_base),
                               self.max_batch, self.stripe_rows, self.node_num, s, status + s * 64, stream, block_dim)
        return status

    # ── 完成等待：聚合 N 段状态槽（doneSeq==1 且 errCode==0）────────────────
    def wait(self, status: int, timeout_us: int = 60_000_000) -> None:
        if _kvops is None:
            raise RuntimeError("kvops ext missing")
        for s in range(self.node_num):
            rc = _kvops.poll_status(status + s * 64, timeout_us)
            if rc != 0:
                raise RuntimeError("kv op failed on node {}: errCode=0x{:x}".format(s, rc & 0xFFFFFFFF))
