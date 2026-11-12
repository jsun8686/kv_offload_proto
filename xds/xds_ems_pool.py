# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# EmsPool（替换原 mempoolhost.py）：对 xds_offload 提供初始化池 / 行分配 / 批量读写接口。
# 内部：EMS 控制面（建池+建链）+ kvops 批量算子（AIV 直驱，图捕获安全）。
#
# 行分配连续性契约：alloc 优先返回连续行区间（decode 追加行同段延伸），离散段 ≤ 4 兜底
# —— 内核侧 src 行号 = req_host_base[req] + tokenPos 的公式依赖此契约。

from typing import List, Optional

import torch

try:
    from ems_client import EmsClient
except ImportError:  # 本地离线单测（未安装扩展包，NPU 机 pip -e 安装后可用）
    EmsClient = None
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


def next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p <<= 1
    return p


def compute_ht_size(padded_buffer_size: int, top_k: int) -> int:
    """哈希表容量：2 的幂且 ≥ 2×(padded+topK)（装载因子 ≤0.5，与 sglang-0730 host 派生一致）"""
    p = 1024
    while p < 2 * (padded_buffer_size + top_k):
        p <<= 1
    return p


def align(v: int, a: int) -> int:
    return (v + a - 1) // a * a


class RowAllocator:
    """框架侧行号分配器：连续区间优先（连续性契约），碎片化时拼接离散行（≤4 段由调用方保证容量）。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._free = [(0, capacity)]
        self._used = 0

    def alloc(self, n: int) -> List[int]:
        if n <= 0:
            raise ValueError("n must be > 0")
        if n > self.free_rows():
            raise MemoryError("allocator exhausted: free={} need={}".format(self.free_rows(), n))
        for i, (s, e) in enumerate(self._free):
            if e - s >= n:
                del self._free[i]
                if e - s > n:
                    self._free.insert(i, (s + n, e))
                self._used += n
                return list(range(s, s + n))
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


class EmsPool:
    """EMS 远端内存池 + 批量读写算子门面（设计文档 §4.2.4 接口语义）。

    对 XDS 的唯一接口面：master 地址传入即可（EmsClient 为内部实现，外部不感知）。
    """

    def __init__(self, master_ip: str, master_port: int, device_id: int, layer_num: int, layer_rows: int,
                 row_bytes: int, stripe_rows: int = 1, acl_lib: str = "libascendcl.so",
                 hcomm_lib: str = "libhcomm.so", rootinfo_path: str = "/etc/hccl_rootinfo.json",
                 timeout_sec: float = 120.0):
        if layer_num < 1 or layer_rows < 1 or row_bytes <= 0 or row_bytes % 128 != 0:
            raise ValueError("invalid pool dims: L={} rows={} rowBytes={}".format(layer_num, layer_rows, row_bytes))
        if EmsClient is None:
            raise RuntimeError("ems_client package not installed (NPU host only)")
        self._ems = EmsClient(master_ip, master_port, timeout_sec)
        self.device_id = device_id
        self.rootinfo_path = rootinfo_path  # EID 自动：device→phyId→rootinfo 匹配（与 emsd 同链路）
        self.layer_num = layer_num
        self.layer_rows = layer_rows
        self.row_bytes = row_bytes
        self.stripe_rows = stripe_rows
        self.acl_lib = acl_lib
        self.hcomm_lib = hcomm_lib
        self.pool = None
        self.node_num = 0
        self.per_layer_seg_rows = 0
        self.channels = []
        self._status_base = 0
        self._status_count = 0
        self._status_cursor = 0

    # ── 生命周期 ────────────────────────────────────────────────────────────
    def create_pool(self, hbm_pool_base: int, hbm_pool_size: int, fatbin_path: str,
                    ems_size_bytes: Optional[int] = None) -> None:
        """EMS 建池（master 打散）→ NPU 侧注册 HBM 池 → 逐段 LINK_EXCHANGE + AIV 通道 → kvops.init。"""
        if _cli is None or _kvops is None:
            raise RuntimeError("kvops_ext / ems_client native ext not installed (NPU host only)")
        self._ems.connect()
        self._ems.ping()  # 建池前连通性自检（失败含 master 地址，便于定位）
        if ems_size_bytes is None:
            ems_size_bytes = self.layer_num * self.layer_rows * self.row_bytes
        self.pool = self._ems.pool_create(ems_size_bytes, self.layer_num, self.layer_rows, self.row_bytes,
                                          self.stripe_rows)
        self.node_num = self.pool.node_num
        self.per_layer_seg_rows = (self.layer_rows + self.node_num - 1) // self.node_num

        _cli.init(self.device_id, self.rootinfo_path, self.acl_lib, self.hcomm_lib)
        _cli.mem_reg(hbm_pool_base, hbm_pool_size, "kv_hbm")
        local_desc = _cli.local_desc()
        mem_desc = _cli.mem_desc()

        self.channels = []
        for seg in self.pool.segments:
            status = self._ems.link_exchange(seg, local_desc, mem_desc)
            if status != 0:
                raise RuntimeError("link_exchange failed on node {} {}: status={}".format(
                    seg.node_id, seg.ctrl_ip, status))
            remote_addr, _size, _t = _cli.mem_import(seg.mem_desc)
            if remote_addr != seg.seg_base:
                raise RuntimeError("seg base mismatch, node {}: ems={} import={}".format(
                    seg.node_id, seg.seg_base, remote_addr))
            # remoteEndpoint 必须是该资源节点（server 侧）的 endpoint 描述——
            # 传本端 local_desc 会让 client 向自己发起建链，channel 永远 READY 不了
            if not seg.ep_desc:
                raise RuntimeError("seg {} has no endpoint desc (emsd too old?): ctrl={}:{}".format(
                    seg.node_id, seg.ctrl_ip, seg.ctrl_port))
            handle, _st = _cli.link_channel(seg.ep_desc, seg.hcomm_listen_port, seg.channel_tag, 60)
            self.channels.append(handle)

        _kvops.init(self.device_id, fatbin_path)
        self._status_count = 8 * self.node_num
        self._status_base = _kvops.alloc_status(self._status_count)

    def next_status(self, slots: int = 1) -> int:
        # 每 block 占 64B 槽（与内核 KvcStatusInit 的 blockIdx*64、KvcStatusFlush 整页 64B 刷写一致）；
        # blockDim>1 时按方法预订 slots 槽（跨 op 复用，上 op 已 wait 完成后其槽可复用）。
        # 环回保护：剩余槽不足时回卷到 0——否则从 cursor 起连续 slots 会写出缓冲尾部
        # （实测症状：1 槽 op 后接 8 槽 op，memset 512B 越界 64B，driver 报 Invalid addr）。
        if self._status_cursor + slots > self._status_count:
            self._status_cursor = 0
        addr = self._status_base + self._status_cursor * 64
        self._status_cursor = (self._status_cursor + slots) % self._status_count
        return addr

    def destroy(self) -> None:
        if _cli is not None:
            for h in self.channels:
                _cli.channel_destroy(h)
            if self.pool is not None:
                _cli.close()
        if _kvops is not None:
            _kvops.close()
        if self.pool is not None:
            self._ems.pool_destroy(self.pool.pool_id)
            self.pool = None
        self._ems.close()

    def make_allocator(self) -> RowAllocator:
        return RowAllocator(self.layer_rows)

    # ── 批量读（decode miss 回取，稠密接口）：N 段 launch 共享 sieve 输出 ────
    def batch_read(self, layer_id: int, topk_indices, req_pool_indices, req_host_base, topk_device_slots, is_miss,
                   num_real, hot_row_base, max_num_reqs: int, top_k: int, stream: int = 0,
                   block_dim: int = 1) -> int:
        if self.pool is None:
            raise RuntimeError("pool not created")
        # stream=0（未指定）改用 torch_npu 当前流：与 torch 侧 fill_/copy_/to(dev) 同流有序
        # （对齐 sglang hisparse_lru_host.cpp 的 get_npu_stream；stream 0 上内核会读到 torch 写入前的旧值）
        if stream == 0:
            stream = torch.npu.current_stream().npu_stream
        status = self.next_status(block_dim)
        layer_offset = layer_id * self.per_layer_seg_rows
        for s in range(self.node_num):
            seg = self.pool.segments[s]
            _kvops.import_seg(self.channels[s], seg.seg_base, layer_offset, self.per_layer_seg_rows,
                              self.row_bytes, _ptr(topk_indices), _ptr(req_pool_indices), _ptr(req_host_base),
                              _ptr(topk_device_slots), _ptr(is_miss), _ptr(num_real), _ptr(hot_row_base), top_k,
                              max_num_reqs, self.stripe_rows, self.node_num, s, status, stream, block_dim)
        return status

    # ── 批量写（prefill 全层连续区间导出）───────────────────────────────────
    def batch_write_alllayers(self, row_lo: int, row_cnt: int, src_layer0_base, src_layer_stride: int, stream: int = 0,
                              block_dim: int = 1) -> int:
        if self.pool is None:
            raise RuntimeError("pool not created")
        if stream == 0:  # 同 batch_read：对齐 torch_npu 当前流
            stream = torch.npu.current_stream().npu_stream
        status = self.next_status(block_dim)
        for s in range(self.node_num):
            seg = self.pool.segments[s]
            _kvops.export_seg_alllayer(self.channels[s], seg.seg_base, self.per_layer_seg_rows, self.row_bytes,
                                       row_lo, row_cnt, self.layer_num, _ptr(src_layer0_base), src_layer_stride,
                                       self.stripe_rows, self.node_num, s, status, stream, block_dim)
        return status

    # ── 完成等待：聚合 N 段状态槽 ────────────────────────────────────────────
    # ==== V1.1/V2：full 池两区导出（每段 kv 大区 + rope 大区）并生成 full_rmt_* 三表 ====
    # 段内布局：[kv 大区: per_node 块 × BS*kvDim fp16][rope 大区: per_node 块 × BS*ropeDim fp16]（满容量对齐，尾段留洞）
    # 行交错分布（kv_layout 语义，stripe_rows=1）：全局行 r → 段 r%n、段内行 r//n，
    # 段 s 的段内连续区 = 全局行 {t*n+s}。导出构造"全局交错源"（[段][段内行] 转置为 [段内行][段]），
    # 一次全区 batch_write 由内核 KvcNodeOf 自动分发到各段；段内两区连续故表寻址为段内字节偏移。
    # 内核寻址约定（gather_selection_rmt）：块内 kv 连续 / 块内 rope 连续，addr 表给每块基址。
    # perm[q] = 块 q 的全局槽位（None 恒等：段 q//per_node、槽 q%per_node）。
    def export_full_rmt_pool(self, full_kv, full_rope, block_size: int, kv_dim: int, rope_dim: int,
                             perm=None, stream: int = 0):
        if self.pool is None or not self.channels:
            raise RuntimeError("pool not created")
        n = self.node_num
        f_blk = full_kv.shape[0]
        if full_kv.shape != (f_blk, block_size, kv_dim) or full_rope.shape != (f_blk, block_size, rope_dim):
            raise ValueError("pool/source shape mismatch: kv={} rope={}".format(full_kv.shape, full_rope.shape))
        if self.row_bytes != kv_dim * 2:
            raise ValueError("pool row_bytes {} != kv_dim*2 {}".format(self.row_bytes, kv_dim * 2))
        if block_size * rope_dim * 2 % self.row_bytes:
            raise ValueError("rope block bytes not multiple of row_bytes %d" % self.row_bytes)
        if self.stripe_rows != 1:
            raise ValueError("export layout requires stripe_rows=1 (row interleave), got %d" % self.stripe_rows)

        per_node = (f_blk + n - 1) // n
        seg_rows = per_node * block_size * (kv_dim + rope_dim) // kv_dim  # 每段满容量行数
        if self.layer_rows < n * seg_rows:
            raise ValueError("pool rows {} < {}x{} (per-node seg rows)".format(self.layer_rows, n, seg_rows))

        dev = full_kv.device
        if perm is None:
            gslot = torch.arange(f_blk, dtype=torch.int64, device=dev)
        else:
            gslot = perm.to(torch.int64).to(dev)
            if gslot.shape[0] != f_blk or sorted(gslot.tolist()) != list(range(f_blk)):
                raise ValueError("perm must be a permutation of [0, f_blk)")
        route = gslot // per_node   # 块 q 所在段
        slot = gslot % per_node     # 段内槽位

        # 段内容：[n, per_node*BS, D]（kv/rope 各一份，槽序连续；尾段不满留 0）
        seg_kv = torch.zeros((n, per_node * block_size, kv_dim), dtype=full_kv.dtype, device=dev)
        seg_rope = torch.zeros((n, per_node * block_size, rope_dim), dtype=full_rope.dtype, device=dev)
        for s in range(n):
            mask = (route == s)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            idx = torch.nonzero(mask).flatten()   # 块 q 列表
            order = torch.argsort(slot[mask])     # 段内槽序 → 连续源重排
            seg_kv[s, :cnt * block_size] = full_kv[idx[order]].reshape(cnt * block_size, kv_dim)
            seg_rope[s, :cnt * block_size] = full_rope[idx[order]].reshape(cnt * block_size, rope_dim)

        # 段内行流：[kv 区 per_node*BS 行（每行 kv_dim 元素=整行）][rope 区字节流按行切]
        row_elems = self.row_bytes // 2
        kv_rows_full = per_node * block_size
        rope_rows_full = per_node * block_size * rope_dim * 2 // self.row_bytes
        seg_stream = torch.zeros((n, seg_rows, row_elems), dtype=full_kv.dtype, device=dev)
        seg_stream[:, :kv_rows_full, :kv_dim] = seg_kv
        seg_stream[:, kv_rows_full:, :] = seg_rope.reshape(n, rope_rows_full, row_elems)
        # 全局交错源：[段][段内行] → 转置 [段内行][段] → 展平（全局行 x = t*n + s）
        gsrc = seg_stream.transpose(0, 1).reshape(n * seg_rows, row_elems).contiguous()
        st = self.batch_write_alllayers(0, n * seg_rows, gsrc.data_ptr(), 0, stream=stream)
        self.wait(st)

        # 三表：段内字节偏移（kv 区满容量偏移对齐 rope 区，与段内容构造一致）
        blk_kv_bytes = block_size * kv_dim * 2
        blk_rope_bytes = block_size * rope_dim * 2
        rope_region_off = kv_rows_full * self.row_bytes
        seg_base = torch.tensor([s.seg_base for s in self.pool.segments[:n]], dtype=torch.int64, device=dev)
        chn = torch.tensor([int(c) for c in self.channels[:n]], dtype=torch.int64, device=dev)
        rmt_channel = chn[route]
        rmt_kv_addr = seg_base[route] + slot * blk_kv_bytes
        rmt_rope_addr = seg_base[route] + rope_region_off + slot * blk_rope_bytes
        return rmt_channel, rmt_kv_addr, rmt_rope_addr

    def wait(self, status: int, timeout_us: int = 60_000_000, block_dim: int = 1) -> None:
        if _kvops is None:
            raise RuntimeError("kvops ext missing")
        for s in range(self.node_num):
            for b in range(block_dim):
                slot = status + (s * block_dim + b) * 64
                rc = _kvops.poll_status(slot, timeout_us)
                if rc != 0:
                    # 负哨兵为 poll 自身失败（与内核 errCode 分开报，避免误读为内核错误）
                    if rc == -0x1000:
                        raise RuntimeError(
                            "kv op poll memcpy failed on node {} block {}".format(s, b))
                    if rc == -0x2000:
                        # 超时兜底诊断：读回整个 64B 槽定位内核卡在哪一步。
                        # word0 = 阶段码（0=未初始化 / 0xE2=INIT_OK / 0xE3=QUEUE_OK /
                        # 0xE4=COMMIT_OK）或终态字；(word0>>32)==1 即其实已完成、
                        # 是轮询侧没看到；word1 = gather 的搬运行数诊断。
                        words = [_kvops.peek_status(slot + i * 8) for i in range(4)]
                        raise RuntimeError(
                            "kv op poll timeout on node {} block {} (kernel doneSeq not set in {:.0f}s, "
                            "slot=[0x{:016x} 0x{:016x} 0x{:016x} 0x{:016x}])".format(
                                s, b, timeout_us / 1e6, words[0], words[1], words[2], words[3]))
                    raise RuntimeError("kv op failed on node {} block {}: errCode=0x{:x}".format(
                        s, b, rc & 0xFFFFFFFF))
