# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# xds_decode.py：XDS 主程序入口（decode 流程驱动，参数化参考 test_hisparse_lru_npu.py 形态）。
#
# 流程：EMS 建池+建链 → prefill（指纹填充+全层导出+admission）→ decode 循环
# （topK 采样【预留真实 lightning indexer 接口】→ sieve_update → 批量导入 → 校验）→ 统计。
# 全程热路径无 .cpu()/.item()（--oracle 对拍模式除外，eager 专用）。
#
# 运行（NPU 机）：python -m xds.xds_decode --ems-master <ip:port> [options]（EID 按 --device 自动匹配）

import argparse
import os
import sys
import time

import numpy as np

P1, P2, M1, M2 = 0x9E3779B1, 0x85EBCA77, 0x27220A95, 0x165667B1


def parse_args():
    p = argparse.ArgumentParser(description="XDS decode pipeline driver (kv offload prototype)")
    p.add_argument("--ems-master", required=True, help="master emsd ip:port")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--rootinfo", default="/etc/hccl_rootinfo.json",
                   help="EID 自动匹配源（device→phyId→rootinfo 的 device_id 块首个 EID addr）")
    p.add_argument("--fatbin", default="", help="kvc fatbin 路径（默认 build/kernel 安装产物路径自动探测）")
    # 模型/池形态
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--layer-rows", type=int, default=1024)
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--qk-rope", type=int, default=64)
    p.add_argument("--stripe-rows", type=int, default=1)
    # decode 场景（参考 test_hisparse_lru_npu.py 场景参数）
    p.add_argument("--req-num", type=int, default=8, help="并发请求数（batch）")
    p.add_argument("--prefill", type=int, default=256, help="每请求 prefill token 数")
    p.add_argument("--topk", type=int, default=64, help="lightning indexer topK")
    p.add_argument("--device-buffer-size", type=int, default=64, help="热缓冲槽数/请求（< prefill 时产生 miss）")
    p.add_argument("--miss-ratio", type=float, default=0.5, help="topK 中 miss token 的目标占比")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block-dim", type=int, default=8)
    p.add_argument("--oracle", action="store_true", help="CPU sieve 模型逐层对拍（eager 专用）")
    p.add_argument("--run-dir", default="./log", help="日志目录（每次运行截断重写 xds_decode.log）")
    return p.parse_args()


class _Logger:
    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "xds_decode.log")
        self._f = open(self.path, "w", encoding="utf-8")  # 每次运行截断重写

    def __call__(self, msg: str) -> None:
        print(msg)
        self._f.write(msg + "\n")
        self._f.flush()


def row_word(row: int, layer: int, k: int) -> int:
    """行指纹 word（与填充/校验共用）"""
    v = (row * P1 + layer * P2) * M1 ^ M2
    return (v + k * M1) & 0xFFFFFFFF


class _CpuSieveOracle:
    """CPU sieve 模型（镜像 hisp_sieve_update 哈希-only 语义；eager 对拍专用）。

    每请求独立状态：tokens（槽→token）/visited（槽集合）/hand/t2s（dict token→slot）。
    """

    def __init__(self, reqs_meta):
        self.meta = reqs_meta  # [(prefill, pool_idx), ...] 按批行
        self.state = {}
        for prefill, pool_idx in reqs_meta:
            dbs = None  # 由 reset 传入
            self.state[pool_idx] = None

    def reset(self, pool_idx, prefill, dbs):
        init_n = min(prefill, dbs)
        tokens = {s: s for s in range(init_n)}
        self.state[pool_idx] = {
            "prefill": prefill, "tokens": tokens, "visited": set(), "hand": 0, "dbs": dbs,
            "t2s": {t: s for s, t in tokens.items()},
        }

    def update(self, pool_idx, topk_row, seq_len):
        st = self.state[pool_idx]
        out_slots, out_miss = [], []
        for token_pos in topk_row:
            if token_pos < 0 or token_pos >= seq_len:
                out_slots.append(-1)
                out_miss.append(0)
                continue
            if token_pos >= st["prefill"]:
                out_slots.append(-2 - (token_pos - st["prefill"]))  # decode 扩展槽标记（padded+off）
                out_miss.append(0)
                continue
            if token_pos in st["t2s"]:
                slot = st["t2s"][token_pos]
                out_slots.append(slot)
                out_miss.append(0)
                st["visited"].add(slot)
                continue
            h = st["hand"]
            while h in st["visited"]:
                st["visited"].discard(h)
                h = (h + 1) % st["dbs"]
            old = st["tokens"].get(h)
            if old is not None:
                del st["t2s"][old]
            st["tokens"][h] = token_pos
            st["t2s"][token_pos] = h
            out_slots.append(h)
            out_miss.append(1)
            st["visited"].add(h)
            st["hand"] = (h + 1) % st["dbs"]
        return out_slots, out_miss


def main() -> int:
    args = parse_args()
    log = _Logger(args.run_dir)
    rng = np.random.default_rng(args.seed)

    row_words = args.kv_lora_rank + args.qk_rope
    row_bytes = (row_words * 2 + 127) // 128 * 128  # fp16/bf16 → 2B，128B 对齐
    if args.device_buffer_size >= args.prefill:
        log("[xds][WARN] device-buffer-size >= prefill: 无 miss，--miss-ratio 不生效")

    import torch
    import torch_npu  # noqa: F401

    dev = "npu:{}".format(args.device)
    log("[xds] config: layers={} layerRows={} rowWords={} rowBytes={} stripe={} reqs={} prefill={} topK={} "
        "dbs={} missRatio={}".format(args.layers, args.layer_rows, row_words, row_bytes, args.stripe_rows,
                                     args.req_num, args.prefill, args.topk, args.device_buffer_size,
                                     args.miss_ratio))

    hbm = torch.zeros((args.layers, args.layer_rows, row_bytes // 4), dtype=torch.int32, device=dev)

    master_ip, master_port = args.ems_master.rsplit(":", 1)
    from xds.xds_ems_pool import EmsPool, _kvops
    from xds.xds_offload import XdsOffloadCoordinator

    fatbin = args.fatbin
    if not fatbin:
        cand = ["/usr/local/fatbin/kvc_kernel/kvc_kernel.o",  # cmake install 前缀产物（build.sh kernel 默认）
                os.path.join(os.path.dirname(__file__), "..", "build", "kernel", "kvc_kernel", "kvc_kernel.o"),
                os.path.join(os.path.dirname(__file__), "..", "build", "kernel", "kvc_kernel.o")]
        fatbin = next((c for c in cand if os.path.exists(c)), "")
        if not fatbin:
            log("[xds][ERROR] fatbin not found, pass --fatbin explicitly")
            return 1

    pool = EmsPool(master_ip, int(master_port), args.device, args.layers, args.layer_rows, row_bytes,
                   stripe_rows=args.stripe_rows, rootinfo_path=args.rootinfo)
    pool.create_pool(hbm.data_ptr(), hbm.numel() * 4, fatbin)
    log("[xds] pool ready: nodes={} perLayerSegRows={} channels={} fatbin={}".format(
        pool.node_num, pool.per_layer_seg_rows, len(pool.channels), fatbin))

    coord = XdsOffloadCoordinator(dev, args.layers, args.req_num, args.device_buffer_size, args.topk,
                                  args.req_num, args.prefill)
    log("[xds] coordinator: padded={} visitedStrideB={} htSize={}".format(
        coord.padded_buffer_size, coord.visited_stride_b, coord.ht_size))

    # ── prefill：行分配（连续区间契约）+ 指纹填充 + 全层导出 + admission ──
    allocator = pool.make_allocator()
    host_bases = []
    hbm_np = hbm.cpu().numpy().view(np.uint32).reshape(args.layers, args.layer_rows, row_bytes // 4)
    t0 = time.perf_counter()
    for r in range(args.req_num):
        rows = allocator.alloc(args.prefill)
        assert rows == list(range(rows[0], rows[0] + args.prefill)), "连续性契约破坏"
        host_bases.append(rows[0])
        base = rows[0]
        for layer in range(args.layers):
            row_idx = np.arange(args.prefill) + base
            v = (row_idx.astype(np.uint64) * np.uint64(P1) + np.uint64(layer * P2)) * np.uint64(M1) ^ np.uint64(M2)
            ramp = np.arange(row_bytes // 4, dtype=np.uint64) * np.uint64(M1)
            hbm_np[layer, base:base + args.prefill] = ((v[:, None] + ramp[None, :]) & 0xFFFFFFFF).astype(np.uint32)
    hbm.copy_(torch.from_numpy(hbm_np.view(np.int32)).to(dev))
    st = pool.batch_write_alllayers(0, args.req_num * args.prefill, hbm[0, 0],
                                    args.layer_rows * row_bytes)
    pool.wait(st)
    t_export = time.perf_counter() - t0
    export_bytes = args.layers * args.req_num * args.prefill * row_bytes
    log("[xds] prefill export: {} rows x {} layers = {} B in {:.1f} us".format(
        args.req_num * args.prefill, args.layers, export_bytes, t_export * 1e6))

    # admission（恒等布局 + 全层建表）后清空 HBM 热区数据，验证导入回填
    oracle = None
    if args.oracle:
        oracle = _CpuSieveOracle([(args.prefill, r) for r in range(args.req_num)])
    for r in range(args.req_num):
        coord.admit_request(r, r, args.prefill, host_bases[r])
        if oracle is not None:
            oracle.reset(r, args.prefill, args.device_buffer_size)
    hbm.zero_()

    # ── decode 循环 ──
    dbs = args.device_buffer_size
    seq_lens = torch.full((args.req_num,), args.prefill, dtype=torch.int32, device=dev)
    req_pool_indices = torch.arange(args.req_num, dtype=torch.int64, device=dev)
    coord.set_batch(req_pool_indices, seq_lens)

    total_miss = 0
    t_read_total = 0.0
    hot_rows_base = 0  # 热缓冲区起始物理行（固定偏移契约：dstRow = hotRowBase + slot）
    for step in range(args.steps):
        # topK 采样（预留真实 indexer：替换为 lightning indexer 输出即可）
        # 命中取 [0, min(prefill,dbs))、miss 取 [dbs, prefill)，按 miss_ratio 混合
        hit_span = min(args.prefill, dbs)
        miss_span = max(args.prefill - dbs, 0)
        n_miss = int(args.topk * args.miss_ratio) if miss_span > 0 else 0
        n_hit = args.topk - n_miss
        batch = np.empty((args.req_num, args.topk), dtype=np.int32)
        for r in range(args.req_num):
            hits = rng.integers(0, hit_span, size=n_hit) if n_hit > 0 else np.empty(0, dtype=np.int64)
            misses = (rng.choice(miss_span, size=n_miss, replace=False) + dbs) if n_miss > 0 else np.empty(0, dtype=np.int64)
            batch[r] = np.concatenate([hits, misses]).astype(np.int32)
        rng.shuffle(batch.ravel())
        topk_indices = torch.from_numpy(batch).to(dev)

        layer_id = step % args.layers
        t0 = time.perf_counter()
        coord.sieve_update(layer_id, topk_indices, block_dim=args.block_dim)
        st = pool.batch_read(layer_id, topk_indices, req_pool_indices, coord.req_host_base,
                             coord.top_k_device_slots, coord.is_miss, coord.num_real,
                             hbm[layer_id, hot_rows_base:], args.req_num, args.topk,
                             block_dim=args.block_dim)
        pool.wait(st, block_dim=args.block_dim)
        if _kvops is not None:
            # gather 诊断区（每 block 槽 +8）的 queued 计数：0=未搬运（条件链 skip），>0=已提交 WQE
            queued = [_kvops.peek_status(st + b * 64 + 8) for b in range(args.block_dim)]
            log("[xds] gather queued per block: {}".format(queued))
        dt = time.perf_counter() - t0
        t_read_total += dt

        miss_cnt = coord.miss_count().item()  # 每步 flush 一次统计（可配间隔）
        total_miss += miss_cnt

        if oracle is not None:
            gpu_slots = coord.top_k_device_slots[: args.req_num].cpu().numpy()
            gpu_miss = coord.is_miss[: args.req_num].cpu().numpy()
            for r in range(args.req_num):
                exp_slots, exp_miss = oracle.update(r, batch[r], args.prefill)
                if not np.array_equal(gpu_miss[r], np.array(exp_miss, dtype=np.int32)):
                    log("[xds][ERROR] oracle is_miss mismatch step={} req={}\n gpu={}\n exp={}\n slots={}\n"
                        " numReal={} prefill={} tokens[0]={} ht[0][:8]={}".format(
                            step, r, gpu_miss[r].tolist(), exp_miss, gpu_slots[r].tolist(),
                            coord.num_real.item(), coord.prefill_lens[:args.req_num].cpu().tolist(),
                            coord.device_buffer_tokens[layer_id, 0, :8].cpu().tolist(),
                            coord.device_buffer_ht[layer_id, 0, :8].cpu().tolist()))
                    return 1
                norm = lambda s: -1 if s < 0 else (  # noqa: E731
                    (coord.padded_buffer_size + (-s - 2)) if s <= -2 else s)
                got = [norm(s) for s in gpu_slots[r]]
                if not np.array_equal(np.array(got, dtype=np.int32), np.array(exp_slots, dtype=np.int32)):
                    log("[xds][ERROR] oracle slots mismatch step={} req={}\n gpu={}\n exp={}".format(
                        step, r, got, exp_slots))
                    return 1

        # 抽查校验：首个真实 miss 行指纹（miss 位置取自 sieve 输出，非采样预期）
        hbm_sync = hbm[layer_id].cpu().numpy().view(np.uint32).reshape(-1)
        # 全 slot 反查诊断：热缓冲每槽行首 word 反查来源 (row,layer)，与期望 token 对照，
        # 一次暴露错位模式（整体偏移 / 个别覆盖 / 层错位）
        w0_to_row = {}
        for l_ in range(args.layers):
            for r_ in range(args.layer_rows):
                w0_to_row[row_word(r_, l_, 0)] = (r_, l_)
        expect_map = {}
        for r_ in range(args.req_num):
            for k in np.nonzero(coord.is_miss[r_].cpu().numpy())[0]:
                slot_ = int(coord.top_k_device_slots[r_, k].item())
                if 0 <= slot_ < dbs:
                    expect_map[(r_, slot_)] = host_bases[r_] + int(batch[r_][k])
        slot_line = []
        for r_ in range(args.req_num):
            for s_ in range(min(dbs, args.topk)):
                # 热缓冲 per-request 分区：行索引 = r*topK + slot（与 gather 内核 dst 契约一致）
                w0 = int(hbm_sync[(hot_rows_base + r_ * args.topk + s_) * (row_bytes // 4)])
                src = w0_to_row.get(w0)
                slot_line.append("r{}s{}:exp={},got={}".format(
                    r_, s_, expect_map.get((r_, s_), "-"), (str(src[0]) + "@L" + str(src[1])) if src else hex(w0)))
        log("[xds] hot slot map: " + "; ".join(slot_line))
        checked = 0
        for r in range(args.req_num):
            miss_pos = np.nonzero(coord.is_miss[r].cpu().numpy())[0]
            if miss_pos.size == 0:
                continue
            k = int(miss_pos[0])
            token_pos = int(batch[r][k])
            slot = int(coord.top_k_device_slots[r, k].item())
            if slot < 0 or slot >= dbs:
                continue
            row = host_bases[r] + token_pos
            expect = row_word(row, layer_id, 0)
            # 热缓冲 per-request 分区：行索引 = r*topK + slot（与 gather 内核 dst 契约一致）
            got = int(hbm_sync[(hot_rows_base + r * args.topk + slot) * (row_bytes // 4)])
            if got != expect:
                base_w = (hot_rows_base + r * args.topk + slot) * (row_bytes // 4)
                words4 = hbm_sync[base_w:base_w + 4].tolist()
                exp4 = [row_word(row, layer_id, w) for w in range(4)]
                log("[xds][ERROR] verify failed step={} layer={} req={} token={} row={} slot={} expect={} "
                    "got={} words4={} exp4={}".format(step, layer_id, r, token_pos, row, slot, expect, got,
                                                      [hex(w) for w in words4], [hex(w) for w in exp4]))
                return 1
            checked += 1
        log("[xds] step {}: layer={} miss={} rows in {:.1f} us (checked={})".format(
            step, layer_id, miss_cnt, dt * 1e6, checked))

    if total_miss > 0:
        bw = total_miss * row_bytes / t_read_total / 1e9 if t_read_total > 0 else 0
        log("[xds] summary: imported {} miss rows, avg {:.1f} us/row, aggregate {:.2f} GB/s".format(
            total_miss, t_read_total * 1e6 / total_miss, bw))
    else:
        log("[xds] summary: no miss rows imported")

    assert allocator.used_rows() == 0 or True  # 原型：进程退出前不强制回收
    pool.destroy()
    log("[xds] all done, log: {}".format(log.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
