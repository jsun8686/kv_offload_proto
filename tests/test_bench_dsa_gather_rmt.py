# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""dsa_gather_rmt 性能摸底：blockDim × 规模档 × {eager, graph} 矩阵。

计时：torch.npu.Event 量 stream 上内核数据面时间（submit 后 record，不含 host poll 开销）；
     warmup×3 + repeat×N 取中位。输出 time_us / records / moved_bytes / 有效带宽 / vs bd1 加速比。

规模档（--scale）：
  s1 正确性档：query=256,  key=4096,   miss 25%  → records≈256
  s2 中档：    query=2048, key=32768,  miss 25%  → records≈2048
  s3 重压档：  query=2048, key=131072, miss 100% → records≈8192（全 miss 带宽压测，对齐原版生产形状）

用法：python -m tests.test_bench_dsa_gather_rmt --device 0 --ems-master <ip:27000>
     [--block-dims 1,2,4,8] [--scale s1|s2|s3] [--mode eager|graph|both] [--repeat 10]
"""

import argparse
import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchops import load_dsa_gather_rmt_op  # noqa: E402
from xds.xds_ems_pool import EmsPool  # noqa: E402

FATBIN_CANDIDATES = [
    "/usr/local/fatbin/kvc_kernel/kvc_kernel.o",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build/kernel/kvc_kernel.o"),
]

BLOCK_SIZE = 128
KV_DIM, ROPE_DIM = 512, 64
REC_BYTES = (BLOCK_SIZE * KV_DIM + BLOCK_SIZE * ROPE_DIM) * 2  # fp16：每 miss 记录搬运字节（kv+rope）

SCALES = {
    "s1": dict(bs=4, query=256, key=4096, total_slots=2048, miss_frac=0.25),
    "s2": dict(bs=4, query=2048, key=32768, total_slots=2048, miss_frac=0.25),
    "s3": dict(bs=4, query=2048, key=131072, total_slots=2048, miss_frac=1.0),
}


def median_us(launch, repeat):
    """Event 计时中位数（μs）：量 stream 上内核数据面时间（submit→record，内核含 ReadNbi+Drain 收尾）。"""
    for _ in range(3):  # warmup（含 hcomm.Init 首执行、fatbin lazy load）
        launch()
    torch.npu.synchronize()
    ts = []
    for _ in range(repeat):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        launch()
        e.record()
        torch.npu.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # ms → us
    ts.sort()
    return ts[len(ts) // 2]


def build_case(sc, dev):
    """按档位构造张量组与索引元数据（数据公式对齐 golden.py）。"""
    bs, query, key, total_slots, miss_frac = sc["bs"], sc["query"], sc["key"], sc["total_slots"], sc["miss_frac"]
    host_blk_per_req = key // BLOCK_SIZE
    host_blk_num = bs * host_blk_per_req
    res_blk_per_req = (total_slots + BLOCK_SIZE - 1) // BLOCK_SIZE
    res_blk_num = bs * res_blk_per_req
    pool_rows = host_blk_num * BLOCK_SIZE * (KV_DIM + ROPE_DIM) // KV_DIM

    hbm_elems = (host_blk_num + res_blk_num) * BLOCK_SIZE * (KV_DIM + ROPE_DIM)
    hbm = torch.empty(hbm_elems, dtype=torch.float16, device=dev)
    h_num = host_blk_num * BLOCK_SIZE
    host_kv = hbm[:h_num * KV_DIM].view(host_blk_num, BLOCK_SIZE, KV_DIM)
    host_rope = hbm[h_num * KV_DIM:h_num * (KV_DIM + ROPE_DIM)].view(host_blk_num, BLOCK_SIZE, ROPE_DIM)
    r_off = h_num * (KV_DIM + ROPE_DIM)
    res_kv = hbm[r_off:r_off + res_blk_num * BLOCK_SIZE * KV_DIM].view(res_blk_num, BLOCK_SIZE, KV_DIM)
    res_rope = hbm[r_off + res_blk_num * BLOCK_SIZE * KV_DIM:].view(res_blk_num, BLOCK_SIZE, ROPE_DIM)

    tok = torch.arange(h_num, dtype=torch.int32).remainder(512)
    host_kv.view(-1, KV_DIM).copy_(tok.view(-1, 1).to(torch.float16))
    host_rope.view(-1, ROPE_DIM).fill_(1024.0)
    res_kv.fill_(-1000.0)
    res_rope.fill_(-1000.0)

    host_tbl = torch.arange(host_blk_num, dtype=torch.int32, device=dev).view(bs, host_blk_per_req)
    res_tbl = torch.arange(res_blk_num, dtype=torch.int32, device=dev).view(bs, res_blk_per_req)
    pool_idx = torch.roll(torch.arange(bs, dtype=torch.int32), shifts=1).to(dev)
    step = max(1, key // query)
    topk = (torch.arange(query, dtype=torch.int32) * step).remainder(key).view(1, -1).repeat(bs, 1).to(dev)
    slots = (torch.arange(query, dtype=torch.int32) * 5).remainder(total_slots).view(1, -1).repeat(bs, 1).to(dev)
    miss_stride = max(1, int(round(1.0 / miss_frac)))
    miss = torch.zeros(query, dtype=torch.int32)
    miss[::miss_stride] = 1
    mask = miss.view(1, -1).repeat(bs, 1).to(dev)

    records = int(mask.sum().item())
    return dict(hbm=hbm, host_kv=host_kv, host_rope=host_rope, res_kv=res_kv, res_rope=res_rope,
                host_tbl=host_tbl, res_tbl=res_tbl, pool_idx=pool_idx, topk=topk, slots=slots, mask=mask,
                pool_rows=pool_rows, records=records)


def run_bd(bd, sc, case, pool_kwargs, fatbin, dev, mode, repeat):
    """单 (bd, 档位) 组：建池（n_channels=bd）→ 导出三表 → eager/graph 计时。"""
    pool = EmsPool(pool_kwargs["ip"], pool_kwargs["port"], pool_kwargs["device"],
                   layer_num=1, layer_rows=case["pool_rows"], row_bytes=KV_DIM * 2)
    pool.create_pool(case["hbm"].data_ptr(), case["hbm"].numel() * 2, fatbin, n_channels=bd)
    rmt_ch, rmt_kv, rmt_rope = pool.export_full_rmt_pool(case["host_kv"], case["host_rope"],
                                                         BLOCK_SIZE, KV_DIM, ROPE_DIM, None, n_core=bd)
    kv4 = case["res_kv"].view(case["res_kv"].shape[0], BLOCK_SIZE, 1, KV_DIM)
    rope4 = case["res_rope"].view(case["res_rope"].shape[0], BLOCK_SIZE, 1, ROPE_DIM)

    def eager_launch():
        return torch_npu.npu_dsa_gather_rmt_submit(
            kv4, rope4, case["res_tbl"], case["host_tbl"], case["pool_idx"], case["topk"],
            case["slots"], case["mask"], BLOCK_SIZE, case["mask"].shape[0],
            rmt_ch, rmt_kv, rmt_rope, block_dim=bd)

    results = {}
    if mode in ("eager", "both"):
        h = eager_launch()
        torch_npu.npu_dsa_gather_rmt_wait(h)  # 错误码校验（warmup 首发的真 handle）
        torch.npu.synchronize()
        results["eager"] = median_us(eager_launch, repeat)

    if mode in ("graph", "both"):
        eager_launch()  # 图外 warmup（hcomm.Init 首执行纪律）
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            eager_launch()
        results["graph"] = median_us(lambda: graph.replay(), repeat)

    pool.destroy()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ems-master", type=str, default="10.83.40.200:27000")
    ap.add_argument("--block-dims", type=str, default="1,2,4,8")
    ap.add_argument("--scale", type=str, default="s1", choices=list(SCALES))
    ap.add_argument("--mode", type=str, default="both", choices=["eager", "graph", "both"])
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(42)
    dev = torch.device("npu", args.device)
    torch.npu.set_device(dev)
    fatbin = next((p for p in FATBIN_CANDIDATES if os.path.exists(p)), None)
    assert fatbin, "fatbin not found: %s" % FATBIN_CANDIDATES
    load_dsa_gather_rmt_op()
    ip, port = args.ems_master.rsplit(":", 1)

    sc = SCALES[args.scale]
    case = build_case(sc, dev)
    bds = [int(x) for x in args.block_dims.split(",") if x.strip()]
    print("[bench] scale=%s bs=%d query=%d key=%d miss=%.0f%% records=%d rec_bytes=%d moved=%.1f MB" % (
        args.scale, sc["bs"], sc["query"], sc["key"], sc["miss_frac"] * 100,
        case["records"], REC_BYTES, case["records"] * REC_BYTES / 1e6))
    print("[bench] matrix: block_dims=%s mode=%s repeat=%d（每点 warmup×3 + 中位）" % (bds, args.mode, args.repeat))

    rows = []
    base = {}
    for bd in bds:
        res = run_bd(bd, sc, case, dict(ip=ip, port=int(port), device=args.device), fatbin, dev,
                     args.mode, args.repeat)
        for m, us in res.items():
            if bd == bds[0]:
                base[m] = us
            rows.append((bd, m, us))
        print("[bench] bd=%d %s" % (bd, " ".join("%s=%.1fus" % (m, us) for m, us in res.items())))

    print("\n| blockDim | mode | time_us | records | moved_MB | BW_GB/s | speedup_vs_bd1 |")
    print("|---|---|---|---|---|---|---|")
    for bd, m, us in rows:
        moved = case["records"] * REC_BYTES
        bw = moved / (us / 1e6) / 1e9
        sp = base.get(m, us) / us if us > 0 else 0.0
        print("| %d | %s | %.1f | %d | %.1f | %.2f | %.2fx |" % (bd, m, us, case["records"], moved / 1e6, bw, sp))
    print("\n[bench] done. （time=stream 内核数据面中位；moved=miss 记录 kv+rope 字节；"
          "参照基线：原版 910_93 H2D 164 records 98.78us）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
