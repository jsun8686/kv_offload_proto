# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""dsa_gather_rmt 唯一测试：torch_npu 两件套原语 + 正确性对拍 + 性能摸底矩阵。

接口面：仅 torch_npu 属性（load 挂载两件套原语）——
    torch_npu.npu_dsa_gather_rmt_submit(...)   纯提交（图捕获区间 / 性能计时通用）
    torch_npu.npu_dsa_gather_rmt_result(h)     结果校验（调用方须已 synchronize）
eager 语义 = 显式三件组合：submit → torch.npu.synchronize() → result。

流程（单 (blockDim, mode) 点）：建池(n_channels=bd) → 导出三表 → eager warmup 首跑
    → golden 对拍一次（PASS 行，不计入计时）→ Event 计时（warmup×3 + repeat 中位）
    → 结果行（time_us / records / moved_MB / BW_GB/s）。

多 blockDim 矩阵 = 外层循环多次调用（每 bd 独立进程）——规避进程内多池重建的
    hcomm device 状态问题（见 doc/dsa_gather_rmt.md 遗留问题）：
    for bd in 1 2 4 8; do python -m tests.test_dsa_gather_rmt --block-dim $bd ...; done

规模全参数化（对齐 vllm-ascend-csrc dsa_gather golden.py 命名）：
    --bs（num_requests）--query-count（每请求 top-K query 槽位）
    --key-len（每池行 KV 序列长度）--misses-per-req（每请求 miss 记录数，均匀散布）    --perm：远端块布局随机洗牌（验证查表寻址；多节点下即"打散"placement 策略，
    任意 miss 局部性被平均摊到各段 → 带宽聚合）。

用法：python -m tests.test_dsa_gather_rmt --device 0 --ems-master <ip:port>
    [--bs 4 --query-count 256 --key-len 4096 --misses-per-req 64]
    [--block-dim 1] [--mode eager|graph|both] [--repeat 10] [--perm]
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
SENTINEL = -1000.0
# 每 miss 记录搬运字节 = 单 token 的 kv+rope（fp16）。
# 粒度铁证（对齐原版 dsa_gather）：原版内核 DataCopy len = kvRecordElements = 张量 size(3)（单
# token 向量，dsa_gather.cpp:130-139 / dsa_gather_torch_adpt.h:293-296），per-token 换入；
# 本仓内核 ReadNbi len = kvDim*elemSize 同粒度。早期版本误乘 BLOCK_SIZE（虚高 128 倍，已修正）。
REC_BYTES = (KV_DIM + ROPE_DIM) * 2


def fill_staging(flat, offset):
    """按 golden.py 公式填充：value = (token % 512) + channel_idx + offset。"""
    tok = torch.arange(flat.shape[0], dtype=torch.int32).remainder(512)
    flat.copy_(tok.view(-1, 1).to(flat.dtype))
    flat.add_(torch.arange(flat.shape[1], dtype=flat.dtype, device=flat.device).view(1, -1))
    flat.add_(offset)


def golden_check(host_kv, host_rope, res_kv, res_rope, pool_idx, topk, slots, mask, ctx):
    """对拍 golden.py：written 集合数据 + sentinel untouched 精确一致（CPU 标准答案）。"""
    bs = ctx["bs"]
    active = mask.cpu().to(torch.bool)
    batch_rows = torch.arange(bs, dtype=torch.int64).view(-1, 1).expand_as(active)
    source_slots = (pool_idx.cpu().view(-1, 1).to(torch.int64) * ctx["host_slots_per_entry"] +
                    topk.cpu().to(torch.int64))[active]
    dest_flat = (batch_rows * ctx["res_slots_per_req"] + slots.cpu().to(torch.int64))[active]
    expected_sources = torch.full((ctx["res_blk_num"] * BLOCK_SIZE,), -1, dtype=torch.int64)
    expected_sources[dest_flat] = source_slots
    written = expected_sources.ge(0).nonzero(as_tuple=False).flatten()
    src = expected_sources.index_select(0, written)

    kv_exp = src.remainder(512).to(torch.float16).view(-1, 1) + torch.arange(KV_DIM, dtype=torch.float16).view(1, -1)
    rope_exp = (src.remainder(512).to(torch.float16).view(-1, 1) +
                torch.arange(ROPE_DIM, dtype=torch.float16).view(1, -1) + 1024)
    kv_flat = res_kv.view(-1, KV_DIM)
    rope_flat = res_rope.view(-1, ROPE_DIM)
    dev_idx = written.to(res_kv.device)
    assert torch.equal(kv_flat.index_select(0, dev_idx).cpu(), kv_exp), "kv records mismatch"
    assert torch.equal(rope_flat.index_select(0, dev_idx).cpu(), rope_exp), "rope records mismatch"
    touched = kv_flat.ne(SENTINEL).any(dim=1) | rope_flat.ne(SENTINEL).any(dim=1)
    assert torch.equal(touched.cpu(), expected_sources.ge(0)), "untouched destinations mismatch (zero-copy broken)"
    return int(written.numel())


def build_case(bs, query_count, key_len, misses_per_req, perm, dev):
    """按参数构造张量组与索引元数据（推导量与 golden 语义对齐）。"""
    host_blk_per_req = key_len // BLOCK_SIZE
    host_blk_num = bs * host_blk_per_req
    total_slots = query_count  # resident slot 容量 = 每请求 query 槽位（恒等分配）
    res_blk_per_req = (total_slots + BLOCK_SIZE - 1) // BLOCK_SIZE
    res_blk_num = bs * res_blk_per_req
    pool_rows = host_blk_num * BLOCK_SIZE * (KV_DIM + ROPE_DIM) // KV_DIM
    ctx = dict(bs=bs, host_slots_per_entry=host_blk_per_req * BLOCK_SIZE,
               res_slots_per_req=res_blk_per_req * BLOCK_SIZE, res_blk_num=res_blk_num)

    hbm_elems = (host_blk_num + res_blk_num) * BLOCK_SIZE * (KV_DIM + ROPE_DIM)
    hbm = torch.empty(hbm_elems, dtype=torch.float16, device=dev)
    h_num = host_blk_num * BLOCK_SIZE
    host_kv = hbm[:h_num * KV_DIM].view(host_blk_num, BLOCK_SIZE, KV_DIM)
    host_rope = hbm[h_num * KV_DIM:h_num * (KV_DIM + ROPE_DIM)].view(host_blk_num, BLOCK_SIZE, ROPE_DIM)
    r_off = h_num * (KV_DIM + ROPE_DIM)
    res_kv = hbm[r_off:r_off + res_blk_num * BLOCK_SIZE * KV_DIM].view(res_blk_num, BLOCK_SIZE, KV_DIM)
    res_rope = hbm[r_off + res_blk_num * BLOCK_SIZE * KV_DIM:].view(res_blk_num, BLOCK_SIZE, ROPE_DIM)
    fill_staging(host_kv.view(-1, KV_DIM), 0)
    fill_staging(host_rope.view(-1, ROPE_DIM), 1024)
    res_kv.fill_(SENTINEL)
    res_rope.fill_(SENTINEL)

    host_tbl = torch.arange(host_blk_num, dtype=torch.int32, device=dev).view(bs, host_blk_per_req)
    res_tbl = torch.arange(res_blk_num, dtype=torch.int32, device=dev).view(bs, res_blk_per_req)
    pool_idx = torch.roll(torch.arange(bs, dtype=torch.int32), shifts=1).to(dev)  # 两跳寻址验证
    step = max(1, key_len // query_count)
    topk = (torch.arange(query_count, dtype=torch.int32) * step).remainder(key_len).view(1, -1).repeat(bs, 1).to(dev)
    slots = torch.arange(query_count, dtype=torch.int32).view(1, -1).repeat(bs, 1).to(dev)
    miss_pos = torch.arange(misses_per_req, dtype=torch.int64) * query_count // max(misses_per_req, 1)
    mask_row = torch.zeros(query_count, dtype=torch.int32)
    mask_row[miss_pos] = 1
    mask = mask_row.view(1, -1).repeat(bs, 1).to(dev)

    records = int(mask.sum().item())
    return dict(hbm=hbm, host_kv=host_kv, host_rope=host_rope, res_kv=res_kv, res_rope=res_rope,
                host_tbl=host_tbl, res_tbl=res_tbl, pool_idx=pool_idx, topk=topk, slots=slots, mask=mask,
                pool_rows=pool_rows, ctx=ctx, records=records,
                perm=torch.randperm(host_blk_num) if perm else None)


def median_us(launch, repeat):
    """Event 计时中位数（μs）：量 stream 上内核数据面时间（submit→record，含 ReadNbi+Drain 收尾）。"""
    for _ in range(3):  # warmup（hcomm.Init 首执行、fatbin lazy load 已由外层 eager 首跑覆盖）
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


def run_bd(bd, case, ip, port, device, fatbin, mode, repeat):
    """单 bd 点：建池 → 导出 → eager warmup+对拍 → eager/graph 计时。

    pool.destroy 进 finally：任何失败（内核 errCode/poll 超时）都保证通道/池清理执行
    （实测异常退出时跳过 destroy 会残留 hcomm 通道状态，进程退出触发 Bus error）。
    """
    pool = EmsPool(ip, port, device, layer_num=1, layer_rows=case["pool_rows"], row_bytes=KV_DIM * 2)
    try:
        return _run_bd_inner(bd, case, ip, port, device, fatbin, mode, repeat, pool)
    finally:
        pool.destroy()


def _run_bd_inner(bd, case, ip, port, device, fatbin, mode, repeat, pool):
    pool.create_pool(case["hbm"].data_ptr(), case["hbm"].numel() * 2, fatbin, n_channels=bd)
    rmt_ch, rmt_kv, rmt_rope = pool.export_full_rmt_pool(case["host_kv"], case["host_rope"],
                                                         BLOCK_SIZE, KV_DIM, ROPE_DIM, case["perm"],
                                                         n_core=bd)
    kv4 = case["res_kv"].view(case["res_kv"].shape[0], BLOCK_SIZE, 1, KV_DIM)
    rope4 = case["res_rope"].view(case["res_rope"].shape[0], BLOCK_SIZE, 1, ROPE_DIM)
    bs = case["mask"].shape[0]

    def submit():
        return torch_npu.npu_dsa_gather_rmt_submit(
            kv4, rope4, case["res_tbl"], case["host_tbl"], case["pool_idx"], case["topk"],
            case["slots"], case["mask"], BLOCK_SIZE, bs, rmt_ch, rmt_kv, rmt_rope, block_dim=bd)

    # eager warmup 首跑（hcomm.Init 图外纪律）：显式三件组合 + 正确性对拍（不计入计时）
    h = submit()
    torch.npu.synchronize()
    torch_npu.npu_dsa_gather_rmt_result(h)
    n = golden_check(case["host_kv"], case["host_rope"], case["res_kv"], case["res_rope"],
                     case["pool_idx"], case["topk"], case["slots"], case["mask"], case["ctx"])
    print("[test] golden PASS (bd=%d records=%d perm=%s)" % (bd, n, "on" if case["perm"] is not None else "off"))

    results = {}
    if mode in ("eager", "both"):
        results["eager"] = median_us(submit, repeat)

    if mode in ("graph", "both"):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            submit()
        results["graph"] = median_us(lambda: graph.replay(), repeat)

    # pool.destroy 由 run_bd 的 finally 单点负责（此处再调会双执行——实测两轮 channel_destroy）
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ems-master", type=str, default="10.83.40.200:27000")
    ap.add_argument("--bs", type=int, default=4, help="num_requests")
    ap.add_argument("--query-count", type=int, default=256, help="每请求 top-K query 槽位数")
    ap.add_argument("--key-len", type=int, default=4096, help="每池行 KV 序列长度（token 数）")
    ap.add_argument("--misses-per-req", type=int, default=64, help="每请求 miss 记录数（均匀散布）")
    ap.add_argument("--perm", action="store_true", help="远端块布局随机洗牌（打散 placement）")
    ap.add_argument("--block-dim", type=int, default=1, help="每进程单值（多 bd 用外层循环独立进程）")
    ap.add_argument("--mode", type=str, default="both", choices=["eager", "graph", "both"])
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()
    assert args.query_count > 0 and args.key_len % BLOCK_SIZE == 0 and 0 < args.misses_per_req <= args.query_count, \
        "query_count>0, key_len must be multiple of BLOCK_SIZE(128), 0<misses_per_req<=query_count"

    torch.manual_seed(42)
    dev = torch.device("npu", args.device)
    torch.npu.set_device(dev)
    fatbin = next((p for p in FATBIN_CANDIDATES if os.path.exists(p)), None)
    assert fatbin, "fatbin not found: %s" % FATBIN_CANDIDATES
    load_dsa_gather_rmt_op()
    ip, port = args.ems_master.rsplit(":", 1)

    case = build_case(args.bs, args.query_count, args.key_len, args.misses_per_req, args.perm, dev)
    bd = args.block_dim
    assert bd >= 1, "block-dim must be >= 1"
    print("[test] bs=%d query_count=%d key_len=%d misses_per_req=%d → records=%d rec_bytes=%d moved=%.1f MB" % (
        args.bs, args.query_count, args.key_len, args.misses_per_req,
        case["records"], REC_BYTES, case["records"] * REC_BYTES / 1e6))
    print("[test] point: block_dim=%d mode=%s repeat=%d perm=%s" % (
        bd, args.mode, args.repeat, "on" if args.perm else "off"))

    res = run_bd(bd, case, ip, int(port), args.device, fatbin, args.mode, args.repeat)
    moved = case["records"] * REC_BYTES
    for m, us in res.items():
        bw = moved / (us / 1e6) / 1e9
        print("[test] bd=%d %s=%.1fus BW=%.2f GB/s (%d records, %.1f MB moved)" % (
            bd, m, us, bw, case["records"], moved / 1e6))
    print("\n[test] done. （time=stream 内核数据面中位；moved=miss 记录 per-token kv+rope 净载荷字节；"
          "参照基线：原版 910_93 H2D 164 token 记录 98.78us≈1.9 GB/s（同 per-token 口径））")
    return 0


if __name__ == "__main__":
    sys.exit(main())
