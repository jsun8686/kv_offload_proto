# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""torch 层 e2e：torch_npu.npu_gather_selection_kv_cache_rmt 挂载 + golden 对拍（V3 验收）。

路径 A（本地，无 emsd）：channel 表全 0 → 生产原算子行为（full 本地张量）。
路径 B（远端，emsd）：EmsPool.export_full_rmt_pool 三表 + full 传空 tensor（纯远端惯例）。
用法：python -m tests.test_torch_gather_selection_rmt --device 0 [--ems-master ip:port]
"""

import argparse
import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kvops_ext._kvops as _kvops  # noqa: E402
from tests.test_gather_selection_rmt import B, S, H, TOPK, golden  # noqa: E402
from torchops import load_gather_rmt_op  # noqa: E402

FATBIN_CANDIDATES = [
    "/usr/local/fatbin/kvc_kernel/kvc_kernel.o",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build/kernel/kvc_kernel.o"),
]

BLOCK_SIZE = 8
TOPK_BS = 1
KV_DIM, ROPE_DIM = 256, 64
F_MAX_BLK, F_BLK_NUM = 16, 64
S_MAX_BLK = 16


def build_case(dev):
    full_kv = torch.randn((F_BLK_NUM, BLOCK_SIZE, KV_DIM), dtype=torch.float16, device=dev)
    full_rope = torch.randn((F_BLK_NUM, BLOCK_SIZE, ROPE_DIM), dtype=torch.float16, device=dev)
    sel_kv = torch.zeros_like(full_kv)
    sel_rope = torch.zeros_like(full_rope)
    full_tbl = torch.arange(B * F_MAX_BLK, dtype=torch.int32, device=dev).reshape(B, F_MAX_BLK) % F_BLK_NUM
    sel_tbl = torch.arange(B * S * H * S_MAX_BLK, dtype=torch.int32, device=dev).reshape(
        B * S * H, S_MAX_BLK) % F_BLK_NUM
    topk_idx = torch.randint(0, 24, (B, S, H, TOPK), dtype=torch.int32, device=dev)
    topk_idx[..., 10:] = -1
    kv_seq = torch.tensor([32, 17], dtype=torch.int32, device=dev)
    q_seq = torch.tensor([2, 2], dtype=torch.int32, device=dev)
    status = torch.full((B, S, H, TOPK + 1), -77, dtype=torch.int32, device=dev)
    return full_kv, full_rope, sel_kv, sel_rope, full_tbl, sel_tbl, topk_idx, kv_seq, q_seq, status


def check(g, sel_kv, sel_rope, sel_tbl, status_out, actual_out, tag):
    g_kv, g_rope, g_status, g_actual = g
    assert torch.equal(actual_out.cpu(), g_actual), "[%s] actual_seq mismatch" % tag
    assert torch.equal(status_out.cpu(), g_status), "[%s] block_status mismatch" % tag
    rows = B * S * H
    for row in range(rows):
        for k in range(TOPK):
            seg = int(g_status[row, k])
            if seg < 0:
                break
            tok = k * TOPK_BS
            s_blk = int(sel_tbl[row, tok // BLOCK_SIZE])
            s_off = tok % BLOCK_SIZE
            assert torch.equal(sel_kv[s_blk, s_off:s_off + TOPK_BS].cpu(),
                               g_kv[s_blk, s_off:s_off + TOPK_BS]), "[%s] kv mismatch row=%d k=%d" % (tag, row, k)
            assert torch.equal(sel_rope[s_blk, s_off:s_off + TOPK_BS].cpu(),
                               g_rope[s_blk, s_off:s_off + TOPK_BS]), "[%s] rope mismatch row=%d k=%d" % (
                tag, row, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ems-master", type=str, default="141.61.49.198:27000")
    args = ap.parse_args()

    torch.manual_seed(42)
    dev = torch.device("npu", args.device)
    torch.npu.set_device(dev)
    fatbin = next((p for p in FATBIN_CANDIDATES if os.path.exists(p)), None)
    assert fatbin, "fatbin not found"
    op = load_gather_rmt_op()
    assert hasattr(torch_npu, "npu_gather_selection_kv_cache_rmt"), "mount failed"

    # ── 路径 A：本地（channel 全 0，生产原算子行为） ──
    _kvops.init(args.device, fatbin)
    full_kv, full_rope, sel_kv, sel_rope, full_tbl, sel_tbl, topk_idx, kv_seq, q_seq, status = build_case(dev)
    rmt_zero = [torch.zeros(F_BLK_NUM, dtype=torch.int64, device=dev) for _ in range(3)]
    actual = op(selection_k_rope=sel_rope, selection_kv_cache=sel_kv,
                selection_kv_block_table=sel_tbl, selection_kv_block_status=status,
                selection_topk_indices=topk_idx, full_k_rope=full_rope, full_kv_cache=full_kv,
                full_kv_block_table=full_tbl, full_kv_actual_seq=kv_seq, full_q_actual_seq=q_seq,
                full_rmt_channel=rmt_zero[0], full_rmt_kv_addr=rmt_zero[1], full_rmt_rope_addr=rmt_zero[2],
                selection_topk_block_size=TOPK_BS, block_size=BLOCK_SIZE)
    g = golden(topk_idx.view(B * S * H, TOPK), full_tbl, kv_seq, q_seq, full_kv.cpu(), full_rope.cpu(),
               sel_tbl.cpu(), bs=BLOCK_SIZE, topk_bs=TOPK_BS, kv_dim=KV_DIM, rope_dim=ROPE_DIM)
    check(g, sel_kv, sel_rope, sel_tbl, status, actual, "local")
    print("[test] torch-op local path PASS")

    # ── 路径 B：远端（emsd + 空 full 惯例） ──
    from xds.xds_ems_pool import EmsPool
    ip, port = args.ems_master.rsplit(":", 1)
    hbm = torch.randn(F_BLK_NUM * BLOCK_SIZE * (KV_DIM + ROPE_DIM) * 2, dtype=torch.float16, device=dev)
    f_num = F_BLK_NUM * BLOCK_SIZE
    full_kv2 = hbm[:f_num * KV_DIM].view(F_BLK_NUM, BLOCK_SIZE, KV_DIM)
    full_rope2 = hbm[f_num * KV_DIM:f_num * (KV_DIM + ROPE_DIM)].view(F_BLK_NUM, BLOCK_SIZE, ROPE_DIM)
    sel_kv2 = hbm[f_num * (KV_DIM + ROPE_DIM):f_num * (KV_DIM + ROPE_DIM) + f_num * KV_DIM].view(
        F_BLK_NUM, BLOCK_SIZE, KV_DIM).zero_()
    sel_rope2 = hbm[f_num * (KV_DIM + ROPE_DIM) + f_num * KV_DIM:].view(
        F_BLK_NUM, BLOCK_SIZE, ROPE_DIM).zero_()
    pool_rows = F_BLK_NUM * BLOCK_SIZE * (KV_DIM + ROPE_DIM) // KV_DIM
    pool = EmsPool(ip, int(port), args.device, layer_num=1, layer_rows=pool_rows, row_bytes=KV_DIM * 2)
    pool.create_pool(hbm.data_ptr(), hbm.numel() * 2, fatbin)
    rmt_ch, rmt_kv, rmt_rope = pool.export_full_rmt_pool(full_kv2, full_rope2, BLOCK_SIZE, KV_DIM, ROPE_DIM)

    _, _, _, _, _, sel_tbl2, topk_idx2, kv_seq2, q_seq2, status2 = build_case(dev)
    empty = torch.empty(0, dtype=torch.int8, device=dev)  # 纯远端空 full 惯例
    actual2 = op(selection_k_rope=sel_rope2, selection_kv_cache=sel_kv2,
                 selection_kv_block_table=sel_tbl2, selection_kv_block_status=status2,
                 selection_topk_indices=topk_idx2, full_k_rope=empty, full_kv_cache=empty,
                 full_kv_block_table=full_tbl, full_kv_actual_seq=kv_seq2, full_q_actual_seq=q_seq2,
                 full_rmt_channel=rmt_ch, full_rmt_kv_addr=rmt_kv, full_rmt_rope_addr=rmt_rope,
                 selection_topk_block_size=TOPK_BS, block_size=BLOCK_SIZE)
    g2 = golden(topk_idx2.view(B * S * H, TOPK), full_tbl, kv_seq2, q_seq2, full_kv2.cpu(), full_rope2.cpu(),
                sel_tbl2.cpu(), bs=BLOCK_SIZE, topk_bs=TOPK_BS, kv_dim=KV_DIM, rope_dim=ROPE_DIM)
    check(g2, sel_kv2, sel_rope2, sel_tbl2, status2, actual2, "remote")
    pool.destroy()
    print("[test] torch-op remote path PASS (empty-full convention)")
    print("[test] torch_npu.npu_gather_selection_kv_cache_rmt e2e PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
