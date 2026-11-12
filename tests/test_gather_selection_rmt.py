# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""gather_selection_rmt 本地路径 golden 对拍（V1.0 验收，NPU 执行）。

语义基准：cann-recipes-infer test_npu_gather_selection_kv_cache.py golden 基础路径
（有效性过滤 + 因果裁剪 + 紧凑追加 + 账本回写；命中免搬为 V1.1 优化，本测试每步独立账本）。
用法：python -m test.test_gather_selection_rmt [--device 0]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kvops_ext._kvops as _kvops  # noqa: E402

FATBIN_CANDIDATES = [
    "/usr/local/fatbin/kvc_kernel/kvc_kernel.o",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build/kernel/kvc_kernel.o"),
]

B, S, H = 2, 2, 1          # batch / query token 数 / head（恒 1）
TOPK = 16
BLOCK_SIZE = 8             # 物理块 token 数（可被 topk_bs 整除）
TOPK_BS = 1                # 选择粒度（GLM 生产形态）
KV_DIM, ROPE_DIM = 64, 16
F_MAX_BLK, F_BLK_NUM = 16, 64      # full 页表列数 / 物理块数
S_MAX_BLK = 16                      # 每行 selection 页表列数


def golden(topk_idx, full_tbl, kv_seq, q_seq, full_kv, full_rope, sel_tbl, bs=BLOCK_SIZE, topk_bs=TOPK_BS):
    rows = B * S * H
    sel_kv = torch.zeros((F_BLK_NUM, bs, KV_DIM), dtype=full_kv.dtype)
    sel_rope = torch.zeros((F_BLK_NUM, bs, ROPE_DIM), dtype=full_rope.dtype)
    status = torch.full((rows, TOPK + 1), -1, dtype=torch.int32)
    actual = torch.zeros(rows, dtype=torch.int32)
    for row in range(rows):
        b = row // (S * H)
        s = (row // H) % S
        kv_len = int(kv_seq[b])
        if s < int(q_seq[b]):
            kv_len -= int(q_seq[b]) - 1 - s
        if kv_len <= 0:
            status[row, TOPK] = 0
            continue
        max_sel = (kv_len + topk_bs - 1) // topk_bs
        last_bs = kv_len - (max_sel - 1) * topk_bs
        valid = 0
        acc = 0
        for k in range(TOPK):
            seg = int(topk_idx[row, k])
            if seg < 0:
                break
            if seg >= max_sel:
                continue
            cu = topk_bs if seg != max_sel - 1 else last_bs
            f_tok = seg * topk_bs
            f_blk = int(full_tbl[b, f_tok // bs])
            f_off = f_tok % bs
            s_tok = valid * topk_bs
            s_blk = int(sel_tbl[row, s_tok // bs])
            s_off = s_tok % bs
            sel_kv[s_blk, s_off:s_off + cu] = full_kv[f_blk, f_off:f_off + cu]
            sel_rope[s_blk, s_off:s_off + cu] = full_rope[f_blk, f_off:f_off + cu]
            status[row, valid] = seg
            valid += 1
            acc += cu
        status[row, valid:TOPK] = -1
        status[row, TOPK] = acc
        actual[row] = acc
    return sel_kv, sel_rope, status, actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(42)
    dev = torch.device("npu", args.device)
    torch.npu.set_device(dev)

    fatbin = next((p for p in FATBIN_CANDIDATES if os.path.exists(p)), None)
    assert fatbin, "fatbin not found: %s" % FATBIN_CANDIDATES
    _kvops.init(args.device, fatbin)

    rows = B * S * H
    full_kv = torch.randn((F_BLK_NUM, BLOCK_SIZE, KV_DIM), dtype=torch.float16, device=dev)
    full_rope = torch.randn((F_BLK_NUM, BLOCK_SIZE, ROPE_DIM), dtype=torch.float16, device=dev)
    sel_kv = torch.zeros_like(full_kv)
    sel_rope = torch.zeros_like(full_rope)
    full_tbl = torch.arange(B * F_MAX_BLK, dtype=torch.int32, device=dev).reshape(B, F_MAX_BLK) % F_BLK_NUM
    sel_tbl = torch.arange(rows * S_MAX_BLK, dtype=torch.int32, device=dev).reshape(rows, S_MAX_BLK) % F_BLK_NUM
    topk_idx = torch.randint(0, 24, (rows, TOPK), dtype=torch.int32, device=dev)
    topk_idx[:, 10:] = -1  # 截断语义
    kv_seq = torch.tensor([32, 17], dtype=torch.int32, device=dev)
    q_seq = torch.tensor([2, 2], dtype=torch.int32, device=dev)
    rmt_ch = torch.zeros(F_BLK_NUM, dtype=torch.int64, device=dev)  # V1.0 本地路径：全 0
    rmt_kv = torch.zeros(F_BLK_NUM, dtype=torch.int64, device=dev)
    rmt_rope = torch.zeros(F_BLK_NUM, dtype=torch.int64, device=dev)
    status_out = torch.full((rows, TOPK + 1), -77, dtype=torch.int32, device=dev)
    actual_out = torch.zeros(rows, dtype=torch.int32, device=dev)

    st = _kvops.alloc_status(1)
    stream = torch.npu.current_stream().npu_stream
    _kvops.gather_selection_rmt(
        sel_rope.data_ptr(), sel_kv.data_ptr(), sel_tbl.data_ptr(), status_out.data_ptr(),
        topk_idx.data_ptr(), full_rope.data_ptr(), full_kv.data_ptr(), full_tbl.data_ptr(),
        kv_seq.data_ptr(), q_seq.data_ptr(), rmt_ch.data_ptr(), rmt_kv.data_ptr(), rmt_rope.data_ptr(),
        st, actual_out.data_ptr(), B, S, H, TOPK, BLOCK_SIZE, F_MAX_BLK, S_MAX_BLK, F_BLK_NUM,
        TOPK_BS, KV_DIM, ROPE_DIM, stream, 1)
    rc = _kvops.poll_status(st, 30_000_000)
    assert rc == 0, "kernel failed rc=0x%x" % (rc & 0xFFFFFFFF)
    torch.npu.synchronize()

    g_kv, g_rope, g_status, g_actual = golden(topk_idx, full_tbl, kv_seq, q_seq, full_kv, full_rope, sel_tbl)

    assert torch.equal(actual_out.cpu(), g_actual), \
        "actual_seq mismatch:\n got=%s\n exp=%s" % (actual_out.cpu().tolist(), g_actual.tolist())
    assert torch.equal(status_out.cpu(), g_status), "block_status mismatch"
    # 热池数据：按 golden 写入位置比对（其余位置本测试未写，不比对）
    for row in range(rows):
        for k in range(TOPK):
            seg = int(g_status[row, k])
            if seg < 0:
                break
            tok = k * TOPK_BS
            s_blk = int(sel_tbl[row, tok // BLOCK_SIZE])
            s_off = tok % BLOCK_SIZE
            got = sel_kv[s_blk, s_off:s_off + TOPK_BS].cpu()
            exp = g_kv[s_blk, s_off:s_off + TOPK_BS]
            assert torch.equal(got, exp), "kv data mismatch row=%d k=%d" % (row, k)
            got_r = sel_rope[s_blk, s_off:s_off + TOPK_BS].cpu()
            exp_r = g_rope[s_blk, s_off:s_off + TOPK_BS]
            assert torch.equal(got_r, exp_r), "rope data mismatch row=%d k=%d" % (row, k)
    print("[test] gather_selection_rmt local-path golden check PASS (rows=%d topk=%d)" % (rows, TOPK))
    return 0


if __name__ == "__main__":
    sys.exit(main())
