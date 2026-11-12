# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""gather_selection_rmt 远端路径 E2E golden 对拍（V1.1 验收，NPU 执行）。

链路：EmsPool（kv 大区 + rope 大区两区导出，可选 perm 乱序槽位）→ full_rmt_* 三表 →
内核 ReadNbi 远端读点直写本地 sel 池（须在 mem_reg 注册范围内）→ 与 golden 逐位对拍。
用法：python -m tests.test_gather_selection_rmt_rmt --device 0 --ems-master <ip:port> [--perm]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kvops_ext._kvops as _kvops  # noqa: E402
from tests.test_gather_selection_rmt import B, S, H, TOPK, golden  # noqa: E402
from xds.xds_ems_pool import EmsPool  # noqa: E402

FATBIN_CANDIDATES = [
    "/usr/local/fatbin/kvc_kernel/kvc_kernel.o",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build/kernel/kvc_kernel.o"),
]

BLOCK_SIZE = 8
TOPK_BS = 1
KV_DIM = 256                    # 行宽约束：kv_dim*2 须 128 对齐；比例 4:1 贴生产
F_MAX_BLK, F_BLK_NUM = 16, 64     # full 页表列数 / 物理块数
S_MAX_BLK = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ems-master", type=str, default="141.61.49.195:27000")
    ap.add_argument("--perm", action="store_true", help="远端槽位乱序（表驱动灵活性验证）")
    ap.add_argument("--rope-dim", type=int, default=64, help="64=生产形态（128B 读点）；256=512B 读点对照实验")
    args = ap.parse_args()
    rope_dim = args.rope_dim
    pool_rows = F_BLK_NUM * BLOCK_SIZE * (KV_DIM + rope_dim) // KV_DIM  # kv 行 + rope 行（统一 512B 行）
    hbm_elems = F_BLK_NUM * BLOCK_SIZE * (KV_DIM + rope_dim) * 2        # full 池 + sel 池（各按 F_BLK_NUM 块计）

    torch.manual_seed(42)
    dev = torch.device("npu", args.device)
    torch.npu.set_device(dev)
    ip, port = args.ems_master.rsplit(":", 1)

    fatbin = next((p for p in FATBIN_CANDIDATES if os.path.exists(p)), None)
    assert fatbin, "fatbin not found: %s" % FATBIN_CANDIDATES

    # 单一大张量：full 池（导出源）+ sel 池（ReadNbi dst，须整段落入 mem_reg 注册 VA 范围）
    hbm = torch.randn(hbm_elems, dtype=torch.float16, device=dev)
    f_num = F_BLK_NUM * BLOCK_SIZE
    full_kv = hbm[:f_num * KV_DIM].view(F_BLK_NUM, BLOCK_SIZE, KV_DIM)
    full_rope = hbm[f_num * KV_DIM:f_num * (KV_DIM + rope_dim)].view(F_BLK_NUM, BLOCK_SIZE, rope_dim)
    sel_kv = hbm[f_num * (KV_DIM + rope_dim):f_num * (KV_DIM + rope_dim) + f_num * KV_DIM].view(
        F_BLK_NUM, BLOCK_SIZE, KV_DIM).zero_()
    sel_rope = hbm[f_num * (KV_DIM + rope_dim) + f_num * KV_DIM:].view(F_BLK_NUM, BLOCK_SIZE, rope_dim).zero_()

    pool = EmsPool(ip, int(port), args.device, layer_num=1, layer_rows=pool_rows, row_bytes=KV_DIM * 2)
    pool.create_pool(hbm.data_ptr(), hbm.numel() * 2, fatbin)
    perm = torch.randperm(F_BLK_NUM) if args.perm else None
    rmt_ch, rmt_kv, rmt_rope = pool.export_full_rmt_pool(full_kv, full_rope, BLOCK_SIZE, KV_DIM, rope_dim, perm)
    print("[test] rmt tables: ch=0x%x kv[0]=0x%x rope[0]=0x%x perm=%s" % (
        int(rmt_ch[0]), int(rmt_kv[0]), int(rmt_rope[0]), "on" if args.perm else "off"))

    rows = B * S * H
    full_tbl = torch.arange(B * F_MAX_BLK, dtype=torch.int32, device=dev).reshape(B, F_MAX_BLK) % F_BLK_NUM
    sel_tbl = torch.arange(rows * S_MAX_BLK, dtype=torch.int32, device=dev).reshape(rows, S_MAX_BLK) % F_BLK_NUM
    topk_idx = torch.randint(0, 24, (rows, TOPK), dtype=torch.int32, device=dev)
    topk_idx[:, 10:] = -1
    kv_seq = torch.tensor([32, 17], dtype=torch.int32, device=dev)
    q_seq = torch.tensor([2, 2], dtype=torch.int32, device=dev)
    status_out = torch.full((rows, TOPK + 1), -77, dtype=torch.int32, device=dev)
    actual_out = torch.zeros(rows, dtype=torch.int32, device=dev)

    st = pool.next_status(1)
    stream = torch.npu.current_stream().npu_stream
    _kvops.gather_selection_rmt(
        sel_rope.data_ptr(), sel_kv.data_ptr(), sel_tbl.data_ptr(), status_out.data_ptr(),
        topk_idx.data_ptr(), full_rope.data_ptr(), full_kv.data_ptr(), full_tbl.data_ptr(),
        kv_seq.data_ptr(), q_seq.data_ptr(), rmt_ch.data_ptr(), rmt_kv.data_ptr(), rmt_rope.data_ptr(),
        st, actual_out.data_ptr(), B, S, H, TOPK, BLOCK_SIZE, F_MAX_BLK, S_MAX_BLK, F_BLK_NUM,
        TOPK_BS, KV_DIM, rope_dim, stream, 1)
    rc = _kvops.poll_status(st, 30_000_000)
    assert rc == 0, "kernel failed rc=0x%x" % (rc & 0xFFFFFFFF)
    torch.npu.synchronize()

    g_kv, g_rope, g_status, g_actual = golden(topk_idx, full_tbl, kv_seq, q_seq, full_kv.cpu(), full_rope.cpu(),
                                              sel_tbl.cpu(), bs=BLOCK_SIZE, topk_bs=TOPK_BS, kv_dim=KV_DIM,
                                              rope_dim=rope_dim)

    assert torch.equal(actual_out.cpu(), g_actual), \
        "actual_seq mismatch:\n got=%s\n exp=%s" % (actual_out.cpu().tolist(), g_actual.tolist())
    assert torch.equal(status_out.cpu(), g_status), "block_status mismatch"
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
            if not torch.equal(got_r, exp_r):
                # 诊断 dump：区分"读被丢弃(全0) / 错搬(他处数据) / 截断(部分对)"
                f_tok = seg * TOPK_BS
                f_blk = int(full_tbl[row // (S * H), f_tok // BLOCK_SIZE])
                f_off = f_tok % BLOCK_SIZE
                read_addr = int(rmt_rope[f_blk]) + f_off * rope_dim * 2
                print("[diag] rope mismatch row=%d k=%d: fBlk=%d fOff=%d sBlk=%d sOff=%d" % (row, k, f_blk, f_off,
                                                                                             s_blk, s_off))
                print("[diag] rmt_rope[fBlk]=0x%x readAddr=0x%x mod512=%d len=%dB" % (
                    int(rmt_rope[f_blk]), read_addr, read_addr % 512, TOPK_BS * rope_dim * 2))
                print("[diag] got[:8] =", ["%.4f" % v for v in got_r[0, :8].float().tolist()])
                print("[diag] exp[:8] =", ["%.4f" % v for v in exp_r[0, :8].float().tolist()])
                nz = sel_rope.view(-1, rope_dim).abs().sum(dim=1)
                nz_mask = (nz > 0).int()
                print("[diag] sel_rope nonzero token slots: %d/%d; first-32 bitmap: %s" % (
                    int(nz_mask.sum()), nz_mask.numel(), nz_mask[:32].tolist()))
                assert False, "rope data mismatch row=%d k=%d" % (row, k)
    pool.destroy()
    print("[test] gather_selection_rmt remote-path golden check PASS (rows=%d topk=%d ropeDim=%d perm=%s)" % (
        rows, TOPK, rope_dim, "on" if args.perm else "off"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
