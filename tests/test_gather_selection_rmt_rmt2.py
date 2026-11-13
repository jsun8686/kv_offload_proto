# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""gather_selection_rmt 多节点 E2E golden 对拍（V2 验收，NPU 执行）。

拓扑：emsd master + worker 各持一段 HBM（--members 两节点）；EmsPool 单点连 master，
pool_create 自动跨节点分段；export_full_rmt_pool 行交错一次全区导出（内核 KvcNodeOf 分发），
full_rmt_* 三表按段路由；内核多通道收尾（逐通道 Commit+Drain）。
部署：master: --members ip:27000,ip:27001 --device 3；worker: --port 27001 --device 4
用法：python -m tests.test_gather_selection_rmt_rmt2 --device 0 --ems-master <ip:27000> [--perm]
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
KV_DIM = 256
F_MAX_BLK, F_BLK_NUM = 16, 64
S_MAX_BLK = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ems-master", type=str, default="141.61.49.195:27000")
    ap.add_argument("--perm", action="store_true", help="全局槽位乱序（跨段路由+表驱动验证）")
    args = ap.parse_args()
    rope_dim = 64
    pool_rows = F_BLK_NUM * BLOCK_SIZE * (KV_DIM + rope_dim) // KV_DIM  # 全局行数（n 段交错）

    torch.manual_seed(42)
    dev = torch.device("npu", args.device)
    torch.npu.set_device(dev)
    ip, port = args.ems_master.rsplit(":", 1)

    fatbin = next((p for p in FATBIN_CANDIDATES if os.path.exists(p)), None)
    assert fatbin, "fatbin not found: %s" % FATBIN_CANDIDATES

    hbm_elems = F_BLK_NUM * BLOCK_SIZE * (KV_DIM + rope_dim) * 2
    hbm = torch.randn(hbm_elems, dtype=torch.float16, device=dev)
    f_num = F_BLK_NUM * BLOCK_SIZE
    full_kv = hbm[:f_num * KV_DIM].view(F_BLK_NUM, BLOCK_SIZE, KV_DIM)
    full_rope = hbm[f_num * KV_DIM:f_num * (KV_DIM + rope_dim)].view(F_BLK_NUM, BLOCK_SIZE, rope_dim)
    sel_kv = hbm[f_num * (KV_DIM + rope_dim):f_num * (KV_DIM + rope_dim) + f_num * KV_DIM].view(
        F_BLK_NUM, BLOCK_SIZE, KV_DIM).zero_()
    sel_rope = hbm[f_num * (KV_DIM + rope_dim) + f_num * KV_DIM:].view(F_BLK_NUM, BLOCK_SIZE, rope_dim).zero_()

    pool = EmsPool(ip, int(port), args.device, layer_num=1, layer_rows=pool_rows, row_bytes=KV_DIM * 2)
    pool.create_pool(hbm.data_ptr(), hbm.numel() * 2, fatbin)
    n = pool.node_num
    assert n == 2, "V2 dual-node test expects node_num=2, got %d (check emsd --members)" % n
    for s in range(n):
        seg = pool.pool.segments[s]
        print("[test] node%d: segBase=0x%x ctrl=%s:%d hcommPort=%d ch=0x%x" % (
            s, seg.seg_base, seg.ctrl_ip, seg.ctrl_port, seg.hcomm_listen_port, int(pool.channels[s])))

    perm = torch.randperm(F_BLK_NUM) if args.perm else None
    rmt_ch, rmt_kv, rmt_rope = pool.export_full_rmt_pool(full_kv, full_rope, BLOCK_SIZE, KV_DIM, rope_dim, perm)
    route_cnt = [(rmt_ch == int(pool.channels[s])).sum().item() for s in range(n)]
    print("[test] block routing: %s blocks/node, perm=%s" % (
        route_cnt, "on" if args.perm else "off"))

    rows = B * S * H
    # CPU 上算好再搬上卡：NPU 上建 int32 中间张量会触发 internal-format 回退告警（数值无影响）
    full_tbl = (torch.arange(B * F_MAX_BLK, dtype=torch.int32) % F_BLK_NUM).reshape(B, F_MAX_BLK).to(dev)
    sel_tbl = (torch.arange(rows * S_MAX_BLK, dtype=torch.int32) % F_BLK_NUM).reshape(rows, S_MAX_BLK).to(dev)
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
            assert torch.equal(got_r, exp_r), "rope data mismatch row=%d k=%d" % (row, k)
    pool.destroy()
    print("[test] gather_selection_rmt multi-node(%d) golden check PASS (rows=%d topk=%d perm=%s)" % (
        n, rows, TOPK, "on" if args.perm else "off"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
