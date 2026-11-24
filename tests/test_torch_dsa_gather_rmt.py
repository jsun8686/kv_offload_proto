# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""torch 层 e2e：torch_npu.npu_dsa_gather_rmt 挂载 + 远端 golden 对拍（D1.1 验收）。

复用 tests/test_dsa_gather_rmt.py 的用例构造与 golden 对拍；差异仅调用形态：
13 参张量版（维度壳内推导）替代 26 参裸指针版。
用法：python -m tests.test_torch_dsa_gather_rmt --device 0 --ems-master <ip:port> [--perm] [--block-dim N]
"""

import argparse
import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_dsa_gather_rmt import (  # noqa: E402
    BS, QUERY_COUNT, KEY_LEN, BLOCK_SIZE, KV_DIM, ROPE_DIM, HOST_BLK_NUM, RES_BLK_NUM, POOL_ROWS, SENTINEL,
    fill_staging, golden_check)
from torchops import load_dsa_gather_rmt_op, npu_dsa_gather_rmt_submit as submit_op  # noqa: E402
from xds.xds_ems_pool import EmsPool  # noqa: E402

FATBIN_CANDIDATES = [
    "/usr/local/fatbin/kvc_kernel/kvc_kernel.o",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build/kernel/kvc_kernel.o"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ems-master", type=str, default="141.61.49.195:27000")
    ap.add_argument("--perm", action="store_true")
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--graph", action="store_true", help="NPUGraph 捕获回放模式（warmup+capture+replay×N）")
    ap.add_argument("--replays", type=int, default=2, help="图回放轮数（--graph 生效）")
    args = ap.parse_args()

    torch.manual_seed(42)
    dev = torch.device("npu", args.device)
    torch.npu.set_device(dev)
    fatbin = next((p for p in FATBIN_CANDIDATES if os.path.exists(p)), None)
    assert fatbin, "fatbin not found: %s" % FATBIN_CANDIDATES
    op = load_dsa_gather_rmt_op()
    assert hasattr(torch_npu, "npu_dsa_gather_rmt"), "mount failed"
    ip, port = args.ems_master.rsplit(":", 1)

    # 与 pybind 层测试同构：单一大 hbm 张量（staging + resident 均注册）
    hbm_elems = (HOST_BLK_NUM + RES_BLK_NUM) * BLOCK_SIZE * (KV_DIM + ROPE_DIM)
    hbm = torch.empty(hbm_elems, dtype=torch.float16, device=dev)
    h_num = HOST_BLK_NUM * BLOCK_SIZE
    host_kv = hbm[:h_num * KV_DIM].view(HOST_BLK_NUM, BLOCK_SIZE, KV_DIM)
    host_rope = hbm[h_num * KV_DIM:h_num * (KV_DIM + ROPE_DIM)].view(HOST_BLK_NUM, BLOCK_SIZE, ROPE_DIM)
    r_off = h_num * (KV_DIM + ROPE_DIM)
    res_kv = hbm[r_off:r_off + RES_BLK_NUM * BLOCK_SIZE * KV_DIM].view(RES_BLK_NUM, BLOCK_SIZE, KV_DIM)
    res_rope = hbm[r_off + RES_BLK_NUM * BLOCK_SIZE * KV_DIM:].view(RES_BLK_NUM, BLOCK_SIZE, ROPE_DIM)
    fill_staging(host_kv.view(-1, KV_DIM), 0)
    fill_staging(host_rope.view(-1, ROPE_DIM), 1024)
    res_kv.fill_(SENTINEL)
    res_rope.fill_(SENTINEL)

    pool = EmsPool(ip, int(port), args.device, layer_num=1, layer_rows=POOL_ROWS, row_bytes=KV_DIM * 2)
    pool.create_pool(hbm.data_ptr(), hbm.numel() * 2, fatbin, n_channels=args.block_dim)
    perm = torch.randperm(HOST_BLK_NUM) if args.perm else None
    rmt_ch, rmt_kv, rmt_rope = pool.export_full_rmt_pool(host_kv, host_rope, BLOCK_SIZE, KV_DIM, ROPE_DIM, perm,
                                                         n_core=args.block_dim)

    host_tbl = torch.arange(HOST_BLK_NUM, dtype=torch.int32, device=dev).view(BS, KEY_LEN // BLOCK_SIZE)
    res_tbl = torch.arange(RES_BLK_NUM, dtype=torch.int32, device=dev).view(BS, -1)
    pool_idx = torch.roll(torch.arange(BS, dtype=torch.int32), shifts=1).to(dev)
    step = max(1, KEY_LEN // QUERY_COUNT)
    topk = (torch.arange(QUERY_COUNT, dtype=torch.int32) * step).remainder(KEY_LEN).view(1, -1).repeat(
        BS, 1).to(dev)
    from tests.test_dsa_gather_rmt import TOTAL_SLOTS  # noqa: E402
    slots = (torch.arange(QUERY_COUNT, dtype=torch.int32) * 5).remainder(TOTAL_SLOTS).view(1, -1).repeat(
        BS, 1).to(dev)
    miss = torch.zeros(QUERY_COUNT, dtype=torch.int32)
    miss[torch.arange(64, dtype=torch.int64) * QUERY_COUNT // 64] = 1
    mask = miss.view(1, -1).repeat(BS, 1).to(dev)

    kv4 = res_kv.view(RES_BLK_NUM, BLOCK_SIZE, 1, KV_DIM)
    rope4 = res_rope.view(RES_BLK_NUM, BLOCK_SIZE, 1, ROPE_DIM)

    if args.graph:
        # 图模式（Step1 实证姿势）：warmup(eager) → capture(submit) → replay×N + golden
        kv_alias, rope_alias = op(kv4, rope4, res_tbl, host_tbl, pool_idx, topk, slots, mask, BLOCK_SIZE, BS,
                                  rmt_ch, rmt_kv, rmt_rope, block_dim=args.block_dim)
        torch.npu.synchronize()
        golden_check(host_kv, host_rope, res_kv, res_rope, pool_idx, topk, slots, mask)  # warmup 校验
        print("[test] graph warmup(eager) golden PASS")
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            submit_op(kv4, rope4, res_tbl, host_tbl, pool_idx, topk, slots, mask, BLOCK_SIZE, BS,
                      rmt_ch, rmt_kv, rmt_rope, block_dim=args.block_dim)
        for i in range(args.replays):
            res_kv.fill_(SENTINEL)
            res_rope.fill_(SENTINEL)
            graph.replay()
            torch.npu.synchronize()
            golden_check(host_kv, host_rope, res_kv, res_rope, pool_idx, topk, slots, mask)
            print("[test] graph replay#%d golden PASS" % i)
        pool.destroy()
        print("[test] torch_npu.npu_dsa_gather_rmt graph e2e PASS (replays=%d blockDim=%d perm=%s)" % (
            args.replays, args.block_dim, "on" if args.perm else "off"))
        return 0

    # eager 路径：13 参张量版（原地写，返回 alias）
    kv_alias, rope_alias = op(kv4, rope4, res_tbl, host_tbl, pool_idx, topk, slots, mask, BLOCK_SIZE, BS,
                              rmt_ch, rmt_kv, rmt_rope, block_dim=args.block_dim)
    torch.npu.synchronize()
    assert kv_alias.data_ptr() == res_kv.data_ptr() and rope_alias.data_ptr() == res_rope.data_ptr(), \
        "in-place alias broken"

    n = golden_check(host_kv, host_rope, res_kv, res_rope, pool_idx, topk, slots, mask)
    pool.destroy()
    print("[test] torch_npu.npu_dsa_gather_rmt e2e PASS (records=%d blockDim=%d perm=%s)" % (
        n, args.block_dim, "on" if args.perm else "off"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
