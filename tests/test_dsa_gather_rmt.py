# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""dsa_gather_rmt 远端 E2E golden 对拍（D1.0 验收，NPU 执行）。

语义基准：vllm-ascend-csrc dsa_gather golden.py——miss_mask 唯一闸门 + pool_entry 两跳寻址
+ sentinel 零搬运。源池 = staging 张量按公式填充 → export_full_rmt_pool 导出（三表）；
resident 池与 staging 同一大 hbm 张量（ReadNbi dst 须在注册范围）。
用法：python -m tests.test_dsa_gather_rmt --device 7 --ems-master <ip:27000> [--perm] [--block-dim N]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kvops_ext._kvops as _kvops  # noqa: E402
from xds.xds_ems_pool import EmsPool  # noqa: E402

FATBIN_CANDIDATES = [
    "/usr/local/fatbin/kvc_kernel/kvc_kernel.o",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build/kernel/kvc_kernel.o"),
]

BS = 4                 # num_requests
QUERY_COUNT = 256      # 每请求 topk 槽位
KEY_LEN = 4096         # 每池行 token 数
BLOCK_SIZE = 128
KV_DIM, ROPE_DIM = 512, 64
TOTAL_SLOTS = 2048     # resident 池总 slot
MISSES_PER_REQ = 64    # miss 密度（golden.py 对齐：均匀散布）
SENTINEL = -1000.0

HOST_BLK_PER_REQ = KEY_LEN // BLOCK_SIZE              # 32
HOST_BLK_NUM = BS * HOST_BLK_PER_REQ                  # 128
RES_BLK_PER_REQ = (TOTAL_SLOTS + BLOCK_SIZE - 1) // BLOCK_SIZE  # 16
RES_BLK_NUM = BS * RES_BLK_PER_REQ                    # 64
HOST_SLOTS_PER_ENTRY = HOST_BLK_PER_REQ * BLOCK_SIZE  # 4096
RES_SLOTS_PER_REQ = RES_BLK_PER_REQ * BLOCK_SIZE      # 2048
POOL_ROWS = HOST_BLK_NUM * BLOCK_SIZE * (KV_DIM + ROPE_DIM) // KV_DIM


def fill_staging(flat, offset):
    """按 golden.py 公式填充：value = (token % 512) + channel_idx + offset。"""
    tok = torch.arange(flat.shape[0], dtype=torch.int32, device=flat.device).remainder(512)
    base = tok.to(flat.dtype).view(-1, 1)
    ch = torch.arange(flat.shape[1], dtype=flat.dtype, device=flat.device).view(1, -1)
    flat.copy_(base)
    flat.add_(ch)
    flat.add_(offset)


def golden_check(host_kv, host_rope, res_kv, res_rope, pool_idx, topk, slots, mask):
    """对拍 golden.py：written 集合数据 + sentinel untouched 精确一致。"""
    active = mask.cpu().to(torch.bool)
    batch_rows = torch.arange(BS, dtype=torch.int64).view(-1, 1).expand_as(active)
    source_slots = (pool_idx.cpu().view(-1, 1).to(torch.int64) * HOST_SLOTS_PER_ENTRY +
                    topk.cpu().to(torch.int64))[active]
    dest_flat = (batch_rows * RES_SLOTS_PER_REQ + slots.cpu().to(torch.int64))[active]
    expected_sources = torch.full((RES_BLK_NUM * BLOCK_SIZE,), -1, dtype=torch.int64)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=7)
    ap.add_argument("--ems-master", type=str, default="141.61.49.198:27000")
    ap.add_argument("--perm", action="store_true", help="远端槽位乱序")
    ap.add_argument("--block-dim", type=int, default=1, help="AIV 核数（D2 矩阵）")
    args = ap.parse_args()

    torch.manual_seed(42)
    dev = torch.device("npu", args.device)
    torch.npu.set_device(dev)
    fatbin = next((p for p in FATBIN_CANDIDATES if os.path.exists(p)), None)
    assert fatbin, "fatbin not found: %s" % FATBIN_CANDIDATES
    ip, port = args.ems_master.rsplit(":", 1)

    # 单一大 hbm 张量：host staging（导出源）+ resident 池（ReadNbi dst，须整段注册）
    hbm_elems = HOST_BLK_NUM * BLOCK_SIZE * (KV_DIM + ROPE_DIM) + RES_BLK_NUM * BLOCK_SIZE * (KV_DIM + ROPE_DIM)
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
    print("[test] rmt tables: ch=0x%x kv[0]=0x%x rope[0]=0x%x perm=%s n_core=%d" % (
        int(rmt_ch[0]), int(rmt_kv[0]), int(rmt_rope[0]), "on" if args.perm else "off", args.block_dim))

    # 索引元数据（语义对齐原版 test.py 构造）
    host_tbl = torch.arange(HOST_BLK_NUM, dtype=torch.int32, device=dev).view(BS, HOST_BLK_PER_REQ)
    res_tbl = torch.arange(RES_BLK_NUM, dtype=torch.int32, device=dev).view(BS, RES_BLK_PER_REQ)
    pool_idx = torch.roll(torch.arange(BS, dtype=torch.int32), shifts=1).to(dev)  # [3,0,1,2] 验证两跳
    step = max(1, KEY_LEN // QUERY_COUNT)
    topk = (torch.arange(QUERY_COUNT, dtype=torch.int32) * step).remainder(KEY_LEN).view(1, -1).repeat(
        BS, 1).to(dev)
    slots = (torch.arange(QUERY_COUNT, dtype=torch.int32) * 5).remainder(TOTAL_SLOTS).view(1, -1).repeat(
        BS, 1).to(dev)
    miss_pos = torch.arange(MISSES_PER_REQ, dtype=torch.int64) * QUERY_COUNT // MISSES_PER_REQ
    mask_row = torch.zeros(QUERY_COUNT, dtype=torch.int32)
    mask_row[miss_pos] = 1
    mask = mask_row.view(1, -1).repeat(BS, 1).to(dev)

    st = pool.next_status(args.block_dim)
    stream = torch.npu.current_stream().npu_stream
    _kvops.dsa_gather_rmt(
        res_kv.data_ptr(), res_rope.data_ptr(), res_tbl.data_ptr(), host_tbl.data_ptr(),
        pool_idx.data_ptr(), topk.data_ptr(), slots.data_ptr(), mask.data_ptr(),
        rmt_ch.data_ptr(), rmt_kv.data_ptr(), rmt_rope.data_ptr(), st,
        BS, QUERY_COUNT, BLOCK_SIZE, KV_DIM, ROPE_DIM, 2, 2,
        BS, HOST_BLK_PER_REQ, RES_BLK_PER_REQ, HOST_BLK_NUM, RES_BLK_NUM, stream, args.block_dim)
    pool.wait(st, block_dim=args.block_dim)  # 每核独立 64B 槽（status+coreId*64）
    torch.npu.synchronize()

    n = golden_check(host_kv, host_rope, res_kv, res_rope, pool_idx, topk, slots, mask)
    pool.destroy()
    print("[test] dsa_gather_rmt remote golden check PASS (records=%d blockDim=%d perm=%s)" % (
        n, args.block_dim, "on" if args.perm else "off"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
