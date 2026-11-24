# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""Step1 图捕获兼容实验：LaunchKernelWithHostArgs + 自定义 attrs 能否被 NPUGraph 记录。

流程（对齐 MemFabric 用例14 纪律 + sglang 捕获姿势）：
  图外：建池导出三表 + status 槽一次性预分配 + warmup（eager 首跑，触发内核 hcomm.Init）
  捕获：torch.npu.graph(auto_dispatch_capture=True) 内纯 launch（无 poll）——姿势逐级降级探测
  回放：sentinel reset → replay → synchronize → golden 对拍，×2 轮（地址固定复用验证）
用法：python -m tests.test_graph_probe --device 0 --ems-master <ip:port>
"""

import argparse
import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kvops_ext._kvops as _kvops  # noqa: E402
from tests.test_dsa_gather_rmt import (  # noqa: E402
    BS, QUERY_COUNT, KEY_LEN, BLOCK_SIZE, KV_DIM, ROPE_DIM, HOST_BLK_NUM, RES_BLK_NUM, POOL_ROWS,
    SENTINEL, fill_staging, golden_check)
from xds.xds_ems_pool import EmsPool  # noqa: E402

FATBIN_CANDIDATES = [
    "/usr/local/fatbin/kvc_kernel/kvc_kernel.o",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build/kernel/kvc_kernel.o"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ems-master", type=str, default="141.61.49.195:27000")
    args = ap.parse_args()

    torch.manual_seed(42)
    dev = torch.device("npu", args.device)
    torch.npu.set_device(dev)
    fatbin = next((p for p in FATBIN_CANDIDATES if os.path.exists(p)), None)
    assert fatbin, "fatbin not found: %s" % FATBIN_CANDIDATES
    ip, port = args.ems_master.rsplit(":", 1)

    # ── 图外准备：池/三表/输入/status 槽（一次分配终身复用） ──
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

    pool = EmsPool(ip, int(port), args.device, layer_num=1, layer_rows=POOL_ROWS, row_bytes=KV_DIM * 2)
    pool.create_pool(hbm.data_ptr(), hbm.numel() * 2, fatbin)
    rmt_ch, rmt_kv, rmt_rope = pool.export_full_rmt_pool(host_kv, host_rope, BLOCK_SIZE, KV_DIM, ROPE_DIM)

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

    st = _kvops.alloc_status(1)  # 一次性预分配（图外，地址固定）

    def reset_output():
        res_kv.fill_(SENTINEL)
        res_rope.fill_(SENTINEL)

    def submit():  # 纯提交：launch 到当前流（capture 中即记录到 capture stream），无 poll
        stream = torch.npu.current_stream().npu_stream
        _kvops.dsa_gather_rmt(
            res_kv.data_ptr(), res_rope.data_ptr(), res_tbl.data_ptr(), host_tbl.data_ptr(),
            pool_idx.data_ptr(), topk.data_ptr(), slots.data_ptr(), mask.data_ptr(),
            rmt_ch.data_ptr(), rmt_kv.data_ptr(), rmt_rope.data_ptr(), st,
            BS, QUERY_COUNT, BLOCK_SIZE, KV_DIM, ROPE_DIM, 2, 2,
            BS, KEY_LEN // BLOCK_SIZE, RES_BLK_NUM * BLOCK_SIZE // BS // BLOCK_SIZE,
            HOST_BLK_NUM, RES_BLK_NUM, stream, 1)

    # ── warmup（图外 eager 首跑：内核 hcomm.Init / fatbin 首执行不得在 capture 中） ──
    reset_output()
    submit()
    rc = _kvops.poll_status(st, 30_000_000)
    assert rc == 0, "warmup kernel failed rc=0x%x" % (rc & 0xFFFFFFFF)
    torch.npu.synchronize()
    n = golden_check(host_kv, host_rope, res_kv, res_rope, pool_idx, topk, slots, mask)
    print("[probe] warmup(eager) golden PASS (records=%d)" % n)

    # ── 捕获（姿势逐级探测：sglang 姿势 → 标准 CM → 裸 begin/end） ──
    graph = torch.npu.NPUGraph()
    pose = "auto_dispatch"
    try:
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            submit()
    except TypeError:
        pose = "std"
        with torch.npu.graph(graph):
            submit()
    except Exception:
        pose = "raw"
        graph.capture_begin()
        try:
            submit()
        finally:
            graph.capture_end()
    print("[probe] captured, pose=%s" % pose)

    # ── 回放 ×2（每轮 sentinel reset → replay → sync → golden） ──
    for i in range(2):
        reset_output()
        graph.replay()
        torch.npu.synchronize()
        golden_check(host_kv, host_rope, res_kv, res_rope, pool_idx, topk, slots, mask)
        print("[probe] replay#%d golden PASS" % i)

    pool.destroy()
    print("[probe] NPUGraph capture PASS: HostArgs launch recordable, pose=%s, replays=2" % pose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
