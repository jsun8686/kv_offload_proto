# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""torch 层 e2e（v2）：torch_npu.npu_dsa_gather_rmt_v2 挂载 + 远端 golden 对拍（AICPU 数据面）。

与 test_torch_dsa_gather_rmt.py 同构（复用同一用例构造与 golden 对拍），差异：
  - 链路：EmsPool(proto="roce") → ROCE + AICPU 通道（A3 数据面，v2）；
  - emsd 须以 --roce --dev-ip <NPU RoCE IPv4> 启动（两端的 NPU IP 须在 hccn_tool 同平面内互通）；
  - 本端 NPU RoCE IP 经 --dev-ip 参数或 KV_CLI_DEV_IP 环境变量传入；
  - block_dim 恒 1（单 AICPU 实例）。
用法：python -m tests.test_torch_dsa_gather_rmt_v2 --device 0 --ems-master <ip:port> \
        [--dev-ip <NPU RoCE IPv4>] [--perm]
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
from torchops import load_dsa_gather_rmt_op  # noqa: E402
from xds.xds_ems_pool import EmsPool  # noqa: E402

FATBIN_CANDIDATES = [
    "/usr/local/fatbin/kvc_kernel/kvc_kernel.o",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build/kernel/kvc_kernel.o"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ems-master", type=str, default="141.61.49.195:27000")
    ap.add_argument("--dev-ip", type=str, default=os.environ.get("KV_CLI_DEV_IP", ""),
                    help="本 NPU RoCE 参数面 IP（默认取 KV_CLI_DEV_IP）")
    ap.add_argument("--perm", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(42)
    dev = torch.device("npu", args.device)
    torch.npu.set_device(dev)
    fatbin = next((p for p in FATBIN_CANDIDATES if os.path.exists(p)), None)
    assert fatbin, "fatbin not found: %s" % FATBIN_CANDIDATES
    aicpu_json = os.path.join(os.path.dirname(fatbin), "libcann_kvops_aicpu.json")
    assert os.path.exists(aicpu_json), "AICPU json not found: %s（kernel 目标须重编并随 fatbin 部署）" % aicpu_json
    load_dsa_gather_rmt_op()
    assert hasattr(torch_npu, "npu_dsa_gather_rmt_v2"), "mount failed"
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

    pool = EmsPool(ip, int(port), args.device, layer_num=1, layer_rows=POOL_ROWS, row_bytes=KV_DIM * 2,
                   proto="roce", dev_ip=args.dev_ip or None)
    pool.create_pool(hbm.data_ptr(), hbm.numel() * 2, fatbin)
    perm = torch.randperm(HOST_BLK_NUM) if args.perm else None
    rmt_ch, rmt_kv, rmt_rope = pool.export_full_rmt_pool(host_kv, host_rope, BLOCK_SIZE, KV_DIM, ROPE_DIM, perm)

    host_tbl = torch.arange(HOST_BLK_NUM, dtype=torch.int32).view(BS, KEY_LEN // BLOCK_SIZE).to(dev)
    res_tbl = torch.arange(RES_BLK_NUM, dtype=torch.int32).view(BS, -1).to(dev)
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

    # torch 层调用：13 参张量版（原地写，返回 alias）
    kv_alias, rope_alias = torch_npu.npu_dsa_gather_rmt_v2(
        res_kv.view(RES_BLK_NUM, BLOCK_SIZE, 1, KV_DIM), res_rope.view(RES_BLK_NUM, BLOCK_SIZE, 1, ROPE_DIM),
        res_tbl, host_tbl, pool_idx, topk, slots, mask, BLOCK_SIZE, BS,
        rmt_ch, rmt_kv, rmt_rope)
    torch.npu.synchronize()
    assert kv_alias.data_ptr() == res_kv.data_ptr() and rope_alias.data_ptr() == res_rope.data_ptr(), \
        "in-place alias broken"

    n = golden_check(host_kv, host_rope, res_kv, res_rope, pool_idx, topk, slots, mask)
    pool.destroy()
    print("[test] torch_npu.npu_dsa_gather_rmt_v2 e2e PASS (records=%d perm=%s)" % (
        n, "on" if args.perm else "off"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
