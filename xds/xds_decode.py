# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# xds_decode.py：XDS decode 流程驱动（M1 非图路径）。
#
# 流程：初始化（EMS 建池+HBM 池注册+建链）→ prefill（模式填充+全层导出）→ decode 循环
# （topK 随机采样【预留真实 lightning indexer 接口】→ miss 判定（位图热缓冲）→ 批量导入
# → 完成等待+抽查校验）→ 统计。行内容指纹：word = (row*P1 + layer*P2 + k) * M1 ^ M2。

import argparse
import time

import numpy as np

from . import kv_layout
from .mempoolhost import MemPoolHost, RowAllocator


def parse_args():
    p = argparse.ArgumentParser(description="XDS decode pipeline driver (M1)")
    p.add_argument("--ems-master", required=True, help="master emsd ip:port")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--eid", required=True, help="本卡 EID（72 hex 字符）")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--layer-rows", type=int, default=1024)
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--qk-rope", type=int, default=64)
    p.add_argument("--dtype-size", type=int, default=2)
    p.add_argument("--stripe-rows", type=int, default=1)
    p.add_argument("--prefill", type=int, default=256, help="prefill 行数（每请求）")
    p.add_argument("--num-reqs", type=int, default=8)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--hot-buffer", type=int, default=0, help="HBM 热缓冲行数（0=全部 miss 模拟）")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fill_pattern(arr: np.ndarray, layer: int, row0: int, row_cnt: int) -> None:
    """按行指纹填充 HBM 池第 layer 层 [row0, row0+row_cnt) 行（word 粒度）。"""
    p1, p2, m1, m2 = 0x9E3779B1, 0x85EBCA77, 0x27220A95, 0x165667B1
    base = np.arange(row_cnt, dtype=np.uint64) + np.uint64(row0)
    v = (base * np.uint64(p1) + np.uint64(layer * p2)) * np.uint64(m1) ^ np.uint64(m2)
    words = arr.size // row_cnt if row_cnt > 0 else 0
    # 广播到整行 word 数
    row_words = arr.shape[-1]
    ramp = np.arange(row_words, dtype=np.uint64) * np.uint64(m1)
    arr[0:row_cnt] = (v[:, None] + ramp[None, :]).astype(arr.dtype)


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    row_words = args.kv_lora_rank + args.qk_rope
    row_bytes = ((row_words * args.dtype_size + 127) // 128) * 128
    layer_rows = args.layer_rows

    print("[xds] config: layers={} layerRows={} rowWords={} rowBytes={} stripe={}".format(
        args.layers, layer_rows, row_words, row_bytes, args.stripe_rows))

    import torch  # NPU 机上为 torch_npu 环境
    import torch_npu  # noqa: F401

    dev = "npu:{}".format(args.device)
    hbm = torch.zeros((args.layers, layer_rows, row_bytes // 4), dtype=torch.int32, device=dev)
    hbm_np = hbm.cpu().numpy().view(np.uint32).reshape(args.layers, layer_rows, row_bytes // 4)

    master_ip, master_port = args.ems_master.rsplit(":", 1)
    from ems_client import EmsClient

    ems = EmsClient(master_ip, int(master_port))
    ems.connect()
    ems.ping()

    pool = MemPoolHost(ems, args.device, args.eid, args.layers, layer_rows, row_bytes, stripe_rows=args.stripe_rows,
                       max_batch=max(args.topk * args.num_reqs, 16))
    pool.create_pool(hbm.data_ptr(), hbm.numel() * 4)
    print("[xds] pool ready: nodes={} perLayerSegRows={} channels={}".format(
        pool.node_num, pool.per_layer_seg_rows, len(pool.channels)))

    alloc = pool.make_allocator()

    # ── prefill：行分配 + 指纹填充 + 全层导出 ──
    req_rows = [alloc.alloc(args.prefill) for _ in range(args.num_reqs)]
    t0 = time.perf_counter()
    for req_id, rows in enumerate(req_rows):
        for l in range(args.layers):
            fill_pattern(hbm_np[l, rows[0]:rows[0] + args.prefill], l, rows[0], args.prefill)
    hbm.copy_(torch.from_numpy(hbm_np.view(np.int32)).to(dev))
    st = pool.batch_export_alllayers(0, args.prefill, hbm[0], layer_rows * row_bytes)
    pool.wait(st)
    t_export = time.perf_counter() - t0
    export_bytes = args.layers * args.prefill * row_bytes
    print("[xds] prefill export done: {} rows x {} layers = {} B in {:.1f} us".format(
        args.prefill, args.layers, export_bytes, t_export * 1e6))

    # 清空 HBM 池，验证导入正确性
    hbm.zero_()

    hot = set(range(args.hot_buffer)) if args.hot_buffer > 0 else set()
    rng = np.random.default_rng(args.seed)
    total_rows_imported = 0
    t_import_total = 0.0

    for step in range(args.steps):
        # topK 采样（预留真实 indexer：替换为 lightning indexer 输出即可）
        all_rows = np.concatenate([np.array(r) for r in req_rows])
        picks = rng.choice(all_rows, size=min(args.topk, all_rows.size), replace=False)
        # miss 判定：不在 HBM 热缓冲中的行需从远端导入
        miss = [int(r) for r in picks if int(r) not in hot]
        if not miss:
            continue
        # HBM 目标槽位 = 原行号（简化：导入回原位；热缓冲槽位管理见 M3）
        indices_host = torch.tensor(miss, dtype=torch.int64, device=dev)
        indices_dst = indices_host.clone()
        num_real = torch.tensor([len(miss)], dtype=torch.int64, device=dev)
        layer_id = step % args.layers
        t0 = time.perf_counter()
        st = pool.batch_import(layer_id, indices_host, indices_dst, num_real, hbm[layer_id])
        pool.wait(st)
        dt = time.perf_counter() - t0
        t_import_total += dt
        total_rows_imported += len(miss)

        # 抽查校验（首/中/尾 3 行）
        layer_np = hbm_np[layer_id]
        for r in (miss[0], miss[len(miss) // 2], miss[-1]):
            v = (np.uint64(r) * np.uint64(0x9E3779B1) + np.uint64(layer_id * 0x85EBCA77)) * np.uint64(0x27220A95) \
                ^ np.uint64(0x165667B1)
            expect0 = np.uint32(v & np.uint64(0xFFFFFFFF))
            got = layer_np[r, 0]
            if got != expect0:
                print("[xds][ERROR] verify failed, step={} layer={} row={} expect={} got={}".format(
                    step, layer_id, r, expect0, got))
                return 1
        print("[xds] step {}: layer={} miss={} rows in {:.1f} us".format(step, layer_id, len(miss), dt * 1e6))

    if total_rows_imported > 0:
        bw = total_rows_imported * row_bytes / t_import_total / 1e9 if t_import_total > 0 else 0
        print("[xds] summary: imported {} rows, avg {:.1f} us/row, aggregate {:.2f} GB/s".format(
            total_rows_imported, t_import_total * 1e6 / total_rows_imported, bw))

    for rows in req_rows:
        alloc.free(rows)
    assert alloc.used_rows() == 0, "allocator leak"
    pool.destroy()
    ems.close()
    print("[xds] all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
