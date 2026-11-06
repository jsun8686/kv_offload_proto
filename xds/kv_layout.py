# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# 行归属映射（层维度均衡）：与 common/kv_layout.h 语义严格一致（单测对拍锚点）。
#
#   node(row)     = (row / S) % N          # 层无关：任意层的行均匀散到 N 段
#   seg_row(row)  = (row / S) / N          # 段内层内行号（紧凑无空洞）
#   addr(l, row)  = seg_base + (l * per_layer_seg_rows + seg_row(row)) * row_bytes


def per_layer_seg_rows(layer_rows: int, node_num: int) -> int:
    return (layer_rows + node_num - 1) // node_num


def node_of(row: int, stripe_rows: int, node_num: int) -> int:
    return (row // stripe_rows) % node_num


def seg_row_of(row: int, stripe_rows: int, node_num: int) -> int:
    return (row // stripe_rows) // node_num


def seg_offset(layer: int, row: int, stripe_rows: int, node_num: int, plsr: int, row_bytes: int) -> int:
    return (layer * plsr + seg_row_of(row, stripe_rows, node_num)) * row_bytes


def seg_addr(seg_base: int, layer: int, row: int, stripe_rows: int, node_num: int, plsr: int, row_bytes: int) -> int:
    return seg_base + seg_offset(layer, row, stripe_rows, node_num, plsr, row_bytes)


def validate(layer_num: int, layer_rows: int, row_bytes: int, stripe_rows: int, node_num: int) -> None:
    if node_num < 1 or node_num > 64 or layer_num < 1 or layer_rows < 1:
        raise ValueError("invalid layout dims: N={} L={} rows={}".format(node_num, layer_num, layer_rows))
    if row_bytes <= 0 or row_bytes % 128 != 0:
        raise ValueError("row_bytes must be 128B aligned, got {}".format(row_bytes))
    if stripe_rows < 1:
        raise ValueError("stripe_rows must be >= 1")
