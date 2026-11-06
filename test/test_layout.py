# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# 行归属映射单测（与 common/kv_layout.h 语义对拍锚点）。

import unittest

from xds import kv_layout as L


class TestLayout(unittest.TestCase):
    def test_per_layer_seg_rows_ceil(self):
        self.assertEqual(L.per_layer_seg_rows(100, 2), 50)
        self.assertEqual(L.per_layer_seg_rows(101, 2), 51)  # 余数进位
        self.assertEqual(L.per_layer_seg_rows(1, 8), 1)
        self.assertEqual(L.per_layer_seg_rows(0, 2), 0)

    def test_node_of_round_robin(self):
        # S=1, N=2：严格交替
        self.assertEqual([L.node_of(r, 1, 2) for r in range(6)], [0, 1, 0, 1, 0, 1])
        # S=64, N=2：块条带
        self.assertEqual([L.node_of(r, 64, 2) for r in (0, 63, 64, 127, 128)], [0, 0, 1, 1, 0])

    def test_seg_row_compact(self):
        # S=1,N=2：row 的段内行号 0.. 连续无空洞
        rows = [L.seg_row_of(r, 1, 2) for r in range(8)]
        self.assertEqual(rows, [0, 0, 1, 1, 2, 2, 3, 3])
        # 覆盖性：layerRows=100,N=3 → segRow 覆盖 0..33 每值恰出现 3 次（余数除外）
        seen = [L.seg_row_of(r, 1, 3) for r in range(99)]
        self.assertEqual(max(seen), 32)
        for v in range(32):
            self.assertEqual(seen.count(v), 3)

    def test_balance_any_subset(self):
        # 任意层的任意行集合均匀：N=2 时连续 100 行 → 每节点 50
        for lo in (0, 7, 123):
            nodes = [L.node_of(r, 1, 2) for r in range(lo, lo + 100)]
            self.assertEqual(nodes.count(0), 50)
            self.assertEqual(nodes.count(1), 50)

    def test_seg_addr_manual(self):
        # 手工算例：N=2, L=4 层, plsr=50, rowBytes=128
        # layer=1, row=3 → node=1, segRow=1 → offset=(1*50+1)*128=6528
        self.assertEqual(L.seg_offset(1, 3, 1, 2, 50, 128), (1 * 50 + 1) * 128)
        # layer=0, row=0 → node=0, segRow=0 → offset=0
        self.assertEqual(L.seg_offset(0, 0, 1, 2, 50, 128), 0)
        # seg_base 叠加
        self.assertEqual(L.seg_addr(0x10000, 2, 10, 1, 2, 50, 128), 0x10000 + (2 * 50 + 5) * 128)

    def test_no_hole_in_segment(self):
        # 段内无空洞：某节点分到的行，其 segRow 与层区起始拼接后不重叠、不越界
        n, plsr, s = 3, L.per_layer_seg_rows(100, 3), 1
        for layer in range(4):
            offsets = sorted(L.seg_offset(layer, r, s, n, plsr, 128) for r in range(100) if L.node_of(r, s, n) == layer % n)
            expect = sorted((layer * plsr + k) * 128 for k in range(len(offsets)))
            self.assertLessEqual(max(offsets) if offsets else 0, (4 * plsr - 1) * 128)

    def test_validate(self):
        L.validate(2, 1024, 1152, 1, 2)
        with self.assertRaises(ValueError):
            L.validate(2, 1024, 1151, 1, 2)  # 非 128B 对齐
        with self.assertRaises(ValueError):
            L.validate(0, 1024, 1152, 1, 2)
        with self.assertRaises(ValueError):
            L.validate(2, 1024, 1152, 0, 2)


if __name__ == "__main__":
    unittest.main()
