# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# 行分配器单测。

import unittest

from xds.mempoolhost import RowAllocator


class TestRowAllocator(unittest.TestCase):
    def test_basic_alloc_free(self):
        a = RowAllocator(100)
        rows = a.alloc(10)
        self.assertEqual(rows, list(range(10)))
        self.assertEqual(a.free_rows(), 90)
        a.free(rows)
        self.assertEqual(a.free_rows(), 100)
        self.assertEqual(a.used_rows(), 0)

    def test_continuous_preferred(self):
        # 碎片化后再分配大块：应拼接离散行
        a = RowAllocator(20)
        r0 = a.alloc(8)   # [0..8)
        r1 = a.alloc(8)   # [8..16)
        r2 = a.alloc(4)   # [16..20)
        a.free(r1)        # 留出 [8..16) 大洞
        got = a.alloc(6)  # 只能从 [8..16) 取连续
        self.assertEqual(got, list(range(8, 14)))
        a.free(r0)
        a.free(r2)
        a.free(got)
        self.assertEqual(a.free_rows(), 20)

    def test_fragment_merge(self):
        a = RowAllocator(10)
        rows = [a.alloc(1)[0] for _ in range(10)]
        self.assertEqual(rows, list(range(10)))
        for r in rows[:5]:
            a.free([r])  # 逐行释放（碎片）
        self.assertEqual(a.free_rows(), 5)
        got = a.alloc(5)  # 应合并出 [0..5) 连续
        self.assertEqual(sorted(got), [0, 1, 2, 3, 4])

    def test_exhausted(self):
        a = RowAllocator(4)
        with self.assertRaises(MemoryError):
            a.alloc(5)

    def test_out_of_range(self):
        a = RowAllocator(4)
        with self.assertRaises(ValueError):
            a.free([99])

    def test_leak_assert(self):
        a = RowAllocator(64)
        blocks = [a.alloc(16) for _ in range(4)]
        for b in blocks:
            a.free(b)
        self.assertEqual(a.used_rows(), 0)  # V3 断言锚点


if __name__ == "__main__":
    unittest.main()
