# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# 行分配器单测（RowAllocator 位于 xds.xds_ems_pool）。

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from xds.xds_ems_pool import RowAllocator, align, compute_ht_size, next_pow2


class TestRowAllocator(unittest.TestCase):
    def test_basic_alloc_free(self):
        a = RowAllocator(100)
        rows = a.alloc(10)
        self.assertEqual(rows, list(range(10)))
        self.assertEqual(a.free_rows(), 90)
        a.free(rows)
        self.assertEqual(a.free_rows(), 100)
        self.assertEqual(a.used_rows(), 0)

    def test_continuous_contract(self):
        # 连续性契约：默认分配必须连续（gather 内核 req_host_base+pos 公式依赖）
        a = RowAllocator(1000)
        r0 = a.alloc(256)
        r1 = a.alloc(256)
        self.assertEqual(r0, list(range(0, 256)))
        self.assertEqual(r1, list(range(256, 512)))
        a.free(r0)
        r2 = a.alloc(100)  # 应复用 [0..100)
        self.assertEqual(r2, list(range(100)))
        a.free(r1)
        a.free(r2)
        self.assertEqual(a.free_rows(), 1000)

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
        self.assertEqual(a.used_rows(), 0)  # 全流程泄漏断言锚点


class TestDeriveHelpers(unittest.TestCase):
    def test_align(self):
        self.assertEqual(align(1, 64), 64)
        self.assertEqual(align(64, 64), 64)
        self.assertEqual(align(65, 64), 128)

    def test_next_pow2(self):
        self.assertEqual(next_pow2(1), 1)
        self.assertEqual(next_pow2(1023), 1024)
        self.assertEqual(next_pow2(1024), 1024)

    def test_compute_ht_size(self):
        # 与 sglang-0730 host 派生一致：2 的幂，≥2×(padded+topK)，下限 1024
        self.assertEqual(compute_ht_size(16, 8), 1024)
        self.assertEqual(compute_ht_size(4112, 2048), 16384)  # 2×(4112+2048)=12320 → 16384
        self.assertEqual(compute_ht_size(1024, 1024), 4096)
        for padded, topk in ((16, 8), (4112, 2048), (64, 64), (65504, 2048)):
            ht = compute_ht_size(padded, topk)
            self.assertEqual(ht & (ht - 1), 0)  # 2 的幂
            self.assertGreaterEqual(ht, 2 * (padded + topk))
            self.assertGreaterEqual(ht, 1024)

    def test_visited_stride(self):
        # 位图行字节数：ceil(padded/8) 上取 64B
        self.assertEqual(align((4112 + 7) // 8, 64), align(514, 64))  # 514 → 576
        self.assertEqual(align((16 + 7) // 8, 64), 64)


if __name__ == "__main__":
    unittest.main()
