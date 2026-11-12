# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# CPU sieve 模型单测（xds_decode._CpuSieveOracle 是 NPU 内核对拍基准，本测试锁定其语义）。

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from xds.xds_decode import _CpuSieveOracle


def make_oracle(num_reqs, prefill, dbs):
    o = _CpuSieveOracle([(prefill, r) for r in range(num_reqs)])
    for r in range(num_reqs):
        o.reset(r, prefill, dbs)
    return o


class TestSieveOracle(unittest.TestCase):
    def test_identity_hit(self):
        # 恒等布局：token < min(prefill,dbs) 开局即命中
        o = make_oracle(1, prefill=100, dbs=8)
        slots, miss = o.update(0, [0, 3, 7], 100)
        self.assertEqual(slots, [0, 3, 7])
        self.assertEqual(miss, [0, 0, 0])

    def test_miss_and_eviction(self):
        # 恒等布局下 token∈[dbs,prefill) 必 miss：淘汰槽 = token%dbs（hand 从 0 扫，visited 空时首槽）
        o = make_oracle(1, prefill=100, dbs=8)
        slots, miss = o.update(0, [8, 9], 100)
        self.assertEqual(miss, [1, 1])
        self.assertEqual(slots, [0, 1])  # hand=0 起淘汰槽 0、槽 1
        # 淘汰槽 0 后原 token0 需重取也 miss（表已 erase）
        slots2, miss2 = o.update(0, [0], 100)
        self.assertEqual(miss2, [1])
        self.assertEqual(slots2, [2])  # hand 已推进到 2

    def test_second_chance(self):
        # second chance：命中置 visited 后，后续 miss 的 hand 扫会清位跳过已访问槽
        o = make_oracle(1, prefill=100, dbs=4)
        o.update(0, [0, 1], 100)  # 槽 0/1 visited
        slots, miss = o.update(0, [10], 100)  # miss：hand 从 0 扫，清 0/1 位后淘汰槽 2
        self.assertEqual((slots, miss), ([2], [1]))
        slots, miss = o.update(0, [11], 100)  # hand 在 3：槽 3 未访问 → 淘汰 3，hand 回 0
        self.assertEqual((slots, miss), ([3], [1]))
        slots, miss = o.update(0, [12], 100)  # 全部已清位/未访问：淘汰槽 0
        self.assertEqual((slots, miss), ([0], [1]))

    def test_ht_churn_consistency(self):
        # 连续淘汰后 t2s 与 tokens 双向一致（backward-shift 语义的模型级等价）
        o = make_oracle(1, prefill=1000, dbs=16)
        st = o.state[0]
        for i in range(200):  # 全 miss 打满 200 轮
            o.update(0, [1000 - 1 - i], 1000)
        self.assertEqual(len(st["t2s"]), 16)
        for t, s in st["t2s"].items():
            self.assertEqual(st["tokens"][s], t)

    def test_decode_extend_and_invalid(self):
        # decode 扩展槽（tokenPos>=prefill）：永不 miss；无效 token：-1 哨兵
        o = make_oracle(1, prefill=10, dbs=4)
        slots, miss = o.update(0, [12, -1, 3], 20)
        self.assertEqual(miss, [0, 0, 0])
        self.assertEqual(slots[0], -4)  # -2-(12-10)：decode 扩展槽标记
        self.assertEqual(slots[1], -1)
        self.assertEqual(slots[2], 3)  # 恒等命中

    def test_duplicate_token_first_hit(self):
        o = make_oracle(1, prefill=100, dbs=8)
        slots, miss = o.update(0, [8, 8], 100)  # 同 token 两次
        self.assertEqual(miss, [1, 0])
        self.assertEqual(slots, [0, 0])  # 第二次命中首次插入的槽

    def test_multi_req_independent(self):
        # 池行状态独立（请求间不互相淘汰）
        o = make_oracle(2, prefill=100, dbs=4)
        o.update(0, [50, 51, 52, 53], 100)  # req0 全 miss 打满
        slots, miss = o.update(1, [0, 1], 100)  # req1 恒等布局不受影响
        self.assertEqual((slots, miss), ([0, 1], [0, 0]))


if __name__ == "__main__":
    unittest.main()
