# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""torch 算子层挂载：npu_gather_selection_kv_cache_rmt（V3）。

形态对齐 GLM 5.2 custom_op_loader.load_offload_gather_op：属性挂载到 torch_npu
（eager 路径；GE/npugraph converter 不在 V3 范围）。底层经 kvops_ext pybind 执行。
用法：
    from torchops import load_gather_rmt_op
    load_gather_rmt_op()          # 幂等
    actual_seq = torch_npu.npu_gather_selection_kv_cache_rmt(...)
调用前需完成 kvops init（EmsPool.create_pool 或 kvops_ext._kvops.init）。
"""

from torchops.gather_rmt_op import load_gather_rmt_op, npu_gather_selection_kv_cache_rmt

__all__ = ["load_gather_rmt_op", "npu_gather_selection_kv_cache_rmt"]
