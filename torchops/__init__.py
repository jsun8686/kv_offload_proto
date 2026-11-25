# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""torch 算子挂载（npu_dsa_gather_rmt，D2a 多核版）。

幂等挂载到 torch_npu（对齐生产 custom_op_loader 属性挂载形态）。eager 路径全链本地
执行；GE/npugraph converter 适配为后续范围，底层经 kvops_ext pybind 执行。

用法：
    from torchops import load_dsa_gather_rmt_op
    load_dsa_gather_rmt_op()
    torch_npu.npu_dsa_gather_rmt(...)                    # eager 同步
    h = torch_npu.npu_dsa_gather_rmt_submit(...)         # 图模式纯提交（capture 区间内）
使用前置：kvops init（EmsPool.create_pool 内含 kvops_ext._kvops.init）。
"""

from torchops.dsa_gather_rmt_op import (
    load_dsa_gather_rmt_op,
    npu_dsa_gather_rmt,
    npu_dsa_gather_rmt_submit,
    npu_dsa_gather_rmt_wait,
)

__all__ = [
    "load_dsa_gather_rmt_op", "npu_dsa_gather_rmt", "npu_dsa_gather_rmt_submit", "npu_dsa_gather_rmt_wait",
]
