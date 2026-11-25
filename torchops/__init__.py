# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
"""dsa_gather_rmt 的 torch 层原语（两件套，torch_npu 属性挂载）。

幂等挂载到 torch_npu（对齐生产 custom_op_loader 属性挂载形态）。eager 语义 = 调用方
显式三件组合（submit → torch.npu.synchronize() → result）；GE/npugraph converter
适配为后续范围，底层经 kvops_ext pybind 执行。

用法：
    from torchops import load_dsa_gather_rmt_op
    load_dsa_gather_rmt_op()
    h = torch_npu.npu_dsa_gather_rmt_submit(...)   # 纯提交（capture 区间内亦可）
    torch.npu.synchronize()
    torch_npu.npu_dsa_gather_rmt_result(h)         # 错误码校验
使用前置：kvops init（EmsPool.create_pool 内含 kvops_ext._kvops.init）。
"""

from torchops.dsa_gather_rmt_op import (
    load_dsa_gather_rmt_op,
    npu_dsa_gather_rmt_submit,
    npu_dsa_gather_rmt_result,
)

__all__ = [
    "load_dsa_gather_rmt_op", "npu_dsa_gather_rmt_submit", "npu_dsa_gather_rmt_result",
]
