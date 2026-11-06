# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# pybind 扩展打包：ems_client._cli（NPU 侧建链）+ kvops_ext._kvops（AIV launch）。
# 构建需 pybind11（pip install pybind11 setuptools）。
# include 目录使用绝对路径：pip 构建子进程 CWD 不保证为项目根，相对路径会解析失败。

import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name="kv_offload_proto",
    version="0.1.0",
    description="XDS KV cache AIV offload prototype (ems_client / kvops_ext / xds)",
    packages=["ems_client", "kvops_ext", "xds"],
    package_dir={"ems_client": "ems/ems_client", "kvops_ext": "kvops/kvops_ext", "xds": "xds"},
    ext_modules=[
        Pybind11Extension(
            "ems_client._cli",
            sources=[os.path.join("ems", "ems_client", "hcomm_cli_ext.cpp")],
            # include 形如 "common/hcomm/hcomm_res.h"（带前缀）→ 以仓库根为 include 根
            include_dirs=[ROOT],
            cxx_std=17,
        ),
        Pybind11Extension(
            "kvops_ext._kvops",
            sources=[os.path.join("kvops", "kvops_ext", "kvops_ext.cpp")],
            # include 形如 "kvc_args.h"（无前缀）→ kernel 目录
            include_dirs=[os.path.join(ROOT, "kvops", "kernel")],
            cxx_std=17,
        ),
    ],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
)
