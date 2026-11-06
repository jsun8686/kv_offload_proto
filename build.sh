#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# 统一构建入口：./build.sh [emsd|kernel|ext|all]
#   emsd   : 资源节点守护进程（任意 Linux x86/aarch64，无 CANN 依赖）
#   kernel : kvc 内核 fatbin（NPU 机，需 source CANN set_env.sh；SOC_VERSION 可用环境变量覆盖）
#   ext    : pybind 扩展（ems_client._cli / kvops_ext._kvops，NPU 机；资源节点不需要）
#   all    : 三者全部（NPU 机完整环境）
set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-all}"
JOBS="${MF_BUILD_JOBS:-$(nproc)}"

build_emsd() {
    echo "== build emsd =="
    cmake -B build/emsd -D_TARGET=emsd -DCMAKE_BUILD_TYPE=Release
    cmake --build build/emsd -j"${JOBS}"
}

build_kernel() {
    echo "== build kvc kernel fatbin =="
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
        echo "ERROR: ASCEND_HOME_PATH not set, source CANN set_env.sh first" >&2
        exit 1
    fi
    cmake -B build/kernel -D_TARGET=kernel -DCMAKE_BUILD_TYPE=Release
    cmake --build build/kernel -j"${JOBS}"
    echo "fatbin: $(find build/kernel -name '*.o' | head -5)"
}

build_ext() {
    echo "== build pybind exts (editable) =="
    pip install -e . --no-build-isolation
}

case "${TARGET}" in
emsd) build_emsd ;;
kernel) build_kernel ;;
ext) build_ext ;;
all) build_emsd; build_ext; build_kernel ;;
*)
    echo "usage: $0 [emsd|kernel|ext|all]" >&2
    exit 1
    ;;
esac
echo "== done (${TARGET}) =="
