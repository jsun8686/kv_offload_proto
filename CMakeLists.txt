# ----------------------------------------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# ----------------------------------------------------------------------------------------------------------
# 顶层构建：emsd（通用 C++，无 CANN 依赖）+ kvc 内核（fatbin，需 NPU 机 CANN 环境）。
# pybind 扩展（ems_client._cli / kvops_ext._kvops）走 setup.py（build.sh 统一入口）。
#
# 构建：
#   emsd（资源节点/任意 Linux）：cmake -B build-emsd -D_TARGET=emsd && cmake --build build-emsd
#   内核 fatbin（NPU 机，先 source CANN set_env.sh）：
#       cmake -B build-kernel -D_TARGET=kernel && cmake --build build-kernel
#   SOC_VERSION 默认 ascend950pr_9599（A5），板型确认：npu-smi info -t board -i <id>
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(kv_offload_proto LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED _TARGET)
    set(_TARGET "all" CACHE STRING "build target: emsd | kernel | all")
endif()

if(_TARGET STREQUAL "emsd" OR _TARGET STREQUAL "all")
    add_executable(emsd
        ems/emsd/main.cc
        ems/emsd/ctrl_server.cc
        ems/emsd/hcomm_host.cc
        ems/emsd/pool_mgr.cc
        common/ems_eid.cc
    )
    # include 形如 "common/kv_proto.h"（带前缀）→ 以仓库根为 include 根
    target_include_directories(emsd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(emsd PRIVATE dl pthread)
endif()

if(_TARGET STREQUAL "kernel" OR _TARGET STREQUAL "all")
    if(NOT DEFINED ENV{ASCEND_HOME_PATH})
        message(FATAL_ERROR "ASCEND_HOME_PATH not set, please source CANN set_env.sh first (kernel target)")
    endif()
    if(NOT DEFINED SOC_VERSION)
        set(SOC_VERSION "ascend950pr_9599" CACHE STRING "SoC version for ascendc fatbin build")
    endif()
    if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
        if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
            set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
        else()
            set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
        endif()
    endif()
    set(ASCENDC_CMAKE "$ENV{ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
    if(NOT EXISTS ${ASCENDC_CMAKE})
        message(FATAL_ERROR "${ASCENDC_CMAKE} not found, check CANN(>=9.1) install")
    endif()
    include(${ASCENDC_CMAKE})

    if(NOT ASC_DEVKIT_ROOT)
        set(ASC_DEVKIT_ROOT "$ENV{HOME}/asc-devkit" CACHE PATH "asc-devkit root for hcomm.h")
    endif()
    # hcomm 头文件定位（跨版本兼容）：
    #   - CANN <= 9.1：asc-devkit 源码自带 include/adv_api/hcomm/hcomm.h（设备侧 Hcomm API）；
    #   - CANN >= 9.2：asc-devkit 移除 hcomm 目录，设备侧 C++ API 迁至 toolkit 自带目录
    #     ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/hcomm/（platform_v220/v310 双平台）；
    #     对外 C API 头在 ${ASCEND_HOME_PATH}/include/hcomm/（无 adv_api 前缀）。
    # 内核源码统一按 "adv_api/hcomm/hcomm.h" 引用——命中无前缀布局时在构建目录生成
    # adv_api/hcomm -> <cann>/include/hcomm 符号链接兼容目录，源码零改动。
    set(KVOPS_HCOMM_INC_ROOT "")
    foreach(kvops_cand IN ITEMS
            "${ASC_DEVKIT_ROOT}/include"
            "$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include"
            "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include"
            "$ENV{ASCEND_HOME_PATH}/include"
            "${ASCEND_CANN_PACKAGE_PATH}/include")
        if(EXISTS "${kvops_cand}/adv_api/hcomm/hcomm.h")
            set(KVOPS_HCOMM_INC_ROOT "${kvops_cand}")
            break()
        elseif(EXISTS "${kvops_cand}/hcomm/hcomm.h")
            set(KVOPS_HCOMM_INC_ROOT "${kvops_cand}")
            set(KVOPS_HCOMM_NEED_COMPAT 1)
            break()
        endif()
    endforeach()
    if(KVOPS_HCOMM_INC_ROOT STREQUAL "")
        message(FATAL_ERROR
            "hcomm.h not found (checked asc-devkit and CANN include dirs). "
            "For CANN >= 9.2 install the cann-hcomm package, or set -DASC_DEVKIT_ROOT to a kit that bundles it.")
    endif()
    if(KVOPS_HCOMM_NEED_COMPAT)
        set(KVOPS_HCOMM_COMPAT_DIR ${CMAKE_CURRENT_BINARY_DIR}/hcomm_compat)
        file(MAKE_DIRECTORY ${KVOPS_HCOMM_COMPAT_DIR}/adv_api)
        if(NOT EXISTS ${KVOPS_HCOMM_COMPAT_DIR}/adv_api/hcomm)
            execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ${KVOPS_HCOMM_INC_ROOT}/hcomm ${KVOPS_HCOMM_COMPAT_DIR}/adv_api/hcomm)
        endif()
        set(KVOPS_HCOMM_INC_ROOT ${KVOPS_HCOMM_COMPAT_DIR})
    endif()

    ascendc_fatbin_library(kvc_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel/kvc_gather_rows.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel/kvc_scatter_rows_alllayer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel/hisp_sieve_update.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel/hisp_sieve_ht_init.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel/gather_selection_rmt.cpp
${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel/dsa_gather_rmt.cpp
    )
    if(COMMAND ascendc_include_directories)
        ascendc_include_directories(kvc_kernel PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel
            ${KVOPS_HCOMM_INC_ROOT}
        )
    else()
        target_include_directories(kvc_kernel PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel
            ${KVOPS_HCOMM_INC_ROOT}
        )
    endif()
endif()
