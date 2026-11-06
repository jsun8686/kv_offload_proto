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
    )
    target_include_directories(emsd PRIVATE common)
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
    if(NOT EXISTS ${ASC_DEVKIT_ROOT}/include/adv_api/hcomm/hcomm.h)
        message(FATAL_ERROR "${ASC_DEVKIT_ROOT}/include/adv_api/hcomm/hcomm.h not found, set -DASC_DEVKIT_ROOT")
    endif()

    ascendc_fatbin_library(kvc_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel/kvc_gather_rows.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel/kvc_scatter_rows_alllayer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel/hisp_sieve_update.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel/hisp_sieve_ht_init.cpp
    )
    if(COMMAND ascendc_include_directories)
        ascendc_include_directories(kvc_kernel PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel
            ${ASC_DEVKIT_ROOT}/include
        )
    else()
        target_include_directories(kvc_kernel PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel
            ${ASC_DEVKIT_ROOT}/include
        )
    endif()
endif()
