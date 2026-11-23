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
    if(NOT EXISTS ${ASC_DEVKIT_ROOT}/include/adv_api/hcomm/hcomm.h)
        message(FATAL_ERROR "${ASC_DEVKIT_ROOT}/include/adv_api/hcomm/hcomm.h not found, set -DASC_DEVKIT_ROOT")
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
            ${ASC_DEVKIT_ROOT}/include
        )
    else()
        target_include_directories(kvc_kernel PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/kvops/kernel
            ${ASC_DEVKIT_ROOT}/include
        )
    endif()

    # AICPU 搬运 kernel（v2 / Hcomm device-RoCE 数据面，MemFabric hybm 同款构建法）：
    #   aarch64 交叉 g++ 直编 .so + json 描述符，产物与 kvc_kernel.o 同目录输出
    #   （ascendc_fatbin_library 的 fatbin 默认落在 <build>/kvc_kernel.o）；
    #   kvops.init 按 fatbin 路径旁加载 json（libcann_kvops_aicpu.json 缺失仅降级 v2）。
    # 部署提示：三个文件必须一起拷（kvc_kernel.o + libcann_kvops_aicpu.so +
    #   libcann_kvops_aicpu.json），构建 kvops_aicpu_kernel 后有回显提示。
    set(KVOPS_AICPU_CXX "$ENV{ASCEND_HOME_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++")
    set(KVOPS_AICPU_SO ${CMAKE_CURRENT_BINARY_DIR}/libcann_kvops_aicpu.so)
    set(KVOPS_AICPU_JSON ${CMAKE_CURRENT_BINARY_DIR}/libcann_kvops_aicpu.json)
    if(EXISTS ${KVOPS_AICPU_CXX})
        add_custom_command(
            OUTPUT ${KVOPS_AICPU_SO}
            COMMAND ${KVOPS_AICPU_CXX} -shared -O2 -std=c++17 -fPIC
                    -I${CMAKE_CURRENT_SOURCE_DIR}/kvops/aicpu_kernel
                    -o ${KVOPS_AICPU_SO}
                    ${CMAKE_CURRENT_SOURCE_DIR}/kvops/aicpu_kernel/dsa_gather_rmt_v2.cc
                    -ldl
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kvops/aicpu_kernel/dsa_gather_rmt_v2.cc
                    ${CMAKE_CURRENT_SOURCE_DIR}/kvops/aicpu_kernel/dsa_gather_rmt_v2.h
            COMMENT "Compiling AICPU kernel libcann_kvops_aicpu.so (aarch64 cross)")
        add_custom_command(
            OUTPUT ${KVOPS_AICPU_JSON}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CMAKE_CURRENT_SOURCE_DIR}/kvops/aicpu_kernel/libcann_kvops_aicpu.json
                    ${KVOPS_AICPU_JSON}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kvops/aicpu_kernel/libcann_kvops_aicpu.json)
        add_custom_target(kvops_aicpu_kernel ALL DEPENDS ${KVOPS_AICPU_SO} ${KVOPS_AICPU_JSON})
        # 部署提醒：fatbin 与 AICPU 产物必须同目录部署（kvops.init 按 fatbin 旁路加载 json）
        add_custom_command(TARGET kvops_aicpu_kernel POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo
                    "[kvops] deploy together: kvc_kernel.o + libcann_kvops_aicpu.so + libcann_kvops_aicpu.json")
        # 自动部署：构建完成即把 kernel 三件套拷到部署目录（fatbin 与 AICPU 产物同目录，
        # kvops.init 按 fatbin 旁路加载 json）。目录可用 -DKVOPS_KERNEL_DEPLOY_DIR=... 覆盖。
        set(KVOPS_KERNEL_DEPLOY_DIR "/usr/local/fatbin/kvc_kernel" CACHE PATH
            "kernel deploy dir (kvc_kernel.o + AICPU artifacts)")
        # fatbin 由 ascendc_fatbin_library 的 ExternalProject（kvc_kernel_precompile）产出，
        # 顶层 Makefile 无 kvc_kernel.o 的生成规则，不能挂文件级依赖——依赖 target 排序，
        # 拷贝前再判一次文件是否存在（不存在仅告警，不阻断构建）。
        set(KVOPS_FATBIN_O ${CMAKE_CURRENT_BINARY_DIR}/kvc_kernel.o)
        set(KVOPS_DEPLOY_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/kvops_deploy.cmake)
        file(WRITE ${KVOPS_DEPLOY_SCRIPT}
"if(EXISTS \"${KVOPS_FATBIN_O}\")
  file(COPY \"${KVOPS_FATBIN_O}\" DESTINATION \"${KVOPS_KERNEL_DEPLOY_DIR}\")
else()
  message(WARNING \"fatbin not found: ${KVOPS_FATBIN_O}, skipped deploy of kvc_kernel.o\")
endif()
file(COPY \"${KVOPS_AICPU_SO}\" \"${KVOPS_AICPU_JSON}\" DESTINATION \"${KVOPS_KERNEL_DEPLOY_DIR}\")
message(STATUS \"kernel artifacts deployed to ${KVOPS_KERNEL_DEPLOY_DIR}\")
")
        add_custom_target(kvops_kernel_deploy ALL
            COMMAND ${CMAKE_COMMAND} -D__unused=1 -P ${KVOPS_DEPLOY_SCRIPT}
            DEPENDS kvops_aicpu_kernel
            COMMENT "Deploying kernel artifacts to ${KVOPS_KERNEL_DEPLOY_DIR}")
        if(TARGET kvc_kernel_precompile)
            add_dependencies(kvops_kernel_deploy kvc_kernel_precompile)
        elseif(TARGET kvc_kernel)
            add_dependencies(kvops_kernel_deploy kvc_kernel)
        endif()
        # ascendc_fatbin_library 在部分 CANN 版本不产出同名 target——存在才绑排序依赖，
        # 不存在无妨：kvops_aicpu_kernel 为 ALL 目标，默认构建即覆盖
        if(TARGET kvc_kernel)
            add_dependencies(kvc_kernel kvops_aicpu_kernel)
        endif()
    else()
        message(WARNING "AICPU cross compiler not found: ${KVOPS_AICPU_CXX}; v2 kernel skipped")
    endif()
endif()
