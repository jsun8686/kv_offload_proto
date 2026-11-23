/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * dsa_gather_rmt_v2：AICPU 搬运 kernel（Hcomm device-RoCE 数据面，A3 路径）。
 *
 * 与 dsa_gather_rmt（AIV+UBC_CTP）同语义同错误码，仅搬运机制不同：
 *   AICPU 侧 dlopen libccl_kernel.so（与 HCCL device loader 同款，MemFabric 实证），
 *   dlsym Hcomm*OnThread 原语，经 AicpuTsRoceChannel 单边读直写 resident 池
 *   （record 粒度，kv/rope 各一发；ReadOnThread 无对齐/粒度约束）。
 * 收尾：去重通道集逐个 HcommChannelFenceOnThread（等本地 CQ 完成）+ DrainOnThread。
 * 单实例（blockDim 恒 1，thread 全通道共享）；status 64B 槽协议与 AIV 版一致。
 * 错误码：0x100+=dlopen/dlsym 失败；0x2xx=kv Read（xx=pair 低位，[16:24]=hcomm ret）；
 *        0x3xx=rope Read；0x4xx/0x5xx=Fence/Drain（[24]=通道序号，低 24bit=ret）；0x182=通道超上限。
 */

#include "dsa_gather_rmt_v2.h"

#include <dlfcn.h>

#include <cstdint>

namespace {

using ThreadHandleT = uint64_t;
using ChannelHandleT = uint64_t;

using FnReadOnThread = int32_t (*)(ThreadHandleT, ChannelHandleT, void *, const void *, uint64_t);
using FnChannelFenceOnThread = int32_t (*)(ThreadHandleT, ChannelHandleT);
using FnChannelDrainOnThread = int32_t (*)(ThreadHandleT, ChannelHandleT);

void *g_hcomm = nullptr;
FnReadOnThread g_readOnThread = nullptr;
FnChannelFenceOnThread g_fenceOnThread = nullptr;
FnChannelDrainOnThread g_drainOnThread = nullptr;

// 与 HCCL device loader 一致：libccl_kernel.so 提供 Hcomm*OnThread 原语（RTLD_NODELETE 防句柄失效）
constexpr const char *HCOMM_LIB_NAME = "libccl_kernel.so";

int32_t EnsureHcommSymbols()
{
    if (g_hcomm != nullptr) {
        return 0;
    }
    g_hcomm = dlopen(HCOMM_LIB_NAME, RTLD_NOW | RTLD_NODELETE);
    if (g_hcomm == nullptr) {
        return 1;
    }
    (void)dlerror();  // 清历史错误，保证后续 dlerror 反映本次 dlsym
    g_readOnThread = reinterpret_cast<FnReadOnThread>(dlsym(g_hcomm, "HcommReadOnThread"));
    g_fenceOnThread = reinterpret_cast<FnChannelFenceOnThread>(dlsym(g_hcomm, "HcommChannelFenceOnThread"));
    g_drainOnThread = reinterpret_cast<FnChannelDrainOnThread>(dlsym(g_hcomm, "HcommChannelDrainOnThread"));
    if (g_readOnThread == nullptr || g_fenceOnThread == nullptr || g_drainOnThread == nullptr) {
        return 2;
    }
    return 0;
}

// 终态写回：word1=record 数（诊断），word0=(1<<32)|rc（host 完成判定；先 word1 后 word0 保序）
inline void FinishStatus(uint64_t status, uint64_t copied, uint32_t rc)
{
    if (status == 0) {
        return;
    }
    volatile uint64_t *st = reinterpret_cast<volatile uint64_t *>(status);
    st[1] = copied;
    st[0] = (1ULL << 32) | static_cast<uint64_t>(rc);
}

}  // namespace

extern "C" int32_t DsaGatherRmtV2(DsaGatherRmtV2Args *args)
{
    if (args == nullptr) {
        return 0x10F;
    }
    volatile uint64_t *st = reinterpret_cast<volatile uint64_t *>(args->status);
    if (args->status != 0) {
        st[0] = 0;  // 清终态（kvops alloc_status 已 memset，双保险防复用槽残留）
    }
    uint64_t copied = 0;

    const int32_t loadRet = EnsureHcommSymbols();
    if (loadRet != 0) {
        FinishStatus(args->status, copied, 0x100U + static_cast<uint32_t>(loadRet));
        return 0x100 + loadRet;
    }

    const int32_t *resTbl = reinterpret_cast<const int32_t *>(args->residentBlockTable);
    const int32_t *hostTbl = reinterpret_cast<const int32_t *>(args->hostBlockTable);
    const int32_t *poolIdx = reinterpret_cast<const int32_t *>(args->requestPoolIndices);
    const int32_t *topkPos = reinterpret_cast<const int32_t *>(args->topkTokenPositions);
    const int32_t *resSlots = reinterpret_cast<const int32_t *>(args->residentSlots);
    const int32_t *mask = reinterpret_cast<const int32_t *>(args->missMask);
    const int64_t *chTbl = reinterpret_cast<const int64_t *>(args->rmtChannel);
    const int64_t *kvAddrTbl = reinterpret_cast<const int64_t *>(args->rmtKvAddr);
    const int64_t *ropeAddrTbl = reinterpret_cast<const int64_t *>(args->rmtRopeAddr);
    uint8_t *resKv = reinterpret_cast<uint8_t *>(args->residentKv);
    uint8_t *resRope = reinterpret_cast<uint8_t *>(args->residentRope);

    const uint64_t kvRecBytes = args->kvDim * args->kvElemSize;
    const uint64_t ropeRecBytes = args->ropeDim * args->ropeElemSize;
    const uint32_t bs = static_cast<uint32_t>(args->blockSize);
    const uint64_t pairCount = args->numRequests * args->queryCount;

    ChannelHandleT rmtChs[8];
    uint32_t rmtChCnt = 0;
    bool rmtQueued = false;

    // 单实例：pair 全量 [0, pairCount)，行切换逻辑与 AIV 版逐字对齐
    if (pairCount > 0) {
        uint32_t reqId = 0;
        uint32_t queryId = 0;
        int32_t poolEntry = poolIdx[0];
        bool validPool = poolEntry >= 0 && static_cast<uint32_t>(poolEntry) < args->hostPoolCapacity;

        for (uint64_t pair = 0; pair < pairCount; ++pair) {
            const uint64_t idxOff = pair;  // dense decode 布局（紧凑布局，同 AIV 版）
            const uint64_t maskOff = pair;
            if (validPool && mask[maskOff] != 0) {
                const int32_t token = topkPos[idxOff];
                const int32_t slot = resSlots[idxOff];
                if (token >= 0 && slot >= 0) {
                    const uint32_t srcLogical = static_cast<uint32_t>(token) / bs;
                    const uint32_t srcOff = static_cast<uint32_t>(token) % bs;
                    const uint32_t dstLogical = static_cast<uint32_t>(slot) / bs;
                    const uint32_t dstOff = static_cast<uint32_t>(slot) % bs;
                    if (srcLogical < args->hostTableWidth && dstLogical < args->residentTableWidth) {
                        const int32_t srcBlk = hostTbl[static_cast<uint64_t>(poolEntry) * args->hostTableWidth +
                                                       srcLogical];
                        const int32_t dstBlk =
                            resTbl[static_cast<uint64_t>(reqId) * args->residentTableWidth + dstLogical];
                        if (srcBlk >= 0 && dstBlk >= 0 && static_cast<uint32_t>(srcBlk) < args->hostBlkNum &&
                            static_cast<uint32_t>(dstBlk) < args->residentBlkNum) {
                            const uint64_t dstKvOff =
                                (static_cast<uint64_t>(dstBlk) * bs + dstOff) * kvRecBytes;
                            const uint64_t dstRopeOff =
                                (static_cast<uint64_t>(dstBlk) * bs + dstOff) * ropeRecBytes;
                            const ChannelHandleT ch = static_cast<ChannelHandleT>(chTbl[srcBlk]);
                            int32_t ret = g_readOnThread(args->thread, ch, resKv + dstKvOff,
                                                         reinterpret_cast<const void *>(
                                                             kvAddrTbl[srcBlk] +
                                                             static_cast<int64_t>(srcOff) * kvRecBytes),
                                                         kvRecBytes);
                            if (ret != 0) {
                                FinishStatus(args->status, copied,
                                             0x200U + static_cast<uint32_t>(pair & 0xFF) +
                                                 (static_cast<uint32_t>(ret) << 16));
                                return 0x200;
                            }
                            if (ropeRecBytes > 0) {
                                ret = g_readOnThread(args->thread, ch, resRope + dstRopeOff,
                                                     reinterpret_cast<const void *>(
                                                         ropeAddrTbl[srcBlk] +
                                                         static_cast<int64_t>(srcOff) * ropeRecBytes),
                                                     ropeRecBytes);
                                if (ret != 0) {
                                    FinishStatus(args->status, copied,
                                                 0x300U + static_cast<uint32_t>(pair & 0xFF) +
                                                     (static_cast<uint32_t>(ret) << 16));
                                    return 0x300;
                                }
                            }
                            bool known = false;
                            for (uint32_t ci = 0; ci < rmtChCnt; ++ci) {
                                if (rmtChs[ci] == ch) {
                                    known = true;
                                    break;
                                }
                            }
                            if (!known) {
                                if (rmtChCnt >= sizeof(rmtChs) / sizeof(rmtChs[0])) {
                                    FinishStatus(args->status, copied, 0x182U);
                                    return 0x182;
                                }
                                rmtChs[rmtChCnt++] = ch;
                            }
                            rmtQueued = true;
                            copied++;
                        }
                    }
                }
            }
            ++queryId;
            if (queryId == args->queryCount && pair + 1 < pairCount) {  // 行切换：重取池行
                queryId = 0;
                ++reqId;
                poolEntry = poolIdx[reqId];
                validPool = poolEntry >= 0 && static_cast<uint32_t>(poolEntry) < args->hostPoolCapacity;
            }
        }
    }

    if (rmtQueued) {  // 逐通道 Fence（等本地 CQ 完成）+ Drain，保数据落地
        for (uint32_t ci = 0; ci < rmtChCnt; ++ci) {
            int32_t ret = g_fenceOnThread(args->thread, rmtChs[ci]);
            if (ret != 0) {
                FinishStatus(args->status, copied,
                             0x400U + (static_cast<uint32_t>(ci) << 24) + (static_cast<uint32_t>(ret) & 0xFFFFFFU));
                return 0x400;
            }
            ret = g_drainOnThread(args->thread, rmtChs[ci]);
            if (ret != 0) {
                FinishStatus(args->status, copied,
                             0x500U + (static_cast<uint32_t>(ci) << 24) + (static_cast<uint32_t>(ret) & 0xFFFFFFU));
                return 0x500;
            }
        }
    }
    FinishStatus(args->status, copied, 0);
    return 0;
}
