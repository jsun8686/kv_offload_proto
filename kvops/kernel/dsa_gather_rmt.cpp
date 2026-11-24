/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * dsa_gather_rmt：dsa_gather 远端池适配版（纯远端，无本地分支/无 0 哨兵）。
 *
 * 语义基准：vllm-ascend-csrc dsa_gather golden.py——miss_mask 唯一闸门（命中/padding 零搬运）
 * + pool_entry 两跳源寻址（batch 行→稳定池行→源物理块）+ req_id 一跳目标寻址
 * + 三层校验（索引非负/逻辑块不越表宽/物理块不越池，全在 AIV 上，不回 CPU）。
 * 搬运：Hcomm<UBC_CTP> ReadNbi 直写 resident 池（record 粒度，kv/rope 各一发；
 * ReadNbi 无对齐/粒度约束，1KB/128B 已实证）。收尾：去重通道集逐个
 * Commit<PIPE_MTE3> + Drain（V2 多通道成果）。无 UB staging（GM 直写）。
 * 并行：pair（req_id×query_id）按核均分（尾核+1），核间零依赖（数据并行原生多核，
 * blockDim=1 即单核）；status 每核独立 64B 槽。
 * 错误码：0x2xx=kv ReadNbi（xx=pair 序号低位，[16:24]=hcomm ret）
 *        0x3xx=rope ReadNbi；0x4xx/0x5xx=Commit/Drain（[24]=通道序号）；0x180+=通道超上限。
 */

#include "kvc_common.h"
#include "kvc_args.h"

extern "C" __global__ __aicore__ void dsa_gather_rmt(
    uint64_t residentKv, uint64_t residentRope, uint64_t residentBlockTable, uint64_t hostBlockTable,
    uint64_t requestPoolIndices, uint64_t topkTokenPositions, uint64_t residentSlots, uint64_t missMask,
    uint64_t rmtChannel, uint64_t rmtKvAddr, uint64_t rmtRopeAddr, uint64_t statusGm, uint32_t numRequests,
    uint32_t queryCount, uint32_t blockSize, uint32_t kvDim, uint32_t ropeDim, uint32_t kvElemSize,
    uint32_t ropeElemSize, uint32_t hostPoolCapacity, uint32_t hostTableWidth, uint32_t residentTableWidth,
    uint32_t hostBlkNum, uint32_t residentBlkNum, uint32_t residentInit)
{
    DsaGatherRmtArgs argsStorage{};
    argsStorage.residentKv = residentKv;
    argsStorage.residentRope = residentRope;
    argsStorage.residentBlockTable = residentBlockTable;
    argsStorage.hostBlockTable = hostBlockTable;
    argsStorage.requestPoolIndices = requestPoolIndices;
    argsStorage.topkTokenPositions = topkTokenPositions;
    argsStorage.residentSlots = residentSlots;
    argsStorage.missMask = missMask;
    argsStorage.rmtChannel = rmtChannel;
    argsStorage.rmtKvAddr = rmtKvAddr;
    argsStorage.rmtRopeAddr = rmtRopeAddr;
    argsStorage.status = statusGm;
    argsStorage.numRequests = numRequests;
    argsStorage.queryCount = queryCount;
    argsStorage.blockSize = blockSize;
    argsStorage.kvDim = kvDim;
    argsStorage.ropeDim = ropeDim;
    argsStorage.kvElemSize = kvElemSize;
    argsStorage.ropeElemSize = ropeElemSize;
    argsStorage.hostPoolCapacity = hostPoolCapacity;
    argsStorage.hostTableWidth = hostTableWidth;
    argsStorage.residentTableWidth = residentTableWidth;
    argsStorage.hostBlkNum = hostBlkNum;
    argsStorage.residentBlkNum = residentBlkNum;
    argsStorage.residentInit = residentInit;
    const DsaGatherRmtArgs *args = &argsStorage;

    const uint32_t coreId = GetBlockIdx();
    KvcStatusWriter w;
    KvcStatusInit(w, args->status + static_cast<uint64_t>(coreId) * 64);  // 每核独立槽

    const uint64_t pairCount = static_cast<uint64_t>(args->numRequests) * args->queryCount;
    const uint32_t coreCount = GetBlockNum();
    const uint64_t pairsPerCore = pairCount / coreCount;
    const uint64_t tailCores = pairCount % coreCount;
    const uint64_t corePairCount = pairsPerCore + (coreId < tailCores ? 1 : 0);
    const uint64_t pairBegin = static_cast<uint64_t>(coreId) * pairsPerCore + (coreId < tailCores ? coreId : tailCores);
    const uint64_t pairEnd = pairBegin + corePairCount;

    __gm__ int32_t *resTbl = reinterpret_cast<__gm__ int32_t *>(args->residentBlockTable);
    __gm__ int32_t *hostTbl = reinterpret_cast<__gm__ int32_t *>(args->hostBlockTable);
    __gm__ int32_t *poolIdx = reinterpret_cast<__gm__ int32_t *>(args->requestPoolIndices);
    __gm__ int32_t *topkPos = reinterpret_cast<__gm__ int32_t *>(args->topkTokenPositions);
    __gm__ int32_t *resSlots = reinterpret_cast<__gm__ int32_t *>(args->residentSlots);
    __gm__ int32_t *mask = reinterpret_cast<__gm__ int32_t *>(args->missMask);
    __gm__ int64_t *chTbl = reinterpret_cast<__gm__ int64_t *>(args->rmtChannel);
    __gm__ int64_t *kvAddrTbl = reinterpret_cast<__gm__ int64_t *>(args->rmtKvAddr);
    __gm__ int64_t *ropeAddrTbl = reinterpret_cast<__gm__ int64_t *>(args->rmtRopeAddr);
    __gm__ uint8_t *resKvGm = reinterpret_cast<__gm__ uint8_t *>(args->residentKv);
    __gm__ uint8_t *resRopeGm = reinterpret_cast<__gm__ uint8_t *>(args->residentRope);

    const uint64_t kvRecBytes = static_cast<uint64_t>(args->kvDim) * args->kvElemSize;
    const uint64_t ropeRecBytes = static_cast<uint64_t>(args->ropeDim) * args->ropeElemSize;
    const uint32_t bs = args->blockSize;

    TBuf<TPosition::VECOUT> wsBuf;
    w.pipe.InitBuffer(wsBuf, KVC_HCOMM_WS_SIZE);
    LocalTensor<uint8_t> ws = wsBuf.Get<uint8_t>();

    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm;
    bool hcommInited = false;
    bool rmtQueued = false;
    ChannelHandle rmtChs[8];
    uint32_t rmtChCnt = 0;
    uint64_t copied = 0;

    if (pairBegin < pairEnd) {  // 本核有任务才取行首元数据
        uint32_t reqId = static_cast<uint32_t>(pairBegin / args->queryCount);
        uint32_t queryId = static_cast<uint32_t>(pairBegin % args->queryCount);
        int32_t poolEntry = poolIdx[reqId];
        bool validPool = poolEntry >= 0 && static_cast<uint32_t>(poolEntry) < args->hostPoolCapacity;

        for (uint64_t pair = pairBegin; pair < pairEnd; ++pair) {
            const uint64_t idxOff = pair;  // dense decode 布局（residentInit 恒 0；紧凑布局 V1.1）
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
                            // D2a 每核独立通道：chTbl 平铺 [hostBlkNum × coreCount]，块 q 核 c → chTbl[q*coreCount+c]
                            const int64_t ch = chTbl[static_cast<uint64_t>(srcBlk) * coreCount + coreId];
                            if (!hcommInited) {
                                int32_t ret = hcomm.Init(ws, KVC_HCOMM_WS_SIZE);
                                if (ret != 0) {
                                    KvcStatusFinish(w, 0x100 + static_cast<uint64_t>(ret));
                                    KvcKernelEpilogue();
                                    return;
                                }
                                hcommInited = true;
                            }
                            int32_t ret = hcomm.template ReadNbi<false>(
                                static_cast<ChannelHandle>(ch), reinterpret_cast<GM_ADDR>(resKvGm + dstKvOff),
                                reinterpret_cast<GM_ADDR>(kvAddrTbl[srcBlk] +
                                                          static_cast<int64_t>(srcOff) * kvRecBytes),
                                kvRecBytes);
                            if (ret != 0) {
                                KvcStatusFinish(w, 0x200 + (pair & 0xFF) + (static_cast<uint64_t>(ret) << 16));
                                KvcKernelEpilogue();
                                return;
                            }
                            if (ropeRecBytes > 0) {
                                ret = hcomm.template ReadNbi<false>(
                                    static_cast<ChannelHandle>(ch), reinterpret_cast<GM_ADDR>(resRopeGm + dstRopeOff),
                                    reinterpret_cast<GM_ADDR>(ropeAddrTbl[srcBlk] +
                                                              static_cast<int64_t>(srcOff) * ropeRecBytes),
                                    ropeRecBytes);
                                if (ret != 0) {
                                    KvcStatusFinish(w, 0x300 + (pair & 0xFF) +
                                                    (static_cast<uint64_t>(ret) << 16));
                                    KvcKernelEpilogue();
                                    return;
                                }
                            }
                            bool known = false;
                            for (uint32_t ci = 0; ci < rmtChCnt; ++ci) {
                                if (rmtChs[ci] == static_cast<ChannelHandle>(ch)) {
                                    known = true;
                                    break;
                                }
                            }
                            if (!known) {
                                if (rmtChCnt >= sizeof(rmtChs) / sizeof(rmtChs[0])) {
                                    KvcStatusFinish(w, 0x182);
                                    KvcKernelEpilogue();
                                    return;
                                }
                                rmtChs[rmtChCnt++] = static_cast<ChannelHandle>(ch);
                            }
                            rmtQueued = true;
                            copied++;
                        }
                    }
                }
            }
            ++queryId;
            if (queryId == args->queryCount && pair + 1 < pairEnd) {  // 行切换：重取池行
                queryId = 0;
                ++reqId;
                poolEntry = poolIdx[reqId];
                validPool = poolEntry >= 0 && static_cast<uint32_t>(poolEntry) < args->hostPoolCapacity;
            }
        }
    }

    if (rmtQueued) {  // 逐通道 Commit+Drain（WQE 与 MTE3 同管道保序；多通道 V2 成果）
        for (uint32_t ci = 0; ci < rmtChCnt; ++ci) {
            int32_t ret = hcomm.template Commit<PIPE_MTE3>(rmtChs[ci]);
            if (ret != 0) {
                KvcStatusFinish(w, 0x400 + (static_cast<uint64_t>(ci) << 24) +
                                (static_cast<uint64_t>(ret) & 0xFFFFFF));
                KvcKernelEpilogue();
                return;
            }
            ret = hcomm.Drain(rmtChs[ci]);
            if (ret != 0) {
                KvcStatusFinish(w, 0x500 + (static_cast<uint64_t>(ci) << 24) +
                                (static_cast<uint64_t>(ret) & 0xFFFFFF));
                KvcKernelEpilogue();
                return;
            }
        }
    }
    KvcStatusSet(w, 1, copied);  // 诊断区：本核搬运 record 数
    KvcStatusFinish(w, 0);
    KvcKernelEpilogue();
}
