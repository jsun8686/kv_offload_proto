/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * kvc_gather_rows：批量导入（稠密接口，v2）。远端段 → HBM 热缓冲。
 * 输入为 sieve 输出契约（topk/slots/isMiss 稠密张量）；多 block 按 positions 分片，
 * 每 block 独立 Hcomm 实例与 WQE 流水（Init → 逐行 ReadNbi<false> → Commit → Drain）。
 */

#include "kvc_common.h"
#include "kvc_args.h"

// fatbin 纯设备编译路径无 auto-gen 包装：kernel 实参 = host 参数块逐字段自然对齐布局
// （见 kvc_scatter_rows_alllayer.cpp 头注释），非 GM 指针。签名与 KvcGatherArgs 字段
// 顺序/类型严格一致（尾置 reserved 省略），体内重组为结构体。
extern "C" __global__ __aicore__ void kvc_gather_rows(uint64_t channel, uint64_t segBase, uint64_t topkIndices,
                                                      uint64_t reqPoolIndices, uint64_t reqHostBase,
                                                      uint64_t topKDeviceSlots, uint64_t isMiss, uint64_t numRealGm,
                                                      uint64_t hotRowBase, uint64_t status,
                                                      uint32_t layerOffsetRows, uint32_t perLayerSegRows,
                                                      uint32_t rowBytes, uint32_t stripeRows, uint32_t nodeNum,
                                                      uint32_t myNode, uint32_t topK, uint32_t maxNumReqs,
                                                      uint32_t positionsPerCore)
{
    KvcGatherArgs argsStorage{};
    argsStorage.channel = channel;
    argsStorage.segBase = segBase;
    argsStorage.topkIndices = topkIndices;
    argsStorage.reqPoolIndices = reqPoolIndices;
    argsStorage.reqHostBase = reqHostBase;
    argsStorage.topKDeviceSlots = topKDeviceSlots;
    argsStorage.isMiss = isMiss;
    argsStorage.numReal = numRealGm;
    argsStorage.hotRowBase = hotRowBase;
    argsStorage.status = status;
    argsStorage.layerOffsetRows = layerOffsetRows;
    argsStorage.perLayerSegRows = perLayerSegRows;
    argsStorage.rowBytes = rowBytes;
    argsStorage.stripeRows = stripeRows;
    argsStorage.nodeNum = nodeNum;
    argsStorage.myNode = myNode;
    argsStorage.topK = topK;
    argsStorage.maxNumReqs = maxNumReqs;
    argsStorage.positionsPerCore = positionsPerCore;
    const KvcGatherArgs *args = &argsStorage;
    const int32_t blockIdx = GetBlockIdx();

    __gm__ int32_t *numRealPtr = reinterpret_cast<__gm__ int32_t *>(args->numReal);
    int32_t numReal = numRealPtr[0];
    if (numReal > static_cast<int32_t>(args->maxNumReqs)) {
        numReal = args->maxNumReqs;  // 防御
    }
    KvcStatusWriter w;
    // 每 block 独占 64B 槽：KvcStatusFlush 按整页 64B 刷出（DataCopy 最小 32B），
    // 若按 8B 步进分片，任一 block 的阶段刷写会覆盖其他 block 的 doneSeq——
    // 实测症状：blockDim=8 时随机某个 block "doneSeq not set" 超时且无设备报错
    KvcStatusInit(w, args->status + static_cast<uint64_t>(blockIdx) * 64);

    const int32_t totalPos = numReal * static_cast<int32_t>(args->topK);
    const int32_t startPos = blockIdx * static_cast<int32_t>(args->positionsPerCore);
    const int32_t endPos = startPos + static_cast<int32_t>(args->positionsPerCore);
    const int32_t posEnd = (endPos > totalPos) ? totalPos : endPos;
    if (startPos >= posEnd) {
        KvcStatusFinish(w, 0);  // 空分片：直接置完成
        KvcKernelEpilogue();
        return;
    }

    TBuf<TPosition::VECOUT> wsBuf;
    w.pipe.InitBuffer(wsBuf, KVC_HCOMM_WS_SIZE);
    LocalTensor<uint8_t> ws = wsBuf.Get<uint8_t>();

    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm;
    int32_t ret = hcomm.Init(ws, KVC_HCOMM_WS_SIZE);
    if (ret != 0) {
        KvcStatusFinish(w, 0x100 + static_cast<uint64_t>(ret));
        KvcKernelEpilogue();
        return;
    }

    __gm__ int32_t *topkPtr = reinterpret_cast<__gm__ int32_t *>(args->topkIndices);
    __gm__ int64_t *reqPoolPtr = reinterpret_cast<__gm__ int64_t *>(args->reqPoolIndices);
    __gm__ int64_t *hostBasePtr = reinterpret_cast<__gm__ int64_t *>(args->reqHostBase);
    __gm__ int32_t *slotsPtr = reinterpret_cast<__gm__ int32_t *>(args->topKDeviceSlots);
    __gm__ int8_t *isMissPtr = reinterpret_cast<__gm__ int8_t *>(args->isMiss);

    uint64_t queued = 0;
    for (int32_t pos = startPos; pos < posEnd; ++pos) {
        if (isMissPtr[pos] == 0) {
            continue;  // 命中/无效项无需搬运
        }
        const int32_t bid = pos / args->topK;
        const int32_t tokenPos = topkPtr[pos];
        const int32_t slot = slotsPtr[pos];
        // 防御：sieve 输出契约下 miss 项 tokenPos/slot 必有效，保守保留负值检查
        if (tokenPos < 0 || slot < 0) {
            continue;
        }
        const int64_t reqIdx = reqPoolPtr[bid];
        if (reqIdx < 0) {
            continue;
        }
        const int64_t row = hostBasePtr[bid] + tokenPos;  // 行分配连续性契约
        if (KvcNodeOf(static_cast<uint32_t>(row), args->stripeRows, args->nodeNum) != args->myNode) {
            continue;  // 行归属其他资源节点
        }
        const uint64_t segRow = KvcSegRowOf(static_cast<uint32_t>(row), args->stripeRows, args->nodeNum);
        GM_ADDR src = reinterpret_cast<GM_ADDR>(
            args->segBase + (static_cast<uint64_t>(args->layerOffsetRows) + segRow) * args->rowBytes);
        GM_ADDR dst = reinterpret_cast<GM_ADDR>(args->hotRowBase + static_cast<uint64_t>(slot) * args->rowBytes);
        ret = hcomm.template ReadNbi<false>(args->channel, dst, src, args->rowBytes);
        if (ret != 0) {
            KvcStatusFinish(w, 0x200 + (static_cast<uint64_t>(pos) & 0xFF) + (static_cast<uint64_t>(ret) << 16));
            KvcKernelEpilogue();
            return;
        }
        queued++;
    }

    ret = hcomm.Commit(args->channel);
    if (ret != 0) {
        KvcStatusFinish(w, 0x300 + static_cast<uint64_t>(ret));
        KvcKernelEpilogue();
        return;
    }
    ret = hcomm.Drain(args->channel);
    KvcStatusSet(w, 1, queued);  // 诊断区：搬运行数（poll 不读）
    KvcStatusFinish(w, ret == 0 ? 0 : 0x400 + static_cast<uint64_t>(ret));
    KvcKernelEpilogue();
}
