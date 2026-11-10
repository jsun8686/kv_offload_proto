/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * hisp_sieve_update：SIEVE miss 识别/淘汰更新（哈希-only，移植自 sglang-0730 hisparse_sieve_update）。
 *
 * 算法：命中置 visited 位；miss 时 hand 前扫清 visited 位并淘汰首个未访问槽（摊还 O(1)）。
 * 状态（按池行，逐层切片传入）：tokens[pool,padded] int32 / visited[pool,visStrideB] 位图 /
 * ht[pool,htSize] uint16（开放寻址）/ hand[pool,16] int32。
 * 输出契约：全量重写 [0,maxNumReqs) 的 slots/is_miss 行（-1 哨兵），调用方免修正趟（图安全）。
 * 注意：args 字段集中读入本地标量（引用形参 API 要求左值；兼省重复 GM 加载）。
 *
 * 950（c310）缓存语义（联调实证）：标量 GM store 滞留核上缓存对 host 不可见
 * （dsb/dci 序列亦不足），MTE DataCopy 回写可见。故 slots/is_miss/hand 输出全部
 * 走 UB staging + DataCopy（与 tokens/visited/ht 同路径）；is_miss 由 int8 改
 * int32（MTE 32B 粒度约束，topK%8==0 由 host 校验）。
 */

#include "hisp_kernel_common.h"
#include "kvc_args.h"
#include "kvc_common.h"

// 输出行 MTE 回写（slots + is_miss 整行）：c310 上标量 GM store 滞留核上缓存对 host
// 不可见（dsb/dci 亦不足），输出统一经 UB staging 后 DataCopy 刷出（与 tokens/ht 同路径）。
static inline __aicore__ void WriteBackSieveRow(GlobalTensor<int32_t> &outGm, GlobalTensor<int32_t> &missGm,
                                                LocalTensor<int32_t> &outLocal, LocalTensor<int32_t> &missLocal,
                                                int32_t topK)
{
    PipeBarrier<PIPE_ALL>();
    DataCopy(outGm, outLocal, topK);
    DataCopy(missGm, missLocal, topK);
}

// fatbin 纯设备编译路径无 auto-gen 包装：kernel 实参 = host 参数块逐字段自然对齐布局
// （见 kvc_scatter_rows_alllayer.cpp 头注释），非 GM 指针。签名与 HispSieveUpdateArgs
// 字段顺序/类型严格一致（尾置 reserved 省略），体内重组为结构体。
extern "C" __global__ __aicore__ void hisp_sieve_update(
    uint64_t topkIndices, uint64_t reqPoolIndices, uint64_t seqLens, uint64_t prefillLens,
    uint64_t deviceBufferTokens, uint64_t deviceBufferVisited, uint64_t deviceBufferHt, uint64_t sieveHand,
    uint64_t topKDeviceSlots, uint64_t isMiss, uint64_t numRealGm, uint64_t status, uint32_t topKIn,
    uint32_t deviceBufferSize, uint32_t paddedBufferSize, uint32_t visitedStrideB, uint32_t maxNumReqsIn,
    uint32_t reqsPerCoreIn, uint32_t htSizeIn, uint32_t poolSizeIn)
{
    HispSieveUpdateArgs argsStorage{};
    argsStorage.topkIndices = topkIndices;
    argsStorage.reqPoolIndices = reqPoolIndices;
    argsStorage.seqLens = seqLens;
    argsStorage.prefillLens = prefillLens;
    argsStorage.deviceBufferTokens = deviceBufferTokens;
    argsStorage.deviceBufferVisited = deviceBufferVisited;
    argsStorage.deviceBufferHt = deviceBufferHt;
    argsStorage.sieveHand = sieveHand;
    argsStorage.topKDeviceSlots = topKDeviceSlots;
    argsStorage.isMiss = isMiss;
    argsStorage.numReal = numRealGm;
    argsStorage.status = status;
    argsStorage.topK = topKIn;
    argsStorage.deviceBufferSize = deviceBufferSize;
    argsStorage.paddedBufferSize = paddedBufferSize;
    argsStorage.visitedStrideB = visitedStrideB;
    argsStorage.maxNumReqs = maxNumReqsIn;
    argsStorage.reqsPerCore = reqsPerCoreIn;
    argsStorage.htSize = htSizeIn;
    argsStorage.poolSize = poolSizeIn;
    const HispSieveUpdateArgs *args = &argsStorage;
    // GM 解引用集中读入本地标量
    const int32_t topK = args->topK;
    const int32_t dbs = args->deviceBufferSize;
    const int32_t padded = args->paddedBufferSize;
    const int32_t visStrideB = args->visitedStrideB;
    const int32_t maxNumReqs = args->maxNumReqs;
    const int32_t reqsPerCore = args->reqsPerCore;
    const int32_t htSize = args->htSize;
    const int32_t poolSize = args->poolSize;

    __gm__ int32_t *numRealPtr = reinterpret_cast<__gm__ int32_t *>(args->numReal);
    int32_t numReal = numRealPtr[0];
    if (numReal > maxNumReqs) {
        numReal = maxNumReqs;  // 防御：GM 标量可能过期
    }

    __gm__ int64_t *reqPoolPtr = reinterpret_cast<__gm__ int64_t *>(args->reqPoolIndices);
    __gm__ int32_t *seqLenPtr = reinterpret_cast<__gm__ int32_t *>(args->seqLens);
    __gm__ int32_t *prefillPtr = reinterpret_cast<__gm__ int32_t *>(args->prefillLens);
    __gm__ int32_t *topkPtr = reinterpret_cast<__gm__ int32_t *>(args->topkIndices);
    __gm__ int32_t *tokensPtr = reinterpret_cast<__gm__ int32_t *>(args->deviceBufferTokens);
    __gm__ uint8_t *visPtr = reinterpret_cast<__gm__ uint8_t *>(args->deviceBufferVisited);
    __gm__ uint16_t *htPtr = reinterpret_cast<__gm__ uint16_t *>(args->deviceBufferHt);
    __gm__ int32_t *handPtr = reinterpret_cast<__gm__ int32_t *>(args->sieveHand);
    __gm__ int32_t *outSlotsPtr = reinterpret_cast<__gm__ int32_t *>(args->topKDeviceSlots);
    __gm__ int32_t *isMissPtr = reinterpret_cast<__gm__ int32_t *>(args->isMiss);  // int32（MTE 32B 粒度）

    const int32_t blockIdx = GetBlockIdx();
    int32_t startReq = blockIdx * reqsPerCore;
    int32_t endReq = startReq + reqsPerCore;
    if (endReq > maxNumReqs) {
        endReq = maxNumReqs;
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> tokBuf;
    TBuf<TPosition::VECCALC> visBuf;
    TBuf<TPosition::VECCALC> htBuf;
    TBuf<TPosition::VECCALC> outBuf;
    TBuf<TPosition::VECCALC> missBuf;
    TBuf<TPosition::VECCALC> handBuf;
    pipe.InitBuffer(tokBuf, padded * sizeof(int32_t));
    pipe.InitBuffer(visBuf, visStrideB);
    pipe.InitBuffer(htBuf, htSize * sizeof(uint16_t));
    pipe.InitBuffer(outBuf, topK * sizeof(int32_t));
    pipe.InitBuffer(missBuf, topK * sizeof(int32_t));
    pipe.InitBuffer(handBuf, hisp::kHandStride * sizeof(int32_t));
    LocalTensor<int32_t> tokLocal = tokBuf.Get<int32_t>();
    LocalTensor<uint8_t> visLocal = visBuf.Get<uint8_t>();
    LocalTensor<uint16_t> htLocal = htBuf.Get<uint16_t>();
    LocalTensor<int32_t> outLocal = outBuf.Get<int32_t>();
    LocalTensor<int32_t> missLocal = missBuf.Get<int32_t>();
    LocalTensor<int32_t> handLocal = handBuf.Get<int32_t>();

    const int32_t htMask = htSize - 1;

    for (int32_t bid = startReq; bid < endReq; ++bid) {
        __gm__ int32_t *reqOutSlots = outSlotsPtr + static_cast<int64_t>(bid) * topK;
        __gm__ int32_t *reqIsMiss = isMissPtr + static_cast<int64_t>(bid) * topK;
        // 本行输出统一走 UB staging（哑行同路径，保证整行 MTE 回写）
        GlobalTensor<int32_t> outGm;
        GlobalTensor<int32_t> missGm;
        outGm.SetGlobalBuffer(reqOutSlots);
        missGm.SetGlobalBuffer(reqIsMiss);

        // padding 行（图 replay）：只写哑输出，不读未定义内容
        if (bid >= numReal) {
            for (int32_t k = 0; k < topK; ++k) {
                outLocal.SetValue(k, -1);
                missLocal.SetValue(k, 0);
            }
            WriteBackSieveRow(outGm, missGm, outLocal, missLocal, topK);
            continue;
        }

        const int64_t reqIdx = reqPoolPtr[bid];
        // 防御：越界池行写哑输出，不进行 GM 越界寻址
        if (reqIdx < 0 || reqIdx >= poolSize) {
            for (int32_t k = 0; k < topK; ++k) {
                outLocal.SetValue(k, -1);
                missLocal.SetValue(k, 0);
            }
            WriteBackSieveRow(outGm, missGm, outLocal, missLocal, topK);
            continue;
        }

        const int32_t seqLen = seqLenPtr[bid];
        const int32_t prefill = prefillPtr[reqIdx];
        __gm__ int32_t *reqTopk = topkPtr + static_cast<int64_t>(bid) * topK;
        __gm__ int32_t *reqTokens = tokensPtr + reqIdx * padded;
        __gm__ uint8_t *reqVis = visPtr + reqIdx * visStrideB;
        __gm__ uint16_t *reqHt = htPtr + reqIdx * htSize;

        GlobalTensor<int32_t> tokGm;
        GlobalTensor<uint8_t> visGm;
        GlobalTensor<uint16_t> htGm;
        tokGm.SetGlobalBuffer(reqTokens);
        visGm.SetGlobalBuffer(reqVis);
        htGm.SetGlobalBuffer(reqHt);
        DataCopy(tokLocal, tokGm, padded);
        DataCopy(visLocal, visGm, visStrideB);
        DataCopy(htLocal, htGm, htSize);
        PipeBarrier<PIPE_ALL>();

        // hand 经 UB staging 读写（标量 store 对 host/跨内核不可见，同输出约束）
        GlobalTensor<int32_t> handGm;
        handGm.SetGlobalBuffer(handPtr + reqIdx * hisp::kHandStride);
        DataCopy(handLocal, handGm, hisp::kHandStride);
        PipeBarrier<PIPE_ALL>();
        int32_t hand = handLocal.GetValue(0);
        if (hand < 0 || hand >= dbs) {
            hand = 0;  // 防御：脏 hand 钳回有效区间
        }

        bool htDirty = false;  // 仅命中不改表 → 跳过 tokens/ht 写回

        for (int32_t k = 0; k < topK; ++k) {
            const int32_t tokenPos = reqTopk[k];

            if (tokenPos < 0 || tokenPos >= seqLen) {
                outLocal.SetValue(k, -1);
                missLocal.SetValue(k, 0);
                continue;
            }
            if (tokenPos >= prefill) {
                // decode 增量 token：扩展槽，永不淘汰、永不 miss
                outLocal.SetValue(k, padded + (tokenPos - prefill));
                missLocal.SetValue(k, 0);
                continue;
            }

            const int32_t hitSlot = hisp::ht_find(htLocal, tokLocal, tokenPos, htMask);
            if (hitSlot >= 0) {
                outLocal.SetValue(k, hitSlot);
                missLocal.SetValue(k, 0);
                hisp::vis_set(visLocal, hitSlot);
                continue;
            }

            // SIEVE 淘汰：hand 前扫清 visited 位，淘汰首个未访问槽
            int32_t h = hand;
            while (hisp::vis_get(visLocal, h) != 0) {
                hisp::vis_clear(visLocal, h);
                ++h;
                if (h == dbs) {
                    h = 0;
                }
            }

            const int32_t oldToken = tokLocal.GetValue(h);
            if (oldToken >= 0) {
                hisp::ub_ht_erase_bs(htLocal, tokLocal, oldToken, htMask);
            }
            hisp::ht_insert(htLocal, tokLocal, tokenPos, h, htMask);
            htDirty = true;

            outLocal.SetValue(k, h);
            missLocal.SetValue(k, 1);
            tokLocal.SetValue(h, tokenPos);
            hisp::vis_set(visLocal, h);
            hand = h + 1;
            if (hand == dbs) {
                hand = 0;
            }
        }

        handLocal.SetValue(0, hand);

        PipeBarrier<PIPE_ALL>();
        DataCopy(visGm, visLocal, visStrideB);
        if (htDirty) {
            DataCopy(tokGm, tokLocal, padded);
            DataCopy(htGm, htLocal, htSize);
        }
        DataCopy(handGm, handLocal, hisp::kHandStride);
        WriteBackSieveRow(outGm, missGm, outLocal, missLocal, topK);
        PipeBarrier<PIPE_ALL>();
    }
    KvcKernelEpilogue();
}
