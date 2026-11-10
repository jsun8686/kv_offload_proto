/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * hisp_sieve_ht_init：持久哈希表构建（哈希-only，移植自 sglang-0730 hisparse_sieve_ht_init）。
 * admission 时逐请求调用（numReqs=1）或构造期全量初始化（blockDim = layerNum × numReqs）。
 * 注意：args 字段为 GM 解引用（临时值），集中读入本地标量后再传 API——
 * Duplicate 等引用形参（const int32_t&）不接受右值。
 */

#include "hisp_kernel_common.h"
#include "kvc_args.h"
#include "kvc_common.h"

// fatbin 纯设备编译路径无 auto-gen 包装：kernel 实参 = host 参数块逐字段自然对齐布局
// （见 kvc_scatter_rows_alllayer.cpp 头注释），非 GM 指针。签名与 HispSieveHtInitArgs
// 字段顺序/类型严格一致（尾置 reserved 省略），体内重组为结构体。
extern "C" __global__ __aicore__ void hisp_sieve_ht_init(uint64_t deviceBufferTokens, uint64_t deviceBufferHt,
                                                         uint64_t prefillLens, uint64_t status, uint32_t reqStart,
                                                         uint32_t numReqsIn, uint32_t layerNumIn, uint32_t poolSizeIn,
                                                         uint32_t deviceBufferSize, uint32_t paddedBufferSize,
                                                         uint32_t htSizeIn)
{
    HispSieveHtInitArgs argsStorage{};
    argsStorage.deviceBufferTokens = deviceBufferTokens;
    argsStorage.deviceBufferHt = deviceBufferHt;
    argsStorage.prefillLens = prefillLens;
    argsStorage.status = status;
    argsStorage.reqStart = reqStart;
    argsStorage.numReqs = numReqsIn;
    argsStorage.layerNum = layerNumIn;
    argsStorage.poolSize = poolSizeIn;
    argsStorage.deviceBufferSize = deviceBufferSize;
    argsStorage.paddedBufferSize = paddedBufferSize;
    argsStorage.htSize = htSizeIn;
    const HispSieveHtInitArgs *args = &argsStorage;
    // GM 解引用集中读入本地标量（引用形参 API 要求左值；兼省重复 GM 加载）
    const int32_t numReqs = args->numReqs;
    const int32_t layerNum = args->layerNum;
    const int32_t poolSize = args->poolSize;
    const int32_t dbs = args->deviceBufferSize;
    const int32_t padded = args->paddedBufferSize;
    const int32_t htSize = args->htSize;

    const int32_t b = GetBlockIdx();
    if (b >= layerNum * numReqs) {
        KvcKernelEpilogue();  // c310 收尾序列所有核统一执行（对齐 PoC 约定）
        return;
    }
    const int32_t layer = b / numReqs;
    const int64_t reqIdx = args->reqStart + b % numReqs;
    if (reqIdx < 0 || reqIdx >= poolSize) {
        KvcKernelEpilogue();
        return;
    }

    __gm__ int32_t *tokensPtr = reinterpret_cast<__gm__ int32_t *>(args->deviceBufferTokens) +
                                (static_cast<int64_t>(layer) * poolSize + reqIdx) * padded;
    __gm__ uint16_t *htPtr = reinterpret_cast<__gm__ uint16_t *>(args->deviceBufferHt) +
                             (static_cast<int64_t>(layer) * poolSize + reqIdx) * htSize;

    TPipe pipe;
    TBuf<TPosition::VECCALC> htBuf;
    pipe.InitBuffer(htBuf, htSize * sizeof(uint16_t));
    LocalTensor<uint16_t> htLocal = htBuf.Get<uint16_t>();

    const int32_t htMask = htSize - 1;
    GlobalTensor<uint16_t> htGm;
    htGm.SetGlobalBuffer(htPtr);

    Duplicate<uint16_t>(htLocal, hisp::kHtEmpty, htSize);
    PipeBarrier<PIPE_ALL>();
    for (int32_t s = 0; s < dbs; ++s) {
        const int32_t t = tokensPtr[s];
        if (t >= 0) {
            hisp::ub_ht_insert_gm_tok(htLocal, tokensPtr, t, s, htMask);
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(htGm, htLocal, htSize);
    KvcKernelEpilogue();
}
