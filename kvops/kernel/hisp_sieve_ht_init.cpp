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

extern "C" __global__ __aicore__ void hisp_sieve_ht_init(GM_ADDR argsBuf)
{
    const __gm__ HispSieveHtInitArgs *args = reinterpret_cast<const __gm__ HispSieveHtInitArgs *>(argsBuf);
    // GM 解引用集中读入本地标量（引用形参 API 要求左值；兼省重复 GM 加载）
    const int32_t numReqs = args->numReqs;
    const int32_t layerNum = args->layerNum;
    const int32_t poolSize = args->poolSize;
    const int32_t dbs = args->deviceBufferSize;
    const int32_t padded = args->paddedBufferSize;
    const int32_t htSize = args->htSize;

    const int32_t b = GetBlockIdx();
    if (b >= layerNum * numReqs) {
        return;
    }
    const int32_t layer = b / numReqs;
    const int64_t reqIdx = args->reqStart + b % numReqs;
    if (reqIdx < 0 || reqIdx >= poolSize) {
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
