/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * hisp_sieve_ht_init：持久哈希表构建（哈希-only，移植自 sglang-0730 hisparse_sieve_ht_init）。
 * admission 时逐请求调用（numReqs=1）或构造期全量初始化（blockDim = layerNum × numReqs）。
 */

#include "hisp_kernel_common.h"
#include "kvc_args.h"
#include "kvc_common.h"

extern "C" __global__ __aicore__ void hisp_sieve_ht_init(GM_ADDR argsBuf)
{
    const __gm__ HispSieveHtInitArgs *args = reinterpret_cast<const __gm__ HispSieveHtInitArgs *>(argsBuf);
    const int32_t b = GetBlockIdx();
    if (b >= static_cast<int32_t>(args->layerNum) * static_cast<int32_t>(args->numReqs)) {
        return;
    }
    const int32_t layer = b / args->numReqs;
    const int64_t reqIdx = args->reqStart + b % args->numReqs;
    if (reqIdx < 0 || reqIdx >= static_cast<int64_t>(args->poolSize)) {
        return;
    }

    __gm__ int32_t *tokensPtr = reinterpret_cast<__gm__ int32_t *>(args->deviceBufferTokens) +
                                (static_cast<int64_t>(layer) * args->poolSize + reqIdx) * args->paddedBufferSize;
    __gm__ uint16_t *htPtr = reinterpret_cast<__gm__ uint16_t *>(args->deviceBufferHt) +
                             (static_cast<int64_t>(layer) * args->poolSize + reqIdx) * args->htSize;

    TPipe pipe;
    TBuf<TPosition::VECCALC> htBuf;
    pipe.InitBuffer(htBuf, args->htSize * sizeof(uint16_t));
    LocalTensor<uint16_t> htLocal = htBuf.Get<uint16_t>();

    const int32_t htMask = static_cast<int32_t>(args->htSize) - 1;
    GlobalTensor<uint16_t> htGm;
    htGm.SetGlobalBuffer(htPtr);

    Duplicate<uint16_t>(htLocal, hisp::kHtEmpty, args->htSize);
    PipeBarrier<PIPE_ALL>();
    for (int32_t s = 0; s < static_cast<int32_t>(args->deviceBufferSize); ++s) {
        const int32_t t = tokensPtr[s];
        if (t >= 0) {
            hisp::ub_ht_insert_gm_tok(htLocal, tokensPtr, t, s, htMask);
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(htGm, htLocal, args->htSize);
    KvcKernelEpilogue();
}
