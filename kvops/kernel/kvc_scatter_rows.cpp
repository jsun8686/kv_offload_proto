/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * kvc_scatter_rows：单层索引导出（HBM 层池 → 远端段）。decode 增量备份（少行）场景。
 */

#include "kvc_common.h"
#include "kvc_args.h"

extern "C" __global__ __aicore__ void kvc_scatter_rows(GM_ADDR argsBuf)
{
    if (GetBlockIdx() != 0) {
        return;
    }
    const __gm__ KvcGatherArgs *args = reinterpret_cast<const __gm__ KvcGatherArgs *>(argsBuf);
    KvcStatusWriter w;
    KvcStatusInit(w, args->status);

    TBuf<TPosition::VECOUT> wsBuf;
    w.pipe.InitBuffer(wsBuf, KVC_HCOMM_WS_SIZE);
    LocalTensor<uint8_t> ws = wsBuf.Get<uint8_t>();

    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm;
    int32_t ret = hcomm.Init(ws, KVC_HCOMM_WS_SIZE);
    if (ret != 0) {
        KvcStatusSet(w, 1, 0x100 + ret);
        KvcKernelEpilogue();
        return;
    }
    KvcStatusSet(w, 0, KVC_STAGE_INIT_OK);

    const uint64_t numReal = *reinterpret_cast<const __gm__ uint64_t *>(args->numRealAddr);
    const uint64_t count = (numReal < args->maxBatch) ? numReal : args->maxBatch;
    const __gm__ int64_t *hostIdx = reinterpret_cast<const __gm__ int64_t *>(args->indicesHost);
    const __gm__ int64_t *dstIdx = reinterpret_cast<const __gm__ int64_t *>(args->indicesDst);

    uint64_t queued = 0;
    for (uint64_t i = 0; i < count; i++) {
        const int64_t row = hostIdx[i];
        if (row < 0) {
            continue;
        }
        if (KvcNodeOf(static_cast<uint32_t>(row), args->stripeRows, args->nodeNum) != args->myNode) {
            continue;
        }
        const uint64_t segRow = KvcSegRowOf(static_cast<uint32_t>(row), args->stripeRows, args->nodeNum);
        GM_ADDR src = reinterpret_cast<GM_ADDR>(args->dstLayerBase +
                                                static_cast<uint64_t>(dstIdx[i]) * args->rowBytes);
        GM_ADDR dst = reinterpret_cast<GM_ADDR>(
            args->segBase + (static_cast<uint64_t>(args->layerOffsetRows) + segRow) * args->rowBytes);
        ret = hcomm.template WriteNbi<false>(args->channel, dst, src, args->rowBytes);
        if (ret != 0) {
            KvcStatusSet(w, 1, 0x200 + (i & 0xFF) + (static_cast<uint64_t>(ret) << 16));
            KvcKernelEpilogue();
            return;
        }
        queued++;
    }
    KvcStatusSet(w, 0, KVC_STAGE_QUEUE_OK);

    ret = hcomm.Commit(args->channel);
    if (ret != 0) {
        KvcStatusSet(w, 1, 0x300 + ret);
        KvcKernelEpilogue();
        return;
    }
    KvcStatusSet(w, 0, KVC_STAGE_COMMIT_OK);
    ret = hcomm.Drain(args->channel);
    const uint64_t done = (queued << 40) | (1ULL << 32) | static_cast<uint64_t>(ret == 0 ? 0 : 0x400 + ret);
    KvcStatusSet(w, 1, done);
    KvcKernelEpilogue();
}
