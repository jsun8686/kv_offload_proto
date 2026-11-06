/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * kvc_scatter_rows_alllayer：全层连续区间导出（HBM 层池 → 远端段）。
 * 每资源节点一次 launch，内核内层循环 + 行归属过滤；条带聚合大 WQE 留 M3 优化。
 */

#include "kvc_common.h"
#include "kvc_args.h"

extern "C" __global__ __aicore__ void kvc_scatter_rows_alllayer(GM_ADDR argsBuf)
{
    if (GetBlockIdx() != 0) {
        return;
    }
    const __gm__ KvcScatterAllArgs *args = reinterpret_cast<const __gm__ KvcScatterAllArgs *>(argsBuf);
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

    uint64_t queued = 0;
    for (uint32_t l = 0; l < args->layerNum; l++) {
        const uint64_t srcLayer = args->srcLayer0Base + static_cast<uint64_t>(l) * args->srcLayerStride;
        const uint64_t segLayer = static_cast<uint64_t>(l) * args->perLayerSegRows;
        for (uint32_t row = args->rowLo; row < args->rowLo + args->rowCnt; row++) {
            if (KvcNodeOf(row, args->stripeRows, args->nodeNum) != args->myNode) {
                continue;
            }
            const uint64_t segRow = KvcSegRowOf(row, args->stripeRows, args->nodeNum);
            GM_ADDR src = reinterpret_cast<GM_ADDR>(srcLayer + static_cast<uint64_t>(row) * args->rowBytes);
            GM_ADDR dst = reinterpret_cast<GM_ADDR>(args->segBase + (segLayer + segRow) * args->rowBytes);
            ret = hcomm.template WriteNbi<false>(args->channel, dst, src, args->rowBytes);
            if (ret != 0) {
                KvcStatusSet(w, 1, 0x200 + ((l * args->rowCnt + (row - args->rowLo)) & 0xFF) +
                                           (static_cast<uint64_t>(ret) << 16));
                KvcKernelEpilogue();
                return;
            }
            queued++;
        }
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
