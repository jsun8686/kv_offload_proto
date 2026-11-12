/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * gather_selection_rmt：npu_gather_selection_kv_cache 远端池适配版（V2）。
 *
 * 语义基准：cann-recipes-infer test_npu_gather_selection_kv_cache.py golden 基础路径
 * （有效性过滤 + 因果裁剪 + 紧凑追加 + 账本回写；命中免搬/跨 seq 复用为 V1.1 优化）。
 * 数据搬运：本地段（channel==0）DataCopyPad 经 UB staging；远端段 Hcomm ReadNbi 直写热池
 * （单流 Nbi* → Commit<PIPE_MTE3> → Drain，规避多流丢 WQE）。
 * 回写（账本/actual_seq）经 DataCopyPad（行尾非 32B 对齐安全）。
 * 元素宽度固定 2B（fp16/bf16 搬运不解析值；int8 场景走空 rope 惯例，V3）。
 */

#include "kvc_common.h"
#include "kvc_args.h"

namespace {
constexpr uint32_t ELEM_SIZE = 2;  // fp16/bf16
constexpr uint32_t RMT_LOCAL = 0;  // channel==0 预留本地
}  // namespace

// fatbin 纯设备编译路径：kernel 实参 = host 参数块逐字段自然对齐布局（见 kvc_args.h 注释）
extern "C" __global__ __aicore__ void gather_selection_rmt(
    uint64_t selKRope, uint64_t selKvCache, uint64_t selBlockTable, uint64_t selBlockStatus,
    uint64_t topkIndices, uint64_t fullKRope, uint64_t fullKvCache, uint64_t fullBlockTable,
    uint64_t fullKvActualSeq, uint64_t fullQActualSeq, uint64_t rmtChannel, uint64_t rmtKvAddr,
    uint64_t rmtRopeAddr, uint64_t statusGm, uint64_t actualSeqGm, uint32_t batchIn, uint32_t seqIn,
    uint32_t headIn, uint32_t topkIn, uint32_t blockSizeIn, uint32_t fullMaxBlkIn, uint32_t selMaxBlkIn,
    uint32_t fullBlkNumIn, uint32_t topkBsIn, uint32_t kvDimIn, uint32_t ropeDimIn)
{
    GatherSelectionRmtArgs argsStorage{};
    argsStorage.selKRope = selKRope;
    argsStorage.selKvCache = selKvCache;
    argsStorage.selBlockTable = selBlockTable;
    argsStorage.selBlockStatus = selBlockStatus;
    argsStorage.topkIndices = topkIndices;
    argsStorage.fullKRope = fullKRope;
    argsStorage.fullKvCache = fullKvCache;
    argsStorage.fullBlockTable = fullBlockTable;
    argsStorage.fullKvActualSeq = fullKvActualSeq;
    argsStorage.fullQActualSeq = fullQActualSeq;
    argsStorage.rmtChannel = rmtChannel;
    argsStorage.rmtKvAddr = rmtKvAddr;
    argsStorage.rmtRopeAddr = rmtRopeAddr;
    argsStorage.status = statusGm;
    argsStorage.actualSeq = actualSeqGm;
    argsStorage.batch = batchIn;
    argsStorage.seq = seqIn;
    argsStorage.head = headIn;
    argsStorage.topk = topkIn;
    argsStorage.blockSize = blockSizeIn;
    argsStorage.fullMaxBlockNum = fullMaxBlkIn;
    argsStorage.selMaxBlockNum = selMaxBlkIn;
    argsStorage.fullKvBlockNum = fullBlkNumIn;
    argsStorage.selTopKBlockSize = topkBsIn;
    argsStorage.kvCacheDim = kvDimIn;
    argsStorage.kRopeDim = ropeDimIn;
    const GatherSelectionRmtArgs *args = &argsStorage;

    if (GetBlockIdx() != 0) {  // V1.0 单 block 串行（单流约束实证）；多核分片 V1.1 后评估
        KvcKernelEpilogue();
        return;
    }
    KvcStatusWriter w;
    KvcStatusInit(w, args->status);

    const uint32_t rows = args->batch * args->seq * args->head;
    const uint32_t topk = args->topk;
    const uint32_t bs = args->blockSize;
    const uint32_t topkBs = args->selTopKBlockSize;
    const uint32_t kvDim = args->kvCacheDim;
    const uint32_t ropeDim = args->kRopeDim;

    __gm__ int32_t *topkPtr = reinterpret_cast<__gm__ int32_t *>(args->topkIndices);
    __gm__ int32_t *fullTbl = reinterpret_cast<__gm__ int32_t *>(args->fullBlockTable);
    __gm__ int32_t *selTbl = reinterpret_cast<__gm__ int32_t *>(args->selBlockTable);
    __gm__ int32_t *kvSeqPtr = reinterpret_cast<__gm__ int32_t *>(args->fullKvActualSeq);
    __gm__ int32_t *qSeqPtr = reinterpret_cast<__gm__ int32_t *>(args->fullQActualSeq);
    __gm__ int64_t *chTbl = reinterpret_cast<__gm__ int64_t *>(args->rmtChannel);
    __gm__ int64_t *kvAddrTbl = reinterpret_cast<__gm__ int64_t *>(args->rmtKvAddr);
    __gm__ int64_t *ropeAddrTbl = reinterpret_cast<__gm__ int64_t *>(args->rmtRopeAddr);
    __gm__ uint16_t *fullKvGm = reinterpret_cast<__gm__ uint16_t *>(args->fullKvCache);
    __gm__ uint16_t *fullRopeGm = reinterpret_cast<__gm__ uint16_t *>(args->fullKRope);
    __gm__ uint16_t *selKvGm = reinterpret_cast<__gm__ uint16_t *>(args->selKvCache);
    __gm__ uint16_t *selRopeGm = reinterpret_cast<__gm__ uint16_t *>(args->selKRope);
    __gm__ int32_t *statusGmPtr = reinterpret_cast<__gm__ int32_t *>(args->selBlockStatus);
    __gm__ int32_t *actualGmPtr = reinterpret_cast<__gm__ int32_t *>(args->actualSeq);

    // UB staging：kv 段 + rope 段（最大 topkBs 个 token）+ 账本行（TOPK+1 int32）+ actual 单元
    TPipe pipe;
    TBuf<TPosition::VECCALC> kvSegBuf;
    TBuf<TPosition::VECCALC> ropeSegBuf;
    TBuf<TPosition::VECCALC> statusBuf;
    TBuf<TPosition::VECCALC> actualBuf;
    TBuf<TPosition::VECOUT> wsBuf;
    pipe.InitBuffer(kvSegBuf, topkBs * kvDim * ELEM_SIZE);
    pipe.InitBuffer(ropeSegBuf, topkBs * ropeDim * ELEM_SIZE);
    pipe.InitBuffer(statusBuf, (topk + 1) * sizeof(int32_t));
    pipe.InitBuffer(actualBuf, 32);  // 32B 最小分配粒度
    pipe.InitBuffer(wsBuf, KVC_HCOMM_WS_SIZE);
    LocalTensor<uint16_t> kvSeg = kvSegBuf.Get<uint16_t>();
    LocalTensor<uint16_t> ropeSeg = ropeSegBuf.Get<uint16_t>();
    LocalTensor<int32_t> statusLocal = statusBuf.Get<int32_t>();
    LocalTensor<int32_t> actualLocal = actualBuf.Get<int32_t>();

    // 远端回源（仅存在远端块时初始化；单流提交；V1.0 单通道，多通道去重 V2）
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm;
    bool hcommInited = false;
    bool rmtQueued = false;
    ChannelHandle rmtCh = 0;
    LocalTensor<uint8_t> ws = wsBuf.Get<uint8_t>();
    uint64_t totalActual = 0;

    GlobalTensor<uint16_t> fullKvView;
    GlobalTensor<uint16_t> fullRopeView;
    GlobalTensor<uint16_t> selKvView;
    GlobalTensor<uint16_t> selRopeView;
    fullKvView.SetGlobalBuffer(fullKvGm);
    fullRopeView.SetGlobalBuffer(fullRopeGm);
    selKvView.SetGlobalBuffer(selKvGm);
    selRopeView.SetGlobalBuffer(selRopeGm);

    DataCopyPadExtParams<uint16_t> padParams{false, 0, 0, 0};

    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t b = row / (args->seq * args->head);
        const uint32_t s = (row / args->head) % args->seq;

        // 因果裁剪：第 s 个 query 只能看到 kv_len - (q_len-1-s)
        int32_t kvLen = kvSeqPtr[b];
        const int32_t qLen = qSeqPtr[b];
        if (s < static_cast<uint32_t>(qLen)) {
            kvLen -= (qLen - 1 - static_cast<int32_t>(s));
        }
        uint32_t validCnt = 0;
        int32_t actual = 0;
        if (kvLen > 0) {
            const uint32_t maxSelNum = (static_cast<uint32_t>(kvLen) + topkBs - 1) / topkBs;
            const uint32_t lastBs = static_cast<uint32_t>(kvLen) - (maxSelNum - 1) * topkBs;
            for (uint32_t k = 0; k < topk; ++k) {
                const int32_t seg = topkPtr[row * topk + k];
                if (seg < 0) {
                    break;  // 截断
                }
                if (static_cast<uint32_t>(seg) >= maxSelNum) {
                    continue;  // 超界跳过
                }
                const uint32_t cu = (static_cast<uint32_t>(seg) == maxSelNum - 1) ? lastBs : topkBs;

                // full 侧：逻辑 token 偏移 → 物理块 + 块内偏移
                const uint32_t fTok = static_cast<uint32_t>(seg) * topkBs;
                const int32_t fBlk = fullTbl[b * args->fullMaxBlockNum + fTok / bs];
                if (fBlk < 0 || static_cast<uint32_t>(fBlk) >= args->fullKvBlockNum) {
                    continue;  // 页表越界防御
                }
                const uint32_t fOff = fTok % bs;
                // selection 侧：紧凑偏移 → 物理块 + 块内偏移
                const uint32_t sTok = validCnt * topkBs;
                const int32_t sBlk = selTbl[row * args->selMaxBlockNum + sTok / bs];
                const uint32_t sOff = sTok % bs;
                if (sBlk < 0) {
                    continue;
                }

                const int64_t ch = chTbl[fBlk];
                if (kvDim > 0) {
                    DataCopyExtParams cpKv{1, static_cast<uint32_t>(cu * kvDim * ELEM_SIZE), 0, 0, 0};
                    const int64_t dstKvBase = static_cast<int64_t>(sBlk) * bs * kvDim + sOff * kvDim;
                    if (ch == RMT_LOCAL) {
                        DataCopyPad(kvSeg, fullKvView[static_cast<int64_t>(fBlk) * bs * kvDim + fOff * kvDim],
                                    cpKv, padParams);
                        PipeBarrier<PIPE_ALL>();
                        DataCopyPad(selKvView[dstKvBase], kvSeg, cpKv);  // UB→GM 为 3 参（无 pad）
                    } else {
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
                            static_cast<ChannelHandle>(ch), reinterpret_cast<GM_ADDR>(selKvGm + dstKvBase),
                            reinterpret_cast<GM_ADDR>(kvAddrTbl[fBlk] + static_cast<int64_t>(fOff) * kvDim *
                                                              ELEM_SIZE),
                            static_cast<uint64_t>(cu) * kvDim * ELEM_SIZE);
                        if (ret != 0) {
                            KvcStatusFinish(w, 0x200 + (k & 0xFF) + (static_cast<uint64_t>(ret) << 16));
                            KvcKernelEpilogue();
                            return;
                        }
                        rmtCh = static_cast<ChannelHandle>(ch);
                        rmtQueued = true;
                    }
                }
                if (ropeDim > 0) {
                    DataCopyExtParams cpRope{1, static_cast<uint32_t>(cu * ropeDim * ELEM_SIZE), 0, 0, 0};
                    const int64_t dstRopeBase = static_cast<int64_t>(sBlk) * bs * ropeDim + sOff * ropeDim;
                    if (ch == RMT_LOCAL) {
                        DataCopyPad(ropeSeg, fullRopeView[static_cast<int64_t>(fBlk) * bs * ropeDim +
                                                          fOff * ropeDim],
                                    cpRope, padParams);
                        PipeBarrier<PIPE_ALL>();
                        DataCopyPad(selRopeView[dstRopeBase], ropeSeg, cpRope);  // UB→GM 为 3 参（无 pad）
                    } else {
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
                            static_cast<ChannelHandle>(ch), reinterpret_cast<GM_ADDR>(selRopeGm + dstRopeBase),
                            reinterpret_cast<GM_ADDR>(ropeAddrTbl[fBlk] +
                                                      static_cast<int64_t>(fOff) * ropeDim * ELEM_SIZE),
                            static_cast<uint64_t>(cu) * ropeDim * ELEM_SIZE);
                        if (ret != 0) {
                            KvcStatusFinish(w, 0x300 + (k & 0xFF) + (static_cast<uint64_t>(ret) << 16));
                            KvcKernelEpilogue();
                            return;
                        }
                        rmtCh = static_cast<ChannelHandle>(ch);
                        rmtQueued = true;
                    }
                }

                statusLocal.SetValue(validCnt, seg);
                validCnt++;
                actual += static_cast<int32_t>(cu);
            }
        }
        for (uint32_t k = validCnt; k < topk; ++k) {
            statusLocal.SetValue(k, -1);
        }
        statusLocal.SetValue(topk, actual);

        // 账本行 / actual_seq 回写（DataCopyPad：行尾非 32B 对齐安全）
        GlobalTensor<int32_t> statusRow;
        statusRow.SetGlobalBuffer(statusGmPtr + static_cast<int64_t>(row) * (topk + 1));
        DataCopyExtParams cpStatus{1, static_cast<uint32_t>((topk + 1) * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(statusRow, statusLocal, cpStatus);  // UB→GM 为 3 参（无 pad）
        actualLocal.SetValue(0, actual);
        GlobalTensor<int32_t> actualOne;
        actualOne.SetGlobalBuffer(actualGmPtr + row);
        DataCopyExtParams cpActual{1, sizeof(int32_t), 0, 0, 0};
        DataCopyPad(actualOne, actualLocal, cpActual);  // UB→GM 为 3 参（无 pad）
        PipeBarrier<PIPE_ALL>();
        totalActual += static_cast<uint64_t>(actual);
    }

    // 单流收尾：远端提交的全部 WQE 一次 Commit（与 WQE 同管道保序）+ Drain 后才可写终态
    if (rmtQueued) {
        int32_t ret = hcomm.template Commit<PIPE_MTE3>(rmtCh);
        if (ret != 0) {
            KvcStatusFinish(w, 0x400 + static_cast<uint64_t>(ret));
            KvcKernelEpilogue();
            return;
        }
        ret = hcomm.Drain(rmtCh);
        if (ret != 0) {
            KvcStatusFinish(w, 0x500 + static_cast<uint64_t>(ret));
            KvcKernelEpilogue();
            return;
        }
    }
    KvcStatusSet(w, 1, totalActual);  // 诊断区：总选取 token 数（poll 不读）
    KvcStatusFinish(w, 0);
    KvcKernelEpilogue();
}
