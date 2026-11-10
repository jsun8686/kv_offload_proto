/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * kvc 内核公共部分：参数布局 / status 写回（UB+MTE3）/ c310 epilogue / 行级传输。
 * 约束：每个编译单元只定义一个 kernel（fatbin 工具链约束）。
 */

#ifndef KVC_KERNEL_COMMON_H
#define KVC_KERNEL_COMMON_H

#include "kernel_operator.h"
#include "adv_api/hcomm/hcomm.h"

using namespace AscendC;

constexpr uint32_t KVC_HCOMM_WS_SIZE = 512;  // HCOMM_URMA_TMP_BUF_SIZE

// status 槽协议（GM，ub(0) 单槽 8B = statusAddr + blockIdx*64，每 block 独占 64B——
// flush 按整页 64B 刷出，8B 步进分片会互相覆盖 doneSeq）：
//   非终态：低 32 位=阶段码（E2/E3/E4），高 32 位=0（host poll 继续等待）；
//   终态：  (doneSeq=1 << 32) | errCode —— host 判 (val>>32)==1 后取低 32 位。
//   ub(1) 起为诊断区（如 queued 计数），poll 不读；整页 64B 经 UB+DataCopy(MTE3) 写回。
//   注意：终态必须写 ub(0)（联调实证：写 ub(1) 时 poll 永远等不到 doneSeq，60s 超时）。
constexpr uint64_t KVC_STATUS_PENDING = 0;
constexpr uint64_t KVC_STAGE_INIT_OK = 0xE2;
constexpr uint64_t KVC_STAGE_QUEUE_OK = 0xE3;
constexpr uint64_t KVC_STAGE_COMMIT_OK = 0xE4;

// 错误码分段：0x1xx=Init、0x2xx=行 Nbi（编码行内偏移）、0x3xx=Commit、0x4xx=Drain
struct KvcStatusWriter {
    TPipe pipe;
    TBuf<TPosition::VECOUT> buf;
    LocalTensor<uint64_t> ub;
    uint64_t gmAddr = 0;
};

static inline __aicore__ void KvcStatusInit(KvcStatusWriter &w, uint64_t statusAddr)
{
    w.gmAddr = statusAddr;
    if (statusAddr == 0) {
        return;
    }
    w.pipe.InitBuffer(w.buf, 64);
    w.ub = w.buf.Get<uint64_t>();
    for (uint32_t i = 0; i < 8; i++) {
        w.ub(i) = 0;
    }
}

static inline __aicore__ void KvcStatusFlush(KvcStatusWriter &w)
{
    if (w.gmAddr == 0) {
        return;
    }
    constexpr TEventID evtSMte3 = 0;
    constexpr TEventID evtMte3S = 1;
    SetFlag<HardEvent::S_MTE3>(evtSMte3);
    WaitFlag<HardEvent::S_MTE3>(evtSMte3);
    GlobalTensor<uint64_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(w.gmAddr), 8);
    DataCopy(gm, w.ub, 8);
    SetFlag<HardEvent::MTE3_S>(evtMte3S);
    WaitFlag<HardEvent::MTE3_S>(evtMte3S);
}

static inline __aicore__ void KvcStatusSet(KvcStatusWriter &w, uint32_t idx, uint64_t value)
{
    if (w.gmAddr == 0) {
        return;
    }
    w.ub(idx) = value;
    KvcStatusFlush(w);
}

// 终态写回（ub(0) 单槽）：(doneSeq=1 << 32) | (errCode & 0xFFFFFFFF)，host poll 由此判定完成
static inline __aicore__ void KvcStatusFinish(KvcStatusWriter &w, uint64_t errCode)
{
    KvcStatusSet(w, 0, (1ULL << 32) | (errCode & 0xFFFFFFFFULL));
}

// c310 收尾 flush（950 平台约定，fatbin 纯设备编译需手动补齐）
static inline __aicore__ void KvcKernelEpilogue()
{
#if defined(__DAV_C310__)
    pipe_barrier(PIPE_ALL);
    dsb(mem_dsb_t::DSB_ALL);
    dci();
#endif
}

// 行归属映射（与 common/kv_layout.h、xds/kv_layout.py 语义一致）
static inline __aicore__ uint32_t KvcNodeOf(uint32_t row, uint32_t stripeRows, uint32_t nodeNum)
{
    return (stripeRows == 0) ? 0 : ((row / stripeRows) % nodeNum);
}

static inline __aicore__ uint32_t KvcSegRowOf(uint32_t row, uint32_t stripeRows, uint32_t nodeNum)
{
    return (stripeRows == 0) ? row : ((row / stripeRows) / nodeNum);
}

#endif  // KVC_KERNEL_COMMON_H
