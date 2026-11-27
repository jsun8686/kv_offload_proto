/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * kvc_scatter_rows_alllayer：全层连续区间导出（HBM 层池 → 远端段）。
 * 每资源节点一次 launch，内核内层循环 + 行归属过滤；单段 8 行聚合大 WQE（对齐免疫 + 块合并搬运）。
 */

#include "kvc_common.h"
#include "kvc_args.h"

// fatbin 纯设备编译路径（ascendc_fatbin_library）无 auto-gen 包装：kernel 实参就是
// host 参数块本身（kvops_ext LaunchCommon 按结构体 memcpy 后 aclrtLaunchKernelWithHostArgs
// 原样下发），不是"指向参数块的 GM 指针"。旧签名 GM_ADDR argsBuf 会把结构体首字段
// channel 误当 GM 指针解引用，args->status 等字段读出垃圾地址，MTE3 写状态页即
// "DDR address out of range"（aivec errcode 95，kvc_scatter_rows_alllayer+0x34c 实测）。
// 故签名与 KvcScatterAllArgs 字段顺序/类型严格一致（自然对齐打包，尾置 reserved 省略），
// 体内重组为结构体，下游代码不变。
extern "C" __global__ __aicore__ void kvc_scatter_rows_alllayer(uint64_t channel, uint64_t segBase,
                                                                uint64_t srcLayer0Base, uint64_t srcLayerStride,
                                                                uint64_t status, uint32_t perLayerSegRows,
                                                                uint32_t rowBytes, uint32_t rowLo, uint32_t rowCnt,
                                                                uint32_t layerNum, uint32_t stripeRows,
                                                                uint32_t nodeNum, uint32_t myNode)
{
    KvcScatterAllArgs argsStorage{};
    argsStorage.channel = channel;
    argsStorage.segBase = segBase;
    argsStorage.srcLayer0Base = srcLayer0Base;
    argsStorage.srcLayerStride = srcLayerStride;
    argsStorage.status = status;
    argsStorage.perLayerSegRows = perLayerSegRows;
    argsStorage.rowBytes = rowBytes;
    argsStorage.rowLo = rowLo;
    argsStorage.rowCnt = rowCnt;
    argsStorage.layerNum = layerNum;
    argsStorage.stripeRows = stripeRows;
    argsStorage.nodeNum = nodeNum;
    argsStorage.myNode = myNode;
    const KvcScatterAllArgs *args = &argsStorage;
    if (GetBlockIdx() != 0) {
        KvcKernelEpilogue();  // c310 收尾序列所有核统一执行（对齐 PoC 约定）
        return;
    }
    KvcStatusWriter w;
    // 与 gather 同契约：每 block 独占 64B 状态槽（整页 64B 刷出不跨槽），当前 blockDim=1
    KvcStatusInit(w, args->status + static_cast<uint64_t>(GetBlockIdx()) * 64);

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
    KvcStatusSet(w, 0, KVC_STAGE_INIT_OK);

    // 8 行合并 WQE（组起点对齐）：WriteNbi 写路径存在文档未载的隐性约束（实测 0x3ff，
    // 2026-11-27 跨机大池）：远端 dst 端点仅 128B 对齐（row_bytes=1152 的奇数行）且段内偏移
    // >~0.9GiB 时 WQE 无 CQE（PollCq 超时 255）；行字节为 256 倍数（1024/1280/1536/2048）
    // 时任意池大小全过。组起点 k*8*rowBytes 恒 256B 对齐（8*1152=9216=36*256），大偏移免疫。
    // 单段（nodeNum=1）行全连续 → 每组 1 个 9216B WJE（WJE 数 /8，兼块合并搬运收益）；
    // 多段交错时本 node 行隔行不连续 → 聚合自动退化为逐行（dst 段内 segRow 奇偶混合，
    // 对齐需布局级改造，不在本改动范围）。host 侧片行数 512、池行数 8*key_len
    // （key_len 为 128 倍数）→ 组起点恒为 8 的倍数行；尾组不满 8 行按连续段发（防御路径）。
    constexpr uint32_t KVC_SCATTER_MERGE_ROWS = 8;
    uint64_t queued = 0;
    for (uint32_t l = 0; l < args->layerNum; l++) {
        const uint64_t srcLayer = args->srcLayer0Base + static_cast<uint64_t>(l) * args->srcLayerStride;
        const uint64_t segLayer = static_cast<uint64_t>(l) * args->perLayerSegRows;
        for (uint32_t j0 = 0; j0 < args->rowCnt; j0 += KVC_SCATTER_MERGE_ROWS) {
            const uint32_t jEnd = (j0 + KVC_SCATTER_MERGE_ROWS < args->rowCnt) ?
                                      (j0 + KVC_SCATTER_MERGE_ROWS) : args->rowCnt;
            uint32_t j = j0;
            while (j < jEnd) {
                const uint32_t row = args->rowLo + j;
                if (KvcNodeOf(row, args->stripeRows, args->nodeNum) != args->myNode) {
                    j++;
                    continue;
                }
                uint32_t run = 1;
                while (j + run < jEnd &&
                       KvcNodeOf(args->rowLo + j + run, args->stripeRows, args->nodeNum) == args->myNode) {
                    run++;
                }
                const uint64_t segRow = KvcSegRowOf(row, args->stripeRows, args->nodeNum);
                // src 基址语义 = 本次调用区域的第 0 行（row_lo 相对寻址）：
                // export_full_rmt_pool 第二段导出传 rope 区首行 + row_lo=kv_rows，
                // 若按绝对行号乘 rowBytes 会从 src 越界 256KB 读到已被 zero_() 的 sel 区，
                // 远端 rope 表被写成全 0（gather 读回全 0，实测 rope mismatch 且 sel_rope 0/512 非零）
                GM_ADDR src = reinterpret_cast<GM_ADDR>(srcLayer +
                                                        static_cast<uint64_t>(j) * args->rowBytes);
                GM_ADDR dst = reinterpret_cast<GM_ADDR>(args->segBase + (segLayer + segRow) * args->rowBytes);
                ret = hcomm.template WriteNbi<false>(args->channel, dst, src, run * args->rowBytes);
                if (ret != 0) {
                    KvcStatusFinish(w, 0x200 + ((l * args->rowCnt + j) & 0xFF) +
                                            (static_cast<uint64_t>(ret) << 16));
                    KvcKernelEpilogue();
                    return;
                }
                queued += run;
                j += run;
            }
        }
    }
    KvcStatusSet(w, 0, KVC_STAGE_QUEUE_OK);

    // Commit 与 WQE 同管道（PIPE_MTE3）保序——机理见 kvc_gather_rows.cpp 同位置注释
    ret = hcomm.template Commit<PIPE_MTE3>(args->channel);
    if (ret != 0) {
        KvcStatusFinish(w, 0x300 + static_cast<uint64_t>(ret));
        KvcKernelEpilogue();
        return;
    }
    KvcStatusSet(w, 0, KVC_STAGE_COMMIT_OK);
    ret = hcomm.Drain(args->channel);
    KvcStatusSet(w, 1, queued);  // 诊断区：搬运行数（poll 不读）
    KvcStatusFinish(w, ret == 0 ? 0 : 0x400 + static_cast<uint64_t>(ret));
    KvcKernelEpilogue();
}
