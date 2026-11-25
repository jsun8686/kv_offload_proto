/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * 内核下传参数（host 侧 kvops_ext 与 AIV 内核共用，布局两端一致，勿改字段顺序）。
 * launch 约定：args buffer = 结构体 + 8B overflow 尾槽（值 = 8B device buffer 地址）。
 */

#ifndef KVC_ARGS_H
#define KVC_ARGS_H

#include <cstdint>

// ── 全层导出（EmsPool.export_full_rmt_pool 数据链）：HBM → 远端段，行交错多节点分发 ──
struct KvcScatterAllArgs {
    uint64_t channel;
    uint64_t segBase;
    uint64_t srcLayer0Base;   // HBM 层 0 基址
    uint64_t srcLayerStride;  // HBM 层跨距（字节）
    uint64_t status;
    uint32_t perLayerSegRows;
    uint32_t rowBytes;
    uint32_t rowLo;           // 各层同区间 [rowLo, rowLo + rowCnt)
    uint32_t rowCnt;
    uint32_t layerNum;
    uint32_t stripeRows;
    uint32_t nodeNum;
    uint32_t myNode;
    uint32_t reserved;
};

// ── DSA miss 记录回源（纯远端 ReadNbi，原生多核 pair 均分）──
struct DsaGatherRmtArgs {
    uint64_t residentKv;         // [P_dst, BS, 1, kvDim] resident KV 物理块池（原地写）
    uint64_t residentRope;       // [P_dst, BS, 1, ropeDim] resident RoPE 物理块池（原地写）
    uint64_t residentBlockTable; // [numReq, W_dst] int32 batch 行→目标物理块
    uint64_t hostBlockTable;     // [poolCap, W_src] int32 池行→源物理块（远端池页表，原样使用）
    uint64_t requestPoolIndices; // [numReq] int32 batch 行→稳定池行（两跳寻址的桥）
    uint64_t topkTokenPositions; // [numReq, queryCount] int32 源 token 位置
    uint64_t residentSlots;      // [numReq, queryCount] int32 目标 slot（lookup 输出）
    uint64_t missMask;           // [numReq, queryCount] int32 唯一搬运开关（1=miss 才搬）
    uint64_t rmtChannel;         // [P_src] int64 源物理块→ChannelHandle（真实值，无 0 语义）
    uint64_t rmtKvAddr;          // [P_src] int64 源物理块→远端 kv 区基址（块内 [BS, kvDim] 连续）
    uint64_t rmtRopeAddr;        // [P_src] int64 源物理块→远端 rope 区基址（块内连续）
    uint64_t status;             // 每核 64B 槽（statusGm + blockIdx*64，kvc_common 协议）
    uint32_t numRequests;        // batch 行数
    uint32_t queryCount;         // 每请求 topk 槽位宽（pair_count = numRequests*queryCount）
    uint32_t blockSize;          // 物理 block token 数
    uint32_t kvDim;              // KV record 元素数
    uint32_t ropeDim;            // RoPE record 元素数
    uint32_t kvElemSize;         // KV 元素字节（1=int8 / 2=fp16，字节寻址用）
    uint32_t ropeElemSize;       // RoPE 元素字节
    uint32_t hostPoolCapacity;   // 池行数（hostBlockTable dim0）
    uint32_t hostTableWidth;     // 源页表宽（W_src）
    uint32_t residentTableWidth; // 目标页表宽（W_dst）
    uint32_t hostBlkNum;         // 源物理块数（host 池 dim0，越界防御）
    uint32_t residentBlkNum;     // 目标物理块数（resident 池 dim0）
    uint32_t residentInit;       // 0=dense decode 布局（V1）；1=紧凑初始化布局（预留）
    uint32_t reserved;
};

#endif  // KVC_ARGS_H
