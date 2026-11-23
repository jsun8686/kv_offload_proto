/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * dsa_gather_rmt_v2 AICPU kernel 参数结构（host 侧 kvops_ext 与 device 侧 kernel 共用，单一事实源）。
 * 全 u64 字段：8B 天然对齐，host/device 两侧 ABI 恒一致；寻址/校验语义与 DsaGatherRmtArgs 相同。
 */

#ifndef DSA_GATHER_RMT_V2_H
#define DSA_GATHER_RMT_V2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t thread;             // HcommThreadHandle（COMM_ENGINE_AICPU_TS，单实例全通道共享）
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
    uint64_t status;             // 64B 槽：word0=(1<<32)|rc（终态），word1=搬运 record 数（诊断）
    uint64_t numRequests;        // batch 行数
    uint64_t queryCount;         // 每请求 topk 槽位宽（pair_count = numRequests*queryCount）
    uint64_t blockSize;          // 物理 block token 数
    uint64_t kvDim;              // KV record 元素数
    uint64_t ropeDim;            // RoPE record 元素数
    uint64_t kvElemSize;         // KV 元素字节（1=int8 / 2=fp16，字节寻址用）
    uint64_t ropeElemSize;       // RoPE 元素字节
    uint64_t hostPoolCapacity;   // 池行数（hostBlockTable dim0）
    uint64_t hostTableWidth;     // 源页表宽（W_src）
    uint64_t residentTableWidth; // 目标页表宽（W_dst）
    uint64_t hostBlkNum;         // 源物理块数（host 池 dim0，越界防御）
    uint64_t residentBlkNum;     // 目标物理块数（resident 池 dim0）
} DsaGatherRmtV2Args;

#ifdef __cplusplus
}
#endif

#endif  // DSA_GATHER_RMT_V2_H
