/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * 内核下传参数（host 侧 kvops_ext 与 AIV 内核共用，布局两端一致，勿改字段顺序）。
 * launch 约定：args buffer = 结构体 + 8B overflow 尾槽（值 = 8B device buffer 地址）。
 */

#ifndef KVC_ARGS_H
#define KVC_ARGS_H

#include <cstdint>

// ── 批量导入（稠密接口）：远端段 → HBM 热缓冲，多 block 按 positions 分片 ──
// 输入为 sieve 输出契约（稠密张量）：扫 [numReal×topK)，isMiss==1 且行归属本节点才搬运；
// src 行号 = reqHostBase[req] + tokenPos（行分配连续性契约，免 req_to_host_pool 大表）；
// dst 行号 = hotRowBase + slot（热缓冲槽与 KV 池物理行固定偏移契约）。
struct KvcGatherArgs {
    uint64_t channel;
    uint64_t segBase;
    uint64_t topkIndices;      // int32 [maxNumReqs, topK]（device）
    uint64_t reqPoolIndices;   // int64 [maxNumReqs]（device）
    uint64_t reqHostBase;      // int64 [maxNumReqs]：每请求 host 池起始行号（device）
    uint64_t topKDeviceSlots;  // int32 [maxNumReqs, topK]（sieve 输出，device）
    uint64_t isMiss;           // int8  [maxNumReqs, topK]（sieve 输出，device）
    uint64_t numReal;          // int32 标量张量（图 padding 早退）
    uint64_t hotRowBase;       // HBM 热缓冲区起始物理行号（本层）
    uint64_t status;           // GM 状态槽（blockDim 分片：status + blockIdx*64，每槽独占 64B 整页）
    uint32_t layerOffsetRows;  // layerId * perLayerSegRows
    uint32_t perLayerSegRows;
    uint32_t rowBytes;
    uint32_t stripeRows;       // S
    uint32_t nodeNum;          // N
    uint32_t myNode;
    uint32_t topK;
    uint32_t maxNumReqs;
    uint32_t positionsPerCore; // ceil(maxNumReqs*topK / blockDim)，对齐 64
    uint32_t reserved;
};

// ── 全层连续区间导出：HBM 层池 → 远端段（prefill 备份） ──
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

// ── SIEVE miss 识别/淘汰更新（哈希-only；按层切片传入 2D 状态） ──
struct HispSieveUpdateArgs {
    uint64_t topkIndices;         // int32 [maxNumReqs, topK]
    uint64_t reqPoolIndices;      // int64 [maxNumReqs]
    uint64_t seqLens;             // int32 [maxNumReqs]（图静态 int32 约定）
    uint64_t prefillLens;         // int32 [poolSize]（按池行索引）
    uint64_t deviceBufferTokens;  // int32 [poolSize, padded]（本层切片）
    uint64_t deviceBufferVisited; // uint8 位图 [poolSize, visitedStrideB]
    uint64_t deviceBufferHt;      // uint16 [poolSize, htSize]（开放寻址）
    uint64_t sieveHand;           // int32 [poolSize, 16]（只用 [0]，64B 行距）
    uint64_t topKDeviceSlots;     // int32 [maxNumReqs, topK]（输出）
    uint64_t isMiss;              // int32 [maxNumReqs, topK]（输出；MTE 回写 32B 粒度，topK%8==0）
    uint64_t numReal;             // int32 标量张量
    uint64_t status;              // 保留（fire-and-forget，靠 stream 序；0=不写）
    uint32_t topK;
    uint32_t deviceBufferSize;
    uint32_t paddedBufferSize;
    uint32_t visitedStrideB;      // 位图行字节数（64B 对齐，≥ceil(padded/8)）
    uint32_t maxNumReqs;
    uint32_t reqsPerCore;         // 对齐 64/gcd(topK,64) 的倍数（64B 输出 cache line）
    uint32_t htSize;              // 2 的幂，≥2×(padded+topK)，装载因子 ≤0.5
    uint32_t poolSize;
    uint32_t reserved;
};

// ── SIEVE 哈希表构建（全层 3D 一次初始化或 admission 逐请求） ──
struct HispSieveHtInitArgs {
    uint64_t deviceBufferTokens;  // int32 [layerNum, poolSize, padded]（3D 全量）
    uint64_t deviceBufferHt;      // uint16 [layerNum, poolSize, htSize]
    uint64_t prefillLens;         // int32 [poolSize]
    uint64_t status;              // 保留
    uint32_t reqStart;
    uint32_t numReqs;
    uint32_t layerNum;
    uint32_t poolSize;
    uint32_t deviceBufferSize;
    uint32_t paddedBufferSize;
    uint32_t htSize;
    uint32_t reserved;
};

// ── gather_selection_kv_cache 远端池适配版（V2，接口同构 + full_rmt_* 三表）──
// 语义基准：cann-recipes-infer test_npu_gather_selection_kv_cache.py golden（:96-136 基础路径）
// 数据布局：热池/本地池均 [BlkNum, BLOCK_SIZE, D]（D 按 2B 元素计），页表 int32，账本 [TOPK+1] int32。
// V1.0：本地路径（channel==0）+ 远端分支预留；V1.1：ReadNbi 回源接入。
struct GatherSelectionRmtArgs {
    uint64_t selKRope;          // [S_BLOCK_NUM, BS, K_ROPE] 热池 rope 区（原地写）
    uint64_t selKvCache;        // [S_BLOCK_NUM, BS, KV_CACHE] 热池 kv 区（原地写）
    uint64_t selBlockTable;     // [rows, S_MAX_BLOCK_NUM] int32 热池页表（rows=B*S*H）
    uint64_t selBlockStatus;    // [B,S,H,TOPK+1] int32 账本（原地写；末位=长度）
    uint64_t topkIndices;       // [B,S,H,TOPK] int32 topk 块段 id
    uint64_t fullKRope;         // 本地池（纯远端传空：kRopeDim==0 跳过）
    uint64_t fullKvCache;       // 本地池（纯远端传空）
    uint64_t fullBlockTable;    // [B, F_MAX_BLOCK_NUM] int32 逻辑→物理块（原样使用）
    uint64_t fullKvActualSeq;   // [B] int32
    uint64_t fullQActualSeq;    // [B] int32
    uint64_t rmtChannel;        // [F_BLOCK_NUM] int64 物理块→ChannelHandle（0=本地）
    uint64_t rmtKvAddr;         // [F_BLOCK_NUM] int64 物理块→远端 kv 块基址
    uint64_t rmtRopeAddr;       // [F_BLOCK_NUM] int64 物理块→远端 rope 块基址
    uint64_t status;            // 内核 status 槽（kvc_common 64B 协议）
    uint64_t actualSeq;         // [rows] int32 输出
    uint32_t batch, seq, head;  // rows = batch*seq*head（head 恒 1）
    uint32_t topk;
    uint32_t blockSize;         // BLOCK_SIZE（物理块 token 数）
    uint32_t fullMaxBlockNum;   // 页表列数
    uint32_t selMaxBlockNum;
    uint32_t fullKvBlockNum;    // 本地池物理块数（越界防御）
    uint32_t selTopKBlockSize;  // topk 段粒度
    uint32_t kvCacheDim;        // KV_CACHE 维（2B 元素数；0=纯远端）
    uint32_t kRopeDim;          // K_ROPE 维（0=int8 空 rope 惯例）
    uint32_t reserved;
};

// dsa_gather 远端池适配版（纯远端：源恒在 emsd 段，无本地分支/无 0 哨兵）。
// 语义基准：vllm-ascend-csrc dsa_gather golden.py（miss_mask 唯一闸门 + pool_entry 两跳寻址
// + 三层校验 + sentinel 零搬运）；搬运 = ReadNbi 直写 resident 池（须在 cli mem_reg 注册范围）。
// 源池张量不入参（对比原版删 host_kv_cache/host_rope_cache）；新增 host_rmt_* 三表
// （[P_src] int64，src_blk 查表得 channel/远端基址）。
// residentInit 预留（V1 恒 0=dense decode 布局；紧凑初始化布局 V1.1 需要时加）。
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
