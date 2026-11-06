/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * kvc 内核下传参数（host 侧 kvops_ext 与 AIV 内核共用，布局两端一致，勿改字段顺序）。
 * launch 约定：args buffer = 结构体 + 8B overflow 尾槽（值 = 8B device buffer 地址）。
 */

#ifndef KVC_ARGS_H
#define KVC_ARGS_H

#include <cstdint>

// 导入（远端段 → HBM 层池）与单层索引导出共用形态
struct KvcGatherArgs {
    uint64_t channel;
    uint64_t segBase;
    uint64_t indicesHost;      // int64[] host 池逻辑行号（device 地址）
    uint64_t indicesDst;       // int64[] HBM 物理槽位号（device 地址）
    uint64_t numRealAddr;      // 标量张量地址：有效行数（图 padding 早退）
    uint64_t dstLayerBase;     // HBM 层基址（gather 为目的 / scatter 为源 srcLayerBase）
    uint64_t status;           // GM 状态槽地址
    uint32_t layerOffsetRows;  // layerId * perLayerSegRows
    uint32_t perLayerSegRows;
    uint32_t rowBytes;
    uint32_t stripeRows;
    uint32_t nodeNum;
    uint32_t myNode;
    uint32_t maxBatch;         // 录制容量（图模式）
    uint32_t reserved;
};

// 全层连续区间导出
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

#endif  // KVC_ARGS_H
