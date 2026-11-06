/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * 行归属映射（层维度均衡）：C 侧参考实现（语义与 xds/kv_layout.py 严格一致，单测对拍）。
 *
 *   node(row)     = (row / S) % N          // 层无关：任意层的行均匀散到 N 段
 *   segRow(row)   = (row / S) / N          // 段内层内行号（紧凑无空洞）
 *   addr(l, row)  = segBase + (l * perLayerSegRows + segRow(row)) * rowBytes
 *                   // 大块内布局 [层0区][层1区]...[层L-1区]，每层区 perLayerSegRows 行
 */

#ifndef KV_LAYOUT_H
#define KV_LAYOUT_H

#include <cstdint>

struct KvLayout {
    uint32_t nodeNum;           // N：资源节点（段）数
    uint32_t layerNum;          // L：层数
    uint32_t layerRows;         // 每层逻辑行数
    uint32_t perLayerSegRows;   // = ceil(layerRows / N)
    uint32_t rowBytes;          // 128B 对齐后行字节数
    uint32_t stripeRows;        // S：条带行数（默认 1）
};

static inline uint32_t KvLayoutPerLayerSegRows(uint32_t layerRows, uint32_t nodeNum)
{
    return (layerRows + nodeNum - 1) / nodeNum;
}

static inline uint32_t KvLayoutNode(uint32_t row, const KvLayout *lay)
{
    return (lay->stripeRows == 0) ? 0 : ((row / lay->stripeRows) % lay->nodeNum);
}

static inline uint32_t KvLayoutSegRow(uint32_t row, const KvLayout *lay)
{
    return (lay->stripeRows == 0) ? row : ((row / lay->stripeRows) / lay->nodeNum);
}

// 层 l 中逻辑行 row 在其归属段内的绝对字节偏移
static inline uint64_t KvLayoutSegOffset(uint32_t layer, uint32_t row, const KvLayout *lay)
{
    const uint64_t layerBase = static_cast<uint64_t>(layer) * lay->perLayerSegRows;
    return (layerBase + KvLayoutSegRow(row, lay)) * lay->rowBytes;
}

// 校验布局参数（mem池创建时使用；返回 0=ok，负值=KV_ERR_PARAM 语义）
static inline int KvLayoutValidate(const KvLayout *lay)
{
    if (lay == nullptr || lay->nodeNum == 0 || lay->nodeNum > 64 || lay->layerNum == 0 ||
        lay->layerRows == 0 || lay->rowBytes == 0 || (lay->rowBytes & 0x7F) != 0 ||
        lay->stripeRows == 0 || lay->perLayerSegRows != KvLayoutPerLayerSegRows(lay->layerRows, lay->nodeNum)) {
        return -1;
    }
    return 0;
}

#endif  // KV_LAYOUT_H
