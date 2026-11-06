/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * hisp 内核公共部分（移植自 sglang-0730 hisparse_lru_kernel.cpp，哈希-only 裁剪）：
 *   - token→slot 开放寻址哈希（Knuth 乘法散列 + 线性探测 + backward-shift 删除，无 tombstone）
 *   - visited 位图化（原 uint8/槽 → packed bit，HBM/UB 均 1/8；访问经读-改-写单字节）
 * 保留原正确性约束：cache line 64B 对齐族 / hand 行距 16×int32 / padded≤65534（uint16 槽号）。
 */

#ifndef HISP_KERNEL_COMMON_H
#define HISP_KERNEL_COMMON_H

#include "kernel_operator.h"

using namespace AscendC;

namespace hisp {

constexpr uint16_t kHtEmpty = 0;
// sieve_hand 行距 16×int32（64B）：不同 block 的标量 GM store 不得共享 cache line
constexpr int32_t kHandStride = 16;

__aicore__ inline uint32_t ht_hash(int32_t token)
{
    return (static_cast<uint32_t>(token) * 2654435761u) >> 16;
}

// 返回持有 token 的槽号，不存在返回 -1（哈希-only：全表开放寻址）
__aicore__ inline int32_t ht_find(LocalTensor<uint16_t> &ht, LocalTensor<int32_t> &tok, int32_t token,
                                  int32_t htMask)
{
    int32_t idx = static_cast<int32_t>(ht_hash(token)) & htMask;
    while (true) {
        uint16_t e = ht.GetValue(idx);
        if (e == kHtEmpty) {
            return -1;
        }
        if (tok.GetValue(static_cast<int32_t>(e) - 1) == token) {
            return static_cast<int32_t>(e) - 1;
        }
        idx = (idx + 1) & htMask;
    }
}

// 插入 token→slot；重复 token 保留首个（最低）槽位（first-hit 语义）
__aicore__ inline void ht_insert(LocalTensor<uint16_t> &ht, LocalTensor<int32_t> &tok, int32_t token,
                                 int32_t slot, int32_t htMask)
{
    int32_t idx = static_cast<int32_t>(ht_hash(token)) & htMask;
    while (true) {
        uint16_t e = ht.GetValue(idx);
        if (e == kHtEmpty) {
            ht.SetValue(idx, static_cast<uint16_t>(slot + 1));
            return;
        }
        if (tok.GetValue(static_cast<int32_t>(e) - 1) == token) {
            return;
        }
        idx = (idx + 1) & htMask;
    }
}

// backward-shift deletion（Knuth 6.4R）：无 tombstone，探测链保持完整
__aicore__ inline void ub_ht_erase_bs(LocalTensor<uint16_t> &ht, LocalTensor<int32_t> &tok, int32_t token,
                                      int32_t htMask)
{
    int32_t idx = static_cast<int32_t>(ht_hash(token)) & htMask;
    while (true) {
        uint16_t e = ht.GetValue(idx);
        if (e == kHtEmpty) {
            return;  // 防御：token 不在表中
        }
        if (tok.GetValue(static_cast<int32_t>(e) - 1) == token) {
            break;
        }
        idx = (idx + 1) & htMask;
    }
    int32_t j = idx;
    int32_t i = idx;
    while (true) {
        i = (i + 1) & htMask;
        uint16_t e = ht.GetValue(i);
        if (e == kHtEmpty) {
            break;
        }
        int32_t home = static_cast<int32_t>(ht_hash(tok.GetValue(static_cast<int32_t>(e) - 1))) & htMask;
        bool move = (j < i) ? (home <= j || home > i) : (home <= j && home > i);
        if (move) {
            ht.SetValue(j, e);
            j = i;
        }
    }
    ht.SetValue(j, kHtEmpty);
}

// init 内核用：探测同 ht_insert，重复比较读 GM token 行
__aicore__ inline void ub_ht_insert_gm_tok(LocalTensor<uint16_t> &ht, __gm__ int32_t *tok, int32_t token,
                                           int32_t slot, int32_t htMask)
{
    int32_t idx = static_cast<int32_t>(ht_hash(token)) & htMask;
    while (true) {
        uint16_t e = ht.GetValue(idx);
        if (e == kHtEmpty) {
            ht.SetValue(idx, static_cast<uint16_t>(slot + 1));
            return;
        }
        if (tok[static_cast<int32_t>(e) - 1] == token) {
            return;
        }
        idx = (idx + 1) & htMask;
    }
}

// ── visited 位图（packed bit；visStrideB 为每池行字节数，64B 对齐由 host 保证） ──
__aicore__ inline int32_t vis_get(LocalTensor<uint8_t> &vis, int32_t slot)
{
    return (vis.GetValue(slot >> 3) >> (slot & 7)) & 1;
}

__aicore__ inline void vis_set(LocalTensor<uint8_t> &vis, int32_t slot)
{
    int32_t byte = slot >> 3;
    vis.SetValue(byte, static_cast<uint8_t>(vis.GetValue(byte) | (1 << (slot & 7))));
}

__aicore__ inline void vis_clear(LocalTensor<uint8_t> &vis, int32_t slot)
{
    int32_t byte = slot >> 3;
    vis.SetValue(byte, static_cast<uint8_t>(vis.GetValue(byte) & ~(1 << (slot & 7))));
}

}  // namespace hisp

#endif  // HISP_KERNEL_COMMON_H
