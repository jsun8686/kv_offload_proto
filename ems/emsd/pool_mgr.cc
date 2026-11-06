/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * emsd 大块内存管理实现：mmap 匿名映射（MAP_NORESERVE），2MB 对齐。
 */

#include "pool_mgr.h"

#include <cerrno>
#include <cstdio>
#include <sys/mman.h>
#include <unistd.h>

namespace {
constexpr uint64_t SEG_ALIGN = 2ULL * 1024 * 1024;  // 2MB 对齐

uint64_t AlignUp(uint64_t v, uint64_t a)
{
    return (v + a - 1) / a * a;
}
}  // namespace

void *EmsdPoolMgr::SegAlloc(uint64_t sizeBytes, uint32_t poolId, uint32_t segIndex)
{
    if (sizeBytes == 0) {
        return nullptr;
    }
    const uint64_t aligned = AlignUp(sizeBytes, SEG_ALIGN);
    void *base = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) {
        printf("[emsd][ERROR] SegAlloc mmap failed, poolId=%u segIndex=%u size=%llu aligned=%llu errno=%d\n", poolId,
               segIndex, (unsigned long long)sizeBytes, (unsigned long long)aligned, errno);
        return nullptr;
    }
    SegKey key{poolId, segIndex};
    segs_[key] = SegRec{base, aligned};
    totalAllocated_ += aligned;
    printf("[emsd] SegAlloc ok, poolId=%u segIndex=%u base=%p size=%llu\n", poolId, segIndex, base,
           (unsigned long long)aligned);
    return base;
}

int EmsdPoolMgr::SegFree(uint32_t poolId, uint32_t segIndex)
{
    auto it = segs_.find(SegKey{poolId, segIndex});
    if (it == segs_.end()) {
        return -1;
    }
    munmap(it->second.base, it->second.size);
    totalAllocated_ -= it->second.size;
    segs_.erase(it);
    return 0;
}

int EmsdPoolMgr::PoolFree(uint32_t poolId)
{
    int failed = 0;
    for (auto it = segs_.begin(); it != segs_.end();) {
        if (it->first.poolId == poolId) {
            if (munmap(it->second.base, it->second.size) != 0) {
                failed++;
            }
            totalAllocated_ -= it->second.size;
            it = segs_.erase(it);
        } else {
            ++it;
        }
    }
    return failed;
}

void *EmsdPoolMgr::Query(uint32_t poolId, uint32_t segIndex) const
{
    auto it = segs_.find(SegKey{poolId, segIndex});
    return (it == segs_.end()) ? nullptr : it->second.base;
}
