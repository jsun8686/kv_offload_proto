/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * emsd 大块内存管理实现：aclrtMalloc 取 NPU HBM（2MB 对齐）。
 * 注意：aclrtMalloc 依赖当前线程 device 上下文，必须在 DispatchLoop 专用线程
 * （已 BindDeviceCtx）上调用，即 SegAlloc 只能经由命令处理路径触达。
 */

#include "pool_mgr.h"

#include <cstdio>
#include <dlfcn.h>

namespace {
constexpr uint64_t SEG_ALIGN = 2ULL * 1024 * 1024;  // 2MB 对齐

uint64_t AlignUp(uint64_t v, uint64_t a)
{
    return (v + a - 1) / a * a;
}
}  // namespace

int EmsdPoolMgr::Init(const std::string &aclLibPath)
{
    aclLibHandle_ = dlopen(aclLibPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (aclLibHandle_ == nullptr) {
        printf("[emsd][ERROR] pool dlopen %s failed: %s\n", aclLibPath.c_str(), dlerror());
        return -1;
    }
    aclrtMallocFn_ = reinterpret_cast<decltype(aclrtMallocFn_)>(dlsym(aclLibHandle_, "aclrtMalloc"));
    aclrtFreeFn_ = reinterpret_cast<decltype(aclrtFreeFn_)>(dlsym(aclLibHandle_, "aclrtFree"));
    if (aclrtMallocFn_ == nullptr || aclrtFreeFn_ == nullptr) {
        printf("[emsd][ERROR] pool dlsym aclrtMalloc/aclrtFree failed: %s\n", dlerror());
        return -1;
    }
    return 0;
}

void *EmsdPoolMgr::SegAlloc(uint64_t sizeBytes, uint32_t poolId, uint32_t segIndex)
{
    if (sizeBytes == 0 || aclrtMallocFn_ == nullptr) {
        return nullptr;
    }
    const uint64_t aligned = AlignUp(sizeBytes, SEG_ALIGN);
    void *base = nullptr;
    if (aclrtMallocFn_(&base, aligned, 0) != 0 || base == nullptr) {
        printf("[emsd][ERROR] SegAlloc aclrtMalloc failed, poolId=%u segIndex=%u size=%llu aligned=%llu\n", poolId,
               segIndex, (unsigned long long)sizeBytes, (unsigned long long)aligned);
        return nullptr;
    }
    SegKey key{poolId, segIndex};
    segs_[key] = SegRec{base, aligned};
    totalAllocated_ += aligned;
    printf("[emsd] SegAlloc ok, poolId=%u segIndex=%u base=%p size=%llu mem=hbm\n", poolId, segIndex, base,
           (unsigned long long)aligned);
    return base;
}

int EmsdPoolMgr::SegFree(uint32_t poolId, uint32_t segIndex)
{
    auto it = segs_.find(SegKey{poolId, segIndex});
    if (it == segs_.end()) {
        return -1;
    }
    if (aclrtFreeFn_ != nullptr) {
        (void)aclrtFreeFn_(it->second.base);
    }
    totalAllocated_ -= it->second.size;
    segs_.erase(it);
    return 0;
}

int EmsdPoolMgr::PoolFree(uint32_t poolId)
{
    int failed = 0;
    for (auto it = segs_.begin(); it != segs_.end();) {
        if (it->first.poolId == poolId) {
            if (aclrtFreeFn_ != nullptr && aclrtFreeFn_(it->second.base) != 0) {
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
