/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * emsd 大块内存管理：段内存取自 NPU HBM（aclrtMalloc，PoC 实证 AIV 通道 mem key
 * 交换要求 DVA 域——裸 mmap host 页无 DVA，握手无法完成），按段整块申请/释放，段表登记。
 */

#ifndef EMSD_POOL_MGR_H
#define EMSD_POOL_MGR_H

#include <cstdint>
#include <map>
#include <string>

class EmsdPoolMgr {
public:
    // dlopen acl 库并解析 aclrtMalloc/aclrtFree（段内存走 HBM；失败返回 -1，进程退出）
    int Init(const std::string &aclLibPath);
    // 段整块申请（2MB 对齐；失败返回 nullptr）
    void *SegAlloc(uint64_t sizeBytes, uint32_t poolId, uint32_t segIndex);
    // 释放段内存；unregister 回调由调用方（ctrl_server）先行处理 hcomm 侧
    int SegFree(uint32_t poolId, uint32_t segIndex);
    // 池销毁：释放该 pool 全部段
    int PoolFree(uint32_t poolId);
    void *Query(uint32_t poolId, uint32_t segIndex) const;
    uint64_t TotalAllocated() const { return totalAllocated_; }

private:
    struct SegKey {
        uint32_t poolId;
        uint32_t segIndex;
        bool operator<(const SegKey &o) const
        {
            return poolId != o.poolId ? poolId < o.poolId : segIndex < o.segIndex;
        }
    };
    struct SegRec {
        void *base;
        uint64_t size;
    };
    std::map<SegKey, SegRec> segs_;
    uint64_t totalAllocated_ = 0;
    void *aclLibHandle_ = nullptr;
    int32_t (*aclrtMallocFn_)(void **, uint64_t, int32_t) = nullptr;  // policy 0 = HUGE_FIRST（PoC 同款）
    int32_t (*aclrtFreeFn_)(void *) = nullptr;
};

#endif  // EMSD_POOL_MGR_H
