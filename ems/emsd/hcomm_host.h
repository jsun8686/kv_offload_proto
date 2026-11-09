/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * emsd host 侧 HCOMM 建链封装：dlsym libhcomm（noDevice 基线构建），Endpoint(HOST, UB_CTP) +
 * MemReg(host 大块) + Export；server 通道在 LINK_EXCHANGE 时按段创建（AIV 引擎）。
 */

#ifndef EMSD_HCOMM_HOST_H
#define EMSD_HCOMM_HOST_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "common/hcomm/hcomm_res.h"
#include "common/kv_proto.h"
#include "common/ems_eid.h"

struct EmsdDynApi {
    HcommResult (*HcommEndpointCreate)(const EndpointDesc *, EndpointHandle *);
    HcommResult (*HcommEndpointDestroy)(EndpointHandle);
    HcommResult (*HcommMemReg)(EndpointHandle, const char *, const CommMem *, HcommMemHandle *);
    HcommResult (*HcommMemUnreg)(EndpointHandle, HcommMemHandle);
    HcommResult (*HcommMemExport)(EndpointHandle, HcommMemHandle, void **, uint32_t *);
    HcommResult (*HcommMemImport)(EndpointHandle, const void *, uint32_t, CommMem *);
    HcommResult (*HcommMemUnimport)(EndpointHandle, const void *, uint32_t);
    HcommResult (*HcommChannelCreate)(EndpointHandle, CommEngine, HcommChannelDesc *, uint32_t, ChannelHandle *);
    HcommResult (*HcommChannelGetStatus)(const ChannelHandle *, uint32_t, int32_t *);
    HcommResult (*HcommChannelDestroy)(const ChannelHandle *, uint32_t);

    int Load(const std::string &libPath);
};

// 每段状态：大块内存（pool_mgr 分配）注册到本 endpoint 后的句柄与导出描述
struct EmsdSeg {
    bool used = false;
    uint32_t poolId = 0;
    void *base = nullptr;
    uint64_t size = 0;
    HcommMemHandle memHandle = nullptr;
    std::vector<uint8_t> memDesc;
    uint16_t hcommListenPort = 0;
    std::string channelTag;
    bool linked = false;      // LINK_EXCHANGE 完成（server 通道已建）
    ChannelHandle channel = 0;
};

class EmsdHcomm {
public:
    // 初始化链（对齐 aiv_urma_poc，EID 全自动）：
    //   dlopen libascendcl(RTLD_GLOBAL) → aclrtSetDevice(deviceId)（host endpoint 依赖进程 device 上下文）
    //   → aclrtGetPhyDevIdByLogicDevId 取 phyId → rootinfo 按 device_id==phyId 自动匹配 EID
    //   → dlopen libhcomm → EndpointCreate(HOST, UB_CTP, COMM_ADDR_TYPE_EID)
    int Init(const std::string &hcommLib, const std::string &aclLib, int32_t deviceId,
             const std::string &rootinfoPath);
    void Close();

    // 大块注册+导出（base 由 pool_mgr 分配）；memDesc 供 LINK_EXCHANGE 前经 master 转给框架侧
    int SegRegister(uint32_t segIndex, uint32_t poolId, void *base, uint64_t size, uint16_t hcommListenPort,
                    const std::string &channelTag);
    // LINK_EXCHANGE：导入对端（框架/NPU 侧）desc/memDesc，创建本段 AIV server 通道并等待 READY
    int LinkServer(uint32_t segIndex, const uint8_t *peerDesc, uint32_t peerDescLen, const uint8_t *peerMemDesc,
                   uint32_t peerMemDescLen, uint32_t readyTimeoutSec);
    int SegRelease(uint32_t segIndex, bool destroyChannel);

    const EmsdSeg *GetSeg(uint32_t segIndex) const;
    const EndpointDesc &LocalDesc() const { return localDesc_; }
    EmsdDynApi &Api() { return api_; }

private:
    EmsdDynApi api_{};
    void *dlHandle_ = nullptr;
    void *aclLibHandle_ = nullptr;
    std::string libPath_;
    std::string aclLib_;
    EndpointHandle endpoint_ = nullptr;
    EndpointDesc localDesc_{};
    EmsdSeg segs_[KV_MAX_SEG_NUM]{};
};

#endif  // EMSD_HCOMM_HOST_H
