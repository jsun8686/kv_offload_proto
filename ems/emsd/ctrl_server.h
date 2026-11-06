/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * emsd 控制服务：阻塞式 TCP（原型单连接串行）。master 处理 POOL_CREATE/DESTROY/PING 并向
 * worker 下发 SEG_ALLOC；worker 处理 SEG_ALLOC/SEG_FREE/PING；两种角色都处理 LINK_EXCHANGE
 * （框架侧与本节点建立 AIV server 通道）。
 */

#ifndef EMSD_CTRL_SERVER_H
#define EMSD_CTRL_SERVER_H

#include <cstdint>
#include <string>
#include <vector>

#include "common/kv_proto.h"
#include "hcomm_host.h"
#include "pool_mgr.h"

struct EmsdMember {
    uint32_t nodeId;
    std::string ip;       // 该节点 emsd 控制 TCP 地址
    uint16_t ctrlPort;
    uint16_t hcommPort;   // 该节点 AIV 通道监听端口基（每段 +segIndex）
};

struct EmsdConfig {
    bool isMaster = false;
    std::string localIp;              // 本节点对外 IP（endpoint + TCP）
    uint16_t listenPort = 27000;      // 本 emsd 控制 TCP 监听端口
    std::string hcommLib = "libhcomm.so";
    uint32_t poolSizeBytes = 8ULL << 30;  // 本节点可提供的池容量上限
    uint32_t readyTimeoutSec = 60;
    std::vector<EmsdMember> members;  // 静态成员表（master 有效，含自身）
};

class EmsdCtrlServer {
public:
    explicit EmsdCtrlServer(const EmsdConfig &cfg);
    int Run();  // 阻塞运行（listen + accept 循环）

private:
    int HandleConn(int fd);
    int OnPoolCreate(const KvPoolCreateReq &req, int fd);
    int OnPoolDestroy(uint32_t poolId, int fd);
    int OnSegAlloc(const KvSegAllocReq &req, int fd);
    int OnSegAllocImpl(const KvSegAllocReq &req, KvSegAllocResp &resp);  // master 本地/worker 共用
    int OnSegFree(uint32_t poolId, uint32_t segIndex, int fd);
    int OnLinkExchange(int fd);
    int SendResp(int fd, uint16_t cmd, const void *payload, uint32_t payloadLen);
    int WorkerSegAlloc(const EmsdMember &m, const KvSegAllocReq &req, KvSegAllocResp &resp);

    EmsdConfig cfg_;
    EmsdHcomm hcomm_;
    EmsdPoolMgr pool_;
    uint32_t nextPoolId_ = 1;
    bool hcommInited_ = false;
};

#endif  // EMSD_CTRL_SERVER_H
