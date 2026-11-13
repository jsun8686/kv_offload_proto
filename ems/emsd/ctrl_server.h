/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * emsd 控制服务：阻塞式 TCP，每连接一个线程（accept 循环不再被长连接阻塞——框架侧
 * pool_create 的连接会一直保持到 session 结束，期间 LINK_EXCHANGE 等新连接必须可被服务）；
 * 但所有命令处理收拢到单一专用线程（DispatchLoop）执行——libhcomm/HCCP 的上下文
 * （RaInit/RaCtxInit/设备上下文）是线程私有的，工作线程上直接调用会出现"socket 建链
 * 发出了但异步握手无法推进"的半失效（联调实证：ListenStart/ConnectAsync 日志在从未
 * RaInit 的连接线程上打出，channel 60s 不 READY）。
 * master 处理 POOL_CREATE/DESTROY/PING 并向 worker 下发 SEG_ALLOC；worker 处理
 * SEG_ALLOC/SEG_FREE/PING；两种角色都处理 LINK_EXCHANGE（框架侧与本节点建立 AIV server 通道）。
 */

#ifndef EMSD_CTRL_SERVER_H
#define EMSD_CTRL_SERVER_H

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
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
    std::string localIp;              // 本节点对外 IP（控制 TCP/成员表用；endpoint 地址走 EID/IPv6）
    uint16_t listenPort = 27000;      // 本 emsd 控制 TCP 监听端口
    int32_t deviceId = 0;             // NPU 设备号（device 模式：endpoint 需 SetDevice 上下文）
    std::string rootinfoPath = "/etc/hccl_rootinfo.json";  // device 模式：EID 按 device_id==phyId 匹配
    bool hostLoc = false;             // loc 模式：false=device（当前包，卡到卡）；true=host（待跨模式包）
    std::string hostAddr;             // host 模式：endpoint IPv6（ipourma host 地址，app 手填）
    std::string aclLib = "libascendcl.so";
    std::string hcommLib = "libhcomm.so";
    uint64_t poolSizeBytes = 8ULL << 30;  // 本节点可提供的池容量上限（u64：支持 >4GB 池）
    uint32_t readyTimeoutSec = 60;
    std::vector<EmsdMember> members;  // 静态成员表（master 有效，含自身）
};

// 控制连接空闲回收超时（秒）：须覆盖 client 完整会话周期（pool_create → N 节点建链握手
// → 数据面 → pool_destroy）。不可复用 readyTimeoutSec（60s，channel READY 诊断用）——
// 双节点 2×35s 握手即超 60s，会话连接读者被误杀且 fd 泄漏致 client 侧 recv 永久等待。
constexpr uint32_t CTRL_CONN_IDLE_TIMEOUT_SEC = 3600;

class EmsdCtrlServer {
public:
    explicit EmsdCtrlServer(const EmsdConfig &cfg);
    int Run();  // 阻塞运行（listen + accept 循环）

private:
    int HandleConn(int fd);
    int HandleCmd(uint16_t cmd, const std::vector<uint8_t> &payload, int fd);
    int Dispatch(uint16_t cmd, const std::vector<uint8_t> &payload, int fd);  // 投放到专用线程并等待
    void DispatchLoop();                                                      // 专用线程：所有 hcomm 调用在此执行
    int OnPoolCreate(const KvPoolCreateReq &req, int fd);
    int OnPoolDestroy(uint32_t poolId, int fd);
    int OnSegAlloc(const KvSegAllocReq &req, int fd);
    int OnSegAllocImpl(const KvSegAllocReq &req, KvSegAllocResp &resp);  // master 本地/worker 共用
    int OnSegFree(uint32_t poolId, uint32_t segIndex, int fd);
    int OnLinkExchange(const std::vector<uint8_t> &payload, int fd);
    int SendResp(int fd, uint16_t cmd, const void *payload, uint32_t payloadLen);
    int WorkerSegAlloc(const EmsdMember &m, const KvSegAllocReq &req, KvSegAllocResp &resp);

    EmsdConfig cfg_;
    EmsdHcomm hcomm_;
    EmsdPoolMgr pool_;
    // 命令处理专用线程队列（单消费者天然串行；hcomm/HCCP 仅在该线程上被调用）
    std::mutex qMutex_;
    std::condition_variable qCv_;
    std::queue<std::packaged_task<int()>> taskQ_;
    uint32_t nextPoolId_ = 1;
    bool hcommInited_ = false;
};

#endif  // EMSD_CTRL_SERVER_H
