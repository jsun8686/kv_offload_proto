/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * emsd 控制服务实现。协议见 common/kv_proto.h；打散策略：段式连续切分（余数归段 0），
 * 任一节点失败直接报错（不回滚不重试，原型语义）。
 */

#include "ctrl_server.h"

#include <arpa/inet.h>
#include <cerrno>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {
constexpr size_t KV_MAX_MSG = 64 * 1024;

int SendAll(int fd, const void *buf, size_t len)
{
    const uint8_t *p = static_cast<const uint8_t *>(buf);
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) {
            return -1;
        }
        p += n;
        len -= static_cast<size_t>(n);
    }
    return 0;
}

int RecvAll(int fd, void *buf, size_t len)
{
    uint8_t *p = static_cast<uint8_t *>(buf);
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) {
            return -1;
        }
        p += n;
        len -= static_cast<size_t>(n);
    }
    return 0;
}

int Listen(uint16_t port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 || listen(fd, 4) != 0) {
        printf("[emsd][ERROR] bind/listen failed, port=%u errno=%d\n", port, errno);
        close(fd);
        return -1;
    }
    return fd;
}

int Connect(const std::string &ip, uint16_t port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(ip.c_str());
    addr.sin_port = htons(port);
    if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        printf("[emsd][ERROR] connect worker failed, ip=%s port=%u errno=%d\n", ip.c_str(), port, errno);
        close(fd);
        return -1;
    }
    return fd;
}

int SendMsg(int fd, uint16_t cmd, const void *payload, uint32_t payloadLen)
{
    KvMsgHeader hdr{};
    hdr.magic = KV_PROTO_MAGIC;
    hdr.version = KV_PROTO_VERSION;
    hdr.cmd = cmd;
    hdr.payloadLen = payloadLen;
    if (SendAll(fd, &hdr, sizeof(hdr)) != 0 || (payloadLen > 0 && SendAll(fd, payload, payloadLen) != 0)) {
        return -1;
    }
    return 0;
}

int RecvMsg(int fd, KvMsgHeader &hdr, std::vector<uint8_t> &payload)
{
    if (RecvAll(fd, &hdr, sizeof(hdr)) != 0 || hdr.magic != KV_PROTO_MAGIC || hdr.version != KV_PROTO_VERSION ||
        hdr.payloadLen > KV_MAX_MSG) {
        return -1;
    }
    payload.resize(hdr.payloadLen);
    if (hdr.payloadLen > 0 && RecvAll(fd, payload.data(), hdr.payloadLen) != 0) {
        return -1;
    }
    return 0;
}
}  // namespace

EmsdCtrlServer::EmsdCtrlServer(const EmsdConfig &cfg) : cfg_(cfg) {}

int EmsdCtrlServer::SendResp(int fd, uint16_t cmd, const void *payload, uint32_t payloadLen)
{
    return SendMsg(fd, cmd, payload, payloadLen);
}

int EmsdCtrlServer::Run()
{
    if (hcomm_.Init(cfg_.hcommLib, cfg_.aclLib, cfg_.deviceId, cfg_.rootinfoPath, cfg_.hostLoc, cfg_.hostAddr) != 0) {
        return -1;
    }
    hcommInited_ = true;
    int lfd = Listen(cfg_.listenPort);
    if (lfd < 0) {
        return -1;
    }
    printf("[emsd] ctrl server listening, role=%s ip=%s port=%u members=%zu\n", cfg_.isMaster ? "master" : "worker",
           cfg_.localIp.c_str(), cfg_.listenPort, cfg_.members.size());
    while (true) {
        int cfd = accept(lfd, nullptr, nullptr);
        if (cfd < 0) {
            continue;
        }
        timeval tv{};
        tv.tv_sec = static_cast<long>(cfg_.readyTimeoutSec);
        (void)setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        std::thread(&EmsdCtrlServer::HandleConn, this, cfd).detach();
    }
}

int EmsdCtrlServer::HandleConn(int fd)
{
    while (true) {  // 连接上串行处理多条命令（LINK_EXCHANGE 复用连接）
        KvMsgHeader hdr{};
        std::vector<uint8_t> payload;
        if (RecvMsg(fd, hdr, payload) != 0) {
            return 0;  // 对端关闭/协议错/读超时，断连即可
        }
        int ret = 0;
        std::lock_guard<std::mutex> lk(connMutex_);
        switch (hdr.cmd) {
            case KV_CMD_PING: {
                KvGenericResp resp{KV_OK, 0};
                ret = SendResp(fd, KV_CMD_PONG, &resp, sizeof(resp));
                break;
            }
            case KV_CMD_POOL_CREATE:
                if (!cfg_.isMaster || payload.size() != sizeof(KvPoolCreateReq)) {
                    KvGenericResp resp{cfg_.isMaster ? KV_ERR_PROTO : KV_ERR_STATE, 0};
                    ret = SendResp(fd, hdr.cmd, &resp, sizeof(resp));
                    break;
                }
                ret = OnPoolCreate(*reinterpret_cast<const KvPoolCreateReq *>(payload.data()), fd);
                break;
            case KV_CMD_POOL_DESTROY:
                if (!cfg_.isMaster || payload.size() != sizeof(KvPoolDestroyReq)) {
                    KvGenericResp resp{KV_ERR_PROTO, 0};
                    ret = SendResp(fd, hdr.cmd, &resp, sizeof(resp));
                    break;
                }
                ret = OnPoolDestroy(reinterpret_cast<const KvPoolDestroyReq *>(payload.data())->poolId, fd);
                break;
            case KV_CMD_SEG_ALLOC:
                if (payload.size() != sizeof(KvSegAllocReq)) {
                    KvSegAllocResp resp{};
                    resp.status = KV_ERR_PROTO;
                    ret = SendResp(fd, hdr.cmd, &resp, sizeof(resp));
                    break;
                }
                ret = OnSegAlloc(*reinterpret_cast<const KvSegAllocReq *>(payload.data()), fd);
                break;
            case KV_CMD_SEG_FREE:
                if (payload.size() != sizeof(KvPoolDestroyReq)) {
                    KvGenericResp resp{KV_ERR_PROTO, 0};
                    ret = SendResp(fd, hdr.cmd, &resp, sizeof(resp));
                    break;
                }
                ret = OnSegFree(reinterpret_cast<const KvPoolDestroyReq *>(payload.data())->poolId,
                                reinterpret_cast<const KvPoolDestroyReq *>(payload.data())->reserved, fd);
                break;
            case KV_CMD_LINK_EXCHANGE:
                ret = OnLinkExchange(payload, fd);
                break;
            default:
                KvGenericResp resp{KV_ERR_PROTO, 0};
                ret = SendResp(fd, hdr.cmd, &resp, sizeof(resp));
                break;
        }
        if (ret != 0) {
            return ret;
        }
    }
}

int EmsdCtrlServer::OnPoolCreate(const KvPoolCreateReq &req, int fd)
{
    const uint32_t n = static_cast<uint32_t>(cfg_.members.size());
    if (n == 0 || n > KV_MAX_SEG_NUM || req.sizeBytes == 0 || req.layerRows == 0 || req.rowBytes == 0 ||
        req.stripeRows == 0) {
        KvPoolCreateRespHead head{KV_ERR_PARAM, 0, 0, 0};
        return SendResp(fd, KV_CMD_POOL_CREATE, &head, sizeof(head));
    }
    // 段式连续切分：均分，余数归段 0
    std::vector<uint64_t> segSize(n);
    for (uint32_t i = 0; i < n; i++) {
        segSize[i] = req.sizeBytes / n;
    }
    segSize[0] += req.sizeBytes % n;
    const uint32_t poolId = nextPoolId_++;

    std::vector<KvSegInfoRec> infos(n);
    std::vector<int> status(n, KV_OK);
    // 段 0 = master 本地
    {
        KvSegAllocReq areq{};
        areq.poolId = poolId;
        areq.segIndex = 0;
        areq.segSizeBytes = segSize[0];
        areq.hcommListenPort = cfg_.members[0].hcommPort;
        (void)snprintf(areq.ctrlIp, sizeof(areq.ctrlIp), "%s", cfg_.members[0].ip.c_str());
        (void)snprintf(areq.channelTag, sizeof(areq.channelTag), "kv_pool_%u", poolId);
        KvSegAllocResp aresp{};
        status[0] = OnSegAllocImpl(areq, aresp);
        if (status[0] == KV_OK) {
            infos[0].nodeId = 0;
            infos[0].segBase = aresp.segBase;
            infos[0].segSizeBytes = segSize[0];
            (void)snprintf(infos[0].ctrlIp, sizeof(infos[0].ctrlIp), "%s", areq.ctrlIp);
            infos[0].ctrlPort = cfg_.members[0].ctrlPort;
            infos[0].hcommListenPort = areq.hcommListenPort;
            (void)snprintf(infos[0].channelTag, sizeof(infos[0].channelTag), "%s", areq.channelTag);
            infos[0].memDescLen = aresp.memDescLen;
            (void)memcpy(infos[0].memDesc, aresp.memDesc, aresp.memDescLen);
        }
    }
    // 其余段 = 各 worker
    for (uint32_t i = 1; i < n && status[i - 1] == KV_OK; i++) {
        KvSegAllocReq areq{};
        areq.poolId = poolId;
        areq.segIndex = i;
        areq.segSizeBytes = segSize[i];
        areq.hcommListenPort = static_cast<uint16_t>(cfg_.members[i].hcommPort);
        (void)snprintf(areq.ctrlIp, sizeof(areq.ctrlIp), "%s", cfg_.members[i].ip.c_str());
        (void)snprintf(areq.channelTag, sizeof(areq.channelTag), "kv_pool_%u", poolId);
        KvSegAllocResp aresp{};
        status[i] = WorkerSegAlloc(cfg_.members[i], areq, aresp);  // 失败直接报错（不回滚）
        if (status[i] == KV_OK) {
            infos[i].nodeId = i;
            infos[i].segBase = aresp.segBase;
            infos[i].segSizeBytes = segSize[i];
            (void)snprintf(infos[i].ctrlIp, sizeof(infos[i].ctrlIp), "%s", areq.ctrlIp);
            infos[i].ctrlPort = cfg_.members[i].ctrlPort;
            infos[i].hcommListenPort = areq.hcommListenPort;
            (void)snprintf(infos[i].channelTag, sizeof(infos[i].channelTag), "%s", areq.channelTag);
            infos[i].memDescLen = aresp.memDescLen;
            (void)memcpy(infos[i].memDesc, aresp.memDesc, aresp.memDescLen);
        }
    }
    int32_t finalStatus = KV_OK;
    for (uint32_t i = 0; i < n; i++) {
        if (status[i] != KV_OK) {
            finalStatus = status[i];
            break;
        }
    }
    const uint32_t nodeNum = (finalStatus == KV_OK) ? n : 0;
    std::vector<uint8_t> buf(sizeof(KvPoolCreateRespHead) + sizeof(KvSegInfoRec) * nodeNum);
    KvPoolCreateRespHead head{finalStatus, poolId, nodeNum, 0};
    (void)memcpy(buf.data(), &head, sizeof(head));
    if (nodeNum > 0) {
        (void)memcpy(buf.data() + sizeof(head), infos.data(), sizeof(KvSegInfoRec) * nodeNum);
    }
    printf("[emsd] PoolCreate done, poolId=%u status=%d nodeNum=%u size=%llu\n", poolId, finalStatus, nodeNum,
           (unsigned long long)req.sizeBytes);
    return SendResp(fd, KV_CMD_POOL_CREATE, buf.data(), static_cast<uint32_t>(buf.size()));
}

int EmsdCtrlServer::OnPoolDestroy(uint32_t poolId, int fd)
{
    // 广播 SEG_FREE（master 本地 + worker），逐节点尽力释放
    int status = KV_OK;
    for (const auto &m : cfg_.members) {
        if (m.nodeId == 0) {
            (void)hcomm_.SegRelease(0, true);
            (void)pool_.SegFree(poolId, 0);
            continue;
        }
        int fdw = Connect(m.ip, m.ctrlPort);
        if (fdw < 0) {
            status = KV_ERR_IO;
            continue;
        }
        KvPoolDestroyReq req{poolId, m.nodeId};  // reserved 字段承载 segIndex（= nodeId）
        if (SendMsg(fdw, KV_CMD_SEG_FREE, &req, sizeof(req)) != 0) {
            status = KV_ERR_IO;
        } else {
            KvMsgHeader hdr{};
            std::vector<uint8_t> resp;
            if (RecvMsg(fdw, hdr, resp) != 0 || resp.size() < sizeof(KvGenericResp)) {
                status = KV_ERR_IO;
            } else if (reinterpret_cast<KvGenericResp *>(resp.data())->status != KV_OK) {
                status = reinterpret_cast<KvGenericResp *>(resp.data())->status;
            }
        }
        close(fdw);
    }
    KvGenericResp resp{status, 0};
    return SendResp(fd, KV_CMD_POOL_DESTROY, &resp, sizeof(resp));
}

// SEG_ALLOC 统一实现（master 本地与 worker 复用同一逻辑：分配 → 注册 → 导出）
int EmsdCtrlServer::OnSegAllocImpl(const KvSegAllocReq &req, KvSegAllocResp &resp)
{
    resp.status = KV_OK;
    if (req.segIndex >= KV_MAX_SEG_NUM || req.segSizeBytes == 0) {
        resp.status = KV_ERR_PARAM;
        return resp.status;
    }
    if (req.segSizeBytes > cfg_.poolSizeBytes) {
        printf("[emsd][ERROR] seg size exceeds node capacity, segIndex=%u size=%llu capacity=%llu\n", req.segIndex,
               (unsigned long long)req.segSizeBytes, (unsigned long long)cfg_.poolSizeBytes);
        resp.status = KV_ERR_NO_RESOURCE;
        return resp.status;
    }
    void *base = pool_.SegAlloc(req.segSizeBytes, req.poolId, req.segIndex);
    if (base == nullptr) {
        resp.status = KV_ERR_NO_RESOURCE;
        return resp.status;
    }
    int ret = hcomm_.SegRegister(req.segIndex, req.poolId, base, req.segSizeBytes, req.hcommListenPort, req.channelTag);
    if (ret != KV_OK) {
        (void)pool_.SegFree(req.poolId, req.segIndex);
        resp.status = ret;
        return resp.status;
    }
    const EmsdSeg *seg = hcomm_.GetSeg(req.segIndex);
    resp.segBase = reinterpret_cast<uint64_t>(seg->base);
    resp.memDescLen = static_cast<uint32_t>(seg->memDesc.size());
    (void)memcpy(resp.memDesc, seg->memDesc.data(), seg->memDesc.size());
    return KV_OK;
}

int EmsdCtrlServer::OnSegAlloc(const KvSegAllocReq &req, int fd)
{
    KvSegAllocResp resp{};
    (void)OnSegAllocImpl(req, resp);
    return SendResp(fd, KV_CMD_SEG_ALLOC, &resp, sizeof(resp));
}

int EmsdCtrlServer::OnSegFree(uint32_t poolId, uint32_t segIndex, int fd)
{
    int status = hcomm_.SegRelease(segIndex, true);
    if (pool_.SegFree(poolId, segIndex) != 0 && status == KV_OK) {
        status = KV_ERR_NOT_FOUND;
    }
    KvGenericResp resp{status, 0};
    return SendResp(fd, KV_CMD_SEG_FREE, &resp, sizeof(resp));
}

int EmsdCtrlServer::OnLinkExchange(const std::vector<uint8_t> &payload, int fd)
{
    // 帧格式：KvLinkFrame 作为 LINK_EXCHANGE 命令的 payload（HandleConn 已统一读取，
    // 与 client 侧 send_msg(CMD_LINK_EXCHANGE, encode_link_frame(...)) 组帧对齐）
    if (payload.size() != sizeof(KvLinkFrame)) {
        KvGenericResp resp{KV_ERR_PROTO, 0};
        return SendResp(fd, KV_CMD_LINK_EXCHANGE, &resp, sizeof(resp));
    }
    const KvLinkFrame *frame = reinterpret_cast<const KvLinkFrame *>(payload.data());
    if (frame->descLen == 0 || frame->descLen > KV_EP_DESC_BUF || frame->memDescLen == 0 ||
        frame->memDescLen > KV_MAX_MEM_DESC) {
        KvGenericResp resp{KV_ERR_PROTO, 0};
        return SendResp(fd, KV_CMD_LINK_EXCHANGE, &resp, sizeof(resp));
    }
    // segIndex 由 channelTag 解析（kv_pool_<poolId>：本节点单池单段，取最小已注册段）
    uint32_t segIndex = KV_MAX_SEG_NUM;
    for (uint32_t i = 0; i < KV_MAX_SEG_NUM; i++) {
        if (hcomm_.GetSeg(i) != nullptr) {
            segIndex = i;
            break;
        }
    }
    int status = KV_ERR_STATE;
    if (segIndex < KV_MAX_SEG_NUM) {
        status = hcomm_.LinkServer(segIndex, frame->endpointDesc, frame->descLen, frame->memDesc, frame->memDescLen,
                                   cfg_.readyTimeoutSec);
    }
    KvGenericResp resp{status, 0};
    return SendResp(fd, KV_CMD_LINK_EXCHANGE, &resp, sizeof(resp));
}

int EmsdCtrlServer::WorkerSegAlloc(const EmsdMember &m, const KvSegAllocReq &req, KvSegAllocResp &resp)
{
    int fdw = Connect(m.ip, m.ctrlPort);
    if (fdw < 0) {
        return KV_ERR_IO;
    }
    int rc = KV_ERR_IO;
    if (SendMsg(fdw, KV_CMD_SEG_ALLOC, &req, sizeof(req)) == 0) {
        KvMsgHeader hdr{};
        std::vector<uint8_t> r;
        if (RecvMsg(fdw, hdr, r) == 0 && r.size() == sizeof(KvSegAllocResp)) {
            resp = *reinterpret_cast<KvSegAllocResp *>(r.data());
            rc = resp.status;
        }
    }
    close(fdw);
    return rc;
}
