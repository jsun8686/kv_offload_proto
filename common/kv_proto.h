/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * EMS 控制面 TCP 协议定义（C 侧，与 ems/ems_client/kv_proto.py 严格对齐，小端，定长二进制）。
 * 报文 = Header(16B) + payload（cmd 对应定长结构体，见下）。原型零第三方依赖。
 */

#ifndef KV_PROTO_H
#define KV_PROTO_H

#include <cstdint>

constexpr uint32_t KV_PROTO_MAGIC = 0x454D5350u;  // "EMSP"
constexpr uint16_t KV_PROTO_VERSION = 1;

// cmd 值（client<->master 与 master<->worker 共用一张表）
enum KvCmd : uint16_t {
    KV_CMD_POOL_CREATE = 1,    // client -> master
    KV_CMD_POOL_DESTROY = 2,   // client -> master
    KV_CMD_PING = 4,           // 双向
    KV_CMD_PONG = 5,
    KV_CMD_LINK_EXCHANGE = 6,  // client -> 某节点 emsd：交换 EndpointDesc/memDesc，emsd 侧建 server 通道
    KV_CMD_SEG_ALLOC = 16,     // master -> worker
    KV_CMD_SEG_FREE = 17,      // master -> worker
};

// 响应码
enum KvStatus : int32_t {
    KV_OK = 0,
    KV_ERR_PROTO = -1,        // 协议/编解码错误
    KV_ERR_PARAM = -2,        // 参数非法
    KV_ERR_NO_RESOURCE = -3,  // 资源不足（大块 OOM / 无可用 worker）
    KV_ERR_HCOMM = -4,        // hcomm 建链/MemReg 失败
    KV_ERR_NOT_FOUND = -5,    // poolId/段不存在
    KV_ERR_STATE = -6,        // 状态错误（如重复建链）
    KV_ERR_IO = -7,           // TCP IO 错误
};

constexpr uint32_t KV_MAX_MEM_DESC = 4096;
constexpr uint32_t KV_CHANNEL_HANDLE_SIZE = 256;  // ChannelEntity ABI（opaque，透传 AIV 内核）
constexpr uint32_t KV_MAX_SEG_NUM = 8;
constexpr size_t KV_EP_DESC_BUF = 128;            // 与 KvLinkFrame.endpointDesc 一致

#pragma pack(push, 1)
struct KvMsgHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t cmd;
    uint32_t payloadLen;  // payload 字节数（不含 Header）
    uint16_t trailerLen;  // 保留（定长协议下恒 0）
    uint16_t reserved;
};

// POOL_CREATE 请求（client -> master）
struct KvPoolCreateReq {
    uint64_t sizeBytes;   // 池总字节
    uint32_t layerNum;    // L
    uint32_t layerRows;   // 每层逻辑行数
    uint32_t rowBytes;    // 128B 对齐行字节
    uint32_t stripeRows;  // S（默认 1）
    uint32_t reserved;
};

// 每段信息（POOL_CREATE 响应体内，nodeNum 个连续排列）
struct KvSegInfoRec {
    uint32_t nodeId;            // 静态成员表下标（= segIndex）
    uint32_t reserved;
    uint64_t segBase;           // 远端大块基地址
    uint64_t segSizeBytes;
    char ctrlIp[64];            // 该节点 emsd 控制地址（LINK_EXCHANGE 用）
    uint16_t ctrlPort;
    uint16_t hcommListenPort;   // 该节点 AIV 通道 server 监听端口
    char channelTag[64];
    uint32_t memDescLen;
    uint32_t reserved2;
    uint8_t memDesc[KV_MAX_MEM_DESC];
};

// POOL_CREATE 响应（master -> client）：KvGenericResp 头 + nodeNum 个 KvSegInfoRec
struct KvPoolCreateRespHead {
    int32_t status;
    uint32_t poolId;
    uint32_t nodeNum;
    uint32_t reserved;
};

// POOL_DESTROY 请求 / PING（payload 可为空）
struct KvPoolDestroyReq {
    uint32_t poolId;
    uint32_t reserved;
};

// 通用响应
struct KvGenericResp {
    int32_t status;
    uint32_t reserved;
};

// SEG_ALLOC 请求（master -> worker）
struct KvSegAllocReq {
    uint32_t poolId;
    uint32_t segIndex;          // = nodeId
    uint64_t segSizeBytes;
    uint16_t hcommListenPort;   // 该节点 AIV server 监听端口
    uint16_t reserved;
    char ctrlIp[64];            // worker 自身控制地址（回填给 client 做 LINK_EXCHANGE）
    char channelTag[64];
};

// SEG_ALLOC 响应（worker -> master）
struct KvSegAllocResp {
    int32_t status;
    uint32_t reserved;
    uint64_t segBase;
    uint32_t memDescLen;
    uint32_t reserved2;
    uint8_t memDesc[KV_MAX_MEM_DESC];
};

// LINK_EXCHANGE 帧：作为 KV_CMD_LINK_EXCHANGE 消息的 payload 一次发送
// （报文 = Header + KvLinkFrame，与 client 侧 send_msg 组帧严格一致，非独立二次报文）
// EndpointDesc ABI ≤ 60B 内容 + 类型域；128B 覆盖未来扩展
struct KvLinkFrame {
    uint32_t descLen;
    uint8_t endpointDesc[128];
    uint32_t memDescLen;
    uint8_t memDesc[KV_MAX_MEM_DESC];
};
#pragma pack(pop)

#endif  // KV_PROTO_H
