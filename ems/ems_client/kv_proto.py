# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# EMS 控制协议 Python 编解码（与 common/kv_proto.h 严格对齐，小端定长二进制）。

from dataclasses import dataclass, field
import struct

MAGIC = 0x454D5350  # "EMSP"
VERSION = 1

CMD_POOL_CREATE = 1
CMD_POOL_DESTROY = 2
CMD_PING = 4
CMD_PONG = 5
CMD_LINK_EXCHANGE = 6
CMD_SEG_ALLOC = 16
CMD_SEG_FREE = 17

KV_OK = 0
KV_ERR_PROTO = -1
KV_ERR_PARAM = -2
KV_ERR_NO_RESOURCE = -3
KV_ERR_HCOMM = -4
KV_ERR_NOT_FOUND = -5
KV_ERR_STATE = -6
KV_ERR_IO = -7

MAX_MEM_DESC = 4096
CHANNEL_HANDLE_SIZE = 256
MAX_SEG_NUM = 8

# Header: magic u32, version u16, cmd u16, payloadLen u32, trailerLen u16, reserved u16（16B，pack(1) 与 C 一致）
HDR_FMT = "<IHHIHH"
HDR_SIZE = struct.calcsize(HDR_FMT)  # 16

# EP_DESC_BUF 必须 ≥ sizeof(EndpointDesc)=160（common/hcomm/hcomm_res_defs.h，与 C 侧 KV_EP_DESC_BUF 严格对齐）；
# 128 时 struct 128s 会静默截断 desc 内容（descLen 字段却仍写真实长度），服务端 descLen 校验直接拒绝。
# 该缓冲同时用于 KvSegInfoRec/KvSegAllocResp 尾部的 epDesc 字段（AIV client remoteEndpoint 用）
EP_DESC_BUF = 160

# POOL_CREATE req: sizeBytes u64, layerNum u32, layerRows u32, rowBytes u32, stripeRows u32, reserved u32
POOL_CREATE_REQ_FMT = "<QIIIII"
# resp head: status i32, poolId u32, nodeNum u32, reserved u32
POOL_CREATE_RESP_HEAD_FMT = "<iIII"
# 段记录: nodeId u32, reserved u32, segBase u64, segSize u64, ctrlIp char[64], ctrlPort u16,
#         hcommListenPort u16, channelTag char[64], memDescLen u32, reserved2 u32, memDesc u8[4096],
#         epDescLen u32, reserved3 u32, epDesc u8[160]（该节点 endpoint 描述，AIV client remoteEndpoint 用）
SEG_INFO_FMT = "<IIQQ64sHH64sII{}sII{}s".format(MAX_MEM_DESC, EP_DESC_BUF)
SEG_INFO_SIZE = struct.calcsize(SEG_INFO_FMT)

POOL_DESTROY_REQ_FMT = "<II"
GENERIC_RESP_FMT = "<iI"
# SEG_ALLOC req: poolId u32, segIndex u32, segSize u64, hcommListenPort u16, reserved u16, ctrlIp[64], tag[64]
SEG_ALLOC_REQ_FMT = "<IIQHH64s64s"
SEG_ALLOC_RESP_FMT = "<iiQII{}sII{}s".format(MAX_MEM_DESC, EP_DESC_BUF)

# LINK 帧: descLen u32, desc[160], memDescLen u32, memDesc[4096]
LINK_FRAME_FMT = "<I{}sI{}s".format(EP_DESC_BUF, MAX_MEM_DESC)
LINK_FRAME_SIZE = struct.calcsize(LINK_FRAME_FMT)


@dataclass
class SegmentInfo:
    node_id: int = 0
    seg_base: int = 0
    seg_size: int = 0
    ctrl_ip: str = ""
    ctrl_port: int = 0
    hcomm_listen_port: int = 0
    channel_tag: str = ""
    mem_desc: bytes = b""
    ep_desc: bytes = b""  # 该节点 endpoint 描述（= sizeof(EndpointDesc) 字节，AIV client 建链 remoteEndpoint）


@dataclass
class PoolInfo:
    pool_id: int = 0
    node_num: int = 0
    segments: list = field(default_factory=list)


def encode_header(cmd: int, payload_len: int) -> bytes:
    return struct.pack(HDR_FMT, MAGIC, VERSION, cmd, payload_len, 0, 0)


def recv_all(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed, expect {} got {} bytes".format(n, len(buf)))
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock, cmd: int, payload: bytes) -> None:
    sock.sendall(encode_header(cmd, len(payload)) + payload)


def recv_msg(sock):
    hdr = recv_all(sock, HDR_SIZE)
    magic, ver, cmd, plen, _, _ = struct.unpack(HDR_FMT, hdr)
    if magic != MAGIC or ver != VERSION:
        raise ConnectionError("bad header magic=0x{:x} ver={}".format(magic, ver))
    payload = recv_all(sock, plen) if plen > 0 else b""
    return cmd, payload


def encode_pool_create_req(size_bytes: int, layer_num: int, layer_rows: int, row_bytes: int, stripe_rows: int) -> bytes:
    return struct.pack(POOL_CREATE_REQ_FMT, size_bytes, layer_num, layer_rows, row_bytes, stripe_rows, 0)


def decode_pool_create_resp(payload: bytes) -> tuple:
    head = struct.unpack_from(POOL_CREATE_RESP_HEAD_FMT, payload, 0)
    status, pool_id, node_num, _ = head
    pool = PoolInfo(pool_id=pool_id, node_num=node_num)
    off = struct.calcsize(POOL_CREATE_RESP_HEAD_FMT)
    for _ in range(node_num):
        (node_id, _r, seg_base, seg_size, ctrl_ip, ctrl_port, hcomm_port, tag, desc_len, _r2, desc,
         ep_len, _r3, ep_desc) = struct.unpack_from(SEG_INFO_FMT, payload, off)
        pool.segments.append(
            SegmentInfo(
                node_id=node_id,
                seg_base=seg_base,
                seg_size=seg_size,
                ctrl_ip=ctrl_ip.split(b"\x00")[0].decode("utf-8", "replace"),
                ctrl_port=ctrl_port,
                hcomm_listen_port=hcomm_port,
                channel_tag=tag.split(b"\x00")[0].decode("utf-8", "replace"),
                mem_desc=desc[:desc_len],
                ep_desc=ep_desc[:ep_len],
            )
        )
        off += SEG_INFO_SIZE
    return status, pool


def encode_link_frame(endpoint_desc: bytes, mem_desc: bytes) -> bytes:
    # struct 's' 对超长 bytes 静默截断——显式拒绝，避免线上出现 descLen 与内容不一致的半帧
    if len(endpoint_desc) > EP_DESC_BUF:
        raise ValueError("endpoint_desc too long: {} > {}".format(len(endpoint_desc), EP_DESC_BUF))
    if len(mem_desc) > MAX_MEM_DESC:
        raise ValueError("mem_desc too long: {} > {}".format(len(mem_desc), MAX_MEM_DESC))
    return struct.pack(LINK_FRAME_FMT, len(endpoint_desc), endpoint_desc, len(mem_desc), mem_desc)


def encode_pool_destroy_req(pool_id: int) -> bytes:
    return struct.pack(POOL_DESTROY_REQ_FMT, pool_id, 0)


def decode_generic_resp(payload: bytes) -> int:
    return struct.unpack(GENERIC_RESP_FMT, payload)[0]
