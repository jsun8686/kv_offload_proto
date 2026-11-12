# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# 控制面互操作集成测试：EmsClient 对照按 common/kv_proto.h 规格实现的 mock emsd 跑
# ping -> pool_create -> link_exchange -> pool_destroy 全流程。
# 锁定线上契约：每条命令 = 一条消息（Header 16B + payload），
# 特别地 LINK_EXCHANGE 的 KvLinkFrame 必须作为该命令的 payload 一次送达
# （组帧分歧会导致对端在 recv 上死等：client 等响应、server 等第二条消息，最终双双超时）。

import os
import socket
import struct
import sys
import threading
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "ems")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ems_client import kv_proto as kp
from ems_client.client import EmsClient


def _recv_all(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("mock server: peer closed")
        buf.extend(chunk)
    return bytes(buf)


class MockEmsd(threading.Thread):
    """按 common/kv_proto.h 规格实现的最小 emsd：单连接串行处理，逐条消息应答。

    记录收到的每条 (cmd, payloadLen, payload)，供测试断言组帧是否符合规格。
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.records = []          # [(cmd, payload_len, payload)]
        self.link_frame_ok = False
        self._lock = threading.Lock()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(4)
        self.port = self._srv.getsockname()[1]
        self._done = threading.Event()

    def stop(self):
        self._done.set()
        try:
            self._srv.close()
        except OSError:
            pass

    def run(self):
        self._srv.settimeout(0.5)
        while not self._done.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            # 与真实 emsd 同构：每连接一个处理线程，长连接保持期间新连接仍可被服务
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        with conn:
            conn.settimeout(5)
            while True:
                try:
                    hdr = _recv_all(conn, kp.HDR_SIZE)
                except (ConnectionError, socket.timeout, OSError):
                    break
                magic, ver, cmd, plen, _, _ = struct.unpack(kp.HDR_FMT, hdr)
                if magic != kp.MAGIC or ver != kp.VERSION:
                    break
                payload = _recv_all(conn, plen) if plen > 0 else b""
                with self._lock:
                    self.records.append((cmd, plen, payload))
                if not self._reply(conn, cmd, plen, payload):
                    break

    def _reply(self, conn, cmd, plen, payload):
        def send(c, body):
            conn.sendall(kp.encode_header(c, len(body)) + body)

        if cmd == kp.CMD_PING:
            send(kp.CMD_PONG, struct.pack(kp.GENERIC_RESP_FMT, kp.KV_OK, 0))
            return True
        if cmd == kp.CMD_POOL_CREATE:
            # 单段，ctrl 地址指向 mock 自身（LINK_EXCHANGE 会连回来）
            seg = kp.SegmentInfo(node_id=0, seg_base=0x7F0000000000, seg_size=1 << 20,
                                 ctrl_ip="127.0.0.1", ctrl_port=self.port,
                                 hcomm_listen_port=60001, channel_tag="kv_pool_1",
                                 mem_desc=b"\xCD" * 128, ep_desc=b"\xEE" * 160)  # epDesc = sizeof(EndpointDesc)
            head = struct.pack(kp.POOL_CREATE_RESP_HEAD_FMT, kp.KV_OK, 1, 1, 0)
            body = struct.pack(kp.SEG_INFO_FMT, seg.node_id, 0, seg.seg_base, seg.seg_size,
                               seg.ctrl_ip.encode(), seg.ctrl_port, seg.hcomm_listen_port,
                               seg.channel_tag.encode(), len(seg.mem_desc), 0, seg.mem_desc,
                               len(seg.ep_desc), 0, seg.ep_desc)
            send(kp.CMD_POOL_CREATE, head + body)
            return True
        if cmd == kp.CMD_LINK_EXCHANGE:
            # 规格断言点：LINK_EXCHANGE 必须携带完整 KvLinkFrame 作为本条消息的 payload，
            # 而不是只发 Header 再等第二条消息
            if plen == kp.LINK_FRAME_SIZE:
                dlen, _d, mlen, _m = struct.unpack(kp.LINK_FRAME_FMT, payload)
                with self._lock:
                    self.link_frame_ok = (dlen > 0 and mlen > 0)
            send(kp.CMD_LINK_EXCHANGE, struct.pack(kp.GENERIC_RESP_FMT, kp.KV_OK, 0))
            return True
        if cmd == kp.CMD_POOL_DESTROY:
            send(kp.CMD_POOL_DESTROY, struct.pack(kp.GENERIC_RESP_FMT, kp.KV_OK, 0))
            return True
        send(cmd, struct.pack(kp.GENERIC_RESP_FMT, kp.KV_ERR_PROTO, 0))
        return True


class TestCtrlInterop(unittest.TestCase):
    def test_full_flow_wire_contract(self):
        mock = MockEmsd()
        mock.start()
        try:
            cli = EmsClient("127.0.0.1", mock.port, timeout_sec=5)
            cli.connect()
            cli.ping()
            pool = cli.pool_create(1 << 20, layer_num=2, layer_rows=128, row_bytes=1152)
            self.assertEqual(pool.node_num, 1)
            self.assertEqual(pool.segments[0].ctrl_port, mock.port)
            # 回归锚点：client 必须拿到 server 侧 endpoint 描述（AIV remoteEndpoint），
            # 缺失时旧代码会退回传本端 local_desc 导致 client 向自己建链、channel 永不 READY
            self.assertEqual(pool.segments[0].ep_desc, b"\xEE" * 160)
            status = cli.link_exchange(pool.segments[0], endpoint_desc=bytes(72), mem_desc=b"\xAB" * 100)
            self.assertEqual(status, kp.KV_OK)
            cli.pool_destroy(pool.pool_id)
            cli.close()
        finally:
            mock.stop()
            mock.join(timeout=2)

        # 组帧契约：恰好 4 条命令、每条一个消息，按序到达
        cmds = [c for c, _, _ in mock.records]
        self.assertEqual(cmds, [kp.CMD_PING, kp.CMD_POOL_CREATE, kp.CMD_LINK_EXCHANGE, kp.CMD_POOL_DESTROY])
        # LINK_EXCHANGE 的 payload 必须是完整 KvLinkFrame（4232B），一次送达
        link = [r for r in mock.records if r[0] == kp.CMD_LINK_EXCHANGE]
        self.assertEqual(len(link), 1)
        self.assertEqual(link[0][1], kp.LINK_FRAME_SIZE)
        self.assertTrue(mock.link_frame_ok)


if __name__ == "__main__":
    unittest.main()
