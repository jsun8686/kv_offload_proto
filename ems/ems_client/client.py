# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# EmsClient：框架侧 EMS 控制客户端（TCP）。pool_create 打散由 master 完成；
# connect_channels 逐段与资源节点 LINK_EXCHANGE 并在 NPU 侧建立 AIV client 通道（经 _cli 扩展）。

import socket

from . import kv_proto as kp


class EmsError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__("status={} {}".format(status, msg))
        self.status = status


class EmsClient:
    def __init__(self, master_ip: str, master_port: int, timeout_sec: float = 120.0):
        self.master = (master_ip, master_port)
        self.timeout = timeout_sec
        self._sock = None

    def connect(self) -> None:
        self._sock = socket.create_connection(self.master, timeout=self.timeout)
        self._sock.settimeout(self.timeout)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _request(self, cmd: int, payload: bytes):
        kp.send_msg(self._sock, cmd, payload)
        return kp.recv_msg(self._sock)

    def ping(self) -> None:
        cmd, payload = self._request(kp.CMD_PING, b"")
        if cmd != kp.CMD_PONG or kp.decode_generic_resp(payload) != kp.KV_OK:
            raise EmsError(kp.KV_ERR_PROTO, "ping failed, cmd={}".format(cmd))

    def pool_create(self, size_bytes: int, layer_num: int, layer_rows: int, row_bytes: int,
                    stripe_rows: int = 1) -> kp.PoolInfo:
        payload = kp.encode_pool_create_req(size_bytes, layer_num, layer_rows, row_bytes, stripe_rows)
        cmd, resp = self._request(kp.CMD_POOL_CREATE, payload)
        if cmd != kp.CMD_POOL_CREATE:
            raise EmsError(kp.KV_ERR_PROTO, "unexpected resp cmd={}".format(cmd))
        status, pool = kp.decode_pool_create_resp(resp)
        if status != kp.KV_OK:
            raise EmsError(status, "pool_create failed, size={} layerNum={} layerRows={} rowBytes={}".format(
                size_bytes, layer_num, layer_rows, row_bytes))
        return pool

    def pool_destroy(self, pool_id: int) -> None:
        cmd, resp = self._request(kp.CMD_POOL_DESTROY, kp.encode_pool_destroy_req(pool_id))
        status = kp.decode_generic_resp(resp)
        if status != kp.KV_OK:
            raise EmsError(status, "pool_destroy failed, poolId={}".format(pool_id))

    def link_exchange(self, seg: kp.SegmentInfo, endpoint_desc: bytes, mem_desc: bytes,
                      sock=None) -> int:
        """与某资源节点 emsd 交换 desc/memDesc（emsd 侧创建 server 通道）。

        sock 可复用已有连接（每段独立连接也行）。返回 KV_OK=0 表示 emsd 侧通道已创建。
        """
        own = sock is None
        s = sock if sock is not None else socket.create_connection((seg.ctrl_ip, seg.ctrl_port), timeout=self.timeout)
        try:
            s.settimeout(self.timeout)
            kp.send_msg(s, kp.CMD_LINK_EXCHANGE, kp.encode_link_frame(endpoint_desc, mem_desc))
            cmd, resp = kp.recv_msg(s)
            if cmd != kp.CMD_LINK_EXCHANGE:
                raise EmsError(kp.KV_ERR_PROTO, "unexpected link resp cmd={}".format(cmd))
            return kp.decode_generic_resp(resp)
        finally:
            if own:
                s.close()
