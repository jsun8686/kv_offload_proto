# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# EMS 控制协议 Python 编解码单测（round-trip；C 侧对齐锚点见 common/kv_proto.h）。

import os
import struct
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "ems")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ems_client import kv_proto as kp


class TestProto(unittest.TestCase):
    def test_header_roundtrip(self):
        hdr = kp.encode_header(kp.CMD_POOL_CREATE, 32)
        self.assertEqual(len(hdr), kp.HDR_SIZE)
        magic, ver, cmd, plen, trailer, _ = struct.unpack(kp.HDR_FMT, hdr)
        self.assertEqual((magic, ver, cmd, plen, trailer), (kp.MAGIC, kp.VERSION, kp.CMD_POOL_CREATE, 32, 0))

    def test_layout_sizes(self):
        # 与 C 侧 sizeof 锚点（common/kv_proto.h 定长字段求和，pack(1)）
        self.assertEqual(kp.HDR_SIZE, 16)
        self.assertEqual(struct.calcsize(kp.POOL_CREATE_REQ_FMT), 28)  # u64 + 5*u32，pack(1) 无 padding
        self.assertEqual(struct.calcsize(kp.POOL_CREATE_RESP_HEAD_FMT), 16)
        # SEG_INFO: 4+4+8+8+64+2+2+64+4+4+4096 = 4260
        self.assertEqual(kp.SEG_INFO_SIZE, 4260)
        self.assertEqual(struct.calcsize(kp.GENERIC_RESP_FMT), 8)
        self.assertEqual(struct.calcsize(kp.SEG_ALLOC_REQ_FMT), 148)
        # SEG_ALLOC_RESP: 4+4+8+4+4+4096 = 4120
        self.assertEqual(struct.calcsize(kp.SEG_ALLOC_RESP_FMT), 4120)
        # LINK frame: 4+128+4+4096 = 4232
        self.assertEqual(kp.LINK_FRAME_SIZE, 4232)

    def test_pool_create_resp_roundtrip(self):
        seg = kp.SegmentInfo(
            node_id=1, seg_base=0x7F0000000000, seg_size=1 << 30, ctrl_ip="192.168.1.20", ctrl_port=27001,
            hcomm_listen_port=60001, channel_tag="kv_pool_1",
            mem_desc=bytes(range(64)),
        )
        n = 2
        head = struct.pack(kp.POOL_CREATE_RESP_HEAD_FMT, kp.KV_OK, 1, n, 0)
        body = b""
        for s in (seg, kp.SegmentInfo(node_id=0, seg_base=0x7F1000000000, ctrl_ip="192.168.1.10")):
            body += struct.pack(
                kp.SEG_INFO_FMT, s.node_id, 0, s.seg_base, s.seg_size, s.ctrl_ip.encode(), s.ctrl_port,
                s.hcomm_listen_port, s.channel_tag.encode(), len(s.mem_desc), 0, s.mem_desc,
            )
        status, pool = kp.decode_pool_create_resp(head + body)
        self.assertEqual(status, kp.KV_OK)
        self.assertEqual(pool.pool_id, 1)
        self.assertEqual(pool.node_num, 2)
        self.assertEqual(pool.segments[0].seg_base, 0x7F0000000000)
        self.assertEqual(pool.segments[1].ctrl_ip, "192.168.1.10")
        self.assertEqual(pool.segments[0].mem_desc, bytes(range(64)))

    def test_link_frame_roundtrip(self):
        desc = bytes(72)
        md = b"\xAB" * 100
        frame = kp.encode_link_frame(desc, md)
        dlen, d, mlen, m = struct.unpack(kp.LINK_FRAME_FMT, frame)
        self.assertEqual((dlen, mlen), (len(desc), len(md)))
        self.assertEqual(d[:dlen], desc)
        self.assertEqual(m[:mlen], md)

    def test_pool_destroy(self):
        payload = kp.encode_pool_destroy_req(7)
        self.assertEqual(struct.unpack(kp.POOL_DESTROY_REQ_FMT, payload), (7, 0))


if __name__ == "__main__":
    unittest.main()
