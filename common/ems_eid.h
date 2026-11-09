/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * EID 自动获取（移植自 aiv_urma_poc host/poc_eid，emsd 与 ems_client._cli 共用）：
 * 解析 rootinfo（默认 /etc/hccl_rootinfo.json）：在 "device_id":<phyId> 的 rank 块内
 * 找第一个 "addr_type":"EID" 的 "addr" 字段（32 hex = 16B，简化解析，PoC 同款）。
 */

#ifndef COMMON_EMS_EID_H
#define COMMON_EMS_EID_H

#include <array>
#include <cstdint>
#include <string>

constexpr uint32_t EMS_EID_LEN = 16;

int EmsGetEid(const std::string &rootinfoPath, uint32_t phyDeviceId, std::array<uint8_t, EMS_EID_LEN> &eid);

std::string EmsEidToString(const std::array<uint8_t, EMS_EID_LEN> &eid);

#endif  // COMMON_EMS_EID_H
