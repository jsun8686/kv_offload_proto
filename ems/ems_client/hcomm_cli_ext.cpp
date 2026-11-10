/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * 框架侧（NPU 卡上进程）HCOMM 建链 pybind 扩展：dlsym libascendcl + libhcomm。
 * Endpoint(DEVICE, UB_CTP, EID) + MemReg(HBM KV 池) + Export/Import + Channel(AIV, client)。
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/ems_eid.h"
#include "common/hcomm/hcomm_res.h"

namespace py = pybind11;

namespace {
struct DynApi {
    int32_t (*aclrtSetDevice)(int32_t);
    int32_t (*aclrtGetPhyDevIdByLogicDevId)(int32_t, int32_t *);
    int32_t (*rtGetDeviceInfo)(uint32_t, int32_t, int32_t, int64_t *);  // 内部接口，可能缺失（PoC 同款容忍）
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
};

DynApi g_api{};
void *g_aclLib = nullptr;
void *g_hcommLib = nullptr;
EndpointHandle g_endpoint = nullptr;
EndpointDesc g_localDesc{};
HcommMemHandle g_memHandle = nullptr;
void *g_memDesc = nullptr;
uint32_t g_memDescLen = 0;
bool g_inited = false;

template <typename T>
T Sym(void *h, const char *name)
{
    auto fn = reinterpret_cast<T>(dlsym(h, name));
    if (fn == nullptr) {
        throw std::runtime_error(std::string("dlsym ") + name + " failed: " + dlerror());
    }
    return fn;
}
}  // namespace

// 初始化（EID 全自动，对齐 PoC/emsd 链路）：
//   SetDevice(deviceId) → aclrtGetPhyDevIdByLogicDevId 取 phyId
//   → rootinfo 按 device_id==phyId 匹配 EID（16B/32hex）
//   → EndpointCreate(DEVICE, UB_CTP, COMM_ADDR_TYPE_EID)
void cli_init(int deviceId, const std::string &rootinfoPath, const std::string &aclLib, const std::string &hcommLib)
{
    if (g_inited) {
        throw std::runtime_error("cli already inited");
    }
    g_aclLib = dlopen(aclLib.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (g_aclLib == nullptr) {
        throw std::runtime_error(std::string("dlopen acl failed: ") + dlerror());
    }
    g_hcommLib = dlopen(hcommLib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (g_hcommLib == nullptr) {
        throw std::runtime_error(std::string("dlopen hcomm failed: ") + dlerror());
    }
    g_api.aclrtSetDevice = Sym<decltype(g_api.aclrtSetDevice)>(g_aclLib, "aclrtSetDevice");
    g_api.aclrtGetPhyDevIdByLogicDevId =
        Sym<decltype(g_api.aclrtGetPhyDevIdByLogicDevId)>(g_aclLib, "aclrtGetPhyDevIdByLogicDevId");
    // rtGetDeviceInfo 为内部接口：符号缺失时降级为 0（对齐 PoC 的容忍策略，不视为致命错误）
    g_api.rtGetDeviceInfo = reinterpret_cast<decltype(g_api.rtGetDeviceInfo)>(dlsym(g_aclLib, "rtGetDeviceInfo"));
    if (g_api.rtGetDeviceInfo == nullptr) {
        printf("[cli][WARN] rtGetDeviceInfo unavailable, topology fields stay 0: %s\n", dlerror());
    }
    g_api.HcommEndpointCreate = Sym<decltype(g_api.HcommEndpointCreate)>(g_hcommLib, "HcommEndpointCreate");
    g_api.HcommEndpointDestroy = Sym<decltype(g_api.HcommEndpointDestroy)>(g_hcommLib, "HcommEndpointDestroy");
    g_api.HcommMemReg = Sym<decltype(g_api.HcommMemReg)>(g_hcommLib, "HcommMemReg");
    g_api.HcommMemUnreg = Sym<decltype(g_api.HcommMemUnreg)>(g_hcommLib, "HcommMemUnreg");
    g_api.HcommMemExport = Sym<decltype(g_api.HcommMemExport)>(g_hcommLib, "HcommMemExport");
    g_api.HcommMemImport = Sym<decltype(g_api.HcommMemImport)>(g_hcommLib, "HcommMemImport");
    g_api.HcommMemUnimport = Sym<decltype(g_api.HcommMemUnimport)>(g_hcommLib, "HcommMemUnimport");
    g_api.HcommChannelCreate = Sym<decltype(g_api.HcommChannelCreate)>(g_hcommLib, "HcommChannelCreate");
    g_api.HcommChannelGetStatus = Sym<decltype(g_api.HcommChannelGetStatus)>(g_hcommLib, "HcommChannelGetStatus");
    g_api.HcommChannelDestroy = Sym<decltype(g_api.HcommChannelDestroy)>(g_hcommLib, "HcommChannelDestroy");

    if (g_api.aclrtSetDevice(deviceId) != 0) {
        throw std::runtime_error("aclrtSetDevice failed, deviceId=" + std::to_string(deviceId));
    }
    int32_t phyId = -1;
    if (g_api.aclrtGetPhyDevIdByLogicDevId(deviceId, &phyId) != 0 || phyId < 0) {
        throw std::runtime_error("aclrtGetPhyDevIdByLogicDevId failed, deviceId=" + std::to_string(deviceId));
    }
    std::array<uint8_t, EMS_EID_LEN> eid{};
    if (EmsGetEid(rootinfoPath, static_cast<uint32_t>(phyId), eid) != 0) {
        throw std::runtime_error("EmsGetEid failed, phyId=" + std::to_string(phyId) + " rootinfo=" + rootinfoPath);
    }
    printf("[cli] EID resolved, deviceId=%d phyId=%d eid=%s\n", deviceId, phyId, EmsEidToString(eid).c_str());

    if (EndpointDescInit(&g_localDesc, 1) != 0) {
        throw std::runtime_error("EndpointDescInit failed");
    }
    g_localDesc.protocol = COMM_PROTOCOL_UBC_CTP;
    g_localDesc.commAddr.type = COMM_ADDR_TYPE_EID;
    memcpy(g_localDesc.commAddr.eid, eid.data(), EMS_EID_LEN);  // 16B 有效（PoC 同款填法）
    g_localDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    g_localDesc.loc.device.devPhyId = static_cast<uint32_t>(phyId);  // 用 phyId（PoC 同款：desc 填物理 ID）
    // 拓扑四元组对齐 aiv_urma_poc（缺这些字段时 UBC_CTP 握手无法推进，实测卡 IN_PROGRESS）：
    //   sdid/serverIdx/superPodIdx 经 rtGetDeviceInfo 查询（infoType 复刻 MemFabric hybm_define.h）
    if (g_api.rtGetDeviceInfo != nullptr) {
        constexpr int32_t kInfoSdid = 26;
        constexpr int32_t kInfoServerId = 27;
        constexpr int32_t kInfoSuperPodId = 28;
        int64_t v = 0;
        uint32_t sdid = 0, serverIdx = 0, superPodIdx = 0;
        if (g_api.rtGetDeviceInfo(static_cast<uint32_t>(deviceId), 0, kInfoSdid, &v) == 0) {
            sdid = static_cast<uint32_t>(v);
        }
        if (g_api.rtGetDeviceInfo(static_cast<uint32_t>(deviceId), 0, kInfoServerId, &v) == 0) {
            serverIdx = static_cast<uint32_t>(v);
        }
        if (g_api.rtGetDeviceInfo(static_cast<uint32_t>(deviceId), 0, kInfoSuperPodId, &v) == 0) {
            superPodIdx = static_cast<uint32_t>(v);
        }
        g_localDesc.loc.device.superDevId = sdid;
        g_localDesc.loc.device.serverIdx = serverIdx;
        g_localDesc.loc.device.superPodIdx = superPodIdx;
        printf("[cli] topology: phyId=%d sdid=%u serverIdx=%u superPodIdx=%u\n", phyId, sdid, serverIdx, superPodIdx);
    }
    HcommResult ret = g_api.HcommEndpointCreate(&g_localDesc, &g_endpoint);
    if (ret != 0 || g_endpoint == nullptr) {
        throw std::runtime_error("HcommEndpointCreate ret=" + std::to_string(ret) + " eid=" +
                                 EmsEidToString(eid));
    }
    g_inited = true;
}

py::bytes cli_local_desc()
{
    return py::bytes(reinterpret_cast<const char *>(&g_localDesc), sizeof(EndpointDesc));
}

// 注册框架侧 HBM KV 池（exchangeAllMems 用）：addr 为 device 指针（tensor.data_ptr()）
void cli_mem_reg(uint64_t addr, uint64_t size, const std::string &tag)
{
    if (!g_inited || g_memHandle != nullptr) {
        throw std::runtime_error("state error: not inited or already registered");
    }
    CommMem mem{};
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = reinterpret_cast<void *>(addr);
    mem.size = size;
    HcommResult ret = g_api.HcommMemReg(g_endpoint, tag.c_str(), &mem, &g_memHandle);
    if (ret != 0 || g_memHandle == nullptr) {
        throw std::runtime_error("MemReg ret=" + std::to_string(ret));
    }
    ret = g_api.HcommMemExport(g_endpoint, g_memHandle, &g_memDesc, &g_memDescLen);
    if (ret != 0 || g_memDesc == nullptr) {
        throw std::runtime_error("MemExport ret=" + std::to_string(ret));
    }
}

py::bytes cli_mem_desc()
{
    if (g_memDesc == nullptr) {
        throw std::runtime_error("mem not registered");
    }
    return py::bytes(static_cast<const char *>(g_memDesc), g_memDescLen);
}

py::tuple cli_mem_import(const std::string &memDesc)
{
    CommMem remote{};
    HcommResult ret = g_api.HcommMemImport(g_endpoint, memDesc.data(), static_cast<uint32_t>(memDesc.size()), &remote);
    if (ret != 0) {
        throw std::runtime_error("MemImport ret=" + std::to_string(ret));
    }
    return py::make_tuple(reinterpret_cast<uint64_t>(remote.addr), static_cast<uint64_t>(remote.size),
                          static_cast<int>(remote.type));
}

// client 通道：与某资源节点 server 建立 AIV 通道，返回 ChannelHandle（u64，内核入参）
py::tuple cli_link_channel(const std::string &peerDescBytes, uint16_t hcommPort, const std::string &channelTag,
                           uint32_t readyTimeoutSec)
{
    if (!g_inited) {
        throw std::runtime_error("not inited");
    }
    if (peerDescBytes.size() != sizeof(EndpointDesc)) {
        throw std::runtime_error("peer desc size mismatch");
    }
    HcommChannelDesc chDesc;
    if (HcommChannelDescInit(&chDesc, 1) != 0) {
        throw std::runtime_error("HcommChannelDescInit failed");
    }
    chDesc.remoteEndpoint = *reinterpret_cast<const EndpointDesc *>(peerDescBytes.data());
    chDesc.notifyNum = 1;
    chDesc.exchangeAllMems = true;
    chDesc.role = HCOMM_SOCKET_ROLE_CLIENT;
    chDesc.port = hcommPort;  // 对端（资源节点）监听端口
    chDesc.qos = 0;
    chDesc.channelName = const_cast<char *>(channelTag.c_str());
    ChannelHandle channel = 0;
    HcommResult ret = g_api.HcommChannelCreate(g_endpoint, COMM_ENGINE_AIV, &chDesc, 1, &channel);
    if (ret != 0 || channel == 0) {
        throw std::runtime_error("ChannelCreate(client) ret=" + std::to_string(ret) + " port=" +
                                 std::to_string(hcommPort));
    }
    // 状态语义（对齐 MemFabric hcomm_api_wrapper）：READY=0 / IN_PROGRESS=1 / FAILED=2 / TIMEOUT=3。
    // 只认 0 会把 FAILED/TIMEOUT 吞成"等满超时"，握手失败的真实原因不可见——必须区分并报出。
    for (uint32_t waited = 0; waited < readyTimeoutSec * 10U; waited++) {
        int32_t status = -1;
        ret = g_api.HcommChannelGetStatus(&channel, 1, &status);
        if (ret != 0) {
            (void)g_api.HcommChannelDestroy(&channel, 1);
            throw std::runtime_error("ChannelGetStatus ret=" + std::to_string(ret));
        }
        if (status == 0) {  // READY
            printf("[cli] channel READY, port=%u channel=0x%llx\n", hcommPort, (unsigned long long)channel);
            return py::make_tuple(static_cast<uint64_t>(channel), status);
        }
        if (status == 2 || status == 3) {  // FAILED / TIMEOUT：终态，立即报错带状态值
            (void)g_api.HcommChannelDestroy(&channel, 1);
            throw std::runtime_error(std::string("channel handshake ") + (status == 2 ? "FAILED" : "TIMEOUT") +
                                     ", port=" + std::to_string(hcommPort));
        }
        if (waited % 50U == 0U) {  // 每 5s 报一次进展（IN_PROGRESS=1 或其他未知值）
            printf("[cli] channel status=%d (0=READY 1=IN_PROGRESS 2=FAILED 3=TIMEOUT), port=%u waited=%u.%us\n",
                   status, hcommPort, waited / 10U, (waited % 10U) * 100U);
        }
        usleep(100 * 1000);
    }
    (void)g_api.HcommChannelDestroy(&channel, 1);
    throw std::runtime_error("channel not ready in " + std::to_string(readyTimeoutSec) + "s, port=" +
                             std::to_string(hcommPort));
}

void cli_channel_destroy(uint64_t channel)
{
    ChannelHandle h = static_cast<ChannelHandle>(channel);
    (void)g_api.HcommChannelDestroy(&h, 1);
}

void cli_close()
{
    if (g_memHandle != nullptr) {
        (void)g_api.HcommMemUnreg(g_endpoint, g_memHandle);
        g_memHandle = nullptr;
    }
    if (g_endpoint != nullptr) {
        (void)g_api.HcommEndpointDestroy(g_endpoint);
        g_endpoint = nullptr;
    }
    g_inited = false;
}

PYBIND11_MODULE(_cli, m)
{
    m.doc() = "ems client side hcomm binding (NPU device process)";
    m.def("init", &cli_init, py::arg("device_id"), py::arg("rootinfo") = "/etc/hccl_rootinfo.json",
          py::arg("acl_lib") = "libascendcl.so", py::arg("hcomm_lib") = "libhcomm.so");
    m.def("local_desc", &cli_local_desc);
    m.def("mem_reg", &cli_mem_reg, py::arg("addr"), py::arg("size"), py::arg("tag") = "kv_hbm");
    m.def("mem_desc", &cli_mem_desc);
    m.def("mem_import", &cli_mem_import, py::arg("mem_desc"));
    m.def("link_channel", &cli_link_channel, py::arg("peer_desc"), py::arg("hcomm_port"), py::arg("channel_tag"),
          py::arg("ready_timeout_sec") = 60);
    m.def("channel_destroy", &cli_channel_destroy, py::arg("channel"));
    m.def("close", &cli_close);
}
