/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * 框架侧（NPU 卡上进程）HCOMM 建链 pybind 扩展：dlsym libascendcl + libhcomm。
 * UBC 路径：Endpoint(DEVICE, UB_CTP, EID) + MemReg(HBM KV 池) + Export/Import + Channel(AIV, client)。
 * RoCE 路径（v2，A3 + AICPU 数据面）：Endpoint(DEVICE, ROCE, IP_V4) + ThreadAlloc(AICPU_TS)
 *   + Channel(AICPU, client)；thread 全通道共享，经 register_rmt_thread 交给 kvops。
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arpa/inet.h>
#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/ems_eid.h"
#include "common/hcomm/hcomm_res.h"

namespace py = pybind11;

namespace {
// 握手阶段计时：定位建链 35s 花在 ChannelCreate 调用内部还是返回后的状态轮询期
const char *TsNow()
{
    static thread_local char buf[32];
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    tm tmv{};
    localtime_r(&ts.tv_sec, &tmv);
    (void)snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", tmv.tm_hour, tmv.tm_min, tmv.tm_sec,
                   static_cast<int>(ts.tv_nsec / 1000000));
    return buf;
}

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
    HcommResult (*HcommThreadAlloc)(CommEngine, uint32_t, const uint32_t *, ThreadHandle *);
    HcommResult (*HcommThreadFree)(const ThreadHandle *, uint32_t);
};

DynApi g_api{};
void *g_aclLib = nullptr;
void *g_hcommLib = nullptr;
EndpointHandle g_endpoint = nullptr;
EndpointDesc g_localDesc{};
HcommMemHandle g_memHandle = nullptr;
void *g_memDesc = nullptr;
uint32_t g_memDescLen = 0;
ThreadHandle g_thread = 0;  // AICPU_TS 数据面线程（roce 模式全通道共享；cli_close 释放）
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

// 双模式公共前置：dlopen libascendcl + libhcomm 并解析全部符号
void CliLoadLibs(const std::string &aclLib, const std::string &hcommLib)
{
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
    g_api.HcommThreadAlloc = Sym<decltype(g_api.HcommThreadAlloc)>(g_hcommLib, "HcommThreadAlloc");
    g_api.HcommThreadFree = Sym<decltype(g_api.HcommThreadFree)>(g_hcommLib, "HcommThreadFree");
}

void CliSetDeviceOrThrow(int32_t deviceId)
{
    if (g_api.aclrtSetDevice(deviceId) != 0) {
        throw std::runtime_error("aclrtSetDevice failed, deviceId=" + std::to_string(deviceId));
    }
}

void CliResolvePhyId(int32_t deviceId, int32_t &phyId)
{
    phyId = -1;
    if (g_api.aclrtGetPhyDevIdByLogicDevId(deviceId, &phyId) != 0 || phyId < 0) {
        throw std::runtime_error("aclrtGetPhyDevIdByLogicDevId failed, deviceId=" + std::to_string(deviceId));
    }
}

// 填充 device loc 四元组（devPhyId + 拓扑三元组）：
// UBC_CTP 握手必需（缺字段实测卡 IN_PROGRESS）；ROCE 路径同填对齐（MemFabric devPhyId:IP 格式源）。
void CliFillDeviceLoc(int32_t deviceId, int32_t phyId)
{
    g_localDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    g_localDesc.loc.device.devPhyId = static_cast<uint32_t>(phyId);  // 用 phyId（PoC 同款：desc 填物理 ID）
    // 拓扑四元组对齐 aiv_urma_poc：sdid/serverIdx/superPodIdx 经 rtGetDeviceInfo 查询
    // （infoType 复刻 MemFabric hybm_define.h）
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
}

// 初始化（EID 全自动，对齐 PoC/emsd 链路）：
//   SetDevice(deviceId) → phyId → rootinfo 按 device_id==phyId 匹配 EID → EndpointCreate(DEVICE, UB_CTP, EID)
void cli_init(int deviceId, const std::string &rootinfoPath, const std::string &aclLib, const std::string &hcommLib)
{
    if (g_inited) {
        throw std::runtime_error("cli already inited");
    }
    CliLoadLibs(aclLib, hcommLib);
    CliSetDeviceOrThrow(deviceId);
    int32_t phyId = -1;
    CliResolvePhyId(deviceId, phyId);
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
    CliFillDeviceLoc(deviceId, phyId);
    HcommResult ret = g_api.HcommEndpointCreate(&g_localDesc, &g_endpoint);
    if (ret != 0 || g_endpoint == nullptr) {
        throw std::runtime_error("HcommEndpointCreate ret=" + std::to_string(ret) + " eid=" +
                                 EmsEidToString(eid));
    }
    g_inited = true;
}

// v2 初始化（roce 模式，A3 + HCOMM AICPU 数据面）：
//   SetDevice → phyId → EndpointCreate(DEVICE, ROCE, IP_V4=devIp) → ThreadAlloc(AICPU_TS, notify=0)
// devIp = 本 NPU 的 RoCE 参数面 IP（hccn_tool 配置，与 emsd --dev-ip 同平面；经 KV_CLI_DEV_IP 传入）。
// AICPU_TS thread 全通道共享（单实例 kernel），cli_close 统一释放。
void cli_init_roce(int deviceId, const std::string &devIp, const std::string &aclLib, const std::string &hcommLib)
{
    if (g_inited) {
        throw std::runtime_error("cli already inited");
    }
    if (devIp.empty()) {
        throw std::runtime_error("init_roce: devIp empty (pass ems pool dev_ip / env KV_CLI_DEV_IP)");
    }
    CliLoadLibs(aclLib, hcommLib);
    CliSetDeviceOrThrow(deviceId);
    int32_t phyId = -1;
    CliResolvePhyId(deviceId, phyId);
    if (EndpointDescInit(&g_localDesc, 1) != 0) {
        throw std::runtime_error("EndpointDescInit failed");
    }
    g_localDesc.protocol = COMM_PROTOCOL_ROCE;
    g_localDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    if (inet_pton(AF_INET, devIp.c_str(), &g_localDesc.commAddr.addr) != 1) {
        throw std::runtime_error("init_roce: invalid devIp: " + devIp);
    }
    CliFillDeviceLoc(deviceId, phyId);
    HcommResult ret = g_api.HcommEndpointCreate(&g_localDesc, &g_endpoint);
    if (ret != 0 || g_endpoint == nullptr) {
        throw std::runtime_error("HcommEndpointCreate(roce) ret=" + std::to_string(ret) + " devIp=" + devIp);
    }
    constexpr uint32_t notifyNum = 0;
    ThreadHandle thread = 0;
    ret = g_api.HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 1, &notifyNum, &thread);
    if (ret != 0 || thread == 0) {
        throw std::runtime_error("HcommThreadAlloc(AICPU_TS) ret=" + std::to_string(ret));
    }
    g_thread = thread;
    printf("[cli] roce endpoint ok, deviceId=%d phyId=%d devIp=%s thread=0x%llx\n", deviceId, phyId, devIp.c_str(),
           (unsigned long long)thread);
    g_inited = true;
}

// 当前 AICPU_TS 数据面线程（0 = 未走 init_roce）；供框架侧登记 kvops register_rmt_thread
uint64_t cli_thread()
{
    return static_cast<uint64_t>(g_thread);
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

// client 通道公共体：UBC 走 COMM_ENGINE_AIV，RoCE（v2）走 COMM_ENGINE_AICPU（thread 预分配共享）
static ChannelHandle CliLinkChannel(CommEngine engine, const char *engineTag, const std::string &peerDescBytes,
                                    uint16_t hcommPort, const std::string &channelTag, uint32_t readyTimeoutSec)
{
    if (!g_inited) {
        throw std::runtime_error("not inited");
    }
    if (engine == COMM_ENGINE_AICPU && g_thread == 0) {
        throw std::runtime_error("link_channel_aicpu requires init_roce first (thread not allocated)");
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
    printf("[cli] %s ChannelCreate(client, %s) enter, port=%u\n", TsNow(), engineTag, hcommPort);
    HcommResult ret = g_api.HcommChannelCreate(g_endpoint, engine, &chDesc, 1, &channel);
    printf("[cli] %s ChannelCreate(client, %s) ret=%d, port=%u\n", TsNow(), engineTag, ret, hcommPort);
    if (ret != 0 || channel == 0) {
        throw std::runtime_error(std::string("ChannelCreate(client, ") + engineTag + ") ret=" +
                                 std::to_string(ret) + " port=" + std::to_string(hcommPort));
    }
    // 状态语义（对齐 MemFabric hcomm_api_wrapper）：READY=0 / IN_PROGRESS=1 / FAILED=2 / TIMEOUT=3。
    // 只认 0 会把 FAILED/TIMEOUT 吞成"等满超时"，握手失败的真实原因不可见——必须区分并报出。
    // 联调插桩：状态翻转即打时间戳（区分耗时在 Create 内部还是轮询期；轮询间隔 100ms）
    int32_t lastStatus = -99;
    for (uint32_t waited = 0; waited < readyTimeoutSec * 10U; waited++) {
        int32_t status = -1;
        ret = g_api.HcommChannelGetStatus(&channel, 1, &status);
        if (ret != 0) {
            (void)g_api.HcommChannelDestroy(&channel, 1);
            throw std::runtime_error("ChannelGetStatus ret=" + std::to_string(ret));
        }
        if (status != lastStatus) {
            printf("[cli] %s status %d -> %d, port=%u waited=%u.%us\n", TsNow(), lastStatus, status, hcommPort,
                   waited / 10U, (waited % 10U) * 100U);
            lastStatus = status;
        }
        if (status == 0) {  // READY
            printf("[cli] channel READY(%s), port=%u channel=0x%llx\n", engineTag, hcommPort,
                   (unsigned long long)channel);
            return channel;
        }
        if (status == 2 || status == 3) {  // FAILED / TIMEOUT：终态，立即报错带状态值
            (void)g_api.HcommChannelDestroy(&channel, 1);
            throw std::runtime_error(std::string("channel handshake ") + (status == 2 ? "FAILED" : "TIMEOUT") +
                                     ", port=" + std::to_string(hcommPort));
        }
        usleep(100 * 1000);
    }
    (void)g_api.HcommChannelDestroy(&channel, 1);
    throw std::runtime_error("channel not ready in " + std::to_string(readyTimeoutSec) + "s, port=" +
                             std::to_string(hcommPort));
}

// client 通道：与某资源节点 server 建立 AIV 通道，返回 ChannelHandle（u64，内核入参）
py::tuple cli_link_channel(const std::string &peerDescBytes, uint16_t hcommPort, const std::string &channelTag,
                           uint32_t readyTimeoutSec)
{
    ChannelHandle channel = CliLinkChannel(COMM_ENGINE_AIV, "AIV", peerDescBytes, hcommPort, channelTag,
                                           readyTimeoutSec);
    return py::make_tuple(static_cast<uint64_t>(channel), 0);
}

// v2 client 通道：AICPU 引擎（RoCE），返回 (channel, thread, status)；thread 供 kvops register_rmt_thread
py::tuple cli_link_channel_aicpu(const std::string &peerDescBytes, uint16_t hcommPort, const std::string &channelTag,
                                 uint32_t readyTimeoutSec)
{
    ChannelHandle channel = CliLinkChannel(COMM_ENGINE_AICPU, "AICPU", peerDescBytes, hcommPort, channelTag,
                                           readyTimeoutSec);
    return py::make_tuple(static_cast<uint64_t>(channel), static_cast<uint64_t>(g_thread), 0);
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
    if (g_thread != 0) {
        ThreadHandle t = g_thread;
        (void)g_api.HcommThreadFree(&t, 1);
        g_thread = 0;
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
    m.def("init_roce", &cli_init_roce, py::arg("device_id"), py::arg("dev_ip"),
          py::arg("acl_lib") = "libascendcl.so", py::arg("hcomm_lib") = "libhcomm.so");
    m.def("thread", &cli_thread);
    m.def("local_desc", &cli_local_desc);
    m.def("mem_reg", &cli_mem_reg, py::arg("addr"), py::arg("size"), py::arg("tag") = "kv_hbm");
    m.def("mem_desc", &cli_mem_desc);
    m.def("mem_import", &cli_mem_import, py::arg("mem_desc"));
    m.def("link_channel", &cli_link_channel, py::arg("peer_desc"), py::arg("hcomm_port"), py::arg("channel_tag"),
          py::arg("ready_timeout_sec") = 60);
    m.def("link_channel_aicpu", &cli_link_channel_aicpu, py::arg("peer_desc"), py::arg("hcomm_port"),
          py::arg("channel_tag"), py::arg("ready_timeout_sec") = 60);
    m.def("channel_destroy", &cli_channel_destroy, py::arg("channel"));
    m.def("close", &cli_close);
}
