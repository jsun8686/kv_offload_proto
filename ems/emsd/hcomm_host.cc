/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * emsd HCOMM host 侧建链实现（dlsym libhcomm，noDevice 基线）。
 */

#include "hcomm_host.h"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

// LINK_EXCHANGE 帧内 desc 缓冲与 EndpointDesc ABI 的锚定：hcomm 头文件升级导致
// sizeof(EndpointDesc) 变化时此处编译失败，避免再次线上才发现 KV_EP_DESC_BUF 不够用
static_assert(sizeof(EndpointDesc) <= KV_EP_DESC_BUF, "KV_EP_DESC_BUF too small for EndpointDesc ABI");

#define EMSD_DLSYM(var, name)                                                                                \
    do {                                                                                                     \
        (var) = reinterpret_cast<decltype(var)>(dlsym(dlHandle_, (name)));                                    \
        if ((var) == nullptr) {                                                                              \
            printf("[emsd][ERROR] dlsym %s failed, lib=%s, err=%s\n", (name), libPath_.c_str(), dlerror());   \
            return -1;                                                                                       \
        }                                                                                                    \
    } while (0)

int EmsdDynApi::Load(const std::string &libPath)
{
    void *h = dlopen(libPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
        printf("[emsd][ERROR] dlopen %s failed: %s\n", libPath.c_str(), dlerror());
        return -1;
    }
    auto load = [h](auto &fn, const char *name) -> int {
        using FnT = std::remove_reference_t<decltype(fn)>;  // 剥引用：decltype(fn) 为函数指针引用，直接 cast 非法
        fn = reinterpret_cast<FnT>(dlsym(h, name));
        if (fn == nullptr) {
            printf("[emsd][ERROR] dlsym %s failed: %s\n", name, dlerror());
            return -1;
        }
        return 0;
    };
    if (load(HcommEndpointCreate, "HcommEndpointCreate") != 0 || load(HcommEndpointDestroy, "HcommEndpointDestroy") != 0 ||
        load(HcommMemReg, "HcommMemReg") != 0 || load(HcommMemUnreg, "HcommMemUnreg") != 0 ||
        load(HcommMemExport, "HcommMemExport") != 0 || load(HcommMemImport, "HcommMemImport") != 0 ||
        load(HcommMemUnimport, "HcommMemUnimport") != 0 || load(HcommChannelCreate, "HcommChannelCreate") != 0 ||
        load(HcommChannelGetStatus, "HcommChannelGetStatus") != 0 || load(HcommChannelDestroy, "HcommChannelDestroy") != 0) {
        dlclose(h);
        return -1;
    }
    return 0;
}

int EmsdHcomm::Init(const std::string &hcommLib, const std::string &aclLib, int32_t deviceId,
                    const std::string &rootinfoPath, bool hostLoc, const std::string &hostAddr)
{
    libPath_ = hcommLib;
    hostLoc_ = hostLoc;
    deviceId_ = deviceId;
    std::array<uint8_t, EMS_EID_LEN> eid{};
    int32_t phyId = -1;

    if (hostLoc) {
        // host 模式（跨模式/noDevice 构建包环境）：不依赖 NPU runtime；
        // 地址由使用方手填（--host-addr，ipourma 的 host IPv6，V6 即 EID 载体）
        if (hostAddr.empty()) {
            printf("[emsd][ERROR] --loc host requires --host-addr <ipv6>\n");
            return -1;
        }
    } else {
        // device 模式（当前 CANN 包可用）：挂 NPU runtime 上下文（host endpoint 初始化经
        // hrtGetDevice 查当前 device，顺序关键）→ phyId → rootinfo 匹配 EID
        aclLib_ = aclLib;
        aclLibHandle_ = dlopen(aclLib.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (aclLibHandle_ == nullptr) {
            printf("[emsd][ERROR] dlopen %s failed: %s\n", aclLib.c_str(), dlerror());
            return -1;
        }
        auto aclrtSetDevice = reinterpret_cast<int32_t (*)(int32_t)>(dlsym(aclLibHandle_, "aclrtSetDevice"));
        if (aclrtSetDevice == nullptr) {
            printf("[emsd][ERROR] dlsym aclrtSetDevice failed, lib=%s, err=%s\n", aclLib.c_str(), dlerror());
            return -1;
        }
        int32_t retDev = aclrtSetDevice(deviceId);
        if (retDev != 0) {
            printf("[emsd][ERROR] aclrtSetDevice failed, deviceId=%d ret=%d\n", deviceId, retDev);
            return -1;
        }
        // 供 BindDeviceCtx 在工作线程重绑（设备上下文线程私有）
        aclrtSetDeviceFn_ = aclrtSetDevice;
        printf("[emsd] aclrtSetDevice ok, deviceId=%d\n", deviceId);
        // rtGetDeviceInfo（内部接口）查拓扑三元组：缺失时容忍降级为 0（对齐 PoC 策略）
        rtGetDeviceInfoFn_ = reinterpret_cast<int32_t (*)(uint32_t, int32_t, int32_t, int64_t *)>(
            dlsym(aclLibHandle_, "rtGetDeviceInfo"));
        if (rtGetDeviceInfoFn_ == nullptr) {
            printf("[emsd][WARN] rtGetDeviceInfo unavailable, topology fields stay 0: %s\n", dlerror());
        }

        auto aclrtGetPhyDevIdByLogicDevId =
            reinterpret_cast<int32_t (*)(int32_t, int32_t *)>(dlsym(aclLibHandle_, "aclrtGetPhyDevIdByLogicDevId"));
        if (aclrtGetPhyDevIdByLogicDevId == nullptr) {
            printf("[emsd][ERROR] dlsym aclrtGetPhyDevIdByLogicDevId failed, err=%s\n", dlerror());
            return -1;
        }
        if (aclrtGetPhyDevIdByLogicDevId(deviceId, &phyId) != 0 || phyId < 0) {
            printf("[emsd][ERROR] aclrtGetPhyDevIdByLogicDevId failed, deviceId=%d phyId=%d\n", deviceId, phyId);
            return -1;
        }
        if (EmsGetEid(rootinfoPath, static_cast<uint32_t>(phyId), eid) != 0) {
            printf("[emsd][ERROR] EmsGetEid failed, phyId=%d rootinfo=%s\n", phyId, rootinfoPath.c_str());
            return -1;
        }
        printf("[emsd] EID resolved, deviceId=%d phyId=%d eid=%s\n", deviceId, phyId, EmsEidToString(eid).c_str());
    }

    if (api_.Load(hcommLib) != 0) {
        return -1;
    }
    dlHandle_ = nullptr;  // 句柄由 Load 内部持有（原型不重复关闭）

    if (EndpointDescInit(&localDesc_, 1) != 0) {
        printf("[emsd][ERROR] EndpointDescInit failed\n");
        return -1;
    }
    localDesc_.protocol = COMM_PROTOCOL_UBC_CTP;  // = COMM_PROTOCOL_UB_CTP（跨节点 UBC，阶段一）
    if (hostLoc) {
        // HOST 形态：commAddr=IP_V6（noDevice 仓 rank_graph 标准填法）；loc.host.id——
        // 不写 loc.device.*（union 会覆盖 host.id，MemFabric 同款注释）
        localDesc_.commAddr.type = COMM_ADDR_TYPE_IP_V6;
        if (inet_pton(AF_INET6, hostAddr.c_str(), &localDesc_.commAddr.addr6) != 1) {
            printf("[emsd][ERROR] invalid --host-addr ipv6: %s\n", hostAddr.c_str());
            return -1;
        }
        localDesc_.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        localDesc_.loc.host.id = 0;
    } else {
        // DEVICE 形态：commAddr=EID + device 四元组（devPhyId=phyId，PoC 卡到卡同款）
        localDesc_.commAddr.type = COMM_ADDR_TYPE_EID;
        memcpy(localDesc_.commAddr.eid, eid.data(), EMS_EID_LEN);  // 16B 有效（PoC 同款填法）
        localDesc_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        localDesc_.loc.device.devPhyId = static_cast<uint32_t>(phyId);
        // 拓扑四元组对齐 aiv_urma_poc（缺这些字段时 UBC_CTP 握手无法推进，实测卡 IN_PROGRESS）：
        //   sdid/serverIdx/superPodIdx 经 rtGetDeviceInfo 查询（infoType 复刻 MemFabric hybm_define.h）
        if (rtGetDeviceInfoFn_ != nullptr) {
            constexpr int32_t kInfoSdid = 26;
            constexpr int32_t kInfoServerId = 27;
            constexpr int32_t kInfoSuperPodId = 28;
            int64_t v = 0;
            uint32_t sdid = 0, serverIdx = 0, superPodIdx = 0;
            if (rtGetDeviceInfoFn_(static_cast<uint32_t>(deviceId), 0, kInfoSdid, &v) == 0) {
                sdid = static_cast<uint32_t>(v);
            }
            if (rtGetDeviceInfoFn_(static_cast<uint32_t>(deviceId), 0, kInfoServerId, &v) == 0) {
                serverIdx = static_cast<uint32_t>(v);
            }
            if (rtGetDeviceInfoFn_(static_cast<uint32_t>(deviceId), 0, kInfoSuperPodId, &v) == 0) {
                superPodIdx = static_cast<uint32_t>(v);
            }
            localDesc_.loc.device.superDevId = sdid;
            localDesc_.loc.device.serverIdx = serverIdx;
            localDesc_.loc.device.superPodIdx = superPodIdx;
            printf("[emsd] topology: deviceId=%d phyId=%d sdid=%u serverIdx=%u superPodIdx=%u\n", deviceId, phyId,
                   sdid, serverIdx, superPodIdx);
        }
    }

    HcommResult ret = api_.HcommEndpointCreate(&localDesc_, &endpoint_);
    if (ret != 0 || endpoint_ == nullptr) {
        printf("[emsd][ERROR] HcommEndpointCreate ret=%d, loc=%s addr=%s\n", ret, hostLoc ? "host" : "device",
               hostLoc ? hostAddr.c_str() : EmsEidToString(eid).c_str());
        return -1;
    }
    printf("[emsd] EndpointCreate ok, loc=%s handle=%p addr=%s\n", hostLoc ? "host" : "device", endpoint_,
           hostLoc ? hostAddr.c_str() : EmsEidToString(eid).c_str());
    return 0;
}

void EmsdHcomm::Close()
{
    for (uint32_t i = 0; i < KV_MAX_SEG_NUM; i++) {
        if (segs_[i].used) {
            (void)SegRelease(i, true);
        }
    }
    if (endpoint_ != nullptr) {
        (void)api_.HcommEndpointDestroy(endpoint_);
        endpoint_ = nullptr;
    }
}

int EmsdHcomm::SegRegister(uint32_t segIndex, uint32_t poolId, void *base, uint64_t size, uint16_t hcommListenPort,
                           const std::string &channelTag)
{
    if (segIndex >= KV_MAX_SEG_NUM || base == nullptr || size == 0) {
        return KV_ERR_PARAM;
    }
    EmsdSeg &seg = segs_[segIndex];
    if (seg.used) {
        return KV_ERR_STATE;
    }
    char memTag[64];
    (void)snprintf(memTag, sizeof(memTag), "kv_seg_%u_%u", poolId, segIndex);
    CommMem commMem{};
    // DEVICE 内存（HBM）注册：AIV 通道 exchangeAllMems 握手交换 DVA 域 mem key，
    // 裸 mmap host 页无 DVA 时握手无法完成（联调实证：双端 60s IN_PROGRESS、server GetStatus 报错）。
    // 对齐 aiv_urma_poc（aclrtMalloc + COMM_MEM_TYPE_DEVICE）；host DRAM 池待跨模式包/VA manager 后置。
    commMem.type = COMM_MEM_TYPE_DEVICE;
    commMem.addr = base;
    commMem.size = size;
    HcommResult ret = api_.HcommMemReg(endpoint_, memTag, &commMem, &seg.memHandle);
    if (ret != 0 || seg.memHandle == nullptr) {
        printf("[emsd][ERROR] MemReg failed, poolId=%u segIndex=%u addr=%p size=%llu ret=%d\n", poolId, segIndex,
               base, (unsigned long long)size, ret);
        return KV_ERR_HCOMM;
    }
    void *desc = nullptr;
    uint32_t descLen = 0;
    ret = api_.HcommMemExport(endpoint_, seg.memHandle, &desc, &descLen);
    if (ret != 0 || desc == nullptr || descLen == 0 || descLen > KV_MAX_MEM_DESC) {
        printf("[emsd][ERROR] MemExport failed, poolId=%u segIndex=%u ret=%d descLen=%u\n", poolId, segIndex, ret,
               descLen);
        return KV_ERR_HCOMM;
    }
    seg.memDesc.assign(static_cast<uint8_t *>(desc), static_cast<uint8_t *>(desc) + descLen);
    seg.used = true;
    seg.poolId = poolId;
    seg.base = base;
    seg.size = size;
    seg.hcommListenPort = hcommListenPort;
    seg.channelTag = channelTag;
    printf("[emsd] SegRegister ok, poolId=%u segIndex=%u base=%p size=%llu memDescLen=%u\n", poolId, segIndex, base,
           (unsigned long long)size, descLen);
    return KV_OK;
}

int EmsdHcomm::LinkServer(uint32_t segIndex, const uint8_t *peerDesc, uint32_t peerDescLen,
                          const uint8_t *peerMemDesc, uint32_t peerMemDescLen, uint32_t readyTimeoutSec)
{
    if (segIndex >= KV_MAX_SEG_NUM || !segs_[segIndex].used) {
        return KV_ERR_NOT_FOUND;
    }
    EmsdSeg &seg = segs_[segIndex];
    if (seg.linked) {
        return KV_ERR_STATE;
    }
    if (peerDescLen != sizeof(EndpointDesc)) {
        return KV_ERR_PROTO;
    }
    // 导入对端（框架/NPU 侧）内存描述：本端 host 大块对框架可见，框架侧内存对本端可见
    // （数据面单边直驱仅要求对端可写本段，import 为交换协议完整性保留）
    CommMem remoteMem{};
    HcommResult ret = api_.HcommMemImport(endpoint_, peerMemDesc, peerMemDescLen, &remoteMem);
    if (ret != 0) {
        printf("[emsd][ERROR] MemImport failed, poolId=%u segIndex=%u ret=%d\n", seg.poolId, segIndex, ret);
        return KV_ERR_HCOMM;
    }
    HcommChannelDesc chDesc;
    if (HcommChannelDescInit(&chDesc, 1) != 0) {
        return KV_ERR_HCOMM;
    }
    chDesc.remoteEndpoint = *reinterpret_cast<const EndpointDesc *>(peerDesc);
    chDesc.notifyNum = 1;
    chDesc.exchangeAllMems = true;  // 每段单内存
    chDesc.role = HCOMM_SOCKET_ROLE_SERVER;
    chDesc.port = seg.hcommListenPort;
    chDesc.qos = 0;
    chDesc.channelName = const_cast<char *>(seg.channelTag.c_str());
    ret = api_.HcommChannelCreate(endpoint_, COMM_ENGINE_AIV, &chDesc, 1, &seg.channel);
    if (ret != 0 || seg.channel == 0) {
        printf("[emsd][ERROR] ChannelCreate(AIV,server) failed, poolId=%u segIndex=%u port=%u ret=%d\n", seg.poolId,
               segIndex, seg.hcommListenPort, ret);
        return KV_ERR_HCOMM;
    }
    // 创建即返回：server 侧 READY 依赖框架侧 client 通道建链完成后由 hcomm 内部交换推进，
    // 此处阻塞轮询会与 client 侧建链互等（client 需先收到本响应才会发起 ChannelCreate）。
    // READY 以框架侧 client 通道 GetStatus 为准。readyTimeoutSec 保留给联调期诊断开关。
    seg.linked = true;
    printf("[emsd] LinkServer created, poolId=%u segIndex=%u channel=0x%llx\n", seg.poolId, segIndex,
           (unsigned long long)seg.channel);
    // 诊断（联调期可回退）：后台轮询 server 侧通道状态，观察握手推进到哪一步。
    // 状态语义（对齐 MemFabric hcomm_api_wrapper）：READY=0 / IN_PROGRESS=1 / FAILED=2 / TIMEOUT=3。
    {
        ChannelHandle ch = seg.channel;
        const uint32_t poolId = seg.poolId, segIdx = segIndex;
        std::thread([this, ch, poolId, segIdx] {
            (void)BindDeviceCtx();  // GetStatus 依赖线程私有 device 上下文
            for (int i = 0; i < 12; ++i) {
                int32_t status = -1;
                const int r = api_.HcommChannelGetStatus(&ch, 1, &status);
                printf("[emsd] chStatus poolId=%u segIndex=%u r=%d status=%d (0=READY 1=IN_PROGRESS 2=FAILED 3=TIMEOUT)\n",
                       poolId, segIdx, r, status);
                fflush(stdout);
                if (r != 0 || status == 0 || status == 2 || status == 3) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }).detach();
    }
    return KV_OK;
}

int EmsdHcomm::BindDeviceCtx()
{
    if (hostLoc_) {
        return 0;  // host 模式不依赖 NPU runtime 上下文
    }
    if (aclrtSetDeviceFn_ == nullptr || aclrtSetDeviceFn_(deviceId_) != 0) {
        printf("[emsd][ERROR] BindDeviceCtx failed, deviceId=%d\n", deviceId_);
        return -1;
    }
    return 0;
}

int EmsdHcomm::SegRelease(uint32_t segIndex, bool destroyChannel)
{
    if (segIndex >= KV_MAX_SEG_NUM || !segs_[segIndex].used) {
        return KV_ERR_NOT_FOUND;
    }
    EmsdSeg &seg = segs_[segIndex];
    printf("[emsd] SegRelease begin, poolId=%u segIndex=%u linked=%d ch=0x%llx\n", seg.poolId, segIndex,
           seg.linked ? 1 : 0, (unsigned long long)seg.channel);
    if (destroyChannel && seg.linked && seg.channel != 0) {
        printf("[emsd] SegRelease HcommChannelDestroy begin, segIndex=%u\n", segIndex);
        (void)api_.HcommChannelDestroy(&seg.channel, 1);
        printf("[emsd] SegRelease HcommChannelDestroy done, segIndex=%u\n", segIndex);
    }
    if (seg.memHandle != nullptr) {
        printf("[emsd] SegRelease HcommMemUnreg begin, segIndex=%u\n", segIndex);
        (void)api_.HcommMemUnreg(endpoint_, seg.memHandle);
        printf("[emsd] SegRelease HcommMemUnreg done, segIndex=%u\n", segIndex);
    }
    printf("[emsd] SegRelease done, poolId=%u segIndex=%u\n", seg.poolId, segIndex);
    seg = EmsdSeg{};
    return KV_OK;
}

const EmsdSeg *EmsdHcomm::GetSeg(uint32_t segIndex) const
{
    if (segIndex >= KV_MAX_SEG_NUM || !segs_[segIndex].used) {
        return nullptr;
    }
    return &segs_[segIndex];
}
