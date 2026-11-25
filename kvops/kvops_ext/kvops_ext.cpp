/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * kvops_ext：AIV 算子 launch 封装（fatbin 显式加载链，PoC 方案 B）。
 *   内核：dsa_gather_rmt（DSA miss 记录回源，纯远端 ReadNbi）
 *         kvc_scatter_rows_alllayer（全层导出，EmsPool.export_full_rmt_pool 数据链依赖）
 * launch 均不在内部同步/读 device 内存（图捕获安全）；status 槽由调用方管理。
 */

#include <pybind11/pybind11.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kvc_args.h"

namespace py = pybind11;

namespace {
// aclrt 常量（对齐 CANN acl_rt.h）
constexpr int32_t LAUNCH_ATTR_SCHEM_MODE = 1;
constexpr int32_t LAUNCH_ATTR_ENGINE_TYPE = 3;
constexpr int32_t LAUNCH_ATTR_TIMEOUT_US = 8;
constexpr int32_t BINARY_LOAD_OPT_LAZY_LOAD = 1;
constexpr int32_t ENGINE_TYPE_AIV = 1;

union BinOptVal {
    uint32_t isLazyLoad;
    uint32_t rsv[4];
};
struct BinOpt {
    int32_t type;
    BinOptVal value;
};
struct BinOpts {
    BinOpt *options;
    size_t numOpt;
};
union LaunchAttrVal {
    uint8_t schemMode;
    int32_t engineType;
    struct { uint32_t lo, hi; } timeoutUs;
    uint32_t rsv[4];
};
struct LaunchAttr {
    int32_t id;
    LaunchAttrVal value;
};
struct LaunchCfg {
    LaunchAttr *attrs;
    size_t numAttrs;
};

struct RtApi {
    int32_t (*aclrtSetDevice)(int32_t);
    int32_t (*aclrtMalloc)(void **, size_t, int32_t);
    int32_t (*aclrtFree)(void *);
    int32_t (*aclrtMemset)(void *, size_t, int32_t, size_t);
    int32_t (*aclrtMemcpy)(void *, size_t, const void *, size_t, int32_t);
    int32_t (*aclrtBinaryLoadFromFile)(const char *, BinOpts *, void **);
    int32_t (*aclrtBinaryUnLoad)(void *);
    int32_t (*aclrtBinaryGetFunction)(const void *, const char *, void **);
    int32_t (*aclrtLaunchKernelWithHostArgs)(void *, uint32_t, void *, LaunchCfg *, void *, size_t, void *, size_t);
};

RtApi rt{};
void *rtLib = nullptr;
void *g_binary = nullptr;
void *g_fnScatterAll = nullptr;
void *g_fnDsaGatherRmt = nullptr;
void *g_overflowBuf = nullptr;
bool g_inited = false;
int g_deviceId = -1;
std::string g_fatbinPath;

template <typename T>
T Sym(const char *name)
{
    auto fn = reinterpret_cast<T>(dlsym(rtLib, name));
    if (fn == nullptr) {
        throw std::runtime_error(std::string("dlsym ") + name + " failed: " + dlerror());
    }
    return fn;
}

void LaunchCommon(void *fn, const void *argsStruct, size_t argsSize, uint32_t blockDim, uint64_t stream)
{
    uint8_t buf[512];
    // 尾槽（工具链为 kernel 恒追加的 8B device buffer 地址参数）按 8B 对齐放置：
    // 结构体 sizeof 非 8 倍数时（如 KvcScatterAllArgs=76B）直接接在末尾会把尾槽放到
    // 非对齐偏移，kernel 侧按对齐 ABI 读到的就是垃圾地址
    const size_t alignedSize = (argsSize + 7U) & ~static_cast<size_t>(7U);
    if (alignedSize + 8 > sizeof(buf)) {
        throw std::runtime_error("args too large");
    }
    memcpy(buf, argsStruct, argsSize);
    if (alignedSize > argsSize) {
        memset(buf + argsSize, 0, alignedSize - argsSize);
    }
    memcpy(buf + alignedSize, &g_overflowBuf, 8);  // 尾槽 = 8B device buffer 地址

    LaunchAttr attrs[3]{};
    attrs[0].id = LAUNCH_ATTR_SCHEM_MODE;
    attrs[0].value.schemMode = 1;
    attrs[1].id = LAUNCH_ATTR_TIMEOUT_US;
    attrs[1].value.timeoutUs = {60 * 1000 * 1000, 0};
    attrs[2].id = LAUNCH_ATTR_ENGINE_TYPE;
    attrs[2].value.engineType = ENGINE_TYPE_AIV;
    LaunchCfg cfg{attrs, 3};

    int32_t ret = rt.aclrtLaunchKernelWithHostArgs(fn, blockDim, reinterpret_cast<void *>(stream), &cfg, buf,
                                                   alignedSize + 8, nullptr, 0);
    if (ret != 0) {
        throw std::runtime_error("LaunchKernelWithHostArgs ret=" + std::to_string(ret));
    }
}

void EnsureInited()
{
    if (!g_inited) {
        throw std::runtime_error("kvops not inited");
    }
}
}  // namespace

void kv_init(int deviceId, const std::string &fatbinPath, const std::string &rtLibPath)
{
    if (g_inited) {
        // 幂等：同一进程内 local 测试路径与 EmsPool.create_pool 都会调 init；
        // 同 device+同 fatbin 直接复用已加载的 binary/function（重复 SetDevice/Malloc/BinaryLoad
        // 只会报错或泄漏）。参数不一致才是真错误，报错里带上已持有的值便于定位。
        if (deviceId == g_deviceId && fatbinPath == g_fatbinPath) {
            printf("[kvops] init skipped: already inited (device=%d)\n", deviceId);
            return;
        }
        throw std::runtime_error("kvops already inited with device=" + std::to_string(g_deviceId) +
                                 " fatbin=" + g_fatbinPath + ", refusing re-init with device=" +
                                 std::to_string(deviceId) + " fatbin=" + fatbinPath);
    }
    rtLib = dlopen(rtLibPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (rtLib == nullptr) {
        throw std::runtime_error(std::string("dlopen rt failed: ") + dlerror());
    }
    rt.aclrtSetDevice = Sym<decltype(rt.aclrtSetDevice)>("aclrtSetDevice");
    rt.aclrtMalloc = Sym<decltype(rt.aclrtMalloc)>("aclrtMalloc");
    rt.aclrtFree = Sym<decltype(rt.aclrtFree)>("aclrtFree");
    rt.aclrtMemset = Sym<decltype(rt.aclrtMemset)>("aclrtMemset");
    rt.aclrtMemcpy = Sym<decltype(rt.aclrtMemcpy)>("aclrtMemcpy");
    rt.aclrtBinaryLoadFromFile = Sym<decltype(rt.aclrtBinaryLoadFromFile)>("aclrtBinaryLoadFromFile");
    rt.aclrtBinaryUnLoad = Sym<decltype(rt.aclrtBinaryUnLoad)>("aclrtBinaryUnLoad");
    rt.aclrtBinaryGetFunction = Sym<decltype(rt.aclrtBinaryGetFunction)>("aclrtBinaryGetFunction");
    rt.aclrtLaunchKernelWithHostArgs =
        Sym<decltype(rt.aclrtLaunchKernelWithHostArgs)>("aclrtLaunchKernelWithHostArgs");

    if (rt.aclrtSetDevice(deviceId) != 0) {
        throw std::runtime_error("aclrtSetDevice failed");
    }
    if (rt.aclrtMalloc(&g_overflowBuf, 8, 0) != 0 || g_overflowBuf == nullptr) {
        throw std::runtime_error("alloc overflow buf failed");
    }
    if (rt.aclrtMemset(g_overflowBuf, 8, 0, 8) != 0) {
        throw std::runtime_error("memset overflow buf failed");
    }

    BinOpt opt{};
    opt.type = BINARY_LOAD_OPT_LAZY_LOAD;
    opt.value.isLazyLoad = 1;
    BinOpts opts{&opt, 1};
    int32_t ret = rt.aclrtBinaryLoadFromFile(fatbinPath.c_str(), &opts, &g_binary);
    if (ret != 0 || g_binary == nullptr) {
        throw std::runtime_error("BinaryLoadFromFile ret=" + std::to_string(ret) + " path=" + fatbinPath);
    }
    const char *names[2] = {"kvc_scatter_rows_alllayer", "dsa_gather_rmt"};
    void **fns[2] = {&g_fnScatterAll, &g_fnDsaGatherRmt};
    for (int i = 0; i < 2; i++) {
        ret = rt.aclrtBinaryGetFunction(g_binary, names[i], fns[i]);
        if (ret != 0 || *fns[i] == nullptr) {
            throw std::runtime_error(std::string("GetFunction ") + names[i] + " ret=" + std::to_string(ret));
        }
    }
    g_inited = true;
    g_deviceId = deviceId;
    g_fatbinPath = fatbinPath;
}

uint64_t kv_alloc_status(int count)
{
    EnsureInited();
    void *buf = nullptr;
    if (rt.aclrtMalloc(&buf, static_cast<size_t>(count) * 64, 0) != 0 || buf == nullptr) {
        throw std::runtime_error("alloc status failed");
    }
    const size_t statusBytes = static_cast<size_t>(count) * 64;
    if (rt.aclrtMemset(buf, statusBytes, 0, statusBytes) != 0) {
        // memset 失败当场抛出：地址非法/越界在分配阶段即暴露（曾因 (void) 吞错拖到 poll 才发现）
        throw std::runtime_error("alloc_status memset failed, addr=" + std::to_string(reinterpret_cast<uint64_t>(buf)) +
                                 " bytes=" + std::to_string(statusBytes));
    }
    return reinterpret_cast<uint64_t>(buf);
}

int64_t kv_poll_status(uint64_t statusAddr, int64_t timeoutUs)
{
    EnsureInited();
    py::gil_scoped_release release;
    const uint64_t deadline = static_cast<uint64_t>(timeoutUs < 0 ? INT64_MAX : timeoutUs);
    uint64_t waited = 0;
    while (true) {
        uint64_t val = 0;
        if (rt.aclrtMemcpy(&val, 8, reinterpret_cast<const void *>(statusAddr), 8, 3 /*D2O*/) != 0) {
            return -0x1000;  // memcpy 失败
        }
        if ((val >> 32) == 1) {
            return static_cast<int64_t>(val & 0xFFFFFFFF);
        }
        if (waited >= deadline) {
            return -0x2000;  // 超时
        }
        usleep(100);
        waited += 100;
    }
}

// 读回状态槽当前值（超时诊断用：阶段码 0xE2/0xE3/0xE4 定位内核卡在哪一步）
uint64_t kv_peek_status(uint64_t statusAddr)
{
    EnsureInited();
    uint64_t v = 0;
    if (rt.aclrtMemcpy(&v, 8, reinterpret_cast<const void *>(statusAddr), 8, 3 /*D2O*/) != 0) {
        return 0xFFFFFFFFFFFFFFFFULL;
    }
    return v;
}

void kv_export_seg_alllayer(uint64_t channel, uint64_t segBase, uint32_t perLayerSegRows, uint32_t rowBytes,
                            uint32_t rowLo, uint32_t rowCnt, uint32_t layerNum, uint64_t srcLayer0Base,
                            uint64_t srcLayerStride, uint32_t stripeRows, uint32_t nodeNum, uint32_t myNode,
                            uint64_t status, uint64_t stream, uint32_t blockDim)
{
    EnsureInited();
    KvcScatterAllArgs args{};
    args.channel = channel;
    args.segBase = segBase;
    args.srcLayer0Base = srcLayer0Base;
    args.srcLayerStride = srcLayerStride;
    args.status = status;
    args.perLayerSegRows = perLayerSegRows;
    args.rowBytes = rowBytes;
    args.rowLo = rowLo;
    args.rowCnt = rowCnt;
    args.layerNum = layerNum;
    args.stripeRows = stripeRows;
    args.nodeNum = nodeNum;
    args.myNode = myNode;
    if (rt.aclrtMemset(reinterpret_cast<void *>(status), blockDim * 64, 0, blockDim * 64) != 0) {
        throw std::runtime_error("export_seg_alllayer status memset failed, status=" + std::to_string(status) +
                                 " bytes=" + std::to_string(blockDim * 64));
    }
    LaunchCommon(g_fnScatterAll, &args, sizeof(args), blockDim, stream);
}

void kv_dsa_gather_rmt(uint64_t residentKv, uint64_t residentRope, uint64_t residentBlockTable,
                       uint64_t hostBlockTable, uint64_t requestPoolIndices, uint64_t topkTokenPositions,
                       uint64_t residentSlots, uint64_t missMask, uint64_t rmtChannel, uint64_t rmtKvAddr,
                       uint64_t rmtRopeAddr, uint64_t status, uint32_t numRequests, uint32_t queryCount,
                       uint32_t blockSize, uint32_t kvDim, uint32_t ropeDim, uint32_t kvElemSize,
                       uint32_t ropeElemSize, uint32_t hostPoolCapacity, uint32_t hostTableWidth,
                       uint32_t residentTableWidth, uint32_t hostBlkNum, uint32_t residentBlkNum,
                       uint64_t stream, uint32_t blockDim)
{
    EnsureInited();
    if (blockSize == 0 || queryCount == 0 || numRequests == 0) {
        throw std::runtime_error("dsa_gather_rmt: blockSize/queryCount/numRequests must be positive");
    }
    if (kvElemSize == 0 || ropeElemSize == 0 || kvElemSize > 2 || ropeElemSize > 2) {
        throw std::runtime_error("dsa_gather_rmt: elem size must be 1 (int8) or 2 (fp16/bf16)");
    }
    if (kvDim * kvElemSize % 32 != 0 || ropeDim * ropeElemSize % 32 != 0) {
        throw std::runtime_error("dsa_gather_rmt: kv/rope record bytes must be 32B aligned");
    }
    DsaGatherRmtArgs args{};
    args.residentKv = residentKv;
    args.residentRope = residentRope;
    args.residentBlockTable = residentBlockTable;
    args.hostBlockTable = hostBlockTable;
    args.requestPoolIndices = requestPoolIndices;
    args.topkTokenPositions = topkTokenPositions;
    args.residentSlots = residentSlots;
    args.missMask = missMask;
    args.rmtChannel = rmtChannel;
    args.rmtKvAddr = rmtKvAddr;
    args.rmtRopeAddr = rmtRopeAddr;
    args.status = status;
    args.numRequests = numRequests;
    args.queryCount = queryCount;
    args.blockSize = blockSize;
    args.kvDim = kvDim;
    args.ropeDim = ropeDim;
    args.kvElemSize = kvElemSize;
    args.ropeElemSize = ropeElemSize;
    args.hostPoolCapacity = hostPoolCapacity;
    args.hostTableWidth = hostTableWidth;
    args.residentTableWidth = residentTableWidth;
    args.hostBlkNum = hostBlkNum;
    args.residentBlkNum = residentBlkNum;
    args.residentInit = 0;  // V1 dense decode 布局
    LaunchCommon(g_fnDsaGatherRmt, &args, sizeof(args), blockDim, stream);
}

void kv_close()
{
    if (g_binary != nullptr) {
        (void)rt.aclrtBinaryUnLoad(g_binary);
        g_binary = nullptr;
    }
    if (g_overflowBuf != nullptr) {
        (void)rt.aclrtFree(g_overflowBuf);
        g_overflowBuf = nullptr;
    }
    g_inited = false;
}

PYBIND11_MODULE(_kvops, m)
{
    m.doc() = "kvops AIV launch ext (fatbin explicit loading chain)";
    m.def("init", &kv_init, py::arg("device_id"), py::arg("fatbin_path"), py::arg("rt_lib") = "libascendcl.so");
    m.def("alloc_status", &kv_alloc_status, py::arg("count"));
    m.def("poll_status", &kv_poll_status, py::arg("status_addr"), py::arg("timeout_us") = -1);
    m.def("peek_status", &kv_peek_status, py::arg("status_addr"));
    m.def("export_seg_alllayer", &kv_export_seg_alllayer, py::arg("channel"), py::arg("seg_base"),
          py::arg("per_layer_seg_rows"), py::arg("row_bytes"), py::arg("row_lo"), py::arg("row_cnt"),
          py::arg("layer_num"), py::arg("src_layer0_base"), py::arg("src_layer_stride"), py::arg("stripe_rows"),
          py::arg("node_num"), py::arg("my_node"), py::arg("status"), py::arg("stream") = 0,
          py::arg("block_dim") = 1);
    m.def("dsa_gather_rmt", &kv_dsa_gather_rmt, py::arg("resident_kv"), py::arg("resident_rope"),
          py::arg("resident_block_table"), py::arg("host_block_table"), py::arg("request_pool_indices"),
          py::arg("topk_token_positions"), py::arg("resident_slots"), py::arg("miss_mask"),
          py::arg("rmt_channel"), py::arg("rmt_kv_addr"), py::arg("rmt_rope_addr"), py::arg("status"),
          py::arg("num_requests"), py::arg("query_count"), py::arg("block_size"), py::arg("kv_dim"),
          py::arg("rope_dim"), py::arg("kv_elem_size"), py::arg("rope_elem_size"),
          py::arg("host_pool_capacity"), py::arg("host_table_width"), py::arg("resident_table_width"),
          py::arg("host_blk_num"), py::arg("resident_blk_num"), py::arg("stream") = 0,
          py::arg("block_dim") = 1);
    m.def("close", &kv_close);
}
