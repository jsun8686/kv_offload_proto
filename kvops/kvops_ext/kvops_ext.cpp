/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * kvops_ext：AIV 批量算子 launch 封装（fatbin 显式加载链，PoC 方案 B）。
 *   aclrtSetDevice -> aclrtBinaryLoadFromFile(LAZY_LOAD) -> aclrtBinaryGetFunction(原名)
 *   -> aclrtLaunchKernelWithHostArgs（SCHEM_MODE=1 + TIMEOUT_US + ENGINE_TYPE=AIV）
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
void *g_fnGather = nullptr;
void *g_fnScatterAll = nullptr;
void *g_fnScatterRows = nullptr;
void *g_overflowBuf = nullptr;  // 8B overflow 尾槽 device buffer
bool g_inited = false;

template <typename T>
T Sym(const char *name)
{
    auto fn = reinterpret_cast<T>(dlsym(rtLib, name));
    if (fn == nullptr) {
        throw std::runtime_error(std::string("dlsym ") + name + " failed: " + dlerror());
    }
    return fn;
}

int LoadFatbin(const std::string &path)
{
    BinOpt opt{};
    opt.type = BINARY_LOAD_OPT_LAZY_LOAD;
    opt.value.isLazyLoad = 1;
    BinOpts opts{&opt, 1};
    int32_t ret = rt.aclrtBinaryLoadFromFile(path.c_str(), &opts, &g_binary);
    if (ret != 0 || g_binary == nullptr) {
        throw std::runtime_error("BinaryLoadFromFile ret=" + std::to_string(ret) + " path=" + path);
    }
    const char *names[3] = {"kvc_gather_rows", "kvc_scatter_rows_alllayer", "kvc_scatter_rows"};
    void **fns[3] = {&g_fnGather, &g_fnScatterAll, &g_fnScatterRows};
    for (int i = 0; i < 3; i++) {
        ret = rt.aclrtBinaryGetFunction(g_binary, names[i], fns[i]);
        if (ret != 0 || *fns[i] == nullptr) {
            throw std::runtime_error(std::string("GetFunction ") + names[i] + " ret=" + std::to_string(ret));
        }
    }
    return 0;
}

// 通用 launch：args 结构体 + 8B overflow 尾槽
void LaunchCommon(void *fn, const void *argsStruct, size_t argsSize, uint32_t blockDim, uint64_t stream)
{
    uint8_t buf[160];
    if (argsSize + 8 > sizeof(buf)) {
        throw std::runtime_error("args too large");
    }
    memcpy(buf, argsStruct, argsSize);
    memcpy(buf + argsSize, &g_overflowBuf, 8);  // 尾槽 = 8B device buffer 地址

    LaunchAttr attrs[3]{};
    attrs[0].id = LAUNCH_ATTR_SCHEM_MODE;
    attrs[0].value.schemMode = 1;
    attrs[1].id = LAUNCH_ATTR_TIMEOUT_US;
    attrs[1].value.timeoutUs = {60 * 1000 * 1000, 0};
    attrs[2].id = LAUNCH_ATTR_ENGINE_TYPE;
    attrs[2].value.engineType = ENGINE_TYPE_AIV;
    LaunchCfg cfg{attrs, 3};

    int32_t ret = rt.aclrtLaunchKernelWithHostArgs(fn, blockDim, reinterpret_cast<void *>(stream), &cfg, buf,
                                                   argsSize + 8, nullptr, 0);
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
        throw std::runtime_error("kvops already inited");
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
    (void)rt.aclrtMemset(g_overflowBuf, 8, 0, 8);
    LoadFatbin(fatbinPath);
    g_inited = true;
}

// 分配 status 槽（64B/槽），返回 GM 地址
uint64_t kv_alloc_status(int count)
{
    EnsureInited();
    void *buf = nullptr;
    if (rt.aclrtMalloc(&buf, static_cast<size_t>(count) * 64, 0) != 0 || buf == nullptr) {
        throw std::runtime_error("alloc status failed");
    }
    (void)rt.aclrtMemset(buf, static_cast<size_t>(count) * 64, 0, static_cast<size_t>(count) * 64);
    return reinterpret_cast<uint64_t>(buf);
}

// 轮询某槽：等 (doneSeq==1)，返回 errCode（低 32 位）；GIL 释放
int64_t kv_poll_status(uint64_t statusAddr, int64_t timeoutUs)
{
    EnsureInited();
    {
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
}

void kv_import_seg(uint64_t channel, uint64_t segBase, int layerId, uint32_t perLayerSegRows, uint32_t rowBytes,
                   uint64_t indicesHost, uint64_t indicesDst, uint64_t numRealAddr, uint64_t dstLayerBase,
                   uint32_t maxBatch, uint32_t stripeRows, uint32_t nodeNum, uint32_t myNode, uint64_t status,
                   uint64_t stream, uint32_t blockDim)
{
    EnsureInited();
    KvcGatherArgs args{};
    args.channel = channel;
    args.segBase = segBase;
    args.indicesHost = indicesHost;
    args.indicesDst = indicesDst;
    args.numRealAddr = numRealAddr;
    args.dstLayerBase = dstLayerBase;
    args.status = status;
    args.layerOffsetRows = static_cast<uint32_t>(layerId) * perLayerSegRows;
    args.perLayerSegRows = perLayerSegRows;
    args.rowBytes = rowBytes;
    args.stripeRows = stripeRows;
    args.nodeNum = nodeNum;
    args.myNode = myNode;
    args.maxBatch = maxBatch;
    // 状态槽清零（launch 前）；图模式下由 capture 记录，replay 前由框架清理
    (void)rt.aclrtMemset(reinterpret_cast<void *>(status), 8, 0, 8);
    LaunchCommon(g_fnGather, &args, sizeof(args), blockDim, stream);
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
    (void)rt.aclrtMemset(reinterpret_cast<void *>(status), 8, 0, 8);
    LaunchCommon(g_fnScatterAll, &args, sizeof(args), blockDim, stream);
}

void kv_export_rows(uint64_t channel, uint64_t segBase, int layerId, uint32_t perLayerSegRows, uint32_t rowBytes,
                    uint64_t indicesHost, uint64_t indicesDst, uint64_t numRealAddr, uint64_t srcLayerBase,
                    uint32_t maxBatch, uint32_t stripeRows, uint32_t nodeNum, uint32_t myNode, uint64_t status,
                    uint64_t stream, uint32_t blockDim)
{
    EnsureInited();
    KvcGatherArgs args{};
    args.channel = channel;
    args.segBase = segBase;
    args.indicesHost = indicesHost;
    args.indicesDst = indicesDst;
    args.numRealAddr = numRealAddr;
    args.dstLayerBase = srcLayerBase;  // 单层导出复用同结构：此字段为 HBM 源层基址
    args.status = status;
    args.layerOffsetRows = static_cast<uint32_t>(layerId) * perLayerSegRows;
    args.perLayerSegRows = perLayerSegRows;
    args.rowBytes = rowBytes;
    args.stripeRows = stripeRows;
    args.nodeNum = nodeNum;
    args.myNode = myNode;
    args.maxBatch = maxBatch;
    (void)rt.aclrtMemset(reinterpret_cast<void *>(status), 8, 0, 8);
    LaunchCommon(g_fnScatterRows, &args, sizeof(args), blockDim, stream);
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
    m.def("init", &kv_init, py::arg("device_id"), py::arg("fatbin_path"),
          py::arg("rt_lib") = "libascendcl.so");
    m.def("alloc_status", &kv_alloc_status, py::arg("count"));
    m.def("poll_status", &kv_poll_status, py::arg("status_addr"), py::arg("timeout_us") = -1);
    m.def("import_seg", &kv_import_seg, py::arg("channel"), py::arg("seg_base"), py::arg("layer_id"),
          py::arg("per_layer_seg_rows"), py::arg("row_bytes"), py::arg("indices_host"), py::arg("indices_dst"),
          py::arg("num_real_addr"), py::arg("dst_layer_base"), py::arg("max_batch"), py::arg("stripe_rows"),
          py::arg("node_num"), py::arg("my_node"), py::arg("status"), py::arg("stream") = 0,
          py::arg("block_dim") = 1);
    m.def("export_seg_alllayer", &kv_export_seg_alllayer, py::arg("channel"), py::arg("seg_base"),
          py::arg("per_layer_seg_rows"), py::arg("row_bytes"), py::arg("row_lo"), py::arg("row_cnt"),
          py::arg("layer_num"), py::arg("src_layer0_base"), py::arg("src_layer_stride"), py::arg("stripe_rows"),
          py::arg("node_num"), py::arg("my_node"), py::arg("status"), py::arg("stream") = 0,
          py::arg("block_dim") = 1);
    m.def("export_rows", &kv_export_rows, py::arg("channel"), py::arg("seg_base"), py::arg("layer_id"),
          py::arg("per_layer_seg_rows"), py::arg("row_bytes"), py::arg("indices_host"), py::arg("indices_dst"),
          py::arg("num_real_addr"), py::arg("src_layer_base"), py::arg("max_batch"), py::arg("stripe_rows"),
          py::arg("node_num"), py::arg("my_node"), py::arg("status"), py::arg("stream") = 0,
          py::arg("block_dim") = 1);
    m.def("close", &kv_close);
}
