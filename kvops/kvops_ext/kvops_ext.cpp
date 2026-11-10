/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * kvops_ext：AIV 算子 launch 封装（fatbin 显式加载链，PoC 方案 B）。
 *   内核：kvc_gather_rows（稠密导入）/ kvc_scatter_rows_alllayer（全层导出）
 *         hisp_sieve_update（SIEVE miss 识别，哈希-only）/ hisp_sieve_ht_init（哈希表构建）
 * launch 均不在内部同步/读 device 内存（图捕获安全）；status 槽由调用方管理。
 * sieve 派生公式与 sglang-0730 hisparse_lru_host.cpp 严格一致：
 *   htSize = 2 的幂 ≥ 2×(padded+topK)（装载因子 ≤0.5）
 *   reqsPerCore 对齐 64/gcd(topK,64)（每 block 输出段 64B cache line 对齐）
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
constexpr int64_t UB_BUDGET_BYTES = 176 * 1024;  // 192KB/核留余量

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
void *g_fnSieveUpdate = nullptr;
void *g_fnSieveHtInit = nullptr;
void *g_overflowBuf = nullptr;
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

int64_t Gcd(int64_t a, int64_t b)
{
    while (b != 0) {
        int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

void LaunchCommon(void *fn, const void *argsStruct, size_t argsSize, uint32_t blockDim, uint64_t stream)
{
    uint8_t buf[160];
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
    const char *names[4] = {"kvc_gather_rows", "kvc_scatter_rows_alllayer", "hisp_sieve_update",
                            "hisp_sieve_ht_init"};
    void **fns[4] = {&g_fnGather, &g_fnScatterAll, &g_fnSieveUpdate, &g_fnSieveHtInit};
    for (int i = 0; i < 4; i++) {
        ret = rt.aclrtBinaryGetFunction(g_binary, names[i], fns[i]);
        if (ret != 0 || *fns[i] == nullptr) {
            throw std::runtime_error(std::string("GetFunction ") + names[i] + " ret=" + std::to_string(ret));
        }
    }
    g_inited = true;
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

// ── 批量导入（稠密）：N 段循环由 Python 侧逐段调用 ──
void kv_import_seg(uint64_t channel, uint64_t segBase, uint32_t layerOffsetRows, uint32_t perLayerSegRows,
                   uint32_t rowBytes, uint64_t topkIndices, uint64_t reqPoolIndices, uint64_t reqHostBase,
                   uint64_t topKDeviceSlots, uint64_t isMiss, uint64_t numReal, uint64_t hotRowBase,
                   uint32_t topK, uint32_t maxNumReqs, uint32_t stripeRows, uint32_t nodeNum, uint32_t myNode,
                   uint64_t status, uint64_t stream, uint32_t blockDim)
{
    EnsureInited();
    if (blockDim > 8) {
        throw std::runtime_error("import blockDim must be <= 8 (status slot layout)");
    }
    KvcGatherArgs args{};
    args.channel = channel;
    args.segBase = segBase;
    args.topkIndices = topkIndices;
    args.reqPoolIndices = reqPoolIndices;
    args.reqHostBase = reqHostBase;
    args.topKDeviceSlots = topKDeviceSlots;
    args.isMiss = isMiss;
    args.numReal = numReal;
    args.hotRowBase = hotRowBase;
    args.status = status;
    args.layerOffsetRows = layerOffsetRows;
    args.perLayerSegRows = perLayerSegRows;
    args.rowBytes = rowBytes;
    args.stripeRows = stripeRows;
    args.nodeNum = nodeNum;
    args.myNode = myNode;
    args.topK = topK;
    args.maxNumReqs = maxNumReqs;
    const uint32_t totalPos = maxNumReqs * topK;
    args.positionsPerCore = (totalPos + blockDim - 1) / blockDim;  // 对齐由 Python 侧校验（64 元素）
    if (rt.aclrtMemset(reinterpret_cast<void *>(status), blockDim * 64, 0, blockDim * 64) != 0) {
        throw std::runtime_error("import_seg status memset failed, status=" + std::to_string(status) +
                                 " bytes=" + std::to_string(blockDim * 64));
    }  // 每 block 独占 64B 槽
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
    if (rt.aclrtMemset(reinterpret_cast<void *>(status), blockDim * 64, 0, blockDim * 64) != 0) {
        throw std::runtime_error("export_seg_alllayer status memset failed, status=" + std::to_string(status) +
                                 " bytes=" + std::to_string(blockDim * 64));
    }
    LaunchCommon(g_fnScatterAll, &args, sizeof(args), blockDim, stream);
}

// ── SIEVE 更新（哈希-only；htSize/reqsPerCore 由本侧派生，公式与 sglang-0730 一致） ──
void kv_sieve_update(uint64_t topkIndices, uint64_t reqPoolIndices, uint64_t seqLens, uint64_t prefillLens,
                     uint64_t deviceBufferTokens, uint64_t deviceBufferVisited, uint64_t deviceBufferHt,
                     uint64_t sieveHand, uint64_t topKDeviceSlots, uint64_t isMiss, uint64_t numReal,
                     uint32_t topK, uint32_t deviceBufferSize, uint32_t paddedBufferSize, uint32_t visitedStrideB,
                     uint32_t maxNumReqs, uint32_t poolSize, uint64_t stream, uint32_t blockDim)
{
    EnsureInited();
    if (paddedBufferSize % 16 != 0) {
        throw std::runtime_error("sieve_update: padded_buffer_size must be a multiple of 16 (64-byte rows)");
    }
    if (paddedBufferSize > 65534) {
        throw std::runtime_error("sieve_update: padded_buffer_size must be <= 65534 (uint16 hash slots)");
    }
    if (visitedStrideB % 64 != 0 || visitedStrideB < (paddedBufferSize + 7) / 8) {
        throw std::runtime_error("sieve_update: visited bitmap stride must be 64B aligned and cover padded");
    }
    if (topK % 8 != 0) {
        // 输出行（slots/is_miss 各 topK 个 int32）经 MTE DataCopy 回写，字节粒度须为 32B 倍数
        throw std::runtime_error("sieve_update: topK must be a multiple of 8 (32B MTE write-back rows)");
    }
    int64_t htSize = 1024;
    while (htSize < 2 * (static_cast<int64_t>(paddedBufferSize) + topK)) {
        htSize <<= 1;
    }
    // UB 预算：tokens 行 + visited 位图行 + 哈希表 + 输出 staging（slots/is_miss 各 topK×4B）
    // + hand 行（16×int32）——sieve 输出在 950 上必须走 MTE 回写（标量 store 对 host 不可见）
    int64_t ubBytes = paddedBufferSize * 4 + visitedStrideB + htSize * 2 + topK * 4 * 2 + 16 * 4;
    if (ubBytes > UB_BUDGET_BYTES) {
        throw std::runtime_error("sieve_update: UB usage " + std::to_string(ubBytes) +
                                 " bytes exceeds the 176KB budget");
    }
    int64_t rpcAlign = 64 / Gcd(topK, 64);
    int64_t reqsPerCore = (maxNumReqs + blockDim - 1) / blockDim;
    reqsPerCore = (reqsPerCore + rpcAlign - 1) / rpcAlign * rpcAlign;
    if (reqsPerCore < 1) {
        reqsPerCore = 1;
    }

    HispSieveUpdateArgs args{};
    args.topkIndices = topkIndices;
    args.reqPoolIndices = reqPoolIndices;
    args.seqLens = seqLens;
    args.prefillLens = prefillLens;
    args.deviceBufferTokens = deviceBufferTokens;
    args.deviceBufferVisited = deviceBufferVisited;
    args.deviceBufferHt = deviceBufferHt;
    args.sieveHand = sieveHand;
    args.topKDeviceSlots = topKDeviceSlots;
    args.isMiss = isMiss;
    args.numReal = numReal;
    args.status = 0;
    args.topK = topK;
    args.deviceBufferSize = deviceBufferSize;
    args.paddedBufferSize = paddedBufferSize;
    args.visitedStrideB = visitedStrideB;
    args.maxNumReqs = maxNumReqs;
    args.reqsPerCore = static_cast<uint32_t>(reqsPerCore);
    args.htSize = static_cast<uint32_t>(htSize);
    args.poolSize = poolSize;
    LaunchCommon(g_fnSieveUpdate, &args, sizeof(args), blockDim, stream);
}

// ── SIEVE 哈希表构建（blockDim = layerNum × numReqs；htSize 传入须为 2 的幂 ≥2×(padded+topK)） ──
void kv_sieve_ht_init(uint64_t deviceBufferTokens, uint64_t deviceBufferHt, uint64_t prefillLens,
                      uint32_t reqStart, uint32_t numReqs, uint32_t layerNum, uint32_t poolSize,
                      uint32_t deviceBufferSize, uint32_t paddedBufferSize, uint32_t htSize, uint64_t stream,
                      uint32_t blockDim)
{
    EnsureInited();
    HispSieveHtInitArgs args{};
    args.deviceBufferTokens = deviceBufferTokens;
    args.deviceBufferHt = deviceBufferHt;
    args.prefillLens = prefillLens;
    args.status = 0;
    args.reqStart = reqStart;
    args.numReqs = numReqs;
    args.layerNum = layerNum;
    args.poolSize = poolSize;
    args.deviceBufferSize = deviceBufferSize;
    args.paddedBufferSize = paddedBufferSize;
    args.htSize = htSize;
    LaunchCommon(g_fnSieveHtInit, &args, sizeof(args), blockDim, stream);
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
    m.def("import_seg", &kv_import_seg, py::arg("channel"), py::arg("seg_base"), py::arg("layer_offset_rows"),
          py::arg("per_layer_seg_rows"), py::arg("row_bytes"), py::arg("topk_indices"), py::arg("req_pool_indices"),
          py::arg("req_host_base"), py::arg("topk_device_slots"), py::arg("is_miss"), py::arg("num_real"),
          py::arg("hot_row_base"), py::arg("top_k"), py::arg("max_num_reqs"), py::arg("stripe_rows"),
          py::arg("node_num"), py::arg("my_node"), py::arg("status"), py::arg("stream") = 0,
          py::arg("block_dim") = 1);
    m.def("export_seg_alllayer", &kv_export_seg_alllayer, py::arg("channel"), py::arg("seg_base"),
          py::arg("per_layer_seg_rows"), py::arg("row_bytes"), py::arg("row_lo"), py::arg("row_cnt"),
          py::arg("layer_num"), py::arg("src_layer0_base"), py::arg("src_layer_stride"), py::arg("stripe_rows"),
          py::arg("node_num"), py::arg("my_node"), py::arg("status"), py::arg("stream") = 0,
          py::arg("block_dim") = 1);
    m.def("sieve_update", &kv_sieve_update, py::arg("topk_indices"), py::arg("req_pool_indices"),
          py::arg("seq_lens"), py::arg("prefill_lens"), py::arg("device_buffer_tokens"),
          py::arg("device_buffer_visited"), py::arg("device_buffer_ht"), py::arg("sieve_hand"),
          py::arg("topk_device_slots"), py::arg("is_miss"), py::arg("num_real"), py::arg("top_k"),
          py::arg("device_buffer_size"), py::arg("padded_buffer_size"), py::arg("visited_stride_b"),
          py::arg("max_num_reqs"), py::arg("pool_size"), py::arg("stream") = 0, py::arg("block_dim") = 8);
    m.def("sieve_ht_init", &kv_sieve_ht_init, py::arg("device_buffer_tokens"), py::arg("device_buffer_ht"),
          py::arg("prefill_lens"), py::arg("req_start"), py::arg("num_reqs"), py::arg("layer_num"),
          py::arg("pool_size"), py::arg("device_buffer_size"), py::arg("padded_buffer_size"),
          py::arg("ht_size"), py::arg("stream") = 0, py::arg("block_dim") = 0);
    m.def("close", &kv_close);
}
