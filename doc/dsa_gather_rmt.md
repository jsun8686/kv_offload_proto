# 设计：dsa_gather × 远端内存池适配（dsa_gather_rmt）

> 状态：**D1.0 / D1.1 / D1.2(图模式) / D2a(多核) 已验收**（远端 golden + torch e2e + NPUGraph + 
> block-dim {1,2,4,8}×{恒等,perm,graph} 全绿，2026-11-24）
> 上游参照：vllm-ascend-csrc `csrc/attention/dsa_gather`（原算子：Host swap memory → resident HBM，
> H2D 透明搬运）；golden 基准 = 其 `golden.py`（miss_mask 唯一闸门 + sentinel 零搬运对拍）
> 本仓已简化为 dsa_gather_rmt 单算子形态（gather_selection 系列及 V0 流水已移除，git 历史可溯）

## 1. 背景与定位

原版 `dsa_gather`（DSA = DeepSeek Sparse Attention）在 decode 流水回源 miss 记录：源是
**Host swap memory**（swapped memory 的 NPU VA，`DataCopy` 读触发 H2D 引擎透明搬运——慢路径，
与权重加载争 PCIe）。`dsa_gather_rmt` 把源替换为**远端 EMS 池**（emsd 段 HBM，URMA ReadNbi
按需回源，走独立超节点互连平面）——**纯远端**：无本地分支、无 0 哨兵（对比 gather_selection_rmt
保留 full 张量是接口同构包袱，本算子从设计之初砍掉源池张量）。

DSA 三算子流水位置不变：`lookup → (hit SFA ∥ gather(miss)) → miss SFA → DA merge`，gather 独立
stream 与 hit SFA 重叠（原版实测 98%+ 区间重叠）。

## 2. 接口（10 tensor + 2 attr → 9 tensor + 2 attr）

```python
torch_npu.npu_dsa_gather_rmt(
    resident_kv_cache,         # [P_dst, BS, 1, kvDim] 原地写
    resident_rope_cache,       # [P_dst, BS, 1, ropeDim] 原地写
    resident_block_table,      # [numReq, W_dst] int32 batch 行→目标物理块
    host_block_table,          # [poolCap, W_src] int32 池行→源物理块（远端池页表，原样使用）
    request_pool_indices,      # [numReq] int32 batch 行→稳定池行（两跳源寻址的桥）
    topk_token_positions,      # [numReq, queryCount] int32 源 token 位置
    resident_slots,            # [numReq, queryCount] int32 目标 slot（lookup 输出）
    miss_mask,                 # [numReq, queryCount] int32 唯一搬运开关（1=miss 才搬）
    block_size, num_requests,  # attr（原样）
    # ★ 新增三表（源物理块坐标系 [P_src] int64，src_blk 查表，无 0 语义）
    host_rmt_channel,          # 块→ChannelHandle（真实值）
    host_rmt_kv_addr,          # 块→远端 kv 区基址（块内 [BS, kvDim] 连续）
    host_rmt_rope_addr,        # 块→远端 rope 区基址（块内连续）
)
```

**变化**：删 `host_kv_cache/host_rope_cache`（源恒远端）；增三表（`EmsPool.export_full_rmt_pool`
直接复用——远端布局与 gather_selection_rmt 同构：段内 [kv 大区][rope 大区]，行交错多节点）。
attr、resident 侧、pool entry 间接层、mask/校验语义**全部不变**。

**寻址链（miss pair）**：
`pool_entry = request_pool_indices[req_id]` → `src_blk = host_block_table[pool_entry][token/BS]`
→ `ch/addr = host_rmt_*[src_blk]` → `ReadNbi(ch, dst=resident 直写, src=addr+块内偏移×rec_bytes, len=rec_bytes)`。

## 3. 内核（kvops/kernel/dsa_gather_rmt.cpp，自研 kvops 风格）

- 语义基准 = golden.py：miss_mask 闸门（命中/padding 零搬运）→ 索引非负 → 逻辑块不越表宽 →
  物理块不越池（三层校验全在 AIV，不回 CPU）
- **纯 ReadNbi 直写 resident 池**（无 UB staging、无本地 MTE 流水——比 gather_selection_rmt 更薄）；
  record 粒度 kv/rope 各一发（1KB/128B，ReadNbi 无对齐/粒度约束已实证）
- **原生多核数据并行**：pair（req_id×query_id）按核均分（尾核+1），核间零依赖；blockDim=1 即单核
  （同一代码路径，非独立分支）；status 每核独立 64B 槽（status + blockIdx*64）
- 行首元数据（pool_entry）提循环外，行切换重取；`pair+1 < pairEnd` 防尾读
- 收尾：去重通道集（≤8）逐个 `Commit<PIPE_MTE3> + Drain`（V2 多通道成果移植）
- 错误码：`0x1xx` hcomm Init / `0x2xx` kv ReadNbi / `0x3xx` rope ReadNbi（`xx`=pair 序号，
  `[16:24]`=hcomm ret）/ `0x4xx·0x5xx` Commit/Drain（`[24]`=通道序号）/ `0x180+` 通道超上限
- 字节寻址参数化（kvElemSize/ropeElemSize ∈ {1,2}）支持 INT8+BF16 dtype 配对（V1 测试跑 fp16）
- residentInit 预留（恒 0=dense decode 布局；紧凑初始化布局 V1.1）

## 4. 并行策略（D2，三级递进——已收官）

| 级 | 方案 | 结论 |
|---|---|---|
| D2c | 直接多核共享 channel（blockDim=2） | ✗ **复现丢 WQE**：golden mismatch 且 rc=0 假完成（V0 结论坐实：同 channel 并发提交静默丢 WQE，非 9.2.1 特有） |
| D2a | 每核专属 channel（通道表平铺 + 批量建链） | ✅ **全绿收官**：block-dim {1,2,4,8} × {恒等, perm, graph} 全 PASS；建链 7~8s 全 READY；多核×图模式首验通过 |

### D2a 多通道机制（四轮实证踩坑沉淀，后续算子多核化直接复用）

1. **官方批量姿势**：一次 `HcommChannelCreate(descs[N], N, channels)` 数组调用。逐条独立调用
   N 次 (num=1) 时第 2 条在 `SocketMgr::GetSocket` 的 inUse 自旋等 `HCCL_CONNECT_TIMEOUT(120s)-10s`
   超时（SocketConfig 撞等前条未释放）——批量是官方文档明示的多通道用法（channelNum [1,1048576]）。
2. **配对维度是 channelName（tag）非 port**：listen 是 **endpoint 级单端口**（`StartListen→HcommEndpointStartListen`
   幂等），port=base+i 被架空。正确姿势：全部通道同 base port + **tag 唯一化**（`tag_i`，两端同规则）。
3. **GetStatus 是建链状态机推进器（功能件非查询件）**：connect 异步结果的收割（`RaGetAsyncReqResult`）、
   tag 发送、`GetOneSocket` 配对全部由其触发——**两侧都必须数组轮询全部 N 条**。emsd 侧"诊断线程"
   实为 server 侧驱动线程（只轮 ch0 时 ch1 成为孤儿通道：connect 无人收割 → 配对永不发生）。
4. **协议**：client 在 LINK_EXCHANGE 帧尾声明 channelCount（+4B），emsd 按需批量开通——通道数随
   `--block-dim` 单一事实源流动（协议零配置）。三表平铺 `[F_BLK × n_core]`（kv/rope addr 表不变），
   内核 `chTbl[q*coreCount+coreId]`，n_core=1 时与旧布局逐字节兼容。

建链形态：批量 Create 秒回（server/client 两侧）→ 两侧驱动轮询 → 7~8s 全 READY → N 条独立
数据面通道（每核独占 SQ → 无并发提交竞态，根治丢 WQE）。

## 5. golden 对拍（tests/test_dsa_gather_rmt.py）

- 数据公式照搬 golden.py：host 池按 `(global_token % 512) + ch + {0(kv),1024(rope)}` 填充 staging
  → `export_full_rmt_pool` 导出 + 三表；resident 池 fill(-1000) sentinel
- 期望映射：`source_slot = pool_entry × host_slots_per_entry + token`（active=miss 集合）；
  `dest_flat = batch_row × resident_slots_per_req + slot`；数据 + **untouched sentinel 精确对拍**
  （零搬运性质：touched ≡ miss 集合）
- 用例矩阵：恒等 / `--perm`（乱序槽位）/ `--block-dim {1,2,4,8}`（D2 矩阵）
- 规模：缩档（bs=4、QUERY=256、key_len=4096）跑正确性；原版规模（bs=4、QUERY=2048、
  key_len=128K）跑性能对照
- 约束：resident 池须在 cli mem_reg 注册范围（ReadNbi dst）——staging+resident 同一大 hbm 张量

## 6. 阶段计划

| 阶段 | 内容 | 验收 |
|---|---|---|
| **D1.0** | 文档 + 内核（多核结构、纯远端）+ args/launch/pybind + golden 测试 | ✅ 单核远端 golden PASS（恒等+perm，records=256） |
| **D1.1** | torch 壳 `torchops/dsa_gather_rmt_op.py` → `torch_npu.npu_dsa_gather_rmt` + e2e | ✅ e2e PASS（恒等+perm） |
| **D1.2 图模式** | 两层 API（submit/wait 拆分 + status 槽池化）+ `--graph` 用例 | ✅ dsa(恒等+perm) + gather_selection 图 e2e PASS（`tests/test_graph_probe.py` 实证 HostArgs 可捕获；EE1016 值校验陷阱已修，见 §7） |
| **D2** | block-dim 矩阵实测（D2c → 依结果 D2a） | ✅ D2c 负结果（共享 channel 丢 WQE 复现）；D2a 全绿：bd {1,2,4,8}×{恒等,perm}+bd8 graph e2e PASS（多通道机制见 §4） |

## 7. 图模式（NPUGraph，两层 API）

**Step1 实证**：`LaunchKernelWithHostArgs`+attrs 可被 NPUGraph 记录，远端 ReadNbi 数据面随图
重放正确（姿势 `torch.npu.graph(auto_dispatch_capture=True)`，对齐 sglang-0730 `NPUGraphRunner`）。

```python
h = npu_dsa_gather_rmt_submit(...)   # 纯提交（capture 区间内调用；launch 层零改造）
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph, auto_dispatch_capture=True):
    h = npu_dsa_gather_rmt_submit(...)
for _ in range(N):
    reset_outputs()                   # 图外 sentinel 重置
    graph.replay(); torch.npu.synchronize()
    npu_dsa_gather_rmt_wait(h)        # 可选：图外错误码校验（golden 对拍亦兜底）
```

纪律（对齐 MemFabric 用例14）：warmup 图外 eager 首跑（内核 hcomm.Init 首执行不得在 capture 中）；
三表/页表/输出张量引用持有至最后回放；status 槽模块级池化复用（地址固定）。
eager 语义不变：`npu_dsa_gather_rmt` = submit + wait 一行组合（旧调用方零感知）。

## 8. 性能摸底（tests/test_bench_dsa_gather_rmt.py）

矩阵：`--block-dims 1,2,4,8` × `--scale s1|s2|s3` × `--mode eager|graph|both`；
计时 `torch.npu.Event` × warmup3 + repeat10 取中位；输出 `time_us / miss_records / moved_bytes /
有效带宽 GB/s / vs bd1 加速比`（markdown 表）。规模档：

| 档 | queryCount | keyLen | 说明 |
|---|---|---|---|
| s1 | 256 | 4096 | 正确性档（records≈256） |
| s2 | 2048 | 32768 | 中档（records≈8K） |
| s3 | 2048 | 131072 | 原版生产形状（对齐 vllm dsa_gather；staging≈67MB） |

参照基线：原版 910_93 H2D 路径 164 miss 记录 98.78μs（dsa_gather 上游）。
