# 设计：dsa_gather × 远端内存池适配（dsa_gather_rmt）

> 状态：**D1.0 / D1.1 / D1.2(图模式) 已验收**（远端 golden + torch e2e + NPUGraph 捕获回放全绿，2026-11-24）
> 上游参照：vllm-ascend-csrc `csrc/attention/dsa_gather`（原算子：Host swap memory → resident HBM，
> H2D 透明搬运）；golden 基准 = 其 `golden.py`（miss_mask 唯一闸门 + sentinel 零搬运对拍）
> 姊妹篇：`npu_gather_selection_kv_cache_rmt.md`（rmt 系工程范式/实证沉淀，本文复用其基建）

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

## 4. 并行策略（D2，三级递进）

| 级 | 方案 | 说明 |
|---|---|---|
| D1 | blockDim=1 单核 | golden 闭环 + 性能基线（原版 910_93 基线：164 miss 记录 98.78μs） |
| D2c | **直接多核共享 channel**（blockDim {2,4,8} 矩阵） | 最便宜的根因实验：稳 → 原生 pair 均分即成品 |
| D2a | 每核专属 channel | 不稳时切换：emsd 每段建 N channel + 内核通道池选择，只改通道获取一处 |

D2c 一轮矩阵同时回答 V0 遗留问题（多核共享 channel 并发提交是否丢 WQE）——dsa_gather 是理想
试验场（pair 独立 + sentinel 零搬运严格校验）。结论对 gather_selection_rmt 多核化同样适用。

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
| **D2** | block-dim 矩阵实测（D2c → 依结果 D2a） | 多核 golden + 性能报告 + 丢 WQE 根因结论 |

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
    wait_and_check(h)                 # 可选：图外错误码校验（golden 对拍亦兜底）
```

纪律（对齐 MemFabric 用例14）：warmup 图外 eager 首跑（内核 hcomm.Init 首执行不得在 capture 中）；
三表/页表/输出张量引用持有至最后回放；status 槽模块级池化复用（地址固定）。
eager 语义不变：`npu_dsa_gather_rmt` = submit + wait 一行组合（旧调用方零感知）。

## 8. 与 gather_selection_rmt 的关系

| | gather_selection_rmt | dsa_gather_rmt |
|---|---|---|
| 源 | 本地张量（空 tensor 惯例）/ 远端，0 哨兵分流 | 仅远端，无哨兵 |
| 行坐标 | B*S*H query 行 + 账本/因果裁剪 | (req,query) pair + miss_mask |
| 源寻址 | 一跳（batch 行页表） | 两跳（pool_entry 间接） |
| 输出 | 热池 + 账本 + actual_seq | resident 池（原地，无账本） |
| 并行 | 单核（多核挂账） | 原生多核结构（D2 实测） |
| 远端布局/三表/收尾 | 同构（export_full_rmt_pool 复用） | 同左 |
