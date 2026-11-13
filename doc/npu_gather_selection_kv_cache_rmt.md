# 设计：npu_gather_selection_kv_cache × 远端内存池适配（V2 重构）

> 状态：**V1.0 / V1.1 / V2 已验收**（本地 golden / 单节点远端 golden / 双节点多通道 golden 全绿，2026-11-13）
> 上游参照：cann-recipes-infer `ops/ascendc/src/gather_selection_kv_cache`（AscendC 源）、
> `models/glm_5_2`（对接模型）、`ops/ascendc/examples/test_npu_gather_selection_kv_cache.py`（golden 语义）

## 1. 背景与目标

kv_offload_proto V0 已闭环卡到卡 URMA 数据面（行粒度 gather）。V2 对接 GLM 5.2 生产形态：以
`npu_gather_selection_kv_cache`（账本式命中检测 + 紧凑热池）为基础算子，将 full 池从本地/Host
内存替换为远端 EMS 池（emsd HBM 段，URMA 回源）。

## 2. 总体架构

```
indexer topk ──▶ npu_gather_selection_kv_cache_rmt ──▶ SFA
                     │ 账本命中 → 不搬
                     │ miss ──▶ ReadNbi(ch, rmt_addr) 回源（EMS 段，topk 段粒度）
                     ▼
        HBM 热池（selection 四件套，跨步持久，8K 项级）   ← 始终本地
```

两层正交：**热池管命中率**（本地 HBM，SFA 直接读，容量 8K/16K + LRU/组相联），**full 池管容量**
（远端段，回源通道 URMA 替代 H2D）。

## 3. 算子接口

原 10 参完全同构（GLM 5.2 `modeling_glm.py:1565` 调用原样；纯远端时 `full_k_rope/full_kv_cache`
传空 tensor，沿用 int8 空 tensor 惯例）。新增 3 参（物理块坐标系 `[F_BLOCK_NUM]`，与
`full_kv_cache` 同为无 B；per-layer 一份）：

```python
torch_npu.npu_gather_selection_kv_cache_rmt(
    # ── 原 10 参（同构）──
    selection_k_rope,            # [S_BLOCK_NUM, BLOCK_SIZE, K_ROPE] 热池·rope 区（原地写）
    selection_kv_cache,          # [S_BLOCK_NUM, BLOCK_SIZE, KV_CACHE] 热池·kv 区（原地写）
    selection_kv_block_table,    # [B*S*H, S_MAX_BLOCK_NUM] 热池页表（输入）
    selection_kv_block_status,   # [B,S,H,TOPK+1] 热池账本（原地写；[0..TOPK-1]=目录，[TOPK]=长度）
    selection_topk_indices,      # [B,S,H,TOPK] topk 块段 id
    full_k_rope, full_kv_cache,  # 本地池（纯远端传空）
    full_kv_block_table,         # [B, F_MAX_BLOCK_NUM] 逻辑块→物理块（原样使用）
    full_kv_actual_seq, full_q_actual_seq,
    selection_topk_block_size=1,
    # ── 新增 3 参 ──
    full_rmt_channel,            # int64 [F_BLOCK_NUM] 物理块→ChannelHandle（0 预留=本地）
    full_rmt_kv_addr,            # int64 [F_BLOCK_NUM] 物理块→远端 kv 池块基址（块内布局同构）
    full_rmt_rope_addr,          # int64 [F_BLOCK_NUM] 物理块→远端 rope 池块基址（int8 传空）
)
# 返回 selection_kv_actual_seq : int32 [B*S*H]（去重+裁剪后实际选取 token 数）
```

**寻址链**：`p = full_kv_block_table[b][i]`（原样）→ `ch/addr = full_rmt_*[p]` →
`ReadNbi<false>(ch, dst=热池块, src=addr+块内偏移×D, 段长=topk_bs×D)`。
页表/paged KV 管理零改动；远端块槽位置固定（三表一次生成，V1 不做迁移）；混布预留：channel=0 走本地路径。

**标量约束**（launch 校验）：`head==1`（MLA 共享 KV）；`block_size % selection_topk_block_size == 0`。
元素宽度固定 2B（fp16/bf16 按位搬运不解析）；`K_ROPE=0` 合法（int8 场景空 rope 惯例）。

**部署形态**：①纯远端（full 传空 int8 tensor，channel 全非 0）②纯本地（channel 全 0，即生产原算子行为）
③任意混布（per-block channel 决定，热池/账本语义三种形态一致）。

**错误码**（poll 返回，低 16 位）：`0x1xx` hcomm Init；`0x2xx` kv 读（`xx`=k 序号，`[16:24]`=hcomm ret）；
`0x3xx` rope 读；`0x4xx/0x5xx` Commit/Drain（`[24]`=通道序号，`[0:24]`=ret）；`0x180+` 通道超上限（≤8）。

## 4. 内核（kvops/kernel/gather_selection_rmt.cpp）

- 语义基准 = golden 参考（test :96-200）：有效性过滤（seg<0 截断/seg≥max 跳过）→ 因果裁剪
  （kv_len - (q_len-1-s)）→ 紧凑追加 → 账本更新（-1 失效、末位长度回写）
- 读点分流：本地段 `DataCopyPad`（经 UB staging，GM→UB 4 参 / UB→GM 3 参，见 §8）；远端段
  `Hcomm<UBC_CTP>::ReadNbi`（dst=热池块直写，须在 cli mem_reg 注册范围内）
- 提交序列：单核单流 `ReadNbi×段 → 逐通道 Commit<PIPE_MTE3> → Drain`（V2 多通道去重集 ≤8；
  多流共享 channel 丢 WQE 已实证，多核并行=核专属通道，V2+）；尾部 `KvcStatusFinish`（64B 槽）
- fatbin 纯设备编译（逐字段签名 + args 重组，同 hisp_sieve_update 风格）；AscendC 源
  split_bs_reuse 作向量化参考，V1 精简自研（单 block 或少 block），分核策略后置
- 输出（账本/actual_seq/热池）经 MTE 回写（c310 标量 store 对 host 不可见，已实证）

## 5. Host 侧

- `xds_ems_pool.py`：① `create_pool`（emsd 集群建链：pool_create → 逐段 LINK_EXCHANGE →
  mem_import → link_channel；sel 池与 full 源置于同一 hbm 大张量整体 mem_reg——ReadNbi dst 要求）
  ② `export_full_rmt_pool`（两区导出 + `full_rmt_*` 三表生成，支持 perm 乱序槽位；多节点行交错，
  见 §8）③ destroy（channel×N → cli close → kvops close → pool_destroy，逐段告警不抛）
- `kvops_ext.cpp`：`gather_selection_rmt(...)` launch 封装（27 参 pybind，head/整除校验）
- `tests/`：golden 对拍三件套（本地 / 单节点远端 / 双节点多通道）
- torch 挂载（V3）：`custom_op_loader` 同款挂 `torch_npu.npu_gather_selection_kv_cache_rmt`

## 6. 阶段计划与验收

| 阶段 | 内容 | 验收 | 状态 |
|---|---|---|---|
| **V1.0 基建** | 内核骨架（本地回退路径）；args/launch；热池创建 | 本地模式（channel=0）与 golden 一致 | ✅ `tests/test_gather_selection_rmt.py` PASS |
| **V1.1 单节点远端** | 读点分流 ReadNbi；两区导出；三表生成 | 单 emsd golden 对拍全绿 | ✅ `..._rmt.py` PASS（ropeDim 64/256 × perm） |
| **V2 多节点** | 表路由多 channel；emsd 集群 | 跨节点段取回对拍 | ✅ `..._rmt2.py` PASS（恒等 + perm 乱序，[32,32]/node） |
| **V3 生产对接** | torch 挂载；GLM 5.2 modeling +3 参；DSA 流水评估 | GLM 5.2 offload 围跑 | ⬜ |

## 7. 风险与开放问题

- ~~ReadNbi dst 对齐/粒度约束~~ → **已实证无约束**（见 §8 矩阵）
- 单核单流（V0 实证多流共享 channel 丢 WQE）；**多核并行路线已定**：每段建 N channel、核 r 专属
  channel[r]，先实测丢 WQE 根因是否=同 channel 并发提交（V2+）
- int8 量化/空 rope 场景（V3）
- aclgraph 兼容（V3 评估）
- 待定：BLOCK_SIZE 与 GLM 配置对齐值（pa_max_length）；OffloadCache 由 XDS 池管理器扮演的边界

## 8. 实证沉淀（V1.1/V2 联调结论，NPU 950/c310 单机验证）

**ReadNbi 无对齐/粒度约束**（`HcommReadNbi` 官方文档与实测一致，仅要求 len>0 + src/dst 落注册范围）：

| 操作 | 粒度 | 地址对齐 | 结果 |
|---|---|---|---|
| WriteNbi（export） | 512B/行 | 行基址任意 | ✓ |
| ReadNbi | 512B / 1152B / 128B | 128B 对齐（非 512 倍） | ✓ 全对 |

**export 内核 `row_lo` 相对寻址语义**（kvc_scatter_rows_alllayer）：`src = srcLayer + (row-rowLo)*rowBytes`
——src 基址=本次调用区域第 0 行，row_lo 是段内 dst 偏移。绝对行号寻址会越过 src 区读到无关内存
（V1.1 rope 全 0 事故根因，row_lo=0 的历史调用不受影响）。

**行交错布局**（kv_layout 语义，stripe_rows=1 强制）：全局行 r → 段 `r%N`、段内行 `r//N`；
`KvcSegRowOf=(row/S)//N` 在 S>1 时条带内行重叠，**不可用**。多节点导出姿势：构造全局交错源
（`[段][段内行]` 转置为 `[段内行][段]`），一次全区 batch_write 由内核 `KvcNodeOf=row%N` 自动分发；
段内布局 = [kv 大区 per_node×BS 行][rope 大区（字节流按行宽切）]，三表寻址=段内字节偏移。

**per-seg status 槽**：`EmsPool._status_count = 8×node_num`；内核按 `status + myNode*64` 写、
`wait()` 按 `status + (s*blockDim+b)*64` 轮询——多节点下 next_status/blockDim 分配须匹配此布局，
否则 poll 错槽假死（V2 双节点 export 假超时根因之一）。

**emsd 控制连接生命周期**（pool_destroy 120s 假超时根因，三层）：cfd 空闲超时误用
`readyTimeoutSec`(60s) 杀长会话（双节点 2×35s 握手即超）→ `HandleConn` 退出泄漏 fd（无 FIN/RST，
client send 成功 recv 干等）→ 修复：`CTRL_CONN_IDLE_TIMEOUT_SEC=3600` + break 后 `close(fd)`。
另 master↔worker SEG_FREE 走 `Connect()` 独立连接（带超时）。

**双段同 VA 无害**：不同 emsd 段 DVA 可相同（各卡独立地址空间），hcomm 按 channel 区分翻译，
实测双段同基址 0x12004be00000 golden 全绿。

**UB→GM DataCopyPad 为 3 参**（无 padParams；GM→UB 为 4 参）——CANN 签名差异，写反编译报
no matching function。

**多通道收尾**：单核按块路由向 N channel 交替提交 WQE，收尾对去重通道集（≤8）逐个
`Commit<PIPE_MTE3> + Drain`；单/双节点行为一致（V2 实测）。

## 9. 部署与运行（双节点示例）

```bash
# master（device 3，成员表含自身+worker；hcommPort 按 60001+nodeId 自动分配）
./build/emsd/emsd --role master --ip <ip> --port 27000 --device 3 --loc device \
  --members <ip>:27000,<ip>:27001 --pool-size-gb 1
# worker（device 4）
./build/emsd/emsd --role worker --ip <ip> --port 27001 --device 4 --loc device

# 双节点 golden（恒等 / perm 乱序跨段路由）
python -m tests.test_gather_selection_rmt_rmt2 --device 7 --ems-master <ip>:27000 [--perm]
# 单节点回归（V1.1）
python -m tests.test_gather_selection_rmt_rmt --device 7 --ems-master <ip>:27000 [--perm] [--rope-dim 256]
```

基准观测：channel READY ≈8.1s/条（时序探针优化后，曾 35s）；destroy 全链 <0.1s。
