# 设计：npu_gather_selection_kv_cache × 远端内存池适配（V2 重构）

> 状态：实施中（V1.0 基建）
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

## 4. 内核（kvops/kernel/gather_selection_rmt.cpp）

- 语义基准 = golden 参考（test :96-200）：有效性过滤（seg<0 截断/seg≥max 跳过）→ 因果裁剪
  （kv_len - (q_len-1-s)）→ 账本命中检测（含尾块/超界无效判定）→ 未命中搬运 → 紧凑追加 +
  账本更新（-1 失效、末位长度回写）
- fatbin 纯设备编译（逐字段签名 + args 重组，同 hisp_sieve_update 风格）；AscendC 源
  split_bs_reuse 作向量化参考，V1 精简自研（单 block 或少 block），分核策略后置
- 读点分流：本地段 `DataCopy`；远端段 `Hcomm<UBC_CTP>::ReadNbi`（dst=热池块直写，须在
  cli mem_reg 注册范围内）
- 提交序列：全核单流 `ReadNbi×段 → Commit<PIPE_MTE3> → Drain`（规避多流丢 WQE，已实证）；
  尾部 `KvcStatusFinish`（64B 槽，复用 kvc_common）
- 输出（账本/actual_seq/热池）经 MTE 回写（c310 标量 store 对 host 不可见，已实证）

## 5. Host 侧

- `xds_ems_pool.py`：① 热池创建（本地 HBM，置于 cli mem_reg 注册范围内——ReadNbi dst 要求）
  ② block 对齐导出（scatter 改造：框架 paged 池逐块→远端段槽）③ 三张 `full_rmt_*` 表生成（块号同键）
- `kvops_ext.cpp`：`gather_selection_rmt(...)` launch 封装（新 args 打包，stream=当前流）
- `xds_decode.py`：selection 形态驱动 + golden oracle（移植 test :96-200）
- torch 挂载（V3）：`custom_op_loader` 同款挂 `torch_npu.npu_gather_selection_kv_cache_rmt`

## 6. 阶段计划与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| **V1.0 基建** | 文档落盘；内核骨架（本地回退路径）；args/launch；热池创建 | 本地模式（channel=0）与 golden 语义一致（账本/actual_seq/数据） |
| **V1.1 单节点远端** | 读点分流 ReadNbi；block 对齐导出；三表生成 | 单 emsd：golden 对拍全绿 + 远端取回数据校验 |
| **V2 多节点** | 表路由多 channel；双 emsd | 跨节点段取回对拍 |
| **V3 生产对接** | torch 挂载；GLM 5.2 modeling 调用点改造（追加 3 参）；DSA install 流水评估（URMA 替代 H2D） | GLM 5.2 offload 围跑 |

## 7. 风险与开放问题

- ReadNbi dst 对齐：热池块基址 32B 对齐（BLOCK_SIZE×D 天然满足，需在导出/分配侧保证）
- int8 量化/空 rope 场景（V3）
- aclgraph 兼容（V3 评估）
- 多流并发保持单流约束（优化后置；通道级并行留 V2+）
- 待定：BLOCK_SIZE 与 GLM 配置对齐值（pa_max_length）；OffloadCache 由 XDS 池管理器扮演的边界
