# kv_offload_proto — XDS KV Cache AIV 直驱卸载原型

按《doc/kv_offload_design.md》实现：单仓三组件（ems / kvops / xds），数据面 HCOMM AIV 内核直驱
（fatbin 显式加载链 + WithHostArgs + args+overflow 尾槽 + MTE3 status + c310 epilogue）；
miss 识别为 device 侧 SIEVE 算子（移植自 sglang-0730，哈希-only + visited 位图，热路径零 CPU 同步）。

## 场景

- XDS（DSA decode）：lightning indexer 产出 topK → HBM 热缓冲 SIEVE 查 miss（device kernel）
  → miss 行经 AIV 内核 URMA 直读远端资源节点 DRAM 回填热缓冲（每层行空间均衡打散到 N 节点，
  聚合 N×单节点带宽）。
- prefill KV 全层导出到远端（HBM → 远端段，条带聚合 WQE）。

## 拓扑与角色

```
资源节点 A（master emsd）      资源节点 B（worker emsd）        NPU 机（计算节点）
┌──────────────────────┐      ┌──────────────────────┐      ┌────────────────────────────┐
│ TCP master（打散/汇总）│◄────►│ TCP worker            │      │ python -m xds.xds_decode    │
│ host DRAM 段0+MemReg  │      │ host DRAM 段1+MemReg  │◄─────│  xds_decode（入口/驱动）     │
│ Endpoint(HOST)+AIV    │◄─────┼──────────────────────┼──────│  xds_offload（SIEVE 协调器） │
│   server 通道          │ TCP  │ AIV server 通道       │ URMA │  xds_ems_pool（池+批量读写） │
└──────────────────────┘ (控制)└──────────────────────┘(数据)└────────────────────────────┘
```

## 快速开始（跨节点手工分别启动）

```bash
# 0) 构建（资源节点执行 emsd；NPU 机执行 ext+kernel）
./build.sh emsd                                            # 资源节点（无 CANN 依赖）
./build.sh ext && ./build.sh kernel                        # NPU 机（先 source CANN set_env.sh）

# 1) 资源节点 A（master；members=静态成员表 ip:ctrlPort）
#    双 loc 模式：--loc device（默认，当前 CANN 包）= 卡到卡：EID 自动按 --device 的 phyId
#    匹配 rootinfo（PoC 同构）；--loc host（待跨模式/noDevice 包）= --host-addr <ipv6> 手填
#    （ipourma host 地址，noDevice 仓 rank_graph 标准填法）。两种模式段内存均恒 host DRAM
#    + COMM_MEM_TYPE_HOST（MemFabric device_urma 生产实证组合）
./build/emsd/emsd --role master --ip <A_ip> --port 27000 --device 4 --loc device \
    --members <A_ip>:27000,<B_ip>:27001 --pool-size-gb 8

# 2) 资源节点 B（worker）
./build/emsd/emsd --role worker --ip <B_ip> --port 27001 --device 5 --loc device --pool-size-gb 8

# 3) NPU 机跑 decode 驱动（EID 全自动：device→phyId→rootinfo 匹配；默认 direct 模式）
python -m xds.xds_decode --ems-master <A_ip>:27000 --device 0 \
    --req-num 8 --prefill 256 --topk 64 --device-buffer-size 64 --miss-ratio 0.5 --steps 10

#    带 CPU oracle 对拍（eager 专用，逐层校验 sieve slots/is_miss）
python -m xds.xds_decode ... --oracle
```

## 关键实现锚点

| 锚点 | 位置 |
|---|---|
| 行归属映射（每层均衡） | `common/kv_layout.h` ↔ 内核 `KvcNodeOf/KvcSegRowOf`（Python 对拍：`test/kv_layout_ref.py`） |
| SIEVE 哈希（Knuth 散列+backward-shift 删除） | `kvops/kernel/hisp_kernel_common.h`（哈希-only；visited 位图 /8） |
| sieve 派生公式（htSize/reqsPerCore/UB 预算） | `kvops_ext.cpp kv_sieve_update`（与 sglang-0730 host 派生严格一致） |
| launch 参数（host/内核共用布局） | `kvops/kernel/kvc_args.h`（args + 8B overflow 尾槽） |
| 稠密导入契约 | `kvc_gather_rows`：isMiss==1 且 (row/S)%N==myNode 才搬；src=reqHostBase[bid]+tokenPos（行分配连续性契约）；dst=hotRowBase+slot（热缓冲固定偏移契约） |
| status 槽 | 64B/槽；gather 按 blockDim 分片（≤8 block）；`poll_status` 等 doneSeq==1 |

## 参数（xds_decode 常用）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ems-master` | 必填 | master emsd ip:port |
| `--device / --rootinfo` | 0 / /etc/hccl_rootinfo.json | NPU 卡号 / EID 自动匹配源（device→phyId→rootinfo） |
| `--fatbin` | 自动探测 | kvc fatbin 路径 |
| `--layers / --layer-rows` | 2 / 1024 | 层数 / 每层行数 |
| `--kv-lora-rank / --qk-rope` | 512 / 64 | MLA 行字宽（fp16） |
| `--stripe-rows` | 1 | 跨节点打散条带 S（导出聚合可调大） |
| `--req-num / --prefill` | 8 / 256 | 并发请求 / prefill token 数 |
| `--topk` | 64 | indexer topK |
| `--device-buffer-size` | 64 | 热缓冲槽数/请求（< prefill 时产生 miss） |
| `--miss-ratio` | 0.5 | topK 采样中 miss token 目标占比 |
| `--steps / --seed` | 10 / 42 | decode 步数 / 随机种子 |
| `--block-dim` | 8 | sieve/gather 内核 block 数 |
| `--oracle` | 关 | CPU sieve 模型逐层对拍（eager） |
| `--run-dir` | ./log | 日志目录（每次截断重写 xds_decode.log） |

## 验收标准

1. `pool ready: nodes=N perLayerSegRows=... channels=N`（V0 建链通过）
2. `prefill export: ... rows x L layers = ... B in ... us`（全层导出）
3. 每步：`step K: layer=L miss=M rows in ... us (checked=C)` 且无 `[xds][ERROR]`
4. `summary: imported M miss rows, avg ... us/row, aggregate ... GB/s`
5. `--oracle` 模式下全部步骤对拍通过（slots/is_miss 与 CPU 模型一致）
6. `[xds] all done`，exit 0；emsd 侧日志无 ERROR

## 排障（精简）

| 症状 | 处置 |
|---|---|
| `HcommEndpointCreate ret=15`，plog 报 `hrtGetDevice failed, deviceLogicId[65535]` | device 模式进程缺 NPU 上下文：加 `--device N`；host 模式（跨模式包）无此依赖 |
| `HcommEndpointCreate ret=11`，plog 报 `rs_urma_get_device_by_eid failed / errno:19` | endpoint 地址未匹配到 UB 设备：device 模式检查 rootinfo 含 `--device` 对应 `device_id` 的 EID addr（`--rootinfo` 指定正确文件）；host 模式检查 `--host-addr` 为本机 ipourma 上真实存在的 IPv6 地址 |
| MemReg 失败（`[emsd][ERROR] MemReg failed, addr=... ret=...`） | 检查段地址/容量（日志已带 addr/size/ret）；CANN 包下 device endpoint + HOST 内存注册为 MemFabric 生产实证组合，若仍失败回传 plog |
| link_exchange/channel not ready | 跨模式 socket（HDC↔PEER）异步 connect 问题：预案在 `ems_client/hcomm_cli_ext.cpp: cli_link_channel` 同步化适配（参照 hcomm_no_device 提交 9b2cb0c9d 模式） |
| sieve_update 报 UB budget | 降低 `--device-buffer-size`（UB=tokens 行+位图+哈希表 ≤176KB/核） |
| sieve_update 报 padded ≤65534 | uint16 槽号上限：`--device-buffer-size` ≤ 65534-16 |
| gather errCode 0x2xx | URMA 行读失败（errCode 高 16 位为 hcomm ret）：查资源节点 emsd 日志/通道状态 |
| verify failed | 行指纹不一致：确认 `--seed` 一致、热缓冲区未被其他任务占用；emsd 侧查段基址 |
| allocator exhausted | `--layer-rows` 不足（req-num×prefill 上限） |

## 单元测试（任意机器，无 NPU 依赖）

```bash
python -m unittest discover -s test -v   # 映射对拍 / 行分配器 / 协议 round-trip / sieve oracle
```
