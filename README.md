# kv_offload_proto — dsa_gather_rmt：DSA miss 记录 AIV 直驱远端回源原型

EMS（external memory server）池上的 `npu_dsa_gather_rmt_submit` 原型：DSA（DeepSeek Sparse Attention）
decode 流水 miss 记录回源，源为远端 emsd 段 HBM，AIV 内核 URMA `ReadNbi` 直写 resident 池
（纯远端：无本地分支、无 0 哨兵、无 UB staging）。

完整设计/实证沉淀见 `doc/dsa_gather_rmt.md`（含 D2a 每核独立通道多核机制链）。

## 核心特性

- **纯远端 ReadNbi 数据面**：miss_mask 唯一闸门 + 两跳源寻址（pool_entry 间接层）
- **原生多核**：pair（req×query）按核均分 + 每核专属 channel（D2a：根治共享 channel 并发丢 WQE）
- **多通道批量建链**：LINK_EXCHANGE 帧声明 channelCount → emsd 一次 `HcommChannelCreate(descs[N])`
  （tag 唯一化配对）→ 两侧 GetStatus 驱动状态机 → 7~8s 全 READY
- **图模式**：submit/wait 两层 API，NPUGraph 捕获回放全绿（warmup 图外纪律）
- **golden 对拍**：对齐 vllm-ascend-csrc `dsa_gather` golden.py（miss 语义 + sentinel 零搬运）

## 组件

```
ems/      emsd（池服务，master/worker，HBM 段 + AIV server 通道）
          ems_client（控制协议 + HCOMM 建链 pybind ext）
kvops/    dsa_gather_rmt.cpp（AIV 内核）+ kvc_scatter_rows_alllayer.cpp（导出链）
          kvops_ext（fatbin 显式加载 + WithHostArgs launch）
xds/      EmsPool（建池/批量导出/三表 [channel×n_core, kv_addr, rope_addr]）
torchops/ torch_npu 两件套原语挂载（_submit / _result；eager=外层显式组合）
tests/    test_dsa_gather_rmt（唯一测试：golden 对拍 + 性能矩阵）
```

## 快速开始（A5/CANN 9.1.0 单节点）

```bash
# 0) 构建（资源节点与 NPU 机各执行一次；NPU 机需 source CANN set_env.sh）
./build.sh

# 1) 资源节点起 emsd（master，HBM 池 1GB）
./build/emsd/emsd --role master --ip <node_ip> --port 27000 --device 3 --loc device \
  --members <node_ip>:27000 --pool-size-gb 1

# 2) NPU 侧唯一测试：torch_npu 两件套接口 + golden 对拍 + 性能（规模全参数化，每进程单 bd 值）
python -m tests.test_dsa_gather_rmt --device 0 --ems-master <node_ip>:27000 \
  --bs 4 --query-count 256 --key-len 4096 --misses-per-req 64 \
  --block-dim 4 --mode both [--perm]
# 大档带宽（8192 records ≈ 9.4MB 净载荷/次）：--query-count 2048 --key-len 131072 --misses-per-req 2048
# 多 bd 矩阵 = 外层循环独立进程（规避 hcomm 多池重建问题，见 doc/dsa_gather_rmt.md §9）：
for bd in 1 2 4 8; do python -m tests.test_dsa_gather_rmt --block-dim $bd ...; done
```

## 性能参考（跨机 950PR，2026-11-26，净载荷口径）

per-token 回源（每 miss 记录 1152B，与原版 dsa_gather 同粒度）：大档 graph bd=2/4/8 →
**1.41 / 2.94 / 5.78 GB/s（10M records/s，通道线性）**；小档 graph 202us（延迟指标）。
真实瓶颈 = WJE 消息提交率（UB 带宽利用率 1.4%）；**同块 miss 合并整块搬运**（backlog
优化）可达带宽口径 ~400 GB/s。跨机与同机时延无差（UB fabric 零损耗）。详见
`doc/dsa_gather_rmt.md` §8.1。

## 运行前提

- `ASCEND_MF_STORE_URL` 不需要（本原型走自有 emsd 控制协议，端口 27000/60001）
- CANN 9.1.0（950PR 实证；`ASCEND_HOME_PATH` 指向安装目录）
- Python ≥3.8 + torch-npu；NPU 进程须能访问 emsd 控制端口

## 许可证

Mulan PSL v2 — 参见文件头部和 LICENSE 文件。
