# kv_offload_proto — dsa_gather_rmt：DSA miss 记录 AIV 直驱远端回源原型

EMS（external memory server）池上的 `npu_dsa_gather_rmt` 原型：DSA（DeepSeek Sparse Attention）
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
torchops/ torch_npu.npu_dsa_gather_rmt（eager 同步 + submit 两层 API）
tests/    test_dsa_gather_rmt（原生 golden）· test_torch_dsa_gather_rmt（e2e/--graph）
          test_bench_dsa_gather_rmt（性能摸底）
```

## 快速开始（A5/CANN 9.1.0 单节点）

```bash
# 0) 构建（资源节点与 NPU 机各执行一次；NPU 机需 source CANN set_env.sh）
./build.sh

# 1) 资源节点起 emsd（master，HBM 池 1GB）
./build/emsd/emsd --role master --ip <node_ip> --port 27000 --device 3 --loc device \
  --members <node_ip>:27000 --pool-size-gb 1

# 2) NPU 侧跑 golden / e2e（--block-dim 声明每段通道数=核数）
python -m tests.test_dsa_gather_rmt     --device 0 --ems-master <node_ip>:27000 --block-dim 8
python -m tests.test_torch_dsa_gather_rmt --device 0 --ems-master <node_ip>:27000 --block-dim 8 [--perm] [--graph]

# 3) 性能摸底（矩阵：blockDim × 规模档 × eager/graph）
python -m tests.test_bench_dsa_gather_rmt --device 0 --ems-master <node_ip>:27000 \
  --block-dims 1,2,4,8 --scale s2 --mode both
```

## 运行前提

- `ASCEND_MF_STORE_URL` 不需要（本原型走自有 emsd 控制协议，端口 27000/60001）
- CANN 9.1.0（950PR 实证；`ASCEND_HOME_PATH` 指向安装目录）
- Python ≥3.8 + torch-npu；NPU 进程须能访问 emsd 控制端口

## 许可证

Mulan PSL v2 — 参见文件头部和 LICENSE 文件。
