# XDS KV Cache AIV 直驱卸载原型（M1 骨架）

按《XDS_KVCache_AIV直驱卸载_软件设计方案.md》M1 范围搭建：单仓三组件（ems / kvops / xds），
数据面为 HCOMM AIV 内核直驱（复用 aiv_urma_poc 方案 B 全套：fatbin 显式加载链 + WithHostArgs +
args 扁平化 + overflow 尾槽 + MTE3 status 写回 + c310 epilogue）。

## 目录

```text
common/     协议（kv_proto.h）+ 行归属映射（kv_layout.h）+ hcomm 复用头
ems/
  emsd/     守护进程（C++）：master/worker、段式切分、host 池、UB_CTP host 建链（server）
  ems_client/  框架侧客户端（Python）+ _cli pybind（NPU 卡侧 AIV client 建链）
kvops/
  kernel/   AIV 内核（fatbin）：kvc_gather_rows / kvc_scatter_rows_alllayer / kvc_scatter_rows
  kvops_ext/   _kvops pybind（launch 封装，图捕获安全：无内部同步/无 host 读 device）
xds/        mempoolhost.py（适配层单文件）+ xds_decode.py（decode 流程驱动）+ kv_layout.py
test/       本地单测（映射/分配器/协议，无 NPU 依赖：python -m unittest discover -s test）
```

## 构建与部署

前置：资源节点部署 noDevice 基线 libhcomm（hcomm_no_device 构建，`HCOMM_FORCE_NODEVICE=1`）。

```bash
# 1) emsd（资源节点 × N，第一台为 master；members=静态成员表 ip:ctrlPort[:hcommPort]）
./build.sh emsd
./build/emsd/emsd --role master --ip <nodeA_ip> --port 27000 \
    --members <nodeA_ip>:27000,<nodeB_ip>:27001 --pool-size-gb 8
./build/emsd/emsd --role worker --ip <nodeB_ip> --port 27001 --pool-size-gb 8

# 2) pybind 扩展 + 内核（NPU 机，先 source CANN set_env.sh）
pip install pybind11
./build.sh ext
./build.sh kernel          # 产物 fatbin .o（SOC_VERSION 可 export 覆盖，默认 ascend950pr_9599）

# 3) 本地单测（任意机器）
python -m unittest discover -s test -v
```

## V0 验证步骤（建链）

```bash
# NPU 机：EID 获取参照 aiv_urma_poc（rootinfo 或枚举），随后
python -m xds.xds_decode --ems-master <nodeA_ip>:27000 --device 0 --eid <72hex> \
    --layers 2 --layer-rows 1024 --prefill 256 --steps 5
```

- PoolInfo 下发与逐段 LINK_EXCHANGE + AIV client 通道 READY 即 V0 通过；
  跨模式 socket（HDC↔PEER）若触发异步 connect 问题，预案：hcomm_cli_ext.cpp 的
  `cli_link_channel` 前置同步 connect 适配（参照 hcomm_no_device 个人提交 9b2cb0c9d 模式）。
- M1 出口：V0 全过 + V2 行数 {1,16,256} 数据校验通过（xds_decode 抽查校验内置）。

## 关键实现锚点

- 行归属映射：`common/kv_layout.h` ↔ `xds/kv_layout.py`（单测对拍；层无关，每层独立均衡）
- launch 参数：`kvops/kernel/kvc_args.h`（host/内核共用布局；args+8B overflow 尾槽）
- status 槽：64B/槽，errCode u32 | doneSeq u32<<32，`poll_status` 等 doneSeq==1
- 内核过滤：N 节点共享 indices，`(row/S)%N == myNode` 行归属过滤（免 host 分组，图捕获安全）
