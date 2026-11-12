# 阶段二实施步骤 2：环境、来源与数值基线

执行日期：2026-11-11。本记录仅对应实施说明中的“步骤 2：固定运行基线，先跑通已有框架”。

## 本步执行结果

| 检查项 | 实际结果 |
| --- | --- |
| 仓库路径 | `/home/zhengtk/OpenTPU/LPUv2/lpuv2-tile-lab` |
| 起始提交 | `8b3deb2db5dbda31b100e3416d3f7a12b0e837ae` |
| 本地工作分支 | `b/golden-stage2`，保留用户指定的工作目录 |
| Python | 3.13.9 |
| PyTorch | 2.12.0+cu130，本阶段参考计算使用 CPU |
| pytest | 8.4.2 |
| 已有框架检查 | `python3 -m pytest tests/framework -q`：34 passed，0 failed |

本步使用已有环境，没有安装或升级软件包，没有下载模型权重，没有新增算子、Golden 或 Tile 实现。

## 固定的模型与计算来源

- 模型：`Qwen/Qwen3.6-35B-A3B-FP8`。
- 模型 revision：`c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`。
- 主干计算参考：Transformers commit `f90e93c4f83974f634860e246cdfd2c68666d605`，来源以 `baseline.json` 为准。
- 已获取该 commit 的模型源码并核对 SHA256：`45a0f0d5ad9f6e2a57efec82da1b2b8d7f56f15b219ec741476117e2f58efcac`，与仓库基线完全一致。

本机另有 Transformers 5.8.1，版本与上述源码 pin 不同。现有框架检查不依赖运行该模型；后续做来源实现的执行对照时，需要使用与固定来源相符的独立环境或明确记录适配。本步的源码核对不代表完整模型已执行。

## 沿用的数值规则

已经读取并核对：

- [baseline.json](../configs/models/qwen3_6_35b_a3b_fp8/baseline.json)：模型配置和固定来源。
- [numerics.json](../configs/models/qwen3_6_35b_a3b_fp8/numerics.json)：算子公式、数据类型和转换位置。
- [functional-v0.json](../configs/models/qwen3_6_35b_a3b_fp8/functional-v0.json)：项目功能验证标准。
- [模型与数值基线说明](model-numerical-baseline.md)、[分工](work-allocation.md)、[kernel 组织](kernel-organization.md)和[验证约定](validation.md)。

当前继续采用 `pto-functional-v0`：进入 FP8 矩阵乘的激活按 K 方向每 128 元素分组；FP32 scale、零组和尾组规则、最近偶数舍入、FP32 GEMM 累加与 BF16 输出均沿用现有定义。BF16/FP32 容差、Top-K 同分规则与专家分桶约定保持原定义。此前的 NVFP4 调查不改变本轮基线。

后续 Golden 的运行约定为推理模式、显式 dtype/cast，不统一开启 autocast。此处只记录约定，尚未实施相关算子。

## 检查证据与完成边界

详细环境、配置文件 SHA256 和框架检查结果保存在 `artifacts/golden-stage2/`。该目录遵循仓库现有忽略规则，不提交运行产物。

框架测试确认 Case、逐输出比较、初始输入/状态隔离与产物记录等现有机制可用。它不表示任何模型算子、PTO kernel 或完整模型已经验证。算子登记与数值 profile 未修改。

先前超出步骤 2 的试写代码、用例、测试和运行结果已移出工作仓库，保存在：

```text
/home/zhengtk/Research/pto-learning/golden-stage2-reverted-nqF4UD
```

当前仅保留本步骤的基线记录、本地分支，以及已单独获得授权的 CodeGraph 索引。
