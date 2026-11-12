# A 组 HW PyPTO kernel 与 C++ 仿真

A 组9个完整算子的计算主体统一使用 [hw-native-sys/pypto](https://github.com/hw-native-sys/pypto)，放在 `kernels/pypto/`。独立 PyTorch golden 位于 `pymodel/ref/`；`pymodel/ops/` 检查接口、组织状态及调用实际 kernel。一个 OP 对应一个逻辑 kernel，内部阶段共用同一套数值标准。

## 实现与入口

| OP / kernel | 算子入口 | 计算文件 | 计算范围 |
| --- | --- | --- | --- |
| OP-03 / K-03 | `pymodel.ops.common.tile.layer_norm` | `common.py` | FP32 均值、中心化方差、仿射，BF16 输出 |
| OP-04 / K-04 | `pymodel.ops.common.tile.residual_add` | `common.py` | BF16 残差，窗口偏移、尾块及输出 seed 保持 |
| OP-06 / K-06 | `pymodel.ops.gdn.kernel.gdn` | `gdn.py`、`fp8.py` | FP8 投影、因果 Conv、recurrent/chunk、gated Norm、两类状态 |
| OP-08 / K-08 | `pymodel.ops.vision.kernels.patch_embedding` | `vision_ops.py` | 像素 im2col、矩阵乘、bias、Merge 顺序及网格 |
| OP-09 / K-09 | `pymodel.ops.vision.kernels.vision_attention` | 同上 | QKV、half-split RoPE、分段注意力、Softmax、输出投影 |
| OP-10 / K-10 | `pymodel.ops.vision.kernels.dense_ffn` | 同上 | Linear、GELU(tanh)、Linear |
| OP-11 / K-11 | `pymodel.ops.vision.position.position_embedding` | 同上 | 网格坐标、四邻点索引、FP32 插值权重、位置加法 |
| OP-12 / K-12 | `pymodel.ops.vision.kernels.spatial_merger` | 同上 | LayerNorm、Merge 分组、GELU(erf)、投影 |
| OP-13 / K-13 | `pymodel.ops.vision.kernels.visual_feature_scatter` | 同上 | 视觉特征覆盖指定文本行，保留其余行 |

factory 直接以算子命名，只进行形状特化；返回的 `@pl.jit` 主体按模型计算顺序直接展开，以 `pl.at` 标出编译阶段。im2col、像素投影、位置插值、Attention和GDN主干均在对应入口中顺序阅读；共享FP8原语和独立验证的GDN recurrent/chunk数学算法单独保留。计算使用 `pl.load/store`、归约、显式行/列广播、矩阵乘和转换。Tile 的物理对齐与逻辑 valid region 分开，尾块只读写有效元素。

## 固定执行链

```text
hw-native-sys PyPTO language
    → Tensor/Tile IR
    → PTO IR (.pto)
    → PTOAS 生成 C++
    → GCC 编译共享库 (.so)
    → simpler a2a3sim 执行
    → 与独立 PyTorch golden 比较
```

后端记录为 **`hw_pto_cpp_sim`**。这是实际编译的 PTO C++ 功能模拟，和 TorchCodegen 语义解释、目标硬件精度模拟、NPU/LPU 执行分别登记。执行适配器 [hw_pypto.py](../pymodel/isa/hw_pypto.py) 要求实际 PTO 与 ELF 产物；编译、执行或比较失败直接报错，不回退到 golden。

编译器、运行时、PTOAS 和运行时使用的 PTO ISA 均由 [upstream.lock.json](../data/upstream.lock.json) 固定。`third_party/pto-isa` 保留独立的指令文档参考版本；运行时严格使用它自身锁定的 ISA 库版本，二者分别记录。

## 安装与运行

```bash
bash scripts/setup-hw-pypto.sh
source scripts/hw-env.sh
python -m pytest tests/framework tests/operators -q
python -m pymodel.task_gen.validate_a --case 03
python -m pymodel.task_gen.validate_a --artifact-root artifacts/a-hw
python -m pytest tests/kernels -q
```

本机安装在项目 `.venv` 和独立工具链目录。`scripts/hw-env.sh` 读取忽略入库的 `.env.hw`，可用 `LPUV2_HW_ENV_FILE` 指向本机工具链环境脚本。安装脚本固定上游提交与 PTOAS wheel SHA，使用私有 GCC/Python 工具环境，不向全局 Python 安装依赖。实际完成的安装、编译与数值范围以验证汇总为准。

## 数值与状态

沿用 `pto-functional-v0`：BF16 `atol=rtol=0.01`，FP32 `atol=1e-5, rtol=1e-4`；整数索引、纯搬运、FP8 payload 和激活 scale 精确检查。模型的转换位置与 checkpoint scale 值保持不变。

GDN 的 `bf16_carrier` 路径在 kernel 中完成 K128 动态量化、E4M3FN 解码与分块 scale 乘法。BF16 精确承载有限 FP8 数值，部分和与 State 使用 FP32。主机只扩宽原始权重字节、分配存储和检查接口。该路径验证模型 FP8 数值算法，直接 FP8 硬件指令另行验证；MX 的 K32 量化不替代本项目的 K128 规则。

GDN 同一 K-06 保留 `path="recurrent"` 与 `path="chunk"`。chunk 计算前缀和、因果系数、两个单位下三角系统及块间 State，系数和 State 乘积保持 FP32。Conv 历史保存原始 qkv 投影；State 和历史均对应同一已消费 Token 前缀，输入保持只读。

```bash
python -m pymodel.task_gen.validate_a --case 06 --gdn-decode-continuity
python -m pymodel.task_gen.validate_a --case 06 --gdn-path chunk \
  --gdn-sequence 5 --gdn-chunk-size 3 --gdn-batch 2
python -m pymodel.task_gen.validate_a --case 06 --gdn-path chunk \
  --gdn-chunk-size 3 --gdn-decode-continuity
python -m pymodel.task_gen.validate_fp8 --case quantize
python -m pymodel.task_gen.validate_fp8 --case linear
```

Vision 的像素入口与窗口顺序见 [像素 Conv3D](vision-pixel-conv3d.md)。位置索引在 kernel 内生成，INT32 存储按范围检查后扩宽为接口的 int64，未在主机重新计算位置。

## 产物与验收

每次 case 保存输入、独立 golden、实际输出、逐输出误差及 manifest。`program/` 保存源代码快照与 SHA、特化参数、上游版本、生成的 PTO/C++/ELF 和二进制 SHA。输出、状态、历史和索引分别验收。

迁移完成后的通过范围由 [原生验证汇总](../data/native-validation-summary.json) 管理。结果使用合成输入和权重；模型相关维度、完整模型、真实 checkpoint 与性能分别登记。LPUv2 指令需求见 [A 组指令候选集](lpuv2-a-instruction-candidates.md)。

## 本次验证结果

2026-11-12 顺序展开 kernel 后，统一回归 **155项通过**：116项 reference/框架与39项原生 kernel 检查。9个完整 OP 的基本用例、GDN recurrent/chunk 核心和完整路径、状态交接、Conv3D/im2col、公共窗口及重复执行均通过。新增4项位置网格与 Attention 分段边界检查，位置索引和插值权重精确一致。逐输出误差与源码 SHA 见 [原生验证汇总](../data/native-validation-summary.json)。结果使用固定随机输入和权重；真实 checkpoint、完整模型、性能及直接 FP8 硬件指令分别验收。
