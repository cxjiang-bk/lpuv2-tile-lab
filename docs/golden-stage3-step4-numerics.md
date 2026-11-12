# 阶段3步骤4：公共Tile数值函数

本步提供后续LM Head、Attention、MoE和MTP投影共用的六个函数。计算主体使用PyPTO的Tile读取、计算和写回；测试通过项目内补充适配的Torch codegen在CPU执行，分别与独立Golden比较。已有Golden、上游源码及步骤3实现保持原样。

## 调用接口

所有Tensor入口采用二维逻辑形状，调用方提供输出空间。连续的高维输入可以通过`view(-1,K)`展开前导轴，输出采用对应视图。输入与输出使用独立存储；以下例子均提供合法输入。

| 函数及位置 | 输入 | 输出与规则 |
| --- | --- | --- |
| `bf16_linear`，`kernels/pto/common/linear.py` | BF16 `x[M,K]`、`weight[N,K]` | BF16 `out[M,N]`；计算`x @ weight.T`，当前入口不带bias |
| `dynamic_quantize`，`kernels/pto/common/quantization.py` | BF16 `x[M,K]` | FP8 E4M3FN `payload[M,K]`、FP32 `scales[M,ceil(K/128)]` |
| `fp8_linear`，`kernels/pto/common/linear.py` | BF16 `x[M,K]`、FP8 E4M3FN `weight[N,K]`、BF16或FP32 `weight_scale[ceil(N/128),ceil(K/128)]` | BF16 `out[M,N]`；内部量化输入，按K组缩放、累加 |
| `softmax`，`kernels/pto/common/softmax.py` | BF16或FP32 `x[M,D]` | FP32 `out[M,D]`；沿每行D维归约，允许用`-Inf`屏蔽部分位置，每行至少有一个有限值 |
| `sigmoid`、`silu`，`kernels/pto/common/activations.py` | BF16或FP32 `x[M,D]` | BF16或FP32 `out[M,D]`；中间计算FP32，在输出处按需转换 |

当前PyPTO版本不接受同类型Tile cast，且不能直接以dtype比较选择静态分支，因此以下开关由调用方按实际类型填写，使用`pl.constexpr`在特化时选择转换路径：

| 开关 | 默认值 | 何时设置 |
| --- | --- | --- |
| Softmax的`input_bf16` | `False` | 输入BF16时设为`True` |
| 激活的`input_bf16`、`output_bf16` | 均为`True` | 输入或输出使用FP32时，将对应开关设为`False` |
| FP8 Linear的`scale_bf16` | `True` | 权重scale使用FP32时设为`False` |

这些开关只控制是否生成cast，不是运行时指令参数。形状和dtype仍由实际Tensor确定。模型使用的scale数值原样传入，`weight_scale_inv`后缀不表示需要再次求倒数。

## 计算怎样分块

**BF16 Linear**按M/N/K方向分为16×16×64。每个输出块先沿完整K轴合并FP32部分积，最后一次转BF16。M/N/K尾部补零参与计算，写回只保留有效行列。Left、Right和Acc是PyPTO的操作数类别，本步尚未确定LPUv2 Buffer映射。

**动态激活量化**每个Token沿完整K轴按128元素分组，组起点固定为0、128、256等。每组独立计算FP32 `amax=max(abs(x))`，非零组`scale=amax/448`，零组`scale=1`。随后将`x/scale`限幅到`[-448,448]`，转为FP8 E4M3FN。尾组补零，其payload写回保留真实K列。共享的`_quantize_group`是一个PyPTO辅助函数，独立量化入口和FP8 Linear都调用它。

**FP8 Linear**每次处理1行、16个输出通道和128个K元素。对当前输入组量化后，与对应FP8权重块做点积，再按顺序乘该Token的激活scale、该128×128权重块的scale，最后加到FP32结果。16列输出块不会跨越128列权重scale边界。所有K组完成后转BF16。当前为简洁的功能实现，同一Token在不同输出块可能重复量化；复用payload和scale的优化在后续集成时评估。

**Softmax**采用三遍读取：第一遍得到整行最大值，第二遍合并`exp(x-max)`的整行总和，第三遍归一化写回。统计时尾块填`-Inf`，其指数贡献为0。最大值从第一个真实数据块建立，不依赖任意选取的负数常量。

**Sigmoid/SiLU**按128列逐块展开`exp`、加法、除法和必要乘法。SiLU在FP32中完成`x*sigmoid(x)`，再按输出类型转换；不会在中间先将sigmoid转为BF16。

## 执行适配解决什么问题

PyPTO版本沿用`data/upstream.lock.json`中的`7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b`。其前端已能表达上述Tile操作，但原始Torch codegen存在以下缺口。项目在`pymodel/isa/torch_codegen.py`继承其生成器，补充这些基础语义；不改上游文件或进程全局操作表。

| 已核实的问题 | 本地处理 |
| --- | --- |
| 只有一个嵌套循环的语句体可能漏掉循环头 | 显式分派ForStmt、SeqStmts和IfStmt，保持IR中的循环和分支 |
| BF16点积生成`matmul(BF16,BF16).float()`，造成每个K块先舍入 | 在CPU模拟点积时先将操作数精确扩展为FP32，保留部分积和累加的FP32语义 |
| dtype map缺少FP8 E4M3FN，默认变成FP32 | 补充FP8 cast和shape/dtype检查，使用Torch E4M3FN转换；用已知字节验证最近偶数舍入及次正规数 |
| `maximums/minimums`被生成为Tensor与Python标量的maximum/minimum | 生成等价的标量clamp |

写回前的有效区域仍由kernel中的Tile视图明确表达。整个算子由IR生成的程序计算，Golden只用于比较。实际后端标签为`pypto_torch_codegen_adapted`，不表示原始生成器、PTO CPU/NPU或LPU硬件已通过验证。原生后端还需核对布局、存储通路、转换和累加约束。

## 一个最小调用

在仓库`.venv`中运行Python时，可按以下方式验证BF16投影；其他函数采用相同执行路径：

```python
import torch
from kernels.pto.common.linear import bf16_linear
from pymodel.isa.torch_codegen import torch_codegen

x = torch.randn(3, 130, dtype=torch.bfloat16)
weight = torch.randn(19, 130, dtype=torch.bfloat16)
out = torch.empty(3, 19, dtype=torch.bfloat16)
program = bf16_linear.specialize(x, weight, out)
namespace = {}
exec(compile(torch_codegen(program), "<linear_tile>", "exec"), namespace)
namespace["bf16_linear"](x, weight, out)
```

例如BF16输入的Softmax使用`softmax.specialize(x, out_fp32, input_bf16=True)`；生成函数运行时仍只接收`x, out_fp32`。FP8量化的调用参数为`x, payload, scales`，两份输出均由调用方预分配。

## 本轮验证

```bash
.venv/bin/python -m pytest tests/kernels/test_numeric_tile.py -q --basetemp=artifacts/golden-stage3-step4/pytest
.venv/bin/python -m pytest -q
```

专项12项通过：包括完整矩形块和M/N/K尾块、257列量化的不同幅度与零组、FP8转换中点/次正规数字节、不同N/K权重块scale、BF16/FP32 scale、masked Softmax、BF16/FP32激活，以及能检出K块提前舍入的抵消例子。所有浮点容差沿用功能标准v0，没有降低随机输入幅度来回避误差。

本轮量化payload字节及scale精确一致；BF16/FP8 Linear样本最大绝对误差为0；Softmax最大绝对误差不超过`5.97e-8`，FP32激活不超过`1.20e-7`。全套162项通过，Ruff和差异检查通过。

测试文件为`tests/kernels/test_numeric_tile.py`。IR、生成Torch代码和Case输入/结果/报告保存在`artifacts/golden-stage3-step4/pytest/`，全套XML为`artifacts/golden-stage3-step4/all-test-results.xml`；运行产物不进入Git。

本步没有新增模型OP/K编号。下一步为LM Head、Concat和MTP输入投影的完整Tile入口。
