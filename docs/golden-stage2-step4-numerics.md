# 阶段二步骤4：公共数值参考

本步骤提供双方可以直接调用的PyTorch数学函数。函数接收Tensor和必要参数，返回Tensor或简单元组；测试直接调用。输入生成、比较及运行产物由测试侧按需组织。

计算规则沿用[模型与数值基线](model-numerical-baseline.md)及[功能验证v0](functional-validation-v0.md)，没有修改模型精度和量化标准。以下属于公共函数，不增加kernel编号。完整Attention、MoE、LM Head及MTP仍按后续步骤实现。

## 1. 已实现的入口

| 函数 / 位置 | 输入 | 返回值 |
| --- | --- | --- |
| `bf16_linear` / [linear.py](../pymodel/ref/linear.py) | BF16 `x[...,K]`、`weight[N,K]`、可选`bias[N]` | BF16 `out[...,N]` |
| `dynamic_quantize` / [quantization.py](../pymodel/ref/quantization.py) | BF16 `x[...,K]` | FP8 `payload[...,K]`和FP32 `scale[...,ceil(K/128)]` |
| `fp8_linear` / [linear.py](../pymodel/ref/linear.py) | BF16 `x[...,K]`、FP8 `weight[N,K]`及配套权重scale | BF16 `out[...,N]` |
| `rmsnorm` / [normalization.py](../pymodel/ref/normalization.py) | BF16 `x[...,D]`、`weight[D]`、`eps=1e-6` | 同形BF16结果 |
| `softmax` / [normalization.py](../pymodel/ref/normalization.py) | BF16或FP32 logits，默认最后轴 | 同形FP32概率 |
| `stable_topk` / [selection.py](../pymodel/ref/selection.py) | BF16或FP32 `scores[...,E]`及`k` | 保持输入dtype的`values[...,k]`和int64 `indices[...,k]` |

所有入口均按推理模式执行，保留输入，不依赖Case、TaskGen或PTO执行模块。各输入需位于同一设备；本轮实际验证CPU路径。

## 2. Linear与量化的关键约定

### BF16 Linear

按`x @ weight.T`的逻辑方向调用`F.linear`，支持原有前导维度和可选BF16 bias。输入、权重、bias和输出保持BF16；内部累加由记录的PyTorch/CPU后端实现决定。没有把整条BF16路径替换成FP32排错路径。

### 动态激活量化

1. 每个Token沿完整K轴每128个元素分组。
2. 用FP32求组内`amax`；非零组`scale=amax/448`，全零组`scale=1`。
3. FP32计算`x/scale`，截断到`[-448,448]`，转换为`float8_e4m3fn`，采用最近偶数舍入。
4. 尾组临时补零后计算；零填充不改变有效元素的绝对最大值。返回的payload裁回逻辑K列，scale保留最后一组。

输入仅包含有效逻辑区域，NaN/Inf报错。非连续视图按逻辑顺序读取；调用方传入的是完整K轴。切分Token行可以独立调用，切分K轴时不能改变原先128元素的分组边界。

scale为反量化乘数：`x_approx=payload.float()*scale`，按所属K组广播。payload和scale分开返回，测试分别检查，后续重排也须保持关联。

### FP8 Linear

权重输入为`float8_e4m3fn`，权重scale形状严格为`[ceil(N/128),ceil(K/128)]`，接受BF16或FP32的有限正值。checkpoint的`weight_scale_inv`按原值作为乘数传入。

`fp8_linear`内部调用动态激活量化，然后按K组递增处理：

1. 将本组FP8激活和FP8权重转成FP32，计算该组点积。
2. 依次乘本Token的激活scale和对应权重行块的scale。
3. 将贡献累加到FP32结果，所有K组完成后转为BF16。

这里使用FP32运算表达FP8的数学语义；没有执行原生FP8 GEMM后端。量化引入的误差属于目标profile的一部分，例如输入`[1,3]`经过动态量化后，第一项恢复为约0.9643，而不是精确的1。比较被测程序时应使用同profile Golden。

## 3. Norm、概率与选择

普通RMSNorm沿最后轴计算FP32均方、rsqrt和`1+FP32(weight)`，乘法完成后才回BF16。它可供Decoder、Q/K和Final Norm复用；A的GDN gated Norm具有不同的权重解释和转换顺序，应在K-06中按自己的定义实现。

`softmax`使用PyTorch的稳定实现，返回FP32概率。Attention在既定边界显式转BF16；Router继续用FP32完成Top-K和选中概率重归一化，最后按模型要求转换路由权重。函数支持单个位置的`-Inf` mask，但NaN、`+Inf`和整行全被mask都会报错；无效query由Attention调用方处理。

`stable_topk`要求输入最后轴按专家ID升序排列。稳定降序排序保证分数完全相同时，较小专家ID排在前面。返回的值、ID一一对应，函数本身不做Softmax、权重归一化、分桶或专家计算。

## 4. A的最小调用例子

下面模拟GDN所需的FP8投影和BF16 a/b投影，使用小尺寸验证公共调用。192只表示示例投影宽度，完整GDN的head组织、门控和状态仍由A实现。

```python
import torch
from pymodel.ref.linear import bf16_linear, fp8_linear

x = torch.ones(2, 130, dtype=torch.bfloat16)
qkv_weight = torch.ones(192, 130).to(torch.float8_e4m3fn)
qkv_scale = torch.tensor([[0.5, 0.5], [0.25, 0.25]], dtype=torch.bfloat16)
qkv = fp8_linear(x, qkv_weight, qkv_scale)  # [2,192]，BF16

a_b_weight = torch.zeros(4, 130, dtype=torch.bfloat16)
a_b_weight[:, 0] = 1
a_b = bf16_linear(x, a_b_weight)           # [2,4]，BF16
```

可执行检查位于[test_shared_projection_example_for_a](../tests/reference/test_linear.py)，用两个不同输入行核对了投影值。若需要单独查看激活量化结果，调用`dynamic_quantize(x)`即可；`fp8_linear`已有内部量化，不要求调用方预先量化。

## 5. 怎样验证

在仓库根目录执行：

```bash
python3 -m pytest tests/reference tests/framework -q
python3 -m ruff check pymodel/ref tests/reference
```

| 部分 | 检查依据与范围 |
| --- | --- |
| BF16 Linear | 手算选择矩阵、bias、零值、非连续视图、前导维度及显式内积交叉检查 |
| 激活量化 | K=127/128/129/255/256/257，逐Token/组scale，零组和零尾组；手写FP8字节检查普通与次正规数的最近偶数舍入 |
| FP8 Linear | 手算不同N/K块scale及抵消；独立逐组量化后用完整FP64反量化矩阵乘交叉检查；K/N尾部、空batch、视图、错误shape/scale、非有限输入 |
| RMSNorm | 手算值、提前BF16舍入的反例、D=256/2048；与Torch归一化表达及固定模型来源对照 |
| Softmax | 大logits、平移不变性、指定轴、mask、异常行及FP32返回边界 |
| Top-K | 手算同分顺序、BF16舍入后同分、256专家Top-8，及Python按分数/ID排序交叉检查 |

BF16计算结果采用v0的`atol=1e-2,rtol=1e-2`，FP32概率采用`atol=1e-5,rtol=1e-4`；手算可精确表达的结果和量化payload/scale采用零容差，索引精确一致。

本轮另从固定Transformers源码提取原始`Qwen3_5MoeRMSNorm`类，移除hub装饰器后独立执行；没有修改类内部计算。先核对源码SHA256与仓库基线相同，再用同一权重和输入比较D=256/2048，结果均精确一致。这项对照只验证普通RMSNorm类，不代表整个模型运行。

实际环境：Python 3.13.9、Torch 2.12.0+cu130，CPU，默认64线程。模型语义来源为Transformers commit `f90e93c4f83974f634860e246cdfd2c68666d605`；FP8量化和Top-K同分使用项目v0约定，不宣称已执行或匹配其原生FP8后端。

新增公共函数对应85项测试；加上已有Embedding的14项与框架的34项，共133项。详细结果保存于本地`artifacts/golden-stage2-step4/`，运行产物不进入Git。
