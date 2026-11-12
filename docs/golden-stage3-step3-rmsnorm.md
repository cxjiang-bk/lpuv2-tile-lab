# 阶段3步骤3：RMSNorm Tile程序

本步完成普通RMSNorm（OP-02/K-02）。程序按列分块，但每一行只有一份统计结果，覆盖该行完整D维。入口为[kernels/pto/common/rmsnorm.py](../kernels/pto/common/rmsnorm.py)，运行例子为[tests/kernels/test_rmsnorm_tile.py](../tests/kernels/test_rmsnorm_tile.py)。

## 接口与数值规则

| 参数 | 含义 |
| --- | --- |
| `x` | BF16的`[M,D]`输入，每行独立做RMSNorm |
| `weight` | BF16的`[D]`参数，实际乘数为`1+FP32(weight)` |
| `out` | 调用方分配的BF16 `[M,D]`输出，与输入和权重使用独立存储 |
| `eps` | 正的FP32标量，模型使用`1e-6`；入口默认值也是`1e-6` |

调用方可将连续的`[B,S,D]`或其他前导维视为`[M,D]`，输出采用同样的视图。测试使用`view(-1,D)`，不复制数据；返回结果仍保留原始形状。当前调用约定要求输入、输出能够以这种方式展平。

公式沿用[模型数值基线](model-numerical-baseline.md)：

```text
x32 = FP32(x)
inv_rms = rsqrt(sum(x32*x32, axis=-1) / D + eps)
y = BF16((x32 * inv_rms) * (1 + FP32(weight)))
```

归约、归一化和乘权重都保持FP32，最后只做一次BF16回转。这是普通RMSNorm的定义；GDN gated Norm的权重解释和转换位置由其独立实现维护。

## 每一行如何计算

1. 把该行的FP32平方和清零，每次读一个`[1,128]` BF16 Tile，再转FP32。
2. 将尾块无效列填零，计算当前块的平方和，累加到这一行的平方和中。归约使用一块可重复使用的`[1,128]` FP32 scratch Tile。
3. 所有列块完成后，除以真实D，加eps，再计算rsqrt，得到该行唯一的归一化系数。
4. 第二遍读取输入和对应权重块，把该系数广播到块内每列，完成归一化及乘`1+weight`，最后转BF16写回。

例如D=259，三块有效列数为128、128、3；三块平方和合并后除以259。只需保留行统计量和当前块，不保存整行中间结果。128列是本轮功能分块选择，硬件宽度由后续架构映射确定。

## 本轮PyPTO表达及边界

执行方式沿用[步骤1](golden-stage3-step1-execution.md)：PyPTO `7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b`，通过`specialize`生成IR，再使用官方`torch_codegen`在CPU执行。

| 操作 | 用途 |
| --- | --- |
| `load/store`、`cast` | 分块读写及BF16/FP32转换 |
| `fillpad`、`row_sum`、`add` | 有效尾块处理、块内归约及跨块平方和累加 |
| `mul/add/rsqrt` | 均方、eps及归一化系数计算 |
| `row_expand_mul` | 将该行统计结果广播到当前块 |
| `tile.slice` | 写回前明确当前块的有效列数，保留物理`[1,128]`形状 |

当前Torch生成器的cast和算术结果不会保留Tensor上的动态有效区属性，虽然IR类型中仍有该信息。因此写回前使用带`valid_shape`的同形状slice视图明确写回范围。该操作不增加数值计算或舍入；本轮没有修改上游库或增加自定义Python语义。此处是生成器表达方式的适配，不能直接据此认定LPU硬件需要增加一条指令。

## 如何运行及验证

```bash
.venv/bin/python -m pytest tests/kernels/test_rmsnorm_tile.py -q --basetemp=artifacts/golden-stage3-step3/pytest
```

测试沿用`Case/run_case`，同一输入分别执行独立Golden和生成的Tile程序。适配仅在测试侧展开视图；被测计算不调用Golden。所有输出先填NaN，再逐元素比较形状、BF16类型和数值，使用既定`atol=1e-2, rtol=1e-2`。

| 用例 | 重点 |
| --- | --- |
| `[2,3,256]`，eps=`1e-6` | 前导轴展平，完整两块列归约，保持原形状 |
| `[2,2048]`，eps=`1e-6` | 跨16块累加，归一化范围始终是整行 |
| `[2,259]`，eps=`0.25` | 尾块仅3列；用较大eps检查该参数确实参与计算，模型eps仍为`1e-6` |

输入与权重使用固定随机种子；后续列块放大4倍，以区分整行归一化和错误的逐块归一化，并包含零行检查有限输出。

三个用例均通过，本轮BF16样本与Golden的最大绝对误差均为0。IR、生成代码、输入、两侧结果和比较报告保存在`artifacts/golden-stage3-step3/pytest/`。这是Python层功能验证，未进行PTO CPU/NPU设备执行或性能验收。

全套150项测试通过，项目Ruff及差异检查通过；全套结果保存为`artifacts/golden-stage3-step3/all-test-results.xml`。

后续步骤4实现公共Linear、量化和Softmax；本步只交付RMSNorm及其分块归约、广播用法。
