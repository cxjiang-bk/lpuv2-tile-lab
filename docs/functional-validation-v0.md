# PTO 功能验证标准 v0

Profile ID：`pto-functional-v0`。2026-11-10已确认，适用于当前Python Tile功能验证。机器可读规则见 [functional-v0.json](../configs/models/qwen3_6_35b_a3b_fp8/functional-v0.json)，来源为 [协作确认清单](https://tju-opentpu.feishu.cn/wiki/EhbLwG01Yin5mqk2SfncHKlPnNz) revision 54中的P01、P02、P05、P06、P07。

本标准补齐项目采用的量化、比较和路由规则；模型权重、转换位置及状态精度继续遵循 [模型与数值基线](model-numerical-baseline.md)。Reference与Tile实现使用相同profile，分别执行各自计算。

## 1. 执行与输入

使用hw-native-sys PyPTO language/Tile API构造程序，并在 PTO C++ 模拟器执行功能测试，具体发行版本见 `data/upstream.lock.json`。每次运行保存实际版本或commit、Python/Torch版本、随机种子、输入和权重的生成方式及profile ID；原生编译/仿真与Python语义执行分别记录。

单算子使用随机输入和随机权重，先跑小尺寸，再覆盖模型相关真实维度、逻辑shape、dtype、有效区域、offset、索引和状态。真实checkpoint权重在集成阶段接入；片上物理容量在后续存储与架构映射阶段评估。

## 2. 动态激活量化

仅量化进入FP8矩阵乘的BF16激活，其余BF16计算和FP32状态保持既定精度。

| 项目 | 已确认规则 |
| --- | --- |
| 分组 | 每个Token沿特征维度K，每128个元素一组 |
| FP8格式 | `torch.float8_e4m3fn`，有限范围`[-448,448]` |
| 激活scale | FP32，作为反量化乘数，与FP8数据成套传递 |
| amax与scale | FP32计算有效元素的`amax=max(abs(x.float()))`；非零组`scale=amax/448` |
| 转换 | FP32计算`x/scale`，截断到`[-448,448]`，按最近偶数舍入为FP8 |
| 全零组 | `scale=1`，量化数据全部为0 |
| 尾组 | 有效元素参与amax计算，填充位置置0 |
| 异常输入 | 输入含NaN或Inf判失败 |

恢复关系为：

```text
x_approx = float(x_fp8) * activation_scale
```

数据与scale始终保持对应，MoE重排Token时同步维护该关联。权重和checkpoint scale保留现有值；FP8 GEMM按K分组应用相应激活和权重scale，在FP32中累加，最终输出BF16。

## 3. 误差与精确比较

以下阈值用于Tile实现与采用相同量化规则的独立reference比较。量化引入的误差与实现相对该reference的误差分别记录。

| 比较对象 | 已确认标准 |
| --- | --- |
| BF16浮点输出 | `atol=1e-2`，`rtol=1e-2` |
| FP32输出、归约结果及GDN State | `atol=1e-5`，`rtol=1e-4` |
| 纯搬运、重排 | 精确一致 |
| 整数索引、计数、mask | 精确一致 |
| 量化FP8 payload、激活scale | 按第2节的算法精确一致 |
| shape、dtype | 符合接口约定 |

逐元素判断：

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

任一元素超限即失败。主输出、KV、GDN State和Conv历史分别检查；输出或状态出现NaN/Inf判失败。报告记录结果名称、不匹配位置及差异。

Case填写`numerical_profile="pto-functional-v0"`，并在每个命名输出的`OutputCheck`中显式填写上述规则。精确浮点检查使用`atol=0, rtol=0`；整数和布尔比较不填浮点容差。现有Case加载器按case中的显式规则执行，profile文件提供共同配置依据。

## 4. TopK同分处理

按分数降序选择，分数完全相同时按专家ID升序。参考程序先按专家ID升序组织输入，再稳定地按分数降序排序并取前K项；分数、专家ID和后续路由权重保持对应。

该同分处理属于项目功能验证v0约定。固定模型源码调用的`torch.topk`及其原后端行为作为模型来源另行记录。

## 5. MoE分桶与合并

按专家分桶组织Token路由记录，统计各专家的有效数量Ne，再Gather输入、执行专家计算，按原Token编号及路由权重合并。保留原Token ID、路由权重，必要时保留TopK slot；FP8数据与scale保持关联。

Top-8路径形成`8*T`条记录。空专家跳过，尾行按Ne处理；合并时每项贡献计入一次。检查空桶、热门专家、尾行和重排后的对应关系。桶索引、边界及具体字段布局随Dispatch/Combine公共实现细化。

## 6. 状态与变更

本文件登记已确认的标准。量化、TopK、分桶及各kernel的实现和测试结果，按OP/K与实际case分别记录。变更本profile时更新版本，关联受影响的reference、kernel和用例后重新验证。
