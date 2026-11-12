# 验证约定

共同遵循 [模型与数值基线](model-numerical-baseline.md)、[数值契约](../configs/models/qwen3_6_35b_a3b_fp8/numerics.json)和已确认的 [PTO功能验证标准v0](functional-validation-v0.md)。当前Python Tile与同profile参考程序的量化、误差和TopK规则由 [functional-v0.json](../configs/models/qwen3_6_35b_a3b_fp8/functional-v0.json)固定；实际后端结果另行记录。

| 层级 | 比较对象 | 完成时记录 |
| --- | --- | --- |
| Golden | 固定参考函数/官方实现与独立reference | 来源revision、配置、dtype、公式及中间结果 |
| Kernel/OP | 同输入golden与实际Python Tile程序或后续PTO后端输出 | 执行方式、操作组合、Tile配置、有效范围、输出/索引/状态及容差 |
| 子图/层 | Vision、GDN/FullAttention+MoE、MTP | 节点与边界输出、残差、cache和状态交接 |
| 模型 | Prefill、连续Decode及后续多模态/MTP | 层输出、logits、生成结果及全部可写状态 |

部分Tile、多Tile尾块、零值/常量行、非零初始状态、空专家和多请求隔离随相关实现加入。比较每个命名输出；GDN State、Conv历史、KV和路由索引单独记录。

Python Tile功能验证、实际PTO CPU/NPU及LPU功能模型验证分别登记。Python结果用于确认当前程序的数值、索引和状态语义；硬件同步、周期和资源结论随对应目标环境验证。

## 已确认的功能验收标准

| 输出或检查对象 | `pto-functional-v0`规则 |
| --- | --- |
| BF16浮点输出 | `atol=1e-2, rtol=1e-2` |
| FP32输出、归约结果与GDN State | `atol=1e-5, rtol=1e-4` |
| 搬运、重排、整数索引、计数和mask | 精确一致 |
| 量化FP8 payload与激活scale | 按v0量化算法精确一致 |
| shape、dtype | 符合接口约定 |

按`abs(actual-reference) <= atol + rtol*abs(reference)`逐元素检查，任一元素超限即失败。主输出、KV、GDN State和Conv历史分别检查；输出或状态出现NaN/Inf判失败。TopK按分数降序，同分按专家ID升序，并保持值、索引和路由权重对应。

动态激活量化使用每Token、K轴128元素分组，FP32 scale为反量化乘数，非零组`scale=amax/448`；FP8格式为`float8_e4m3fn`，截断到`[-448,448]`后最近偶数舍入。全零组、尾组及异常输入规则见 [完整标准](functional-validation-v0.md)。

## 当前阶段的Python Tile功能验证

以官方PyPTO main为来源，在Python层编写Tile程序，与独立golden比较。Golden可以直接表达完整算子，Tile程序展开基础操作、分块、归约、类型转换和状态顺序；所需补充能力配套独立语义测试。采用相同的随机输入和权重，先检查小尺寸，再补充模型相关真实维度及边界；真实权重在集成阶段接入。

当前按 [一个OP对应一个逻辑kernel](kernel-organization.md) 组织16个实现入口。内部子阶段可导出中间结果和单独排错，完整kernel的完成依据是对应OP的全部输出与状态通过；Prefill/Decode、recurrent/chunk分别登记路径结果。子图和层由这些OP/kernel组合，跨OP融合在后续架构映射阶段评估。

运行时保存实际PyPTO commit、Python/Torch版本、数值profile、随机种子、程序或IR，以及所有命名输出的结果。Python阶段通过后可据此整理LPUv2指令候选能力；实际PTO后端和LPU架构约束随相应程序接入验证。公共能力的主责、交付顺序及状态记录见 [两人分工与公共协作约束](work-allocation.md)。

## 先运行公共框架检查

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/framework -q
```

当前主线框架测试覆盖比较器、用例配置、输入/状态隔离和复现产物。测试中的小型回调统一标记为 `framework_fixture`。B路线已接入独立reference、Python Tile实现及对应测试，登记为CPU功能验证；该状态不代表原生PTO/LPU后端或完整模型已经验证。

## 公共入口与原pymodel的对应关系

| 原pymodel的职责 | 本仓库入口 |
| --- | --- |
| `test.py` 组织输入、算子、reference与断言 | `tests/operators/`、`tests/kernels/` 等pytest用例；公共执行入口为 `pymodel/task_gen/validate.py::run_case` |
| `utils.py` 生成参数、比较结果 | `pymodel/common/case.py::Case` 固定参数与seed；`pymodel/utils/compare.py::compare_outputs` 比较所有命名输出 |
| `gen_taskbin.py` 保存输入、输出及指令 | `run_case` 保存tensor、case、环境及比较报告；后端有真实指令格式后再接入指令导出 |

Reference和被测实现分别接收一份深拷贝的输入及参数，可以原地更新自己的状态。拷贝保留输入tensor之间的存储别名及stride；同一case的reference、被测实现和外部原始输入分别隔离。框架使用inference模式，保留各算子的dtype/cast，不开启统一autocast。

## Case与比较规则

[示例case](../cases/examples/op04-residual-bf16.json)定义OP/K编号、数值profile、seed、参数及输出检查规则。输入tensor由算子用例显式构造，随机数使用 `case.generator()`；JSON不承担通用计算图或Tensor生成DSL。新功能用例填写`numerical_profile="pto-functional-v0"`，并按上表显式配置每份输出的检查。精确浮点比较填写`atol=0, rtol=0`，整数和布尔输出不填浮点容差。现有OP-04零容差用例继续用于严格残差检查；Case加载器按显式规则执行。

每个回调返回 `dict[str, torch.Tensor]`，例如 `{"out": ..., "state_out": ...}`。case必须声明reference返回的全部输出；比较器报告缺失或多余的输出。

- 浮点输出同时检查shape、dtype与每个元素的 `abs(actual-reference) <= atol + rtol * abs(reference)`。每个输出显式提供atol/rtol；接近零的参考值也参与比较。
- 整数和布尔输出精确比较，不使用浮点容差。专家索引需要比较int64 / uint32等不同存储类型时，显式设置 `logical_indices=true`，在不丢失整数值的int64域比较。
- 默认所有NaN/Inf都导致失败，包含双方同为Inf的情况。需比较含合法无穷值的中间mask时，再为该用例明确专门策略。
- 报告按输出记录最大绝对/相对误差、最坏坐标、不匹配数量/比例和非有限值数量。浮点分析转FP64，执行和保存的tensor保留原dtype；整数比较保留精确性。
- 相同输出形状及数值不能替代dtype检查。TopK同值顺序、有效区域和padding语义由对应算子用例显式验证。

## 后续PTO CPU程序的接入示例

下面展示pytest用例的组织方式。`residual_ref`与`residual_pto_cpu`由OP-04实现任务提供，目前仓库尚未提供这两个函数。

```python
import torch

from pymodel.common.case import load_case
from pymodel.task_gen.validate import run_case


def test_residual_bf16(tmp_path):
    case = load_case("cases/examples/op04-residual-bf16.json")
    p, generator = case.parameters, case.generator()
    shape = (p["rows"], p["cols"])
    inputs = {
        "x": torch.randn(shape, generator=generator, dtype=torch.bfloat16),
        "residual": torch.randn(shape, generator=generator, dtype=torch.bfloat16),
    }
    result = run_case(
        case, inputs=inputs,
        reference=lambda t, p: {"out": residual_ref(t["x"], t["residual"])},
        implementation=lambda t, p: {"out": residual_pto_cpu(t["x"], t["residual"])},
        backend="pto_cpu", artifact_root=tmp_path,
        backend_details={"target_family": "<actual target>", "binary_sha256": "<actual build hash>"},
    )
    assert result.passed, f"{result.report}\nartifacts: {result.run_dir}"
```

`backend`记录调用路径，实际PTO覆盖依据对应适配器和kernel源码审核，并在后端详情中记录实际编译器、binary hash、target和device。当前框架只提供回调接口，不提供PTO编译器或执行适配器；实现缺失会报错，运行异常保存诊断后重新抛出。

## 运行产物与复现

每次运行在 `artifact_root/<case_id>/<run_id>/` 建立独立目录。run_id含UTC时间和随机后缀，两人并行或重复运行时分别保存结果。默认artifact_root为当前目录的 `artifacts/`；pytest中可用 `tmp_path`，排错时改为需要保留的路径。

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | case、数值profile、backend及其详情、Python/Torch版本、repo/PTO版本、工作区状态、回调源码hash和输入shape/dtype/stride/device |
| `inputs.pt` | 本次输入，包含输入状态；在执行前保存 |
| `reference.pt` / `actual.pt` | 对应路径成功返回的全部命名tensor；保留BF16、FP32和整数dtype |
| `report.json` | 各输出检查结果；异常时记录失败阶段、异常类型和信息 |
| `run.log` | 成功/失败摘要，或异常traceback |

读取tensor文件使用 `torch.load(path, map_location="cpu", weights_only=True)`。callback执行失败时保留已完成阶段的产物；比较失败返回 `passed=false`，pytest用例必须断言该字段。框架本身不提升OP/K登记状态，实际通过的测试与后端证据由实现PR记录。
