# Qwen3.6-35B-A3B-FP8 基线

基线ID：`qwen3_6_35b_a3b_fp8-v1`。模型revision：`c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`；Transformers源码commit：`f90e93c4f83974f634860e246cdfd2c68666d605`。

| 文件 | 职责 |
| --- | --- |
| [config.json](config.json) | 官方原始配置快照，保留完整FP8不量化清单 |
| [generation_config.json](generation_config.json) | 官方生成默认值快照 |
| [baseline.json](baseline.json) | 模型/权重/代码版本、结构尺寸、来源hash与执行范围 |
| [numerics.json](numerics.json) | dtype、Norm、Attention、GDN、MoE、Vision与量化数值规则 |
| [functional-v0.json](functional-v0.json) | 已确认的项目功能标准：动态量化、BF16/FP32阈值、精确比较、TopK同分和MoE分桶 |
| [checkpoint_metadata.json](checkpoint_metadata.json) | 全42分片header的dtype/shape核对结果与代表张量 |

详细规则及后续执行记录见 [模型与数值基线](../../../docs/model-numerical-baseline.md)。权重文件保存在仓库外。配置、来源源码和checkpoint header已核对；[A组算子reference](../../../docs/reference-a.md)已部分就绪，完整模型、目标FP8执行环境与模型结果随对应任务验证。

`pto-functional-v0`已确认，说明见 [PTO功能验证标准v0](../../../docs/functional-validation-v0.md)。新功能case记录该ID，并显式声明每个输出的比较规则；实现与测试证据继续按OP/K登记。
