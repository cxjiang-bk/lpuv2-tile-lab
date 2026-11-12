# OP 与 kernel 登记

`operators.json`登记 16 条 OP；`kernels.json`登记一一对应的逻辑 kernel。内部步骤、Prefill/Decode 和 recurrent/chunk 路径保留在同一 kernel。

| 字段 | 含义 |
| --- | --- |
| `golden_status`、`golden_scope` | 独立 reference 与模型来源核对的范围 |
| `pto_implementation_status/pto_validation_status`、kernel 的 `implementation_status/validation_status` | PTO 程序实现与验证状态，包括经 PyPTO 编译生成的 PTO 程序；结合 backend 阅读 |
| `python_implementation_status/python_validation_status` | 沿用已有字段名，登记 Python/PyPTO 前端实现与对应后端的验证状态 |
| `python_entry_point/python_tile_program` | 实际调用入口与 PyPTO 程序文件 |
| `python_backend`、`native_backend` | 实际执行路径；当前 A 组使用 `hw_pto_cpp_sim`，表示 PTOAS 生成 C++ 后在 simpler a2a3sim 执行 |
| `native_validation_status`、`native_validation_evidence` | 原生功能仿真状态及 case、报告来源；阻断项单独记录 |
| `model_validation_status` | 子图和模型集成验证状态，独立于算子验证 |

原生 PTO C++ 仿真 通过结果对应具体输入与数值规则，不表示 NPU、LPU 或完整模型通过。逐项记录见 [native-validation-summary.json](native-validation-summary.json)，候选需求见 [a-instruction-candidates.json](a-instruction-candidates.json)。

指令候选清单使用schema v2：`candidates`保留16类需求索引，`basic_instruction_candidates`、`fusion_candidates`和`programming_model_requirements`分别登记29项基础功能、12项融合项（含三维窗口展开）和6项编程模型约定。`source_index`对应kernel文件与符号；`supporting_case_operator_ids`表示有相关算子用例，不能推导为候选指令的独立一致性测试已通过。`acceptance_to_add`记录后续指令模型需要补的定向验收，FP8直接指令与carrier数值路径分别标记。
