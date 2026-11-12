# pymodel/ref

独立PyTorch golden按算子、子图、层和模型导出输出、索引与状态。

共同输入是 [Qwen3.6-35B-A3B-FP8基线](../../configs/models/qwen3_6_35b_a3b_fp8/README.md)：[模型结构与来源](../../configs/models/qwen3_6_35b_a3b_fp8/baseline.json)、[数值契约](../../configs/models/qwen3_6_35b_a3b_fp8/numerics.json)。计算时保留官方dtype、权重解释及cast位置，实际执行环境和比较规则随case记录。

当前Python功能reference与Tile实现采用 [PTO功能验证标准v0](../../docs/functional-validation-v0.md)，共同配置为 [functional-v0.json](../../configs/models/qwen3_6_35b_a3b_fp8/functional-v0.json)。按已确认的动态量化、误差阈值及TopK规则构造参考结果；测试显式使用对应的检查规则。

当前提供B负责的7个算子ref、[公共数值参考](../../docs/golden-stage2-step4-numerics.md)和[最小MTP组合](../../docs/golden-stage2-steps5-8.md)，测试见 [tests/reference](../../tests/reference/)。接口建议见 [阶段二步骤3记录](../../docs/golden-stage2-step3-interfaces.md)，实际调用方式见对应实现说明。
