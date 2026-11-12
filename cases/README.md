# Case定义

小型JSON配置记录case编号、OP/K、数值profile、seed、参数及逐输出比较规则，通过 `pymodel.common.case.load_case` 读取。[OP-04示例](examples/op04-residual-bf16.json)与[GDN recurrent核心](operators/gdn-recurrent-bf16.json)已用于PyTorch reference检查；PTO路径随kernel接入。

[视觉位置插值元数据case](operators/vision-position-interpolation.json)验证OP-11前置阶段的`interp_indices`和`interp_weights`两个命名输出。

[kernels/](kernels)保存 A 组 9 个算子的输入 shape 预设、随机种子、参数和逐输出比较规则。`python -m pymodel.task_gen.validate_a` 在独立进程读取这些 case，经 HW PyPTO/PTOAS/C++ 仿真 执行，保存源码、二进制与数值报告。具体通过范围见 [A 组说明](../docs/pypto-a1.md)。
