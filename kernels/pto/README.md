# B组 HW PyPTO Tile kernel

当前包含Embedding、RMSNorm、Full Attention、MoE、LM Head、Feature Concat和MTP输入投影，以及公共Linear/量化/激活/Softmax。使用 `pypto.language` 表达Tile程序，经CPU Torch生成路径与独立reference比较。入口、版本与验证范围见[B组实现汇总](../../docs/golden-stage3-step9-summary.md)。

A组的原生PTO C++仿真程序位于 `kernels/pypto/`。两条执行路径分别登记，完整逻辑kernel见 `data/kernels.json`。
