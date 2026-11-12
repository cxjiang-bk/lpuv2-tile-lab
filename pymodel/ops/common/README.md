# 公共算子入口

[tile.py](tile.py)提供 `residual_add` 与 `layer_norm`，调用 `kernels/pypto/common.py` 中同名程序并执行 HW PyPTO/PTOAS/C++ 仿真。输入转换、窗口、输出与状态约束见 [A 组说明](../../../docs/pypto-a1.md)。
