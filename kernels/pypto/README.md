# HW PyPTO kernel

计算主体统一使用 `import pypto.language as pl` 和 `@pl.jit`。`common.py` 实现 K-03/04，`vision_ops.py` 实现 K-08/09/10/11/12/13，`gdn.py` 实现 K-06。`fp8.py` 提供内部量化与投影，保留一个 OP 对应一个逻辑 kernel。

显式 Tile 接口表达读写、归约和广播；矩阵及组合阶段经编译器生成 PTO IR。kernel 中计算坐标、量化、状态与输出，不调用 Torch golden。工厂使用算子名，有独立计算意义的阶段用 inline/incore 函数组织。

数值转换按模型顺序保留。A2/A3 的向量指令不支持某些 BF16 算术形式时，使用 FP32 算术并在原有 BF16 边界立即舍入。布局、有效区域和输出轴明确；广播使用对应行/列操作。

执行链为 PTO → PTOAS → C++ → GCC 共享库 → simpler `a2a3sim`，记录为 `hw_pto_cpp_sim`。接口与逐项覆盖见 [A 组说明](../../docs/pypto-a1.md)。
