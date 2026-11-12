# 原生执行适配

[hw_pypto.py](hw_pypto.py) 编译 hw-native-sys/pypto 的 `@pl.jit` 程序，调用 PTOAS 生成 C++，编译共享库并在 simpler a2a3sim 执行。后端为 `hw_pto_cpp_sim`，产物包含源代码快照、PTO/C++/ELF、版本及 SHA。

编译、仿真或逐输出比较失败均报错。独立 reference 保持在 `pymodel/ref/`，不作为执行后端。安装和运行见 [A组说明](../../docs/pypto-a1.md)。
