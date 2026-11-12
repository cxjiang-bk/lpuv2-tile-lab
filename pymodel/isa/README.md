# Tile 执行适配

[hw_pypto.py](hw_pypto.py) 编译 A 的 HW PyPTO `@pl.jit` 程序，经 PTOAS 生成 C++，在 simpler a2a3sim 执行。后端为 `hw_pto_cpp_sim`，保存 PTO/C++/ELF、源码和版本 SHA；失败直接报错。安装见 [A组说明](../../docs/pypto-a1.md)。

B 使用 [torch_codegen.py](torch_codegen.py) 生成 CPU Torch 程序，[specialize.py](specialize.py)支持公共函数的多形状调用；适配范围见[数值记录](../../docs/golden-stage3-step4-numerics.md)和[Attention记录](../../docs/golden-stage3-step6-attention.md)。后端为 `pypto_torch_codegen` / `pypto_torch_codegen_adapted`。

两条路径分别由 `data/upstream.lock.json` 的 `pypto`、`pypto_torch_codegen` 锁定。独立 reference 位于 `pymodel/ref/`，用于结果比较。
