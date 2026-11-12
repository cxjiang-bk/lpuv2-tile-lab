# GDN 算子入口

[kernel.py](kernel.py)调用完整 K-06：FP8 qkv/z/out 投影、BF16 a/b 和 DWConv、FP32 recurrent State、gated Norm 与历史返回。入口保留 checkpoint 的 E4M3FN payload 和 BF16 block scale。

默认 `bf16_carrier` 路径使用 simpler a2a3sim 原生 PTO C++ 功能仿真：主机扩宽原始 FP8 字节编码，模拟器内完成解码、量化、矩阵乘和状态更新，BF16 精确承载有限 FP8 数值。完整小尺寸 GDN 的三个输出已通过，逐项结果见 [A 组说明](../../../docs/pypto-a1.md)。直接 FP8 硬件指令的 dtype、scale 和累加契约在后续指令级验证中验收。

独立 reference 在 `pymodel/ref/gdn.py` 和 `pymodel/ref/fp8.py`。K-06 包含内部各阶段；它们没有额外的逻辑 kernel 编号。

`gdn(..., path="recurrent")`默认逐Token递推；`path="chunk", chunk_size=64`选择块内系数、两次单位下三角求解和跨块State扫描。投影、Conv、gated Norm及三个命名输出共用。chunk返回的FP32 State和原始BF16 Conv历史可继续交给recurrent Decode，输入状态保持不变。当前验证范围以[A组说明](../../../docs/pypto-a1.md)和原生报告为准。
