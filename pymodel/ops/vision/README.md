# Vision 算子入口

[kernels.py](kernels.py)提供 Patch Embedding、Attention、Dense FFN、Merger 和 Feature Scatter；[position.py](position.py)提供完整视觉位置嵌入。计算主体统一位于 [vision_ops.py](../../../kernels/pypto/vision_ops.py)，经 HW PyPTO/PTOAS/C++ 仿真 执行。Patch Embedding 按 `im2col → pl.matmul → bias` 处理 BF16/FP32 `[B,C,T,H,W]` 像素窗口，返回 `out` 和 `grid_thw`，host 不预切 Patch；预处理规则另定，详见 [像素 Conv3D](../../../docs/vision-pixel-conv3d.md)。覆盖与复现见 [A 组说明](../../../docs/pypto-a1.md)。
