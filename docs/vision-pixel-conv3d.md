# Vision Patch Embedding：像素 im2col 与矩阵乘

OP-08 / K-08 接收 BF16/FP32 像素张量 `pixels[B,C,T,H,W]`，返回 BF16 `out[M,D]` 和 int64 `grid_thw[B,3]`。本阶段输入已是像素张量；解码、resize、归一化与 padding 规则另行定义。

权重为 BF16 `[D,C,Kt,Kh,Kw]`，bias 为 BF16 `[D]`。当前支持无 padding、dilation=1、groups=1 的三维卷积，stride 可与窗口不同。输出网格 `Ot,Oh,Ow` 按普通卷积公式计算，空间网格需能被 `spatial_merge_size` 整除。

## 计算顺序

```text
for each output position (b,t,h,w):
    columns[row,c,dt,dh,dw] = BF16(pixels[b,c,t*st+dt,h*sh+dh,w*sw+dw])
raster32 = FP32_accumulate(columns @ flatten(weight).T)
raster = BF16(raster32 + FP32(bias))
out = reorder(raster, order=[B,Ot,block_h,block_w,intra_h,intra_w])
grid_thw[b] = [Ot,Oh,Ow]
```

计算主体在 [vision_ops.py](../kernels/pypto/vision_ops.py)。`image_windows` 在设备端生成三维窗口坐标，以 `pl.tile.mgather(..., coalesce="elem")` 读取像素。FP32 carrier 保持原始 BF16/FP32 像素值，窗口取出后才转 BF16。主机仅传入原始像素，权重 flatten 只改变连续存储的形状。

窗口地址按每行 K128 计算，原始像素转换按1024元素分块；矩阵乘按 `M16/N128/K128` 分块，FP32 accumulator 跨 K 块累加，最后加 bias 并转 BF16。Token 重排只搬运有效列，支持 hidden 尾块。

当前是 `im2col → pl.matmul/pl.matmul_acc → bias` 的基础操作组合，记录 `dedicated_im2col_instruction=false`。PTO `TIMG2COL` 的目标布局及时间轴映射另行验证；LPUv2 的 `IMG2COL_3D` 仍作为专用窗口搬运候选评估。

## 验证与范围

```bash
source scripts/hw-env.sh
python -m pytest tests/kernels/test_vision_conv3d.py -q
```

5项检查覆盖重叠 stride、独立窗口精确比较、模型参数 `C=3, kernel=stride=(2,16,16), D=1152`，以及 `D=130` 尾块。完整卷积对照独立 `torch.nn.functional.conv3d`，窗口对照独立 `unfold`。输出、网格与输入只读性分别检查；PTO/C++/ELF 随运行保存。

坐标组合使用 FP32，接口要求输入像素数与窗口元素数小于 `2**24`；该范围限制随用例记录。通用大图分块、窗口矩阵驻留与搬运成本在架构映射阶段评估。LPU 索引语义仍要求精确整数运算。
