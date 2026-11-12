# 原生 kernel 验证

先按 [A 组说明](../../docs/pypto-a1.md)安装 HW PyPTO/PTOAS/simpler 并 `source scripts/hw-env.sh`，再执行 `python -m pytest tests/kernels -q`。

每条 OP 在独立进程运行编译生成的 PTO C++ 共享库，与独立 reference 比较，并检查源码、二进制和运行记录。GDN默认用BF16精确承载FP8数值，额外检查recurrent连续Decode、chunk系数与三角求解、多个chunk/尾块、chunk Prefill到recurrent Decode的状态交接和状态只读，以及量化中点/尾块和分块scale。Attention 多头、分段和 FP8 跨 K 块缓冲区的连续重复执行也有原生回归。未配置原生环境时明确跳过。

OP-08 显式采用 `im2col → pl.matmul → bias`，展开由设备端窗口读取组合完成。`test_vision_conv3d.py` 检查三维窗口、重叠 stride、模型卷积参数、生成矩阵代码和实际 C++ 共享库。
