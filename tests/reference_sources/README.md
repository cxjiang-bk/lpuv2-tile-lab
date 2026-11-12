# 固定版本的参考源码

[modeling_qwen3_5_moe.py.txt](modeling_qwen3_5_moe.py.txt)是[Transformers 固定源码](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)的原样测试快照，SHA-256 为 `45a0f0d5ad9f6e2a57efec82da1b2b8d7f56f15b219ec741476117e2f58efcac`。版本与hash来自模型基线的 `baseline.json`。

[vision_utils.py.txt](vision_utils.py.txt)来自[同一提交的视觉元数据实现](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/vision_utils.py)，SHA-256 为 `bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356`，用于对照插值邻居和FP32系数的计算。

算子测试校验整个文件的hash，再通过AST提取所需函数或类，运行其PyTorch fallback。测试入口去除顶层加速器装饰器，固定eager Attention与非export路径；不导入完整Transformers，也不替换所选函数的数学主体。生产reference独立写在 `pymodel/ref/`。

原始版权声明保留在源码中，许可全文见 [LICENSE.transformers](LICENSE.transformers)。
