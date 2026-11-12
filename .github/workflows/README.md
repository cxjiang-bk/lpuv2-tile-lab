# CI接入位置

当前可本地运行 `python -m pytest tests/framework -q`，尚无可执行CI。首个算子实现PR接入：

1. registry与代码风格检查。
2. 按 [功能验证标准v0](../../docs/functional-validation-v0.md) 运行独立golden与Python Tile程序，检查输出、量化结果、索引及状态。
3. 保存实际版本、case和比较结果，失败时保存小型复现产物。

实际PTO CPU/NPU的构建运行和LPU功能模型检查随对应执行环境接入。测试结果按具体路径与case登记。
