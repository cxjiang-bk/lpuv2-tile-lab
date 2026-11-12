# 本地实验产物

`run_case`默认在本目录下按 `<case_id>/<run_id>/` 保存输入、reference、actual、manifest、比较报告和日志。tensor文件保留BF16/FP32/整数dtype；异常时保留已完成阶段的产物。

运行产物默认被git忽略。复现方式见[验证框架](../docs/validation.md)。

保留已确认通过的完整回归及其PTO/C++/共享库证据；过期的失败尝试、重复调试目录和 `build_output/` 可清理。工具链和独立实验源代码分别管理。
