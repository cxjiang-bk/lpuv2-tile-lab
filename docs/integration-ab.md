# A/B 合并验证

A的9条OP与B的7条OP保留独立实现和reference。主线按一个OP对应一个逻辑kernel登记，MTP作为已有kernel的组合验证。

2026-11-12 合并回归 **298项通过、0失败、0跳过**：34项框架、82项A reference、39项A原生PTO C++仿真、110项B reference、31项B CPU Tile生成及2项最小MTP。数值规则沿用 `pto-functional-v0`。输入和权重使用固定随机种子，完整模型、真实checkpoint及目标硬件验证分别推进。

A、B在本次集成中均使用PyPTO `1e1236c187b4d4b971a9f8a4e196aeae349e60df`。B原始验证版本 `7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b` 保留在工具链锁与阶段记录中。A执行 `hw_pto_cpp_sim`，B执行 `pypto_torch_codegen` / `pypto_torch_codegen_adapted`，对应产物与验证范围独立登记。MTP仍使用B的Tile残差辅助，A的OP-04接入另行验证。

```bash
source scripts/hw-env.sh
python -m pytest tests -q --basetemp=artifacts/integration-final
```

测试分组、源码SHA及运行记录见[集成汇总](../data/integration-validation-summary.json)。[完整候选清单](lpuv2-instruction-candidates.md)合并16条OP的计算和数据操作需求，保留A/B原始需求索引用于追溯。
