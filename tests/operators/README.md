# tests/operators

运行 `python -m pytest tests/operators -q`。当前覆盖A组BF16基础/视觉reference及GDN的Torch核心与状态图，详见 [A组reference说明](../../docs/reference-a.md)。固定上游源码快照及许可放在 [reference_sources](../reference_sources/README.md)。

GDN投影图测试采用明确的BF16权重fixture；目标FP8投影reference与PTO执行在对应任务中接入。每个多输出算子的状态单独比较。
