# pymodel/isa

`torch_codegen.py`为固定PyPTO版本补充CPU执行所需的基础语义，范围与调用例子见[步骤4记录](../../docs/golden-stage3-step4-numerics.md)。PTO目录见`data/pto_isa_catalog.json`；原生后端及LPU映射在后续接入。

`specialize.py`支持同一程序内公共函数的不同形状调用；Attention使用方式及新增基础操作见[步骤6记录](../../docs/golden-stage3-step6-attention.md)。
