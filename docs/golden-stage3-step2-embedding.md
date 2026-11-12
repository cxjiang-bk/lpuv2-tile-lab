# 阶段3步骤2：Embedding Tile程序

本步完成K-01 Embedding：根据Token ID，从BF16权重表中读取对应行，输出仍按原来的batch和Token顺序排列。程序位于[kernels/pto/common/embedding.py](../kernels/pto/common/embedding.py)，比较测试位于[tests/kernels/test_embedding_tile.py](../tests/kernels/test_embedding_tile.py)。

## 输入输出与计算流程

| 对象 | 形状与类型 | 含义 |
| --- | --- | --- |
| `token_ids` | `[B,S]`，INT64 | 每个Token在词表中的行号，范围为`[0,V)` |
| `embedding_weight` | `[V,H]`，BF16 | Embedding权重表 |
| `out` | `[B,S,H]`，BF16 | 由调用方提供，kernel填入结果 |

1. 把Token位置视为连续的`B*S`行，把输出视为`[B*S,H]`。这里的reshape只是逻辑视图，不计算Embedding结果。
2. 用`pl.read`读出当前Token ID，作为权重表的行偏移。
3. 每次用`pl.load`读取该行的32列，即一个`[1,32]` Tile。
4. 用`pl.store`把Tile写到当前Token对应的输出行，保持原请求顺序。重复Token可以读取同一权重行，分别写到各自位置。
5. 最后不足32列时，用`valid_shape=[1,有效列数]`限定读写范围。例如H=67时，依次写32、32、3列。

这里采用32列是为了建立清晰的功能用例，不代表LPUv2的硬件Tile宽度已经确定。程序没有算术计算或格式转换，BF16数据按位保留。

## 怎样执行和比较

运行环境沿用[步骤1](golden-stage3-step1-execution.md)：官方PyPTO提交`7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b`，Python 3.13.9、Torch 2.12.0+cu130，实际执行设备为CPU。

```bash
.venv/bin/python -m pytest tests/kernels/test_embedding_tile.py -q --basetemp=artifacts/golden-stage3-step2/pytest
```

测试先用`embedding.specialize`生成Program IR，再由官方`torch_codegen`生成并执行Torch代码。IR中可以看到Token标量读取、带动态行偏移的Tile load，以及Tile store。当前所需操作均由所选上游版本提供，没有补充自定义指令语义。

已有`Case/run_case`负责输入隔离、运行Golden和被测程序、比较与保存报告。Golden侧调用`pymodel/ref/embedding.py`；被测侧执行生成代码。输出初始化为NaN，可在比较时发现未写入的位置。

| 用例 | 输入 | 验证内容 |
| --- | --- | --- |
| `embedding_h64` | IDs为`[[0,10,3],[3,8,0]]`，权重`[11,64]` | 多batch、重复ID、词表首尾行、原位置写回、两个完整列Tile |
| `embedding_h67` | 相同IDs，权重`[11,67]` | 跨三个列Tile，末尾仅3列有效 |

两例采用固定种子生成BF16随机权重，比较全部输出的形状、类型和数值，`atol=0, rtol=0`。实际最大误差均为0。

## 操作使用记录

| 实际PyPTO操作 | 在本kernel中解决什么问题 | 指令集整理时如何理解 |
| --- | --- | --- |
| `pl.reshape` | 把batch和Token轴合成逻辑行 | 逻辑视图操作，不自动认定为一条硬件指令 |
| `pl.read` | 从Token ID表读取当前索引 | 记录索引读取能力，后续确定LPU表达方式 |
| `pl.load` | 按Token ID定位权重行，再分块读取 | 对应Tile读取语义；索引行偏移由软件显式提供，本例没有使用TGATHER |
| `pl.store` | 将数据写回对应Token行 | 对应Tile写回语义，尾块遵守有效区域 |
| `pl.range`、`pl.min` | 遍历Token/列块，计算尾块有效列数 | 程序控制和标量计算，不直接计为Tile算术指令 |

候选需求同步记录在[ISA需求说明](isa-requirements.md)。本步证明的是Python层的Tile程序功能，PTO设备执行、Buffer映射和硬件性能由后续阶段验证。

## 结果与产物

- Embedding专项2项通过，与独立Golden精确一致。
- 全套147项测试通过；项目Ruff检查通过。
- 专项运行的IR、生成Torch程序及`run_case`报告位于`artifacts/golden-stage3-step2/pytest/`，输入和两侧结果同时保存为`.pt`文件。
- [全套测试XML](../artifacts/golden-stage3-step2/all-test-results.xml)记录本轮检查结果。运行产物在Git忽略目录内。

本步交付完整Embedding入口和Golden比较用例。下一步为阶段3步骤3：RMSNorm的分块归约与广播。
