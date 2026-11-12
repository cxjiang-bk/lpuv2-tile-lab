# 阶段3步骤5：LM Head、Concat与MTP输入投影

本步完成OP-14/K-14、OP-15/K-15、OP-16/K-16三个完整入口。两个投影入口都调用步骤4的公共`bf16_linear`，输入准备和结果顺序由各自入口组织。

## 接口与职责

| 入口 | 输入 | 输出 | 计算职责 |
| --- | --- | --- | --- |
| `lm_head`，`kernels/pto/output/lm_head.py` | BF16 `h[B,S,H]`、BF16 `weight[V,H]`、INT64 `output_positions[B,P]` | BF16 `out[B,P,V]` | 从每个请求的hidden中选择指定位置，计算完整词表logits |
| `feature_concat`，`kernels/pto/mtp/input.py` | 两路已归一化、对齐的BF16 `e_norm[M,H]`、`h_norm[M,H]` | BF16 `out[M,2H]` | 每行前H列放Embedding特征，后H列放前序hidden特征 |
| `mtp_input_projection`，同上 | BF16 `eh[M,2H]`、独立BF16 `fc_weight[H,2H]` | BF16 `out[M,H]` | 将拼接特征投影回H维 |

调用方预分配输出，与输入、权重使用独立存储。输入形状、类型及位置应符合上表。LM Head的`h`已经经过Final Norm；Concat的两路输入已经分别归一化。MTP投影使用独立fc参数，LM Head使用其独立词表权重。

Concat和MTP投影沿用公共函数的二维行接口。例如两路输入原为`[B,S,H]`，调用侧使用`view(-1,H)`，输出使用相应的二维视图，逻辑结果仍为`[B,S,2H]`或`[B,S,H]`。当前约定要求这些Tensor可以这样展开。

## LM Head怎样计算

1. 将hidden视为`[B*S,H]`，将输出视为`[B*P,V]`，保持原存储。
2. 按请求和输出位置遍历。`output_positions[b,p]`是请求b内部的局部位置，范围为`[0,S)`。
3. 读取该索引，用Tensor slice视图选出输入的一行，以及输出中对应的一行。
4. 将这两个视图与词表权重传给公共BF16 Linear。Linear沿词表V和hidden H分块计算，合并FP32部分积后转BF16写回。

例如`positions[0]=[3,0,3]`，请求0的输出依次对应其第3、0、3行。每个位置都有独立输出行，重复位置按给定次序保留。本版逐个选中位置调用投影；多位置批量计算及权重复用属于后续优化。

入口使用`@pl.jit`组织选行和调用，实际矩阵计算由公共InCore函数完成。生成的IR中能看到索引读取、输入/输出视图、`bf16_linear`调用及其Tile load/matmul/store。

## Concat和MTP投影怎样计算

Concat每次从同一行的两路输入各读取一个`[1,128]` Tile。Embedding Tile写到输出列`col`，hidden Tile写到列`H+col`。有效区域随Tile传递，H不整除128时只写真实列。整个过程保持BF16值，比较采用精确匹配。

MTP投影将完整`eh[M,2H]`及`fc_weight[H,2H]`交给公共BF16 Linear，计算：

```text
hidden = BF16(eh @ fc_weight.T)
```

Linear的M/N/K分块、FP32累加及最终BF16转换沿用[步骤4](golden-stage3-step4-numerics.md)。本步没有改变共同数值定义。

## 怎样运行及验证

```bash
.venv/bin/python -m pytest tests/kernels/test_output_tile.py -q --basetemp=artifacts/golden-stage3-step5/pytest
```

测试用现有`Case/run_case`，将同一输入分别传给独立Golden及IR生成的程序。输出先填NaN，再比较所有元素的形状、类型和数值。投影使用既定`atol=1e-2,rtol=1e-2`，拼接使用`atol=0,rtol=0`。

| 用例 | 覆盖 |
| --- | --- |
| LM Head：H=64、V=16 | 两个请求、不同局部位置、非顺序选择及重复位置 |
| LM Head：H=67、V=19 | 上述位置关系，以及hidden和词表尾块 |
| Concat：H=128、131 | `[2,3,H]`的行顺序、两路正负标记、拼接边界和尾块 |
| MTP投影：H=32、35 | 独立随机fc的方向、`2H→H`维度及K/N尾块，恢复`[2,3,H]`输出 |

六项专项通过，本轮样本与Golden的最大绝对误差均为0。实际路径沿用PyPTO提交`7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b`及`pymodel/isa/torch_codegen.py`，记录为`pypto_torch_codegen_adapted`，实际设备为CPU；PTO CPU/NPU及LPU硬件后端由后续阶段验证。

全套168项测试通过，项目Ruff及差异检查通过；全套XML保存为`artifacts/golden-stage3-step5/all-test-results.xml`。

每个用例的`program_ir.py`、`program_torch.py`和Case输入/两侧结果/报告保存在`artifacts/golden-stage3-step5/pytest/`。运行产物在Git忽略目录中，执行方式和版本保存在manifest。当前三个入口均以小尺寸随机数据或标记数据验证，真实模型权重及完整MTP连接在后续集成中验证。

下一步为阶段3步骤6：完整Full Attention。
