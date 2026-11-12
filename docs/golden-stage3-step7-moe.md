# 阶段3步骤7：完整MoE

本步完成OP-07/K-07。入口为[kernels/pto/moe/moe.py](../kernels/pto/moe/moe.py)，从hidden开始完成Router、Top-8、专家分桶、Gather、专家FFN、按原Token合并及Shared Expert。主输出和路由、分桶记录分别交付，便于后续与MTP连接。

## 输入输出

入口使用显式Tensor参数，调用方提前分配六份输出。原始`[B,S,H]`输入可以在调用侧用`view(-1,H)`展开为M行，输出使用对应视图；测试中的`_tile_call`提供参数顺序和最小执行方式。

| 参数组 | 形状和类型 | 含义 |
| --- | --- | --- |
| `x` | BF16 `[M,H]` | M个有效Token的输入hidden |
| `router`、`shared_gate_weight` | BF16 `[E,H]`、`[1,H]` | 路由投影及每Token一个标量的shared门控投影 |
| `expert_gate/up/down` | FP8 E4M3FN `[E,I,H]`、`[E,I,H]`、`[E,H,I]` | 各专家的三份独立投影权重 |
| `shared_ffn_gate/up/down` | FP8 E4M3FN `[J,H]`、`[J,H]`、`[H,J]` | Shared Expert的独立FFN权重，J可与I不同 |
| 六份scale | BF16，按各矩阵N/K方向128×128分块 | 专家scale增加最前面的E轴，shared scale为二维；数值原样作为反量化乘数 |
| `out` | BF16 `[M,H]` | routed与shared合并后的主结果 |
| `expert_id`、`route_weight` | INT64、BF16，均为`[M,8]` | 按原Token顺序保存选中的专家及归一化路由权重 |
| `bucket_offset` | INT64 `[E+1]` | 专家e的记录区间为`[offset[e],offset[e+1])`，差值就是有效数量Ne |
| `bucket_token_id`、`bucket_slot` | INT64 `[8*M]` | 每条桶记录对应的原Token行，以及该Token的Top-8位置 |

本版固定Top-8，要求M>0、E≥8、各矩阵和scale形状符合表中关系。输入和输出使用独立存储；所有输入Token都有效。没有跨调用状态或隐含的物理Buffer地址。

## 数据怎样流动

1. **Router产生概率。** 公共BF16 Linear计算`x @ router.T`，得到BF16 logits；公共Softmax沿完整专家轴计算FP32概率。
2. **选Top-8并归一化。** 每个Token选择8个不同专家，按概率降序、完全同分时专家ID升序排列。选中的概率仍为FP32，除以这8项的总和后转BF16。
3. **生成专家桶。** 按专家ID升序遍历路由。每桶内部按原Token行、slot顺序记录，保留`token_id`与`slot`。`bucket_offset[E]`等于`8*M`；空桶的前后offset相同。
4. **Gather专家输入。** 读取当前桶的Token ID，把对应BF16 hidden逐行、逐列Tile复制到可复用的输入工作区。传给FFN的是`[Ne,H]`有效前缀视图，Ne在运行时从offset得到。空桶跳过。
5. **完成该专家FFN。** 从三维权重及scale中选出当前专家，复用公共FP8 Linear计算Gate/Up。Gate经过SiLU，再与Up相乘，作为Down投影的输入。三次投影和激活沿用公共量化、累加和转换规则。
6. **Scatter并合并。** 用桶记录中的Token和slot读取对应BF16路由权重，缩放专家输出，再加回原Token的BF16合并区。专家按ID升序串行处理，每项贡献恰好加入一次。
7. **加入Shared Expert。** 对全部Token执行独立shared FFN；`shared_gate_weight`经BF16投影及BF16 sigmoid得到每Token的门控。shared结果乘该门控后，与routed结果相加得到`out`。

输入工作区和专家输出工作区各为`[M,H]`，按专家复用。因为每个Token不会重复选择同一专家，单个桶的Ne不会超过M。Gate/Up等中间Tensor按当前Ne创建，计算和合并只遍历有效行。本版Gather的是BF16 hidden，随后在公共FP8 Linear内部按Token/K组量化，因此数据和激活scale的对应关系随当前桶重新建立。

## 精度及合并顺序

遵循现有Golden与功能标准v0：

- Router logits为BF16，Softmax、Top-8概率及重归一化为FP32，路由权重最后转BF16。
- Gate/Up输出BF16，SiLU输出BF16，`SiLU(gate)*up`输出BF16，再进入Down动态量化；Down输出BF16。
- 专家输出乘路由权重得到BF16贡献，再与原合并值做BF16加法。按专家ID升序处理，与当前Golden的合并顺序一致。
- shared的门控投影、sigmoid、乘法及最终合并均保留BF16边界。

这里使用顺序读—加—写实现Scatter合并。当前程序没有并发的同地址更新，不将此结果标记为硬件Atomic Add验收；并发合并、FP32累加区及其与模型数值语义的关系在架构映射时另行评估。

## Top-8怎样表达

已核对PyPTO的`sort32`、`row_argmax`和Gather接口。`sort32`返回32元素块的打包值—索引结果，本轮尚未确认其原生同分规则及当前Torch生成路径。因此首版直接用基础读取、标量比较和循环组合选择，完整计算仍在PyPTO程序内。

每个Token复制一行概率到局部Tile，重复8轮：按专家ID升序扫描，只有新概率**严格更大**时才替换当前最佳项；选中后将该位置置为-1，避免重复选择。Softmax概率非负，因此-1仅作为已选标记。这保证同分时保留较小ID，每Token约做`8*E`次比较。

分桶同样以基础索引和循环组合，约扫描`E*M*8`条候选关系。本版用于明确正确行为；排序/选择和计数前缀和等高效实现留待后续候选指令及性能取舍，不把这些循环直接认定为一条LPU指令。现有实现没有使用整Tensor的Torch TopK或MoE替代被测主体。

## 怎样执行和验证

```bash
.venv/bin/python -m pytest tests/kernels/test_moe_tile.py -q --basetemp=artifacts/golden-stage3-step7/pytest
.venv/bin/python -m pytest -q
```

入口使用`specialize(moe, *args)`，由现有`pymodel/isa/specialize.py`为动态桶前缀及公共函数的不同形状生成对应函数，再由`pymodel/isa/torch_codegen.py`生成CPU代码。PyPTO沿用固定提交`7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b`，实际路径为`pypto_torch_codegen_adapted`；本步没有增加执行适配，也未修改Golden或上游源码。

| 用例 | 覆盖 |
| --- | --- |
| 4 Token、12专家，H=17/I=19，随机路由 | 不同路由、热门专家、空桶、动态Ne、特征尾块、shared贡献；全部六份输出与Golden比较 |
| 相同尺寸，Router全零 | 全同分时选择ID 0～7，路由权重精确为1/8，空专家跳过 |
| 2 Token、10专家，H=129/I=131 | 跨128量化组、不同N/K权重scale、列尾部与动态桶前缀 |
| 36专家的定向选择 | 最佳项在ID 35/31，其余门槛同分；检查跨32边界仍按ID升序选择，并保持输入概率不变 |

三个完整入口用例通过`Case/run_case`比较主输出、路由权重和全部整数桶记录。主结果及路由权重使用BF16既定容差；索引、slot和offset精确比较；另检查全部`8*M`条Token/slot记录恰好出现一次。同分权重也单独精确检查。四项专项通过，本轮主输出、路由权重的最大绝对误差均为0，整数记录全部一致。

IR、生成程序和Case输入/两侧结果/报告保存在`artifacts/golden-stage3-step7/pytest/`。本轮为小尺寸Python功能验证，原生PTO/LPU后端、真实模型规模、物理存储容量和性能均在后续集成范围。

全套175项测试通过，Ruff和差异检查通过；全套XML为`artifacts/golden-stage3-step7/all-test-results.xml`。

下一步为阶段3步骤8：组合最小MTP Tile流程。
