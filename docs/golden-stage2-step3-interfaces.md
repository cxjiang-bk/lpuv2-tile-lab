# B路线阶段二步骤3：接口约定与首个Golden用例

记录日期：2026-11-11。基线：`qwen3_6_35b_a3b_fp8-v1`；数值profile：`pto-functional-v0`。

本文记录步骤3的接口建议与当时的实现范围。公共数值入口见[步骤4记录](golden-stage2-step4-numerics.md)，完整算子及MTP的实际调用方式见[步骤5—8记录](golden-stage2-steps5-8.md)。

本步骤把已确定的7个算子整理成可直接讨论的调用约定，并用一个小型Embedding检查调用与测试流程。算子清单沿用阶段一；本次落实的是“怎样传入数据、怎样返回并检查结果”。

## 1. 本次交付及依据

| 内容 | 本次状态 |
| --- | --- |
| OP-01/02/05/07/14/15/16输入、输出及边界 | 下文逐项整理；未实现的函数名称、KV及桶字段标为建议 |
| OP-01最小数学入口 | 已实现`pymodel.ref.embedding.embedding` |
| Embedding自身检查 | pytest直接提供输入、调用Golden，并与手写预期及直接索引比较 |
| 公共测试工具 | 按需复用已有`Case`、`run_case`和比较器，负责命名输出、统一记录和张量保存 |
| 其他算子和公共数值函数 | 留在后续步骤；本文的签名说明不表示已有可调用实现 |

依据：

- [阶段二详细实施步骤第3节](https://tju-opentpu.feishu.cn/docx/J8uQdwJhRobt7BxL6USc7kv1nxh)及[阶段一接口记录](https://tju-opentpu.feishu.cn/docx/OoyId7huHojnTKxuDvzcZJu9nRe)（本轮依据的缓存为revision 7）。
- [OP与kernel边界](kernel-organization.md)、[分工](work-allocation.md)、[已有状态接口登记](../data/interfaces.json)。
- [模型来源](../configs/models/qwen3_6_35b_a3b_fp8/baseline.json)、[数值契约](../configs/models/qwen3_6_35b_a3b_fp8/numerics.json)、[功能验证v0](functional-validation-v0.md)。

模型权重精度、计算规则和OP边界沿用既定基线。下文的Python字段及状态布局是B路线的接口建议，供A/B逐项对齐；本次仅Embedding调用已落地。主线登记中的算子与kernel状态继续按实际完整验收更新。

## 2. 两人共同的调用方式

### 2.1 维度、权重和行号

| 记号 | 含义 |
| --- | --- |
| `B`、`S` | 本次请求数、本次每请求输入的Token行数 |
| `T` | 矩形输入展平后的行数`B*S`；建议`token_id=b*S+s` |
| `H`、`D` | 文本hidden宽度`H=2048`；`D`为当前运算的特征宽度 |
| `V` | 词表大小；模型为248320；小测试可使用小词表 |
| `N`、`K` | Linear输出、输入特征宽度；`x[...,K] @ weight[N,K].T -> out[...,N]` |
| `L`、`P` | 输入KV数组的最大逻辑长度、每请求选取的输出位置数量 |
| `E`、`I` | 专家数256、专家中间宽度512；模型Top-K为8 |

`token_id`是本次输入的行号；`token_ids`是词表中的编号；`seq_pos`是序列中的因果位置。这三个量含义不同。例如`B=2,S=3`，`token_id=4`指请求1的本次第1行，不能据此推断词表ID或绝对位置。

### 2.2 Golden直接接收数据，测试负责检查

数学函数直接接收tensor和必要的数学参数，例如已实现的`embedding(token_ids, embedding_weight) -> Tensor`。测试在Python中准备输入，调用函数并检查结果：

```python
actual = embedding(token_ids, embedding_weight)
torch.testing.assert_close(actual, expected, atol=0, rtol=0)
```

简单算子返回Tensor，Attention、MoE等复杂算子按需返回具名结果和状态。输入生成、比较、日志及文件保存由测试侧组织，Golden专注数学计算。

需要接入公共`run_case`时，测试侧才将输入和结果整理成现有协议：`(inputs: dict[str, Tensor], parameters: dict) -> dict[str, Tensor]`。其中`inputs`放权重、激活、位置和历史tensor，`parameters`放可保存为JSON的维度、eps等参数。公共框架继续承担运行记录和产物保存，Golden无需依赖这些对象。

| 输出命名 | 用法与检查 |
| --- | --- |
| `out` | 当前算子的主结果 |
| `kv_k_out`、`kv_v_out` | Attention更新后的K、V；各自比较 |
| `kv_valid_len_out`、`kv_seq_pos_out` | KV有效长度和键位置；整数精确比较 |
| `expert_id`、`route_weight` | 原Token顺序下的Top-8专家及路由权重；索引精确、权重按BF16规则 |
| `bucket_offset`、`bucket_token_id`、`bucket_slot`、`bucket_route_weight` | 可选分桶检查点；字段建议见第6节 |

直接测试按输出含义逐项断言；接入`run_case`时，`Case.outputs`列出实际导出的全部Tensor及检查规则。可选检查点一旦导出，也要检查。

### 2.3 有效区域、offset与输入更新

当前最小入口接收完整的有效逻辑tensor。若输入来自父tensor的某个区域，调用方先传对应视图。例如`token_ids_parent[:, 1:4]`表示本次3行；这3行的输出位置仍按视图内顺序排列。视图的存储offset与Token绝对位置分别记录，不能混用。

Embedding内每个传入ID都有效，不隐含padding ID。RMSNorm必须覆盖完整归一化轴；投影必须覆盖完整K轴。后续Tile实现的尾部填充不参与统计或运算，数学Golden由有效逻辑数据直接表达。FP8跨块分组和scale与父K轴的对应在步骤4实现，不能通过切片静默改变分组。

本路线建议数学入口均保留输入，返回新结果；状态通过显式的`*_out`返回。当前Embedding已满足此约定，返回新存储。函数在推理模式执行；直接调用后若要原地改输出，应处于`torch.inference_mode()`内，或先在外部克隆。后续测试有可写状态时，应为Golden和被测程序提供各自独立的同一初值；公共`run_case`已有相应隔离机制。

## 3. K-01：Token Embedding（本次可调用）

入口：`embedding(token_ids, embedding_weight) -> out`，位于[pymodel/ref/embedding.py](../pymodel/ref/embedding.py)。

| 参数或输出 | shape / dtype | 含义 |
| --- | --- | --- |
| `token_ids` | `[B,S]` / int64 | 每个请求、每个输入位置对应的词表ID |
| `embedding_weight` | `[V,H]` / BF16 | 每个词表项一行；模型为`[248320,2048]` |
| `out` | `[B,S,H]` / BF16 | `out[b,s]=embedding_weight[token_ids[b,s]]` |

所有ID必须在`[0,V)`内；负数、越界、形状或dtype错误直接报错。两份输入须位于同一设备。`V/H`为正；空`B/S`返回相应空结果。非连续视图按其逻辑内容查表；ID相对于传入表的第0行定义。

无需位置参数或历史状态。重复ID返回相同内容，输出行的请求/Token顺序保持输入顺序。输入不修改，输出独立于权重存储。输出是纯搬运结果，检查shape、dtype及零容差数值一致。

模型的Embedding表为BF16，与LM Head权重独立。当前用小型合成表验证接口和查表，尚未接入真实权重或完整词表维度。

## 4. K-02：普通RMSNorm（接口建议）

拟定数学调用：`rmsnorm(x, weight, *, eps=1e-6) -> out`。本次未实现。

| 参数或输出 | shape / dtype | 含义 |
| --- | --- | --- |
| `x` | `[...,D]` / BF16 | 最后一轴为完整归一化范围；前导轴各行独立 |
| `weight` | `[D]` / BF16 | 该调用点的Norm参数 |
| `eps` | Python浮点数，默认`1e-6` | 加到均方上的稳定项 |
| `out` | 与`x`同形 / BF16 | 归一化、乘权重后的结果 |

权重按`1+FP32(weight)`解释。先转FP32做最后轴均方和rsqrt，乘权重后回BF16，保持原有转换位置。文本使用`D=2048`，Q/K使用`D=256`；例如`[B,S,16,256]`中每个head独立计算。有效区覆盖全部D列；局部列片段不能当成完整归一化范围。

没有状态和位置参数；建议输入保留，返回新tensor，不做隐式原地更新。普通RMSNorm与A负责的GDN gated Norm分别定义，后者的直接权重与转换顺序见数值基线。

## 5. K-05：完整Full Attention（接口建议）

拟定数学调用接收`x`、投影/Norm参数、`seq_pos`、`rope_pos`、`query_valid`及`kv_in`，返回主输出和新KV。具体Python对象封装待D1对齐；本次没有该函数实现。

### 5.1 输入与权重

| 参数名建议 | shape / dtype | 含义 |
| --- | --- | --- |
| `x` | `[B,S,2048]` / BF16 | 已完成Decoder前置Norm的hidden |
| `q_gate_weight` | `[8192,2048]` / FP8 | 每head的Q/gate联合投影 |
| `k_weight`、`v_weight` | 各`[512,2048]` / FP8 | 2个KV head的投影 |
| `out_weight` | `[2048,4096]` / FP8 | 16个query head拼接后的输出投影 |
| 对应的`*_scale` | `[ceil(N/128),ceil(K/128)]` | 各自权重块的反量化乘数；保留checkpoint数值 |
| `q_norm_weight`、`k_norm_weight` | 各`[256]` / BF16 | 内部head-wise普通RMSNorm权重 |
| `seq_pos` | `[B,S]` / int64 | 当前Token的因果顺序位置 |
| `rope_pos` | `[3,B,S]` / int64 | temporal、height、width三路RoPE坐标 |
| `query_valid` | `[B,S]` / bool | 建议每请求有效行构成前缀；矩形首批case全部为True |

权重scale在checkpoint中为BF16；计算时可转FP32但数值保持不变。输入动态量化与FP8 Linear计算规则引用v0，留待公共数值函数实现。

Q/gate先组织成`[B,S,16,512]`，每head拆成256+256；Q/K普通Norm和64维Partial MRoPE位于本OP内部。因果mask依据`seq_pos`，RoPE依据`rope_pos`。外层Decoder前置Norm与残差分别由K-02/K-04组合提供。

### 5.2 初始历史、结果与有效区

| 输入字段建议 | shape / dtype | 含义 |
| --- | --- | --- |
| `kv_k`、`kv_v` | 各`[B,2,L,256]` / BF16 | 已有历史；K已做Norm/RoPE，V为投影值 |
| `kv_valid_len` | `[B]` / int64 | 每请求有效前缀长度，范围`0..L` |
| `kv_seq_pos` | `[B,L]` / int64 | 每个有效键的因果位置；填充位置不参与计算 |

建议从每请求的`kv_valid_len[b]`处追加当前有效K/V，旧有效前缀保留；输出逻辑长度为`L_out=max_b(旧有效长度+当前有效行数)`，填充补零。有效键位置按因果顺序排列，追加的新位置接在原前缀之后。上述布局和不等长处理为待双方确认的具体建议，不表示已经支持该路径。

| 导出输出 | shape / dtype | 检查方式 |
| --- | --- | --- |
| `out` | `[B,S,2048]` / BF16 | 主结果；建议无效query行置零 |
| `kv_k_out`、`kv_v_out` | 各`[B,2,L_out,256]` / BF16 | 分别比较新旧有效内容及约定填充 |
| `kv_valid_len_out` | `[B]` / int64 | 精确比较更新后的长度 |
| `kv_seq_pos_out` | `[B,L_out]` / int64 | 精确比较键位置；填充建议为0 |

无历史时传长度0的K/V及位置、全零有效长度。每个有效query须有合法key；无效query不进入Softmax。建议所有输入状态保留，返回新状态，按请求/层/target或draft分别保存，供下次调用。

例如已有位置0～2，新输入3、4：位置3只能看0～3，位置4可以看0～4；输出两行，最终有效长度为5。用切片传入新Token时，`seq_pos/rope_pos/query_valid`同步切片；不能把视图局部第0行误当成序列位置0。

## 6. K-07：完整MoE FFN（接口建议）

拟定数学入口接收`x`、Router/shared gate参数及各专家参数，返回主结果和路由信息；分桶检查点可选。字段名称及权重容器待D2对齐。本次未实现计算。

### 6.1 输入与输出

| 参数名建议 | shape / dtype | 含义 |
| --- | --- | --- |
| `x` | `[B,S,H]`或`[T,H]` / BF16 | 有效hidden；原始行顺序供Router和shared门控使用 |
| `router_weight` | `[E,H]` / BF16 | Router投影；`E=256` |
| `shared_gate_weight` | `[1,H]` / BF16 | 每Token一个shared分支门控系数 |
| `expert_gate_weight`、`expert_up_weight` | 各`[E,I,H]` / FP8 | 每个专家的独立Gate、Up；`I=512` |
| `expert_down_weight` | `[E,H,I]` / FP8 | 每个专家的独立Down |
| `shared_ffn_gate_weight`、`shared_ffn_up_weight` | 各`[I,H]` / FP8 | shared专家的FFN，与标量门控参数区分 |
| `shared_ffn_down_weight` | `[H,I]` / FP8 | shared专家Down |
| 各矩阵的`*_scale` | 每矩阵`[ceil(N/128),ceil(K/128)]` | routed专家多一个E轴；数据和scale同专家、同块对应 |
| `out` | 与`x`同形 / BF16 | routed贡献与shared贡献合并的结果 |
| `expert_id` | `[T,8]` / int64 | 原Token行顺序下的专家ID，Top-K列为slot |
| `route_weight` | `[T,8]` / BF16 | 与`expert_id`同位置的权重 |

矩形输入按`token_id=b*S+s`展平并按原形恢复。若上游存在padding，应先选出有效Token并保留回填映射；该变长适配待单独实现，当前不假定padding行自动参与或被自动忽略。

Router、Softmax、Top-8及重归一化沿用基线；同分按专家ID升序，最终路由权重回BF16。专家FFN与shared参数独立。专家缩放与跨专家合并保留已定BF16边界；不得把硬件FP32 atomic设想直接当成本轮Golden的数值定义。

### 6.2 分桶表示建议

使用E+1个边界描述各专家桶，`R=8*T`。专家e的记录范围为`[bucket_offset[e], bucket_offset[e+1])`，有效行数`Ne`为两边界之差。

| 检查点名称建议 | shape / dtype | 含义 |
| --- | --- | --- |
| `bucket_offset` | `[E+1]` / int64 | 首项0、末项R、单调不减 |
| `bucket_token_id` | `[R]` / int64 | 每条记录来自哪一行，范围`[0,T)` |
| `bucket_slot` | `[R]` / int64 | 原Token的Top-8列号，范围`[0,8)` |
| `bucket_route_weight` | `[R]` / BF16 | `route_weight[bucket_token_id, bucket_slot]` |

桶按专家ID递增，桶内建议按`token_id,slot`排序。每条`(token_id,slot)`恰好一次；空桶`Ne=0`。若内部暂存有填充，只处理前Ne行。预量化Gather如被采用，payload及scale沿同一索引重排；具体预量化入口在步骤4及MoE实现时确定。

本OP不维护跨调用历史，建议输入不修改、返回新结果。`token_id`恢复本次Token行，跨调用身份由调用方管理。导出检查点时，整数精确比较；权重按其原始路由对应关系检查。

## 7. K-14：LM Head（接口建议）

拟定调用：`lm_head(h, weight, output_positions) -> out`。本次未实现。

| 参数或输出 | shape / dtype | 含义 |
| --- | --- | --- |
| `h` | `[B,S,H]` / BF16 | 已完成Final Norm或MTP输出Norm的hidden |
| `weight` | `[V,H]` / BF16 | 独立LM Head权重，无bias |
| `output_positions` | `[B,P]` / int64 | 每请求要选的本次输入局部行号，范围`[0,S)` |
| `out` | `[B,P,V]` / BF16 | 对所选hidden做`h_selected @ weight.T`得到logits |

例如`S=4, output_positions=[[3,1,3]]`，输出按第3、1、3行顺序保留重复选择。绝对Token位置由调用方换成当前输入局部行号；有有效前缀时只能选择有效行。建议负数和越界报错，输入保留，无状态，返回新结果。

本入口不重复Final Norm；采样由外层完成；输出不统一转FP32。词表分块是实现细节，完整Golden的输出语义仍覆盖全部V列。

## 8. K-15：Feature Concat（接口建议）

拟定调用：`feature_concat(e_norm, h_norm) -> out`。本次未实现。

| 参数或输出 | shape / dtype | 含义 |
| --- | --- | --- |
| `e_norm` | `[...,2048]` / BF16 | 已归一化的embedding路 |
| `h_norm` | 与`e_norm`同形 / BF16 | 已归一化、按同一Token顺序排列的hidden路 |
| `out` | `[...,4096]` / BF16 | 最后轴按`[e_norm;h_norm]`拼接 |

两路前导轴与有效区域必须相同；不能隐式广播或自行错位一行。建议输入不修改，输出新存储；没有内部状态。位置对齐由IF-05调用方负责，单凭两个tensor形状相同无法证明Token对齐。两路有位置标签时，由外层适配器同时检查标签；MTP具体hidden来源仍属于D3。

## 9. K-16：MTP输入投影（接口建议）

拟定调用：`mtp_input_projection(eh, fc_weight) -> out`。本次未实现。

| 参数或输出 | shape / dtype | 含义 |
| --- | --- | --- |
| `eh` | `[...,4096]` / BF16 | K-15按`[e_norm;h_norm]`输出的完整特征 |
| `fc_weight` | `[2048,4096]` / BF16 | 独立的`mtp.fc.weight`，无bias |
| `out` | `[...,2048]` / BF16 | `eh @ fc_weight.T` |

前导轴和行顺序保持不变，完整4096列共同参与投影。建议输入不修改、返回新结果，无独立状态。参数形状和精度已有依据；完整MTP何时调用、输入hidden来自哪个位置，仍在D3单独对齐。

## 10. Embedding怎样运行和检查

在仓库根目录执行：

```bash
python3 -m pytest tests/reference/test_embedding.py -q
python3 -m pytest tests/reference/test_embedding.py tests/framework -q
```

测试位于[tests/reference/test_embedding.py](../tests/reference/test_embedding.py)，只需阅读这个文件和Golden入口即可运行与理解。

1. 手算用例直接写出BF16权重表、Token ID和预期结果，检查重复/乱序ID、多请求、输入保留和输出独立存储。
2. 随机用例在测试内使用种子17的独立生成器，分别检查`[1,1]`与`[2,4]`的Token ID。权重表为`[7,5]`；预期为`weight[token_ids]`，实际调用`embedding(token_ids, weight)`。
3. 比较使用`torch.testing.assert_close(actual, expected, atol=0, rtol=0)`，检查shape、dtype及逐元素数值。Embedding为纯搬运，采用零容差。
4. 定向用例检查非零offset、非连续视图、空输入、负数/越界ID，以及不合法dtype和shape。

本步骤属于PyTorch Golden自身检查；实际使用CPU和小型合成数据。它不代表Tile/PTO或固定Transformers整模型已经执行。张量保存和来源记录按需使用现有公共工具；简单Golden测试直接通过pytest运行。

## 11. 本次验证记录

| 检查 | 结果 |
| --- | --- |
| Embedding专项测试 | 14项通过；手写预期、随机直接索引及关键边界 |
| 既有框架测试 | 原34项全部通过；与专项测试合计48项 |
| Ruff检查 | Golden与专项测试通过 |
| 数值检查 | 随机用例输出`[1,1,5]`与`[2,4,5]`，BF16、零容差检查通过 |
| 环境 | Python 3.13.9、Torch 2.12.0+cu130，实际使用CPU |

本次[测试XML](../artifacts/golden-stage2-step3/simplified-test-results.xml)保存在本地忽略目录。需要生成同类记录时，在pytest命令后添加`--junitxml=artifacts/golden-stage2-step3/simplified-test-results.xml`。历史运行产物保持原记录，当前入口与验收方式以上述直接测试为准。

## 12. 交接时确认什么

| 事项 | 本次给出的内容 | 后续确认方式 |
| --- | --- | --- |
| D1公共调用 | `out`、`[N,K]`权重方向、显式状态输出；Embedding已有可运行例子 | A/B按能力逐项采用；其他函数随实现提供实际调用例子 |
| D1 KV | 因果位置与THW分开；K/V、有效长度、键位置分别返回 | 确认具体字段封装、变长和无效区规则，再实现连续调用测试 |
| D2分桶 | 边界、原Token、slot、对应权重及排序建议 | 用可手算分桶例和Top-8用例交叉检查；此处尚无分桶实现 |
| D3 MTP | K-15/16的独立输入输出已列清 | 补齐支持MTP来源中的hidden/Token对应链；不影响先实现独立算子 |

本次结果只覆盖最小Embedding Golden及测试接入。完整K-01覆盖、公共数值函数、其余Golden和Tile实现分别在后续步骤验收。
