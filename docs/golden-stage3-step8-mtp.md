# 阶段3步骤8：最小MTP Tile组合

本步将B路线已有的7个完整Tile kernel连接为一次MTP计算，并检查Prefill后接续draft时，Token、hidden和draft KV能否正确交接。调用入口为[pymodel/models/qwen36/mtp_tile.py](../pymodel/models/qwen36/mtp_tile.py)，测试为[tests/models/test_mtp_tile.py](../tests/models/test_mtp_tile.py)。

## 这一步负责什么

`mtp_step`是Python组合入口：分配Tensor、建立逻辑视图、依次调用已有kernel，并返回结果和状态。Embedding、Norm、投影、Attention、MoE及残差的数值计算均由PyPTO Tile程序执行，Golden只作为独立的比较对象。

MTP组合不增加第8个B路线kernel。A的OP-04 Residual Tile入口尚未接入，本步用私有`_residual_add`执行BF16 Tile加法，供两个残差位置使用；这部分不登记为A的OP-04验收。

## 调用时需要准备什么

入口参数顺序与Golden的`mtp_step`一致，额外提供可选`artifact_dir`，用于保存各阶段IR及生成的CPU代码。

| 参数 | 含义 |
| --- | --- |
| `token_ids`，INT64 `[B,S]` | 已按MTP规则对齐的Token ID |
| `previous_hidden`，BF16 `[B,S,H]` | 与各行Token配对的上一阶段输出Norm后hidden |
| `seq_pos`，INT64 `[B,S]` | 本次draft的因果位置 |
| `weights` | BF16参数：`embedding`、`lm_head`、`fc`及下表中的五份Norm权重 |
| `attention_weights/scales` | draft独立Attention参数；名称及形状沿用[完整Attention](golden-stage3-step6-attention.md) |
| `moe_weights/scales` | draft独立MoE参数；名称及形状沿用[完整MoE](golden-stage3-step7-moe.md) |
| `rope_pos`，可选INT64 `[3,B,S]` | THW坐标；省略时使用三路相同的`seq_pos`，适用于文本输入 |
| `kv`，可选字典 | draft历史：`k/v[B,K,L,256]`和`seq_pos[B,L]`；省略时从空历史开始 |
| `top_k` | 当前固定为8，与MoE Tile入口一致 |
| `artifact_dir`，可选路径 | 保存每个阶段的`*_ir.py`和`*_torch.py` |

`embedding[V,H]`和`lm_head[V,H]`可传入与target共享的参数；`fc[H,2H]`、五份Norm权重、Attention和MoE参数属于draft。五份Norm权重均为`[H]`，名称如下：

| 参数名 | 使用位置 |
| --- | --- |
| `pre_fc_norm_embedding` | Embedding输出 |
| `pre_fc_norm_hidden` | 上一阶段hidden |
| `input_layernorm` | Attention输入 |
| `post_attention_layernorm` | MoE输入 |
| `final_norm` | MTP层输出，随后送入LM Head及下一轮draft |

所有普通RMSNorm沿用`1+weight`及eps=`1e-6`。Attention/MoE投影使用FP8权重和BF16块scale；Embedding、fc、Norm、LM Head及组合边界沿用已有BF16规则。

## Token和hidden怎样对齐

Prefill时，每个请求的Token左移一位，末尾补入下一Token。例如target输入为`[1,2]`、下一Token为3，则MTP输入为`[2,3]`。

hidden和位置保持原来的行顺序：第0行用Token 2的Embedding配对target第0行hidden，第1行用Token 3的Embedding配对target第1行hidden。测试在输入准备阶段复用`prepare_mtp_tokens`完成这一规则；`mtp_step`接收的已经是对齐后的ID。

## 一次调用的计算顺序

| 阶段名 | 计算及结果 |
| --- | --- |
| `embedding` | 按Token ID读取共享Embedding |
| `embedding_norm` | 归一化Embedding，得到`e_norm` |
| `previous_hidden_norm` | 独立归一化上一阶段hidden，得到`h_norm` |
| `concat` | 按`[e_norm;h_norm]`拼接，得到`[B*S,2H]` |
| `input_projection` | 使用draft独立BF16 fc投影回H维，得到`projected` |
| `attention_input_norm` | 对`projected`做Attention输入Norm |
| `attention` | 完整Full Attention，返回分支结果及追加后的draft KV |
| `attention_residual` | `after_attention = projected + attention_out`，BF16加法 |
| `moe_input_norm` | 对`after_attention`做MoE输入Norm |
| `moe` | 完整Router、Top-8、分桶、专家及shared计算 |
| `moe_residual` | `after_moe = after_attention + moe_out`，BF16加法 |
| `output_norm` | 归一化`after_moe`，得到最终`hidden_out` |
| `lm_head` | 对本次所有S个位置投影，得到logits |

两个残差分别使用`projected`和`after_attention`作为基底。`hidden_out`是输出Norm之后的值，下一次draft直接使用相应行；LM Head内部不再重复Norm。

## 返回结果及下一次draft

| 返回名 | 形状和类型 | 用途 |
| --- | --- | --- |
| `out` | BF16 `[B,S,V]` | 本次各位置的logits |
| `hidden_out` | BF16 `[B,S,H]` | 输出Norm后hidden |
| `kv_k_out/kv_v_out` | BF16 `[B,K,L+S,256]` | 旧历史加当前Token的draft KV |
| `kv_seq_pos_out` | INT64 `[B,L+S]` | 与KV同步追加的位置 |
| `kv_valid_len_out` | INT64 `[B]` | 每个请求的有效长度，当前均为L+S |
| `expert_id` | INT64 `[B*S,8]` | 当前Token的Top-8专家 |
| `route_weight` | BF16 `[B*S,8]` | 对应路由权重 |

下一次draft按以下顺序组织：

1. 调用方从本次logits选择下一Token，形状为`[B,1]`。
2. 取本次`hidden_out[:, -1:]`，与该Token配对。
3. 将返回的K、V和位置组成下一次`kv`字典，并提供下一draft位置。
4. 再次调用`mtp_step`。历史KV只读，新KV另行返回。

这里传递的KV属于draft；target KV由外部保管。采样策略、target验证、候选树、接受/拒绝及状态回滚由后续模型集成负责。

## 执行和本轮验证

各阶段复用`specialize`和本项目的`torch_codegen`适配，生成CPU Torch代码后执行；实际后端标记为`pypto_torch_codegen_adapted`。PyPTO固定提交为`7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b`。本步没有修改Golden、执行适配或上游源码。

```bash
.venv/bin/python -m pytest tests/models/test_mtp_tile.py -q --basetemp=artifacts/golden-stage3-step8/pytest
.venv/bin/python -m pytest -q --junitxml=artifacts/golden-stage3-step8/all-test-results.xml
```

| 用例 | 检查内容 |
| --- | --- |
| 两请求，各2 Token Prefill，再接1次draft | Token移位、完整7类kernel组合、下一Token及输出Norm后hidden传递、draft KV从2行增长到3行、历史前缀保持；Prefill和后续draft的全部8份输出分别与Golden比较 |
| Attention输出投影及MoE Down权重置零 | 保留完整调用，单独核对两路Norm、Concat顺序、fc方向、残差基底、输出Norm及LM Head；另用直接公式验证hidden和logits |

两项专项均通过。BF16结果使用既定`atol=1e-2,rtol=1e-2`，整数结果精确比较；本轮数值样本与Golden的最大绝对误差为0。另精确检查旧KV及新KV的历史前缀，确保连续调用不改写输入历史。

每次MTP调用保存13份IR和13份生成程序，便于逐阶段查看实际Tile操作；Case输入、两侧输出和报告保存在`artifacts/golden-stage3-step8/pytest/`。

当前验证使用CPU小尺寸随机输入，沿用Attention的D=256、Partial RoPE 64维、等长有效请求及MoE Top-8约束。原生PTO/LPU后端、真实权重和性能不属于本步验证结论。

全套177项测试通过，Ruff及差异检查通过；全套结果为`artifacts/golden-stage3-step8/all-test-results.xml`。独立代码审查未发现需要修复的问题。

下一步为阶段3步骤9：汇总7个kernel的实现与操作使用记录。
