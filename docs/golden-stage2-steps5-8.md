# B路线步骤5—8：完整算子ref与最小MTP组合

本轮完成LM Head、Concat、MTP输入投影、Full Attention、MoE及最小MTP draft计算。Embedding、普通RMSNorm和公共数值函数直接复用，所有入口都在`pymodel/ref`，测试直接通过pytest调用。

模型与数值规则沿用`qwen3_6_35b_a3b_fp8-v1`和`pto-functional-v0`。函数支持模型参数形状，本轮用小型合成参数核对完整计算；初期Attention采用每个请求等长、全部Token有效的输入。

## 1. 调用入口

| 步骤 | 函数 / 文件 | 输入与返回 |
| --- | --- | --- |
| 5 | `lm_head` / [output.py](../pymodel/ref/output.py) | 已归一化`h[B,S,H]`、独立`weight[V,H]`和局部`output_positions[B,P]` → BF16 logits`[B,P,V]` |
| 5 | `feature_concat` / [output.py](../pymodel/ref/output.py) | 已归一化且对齐的`e_norm/h_norm[...,H]` → BF16 `[e_norm;h_norm][...,2H]` |
| 5 | `mtp_input_projection` / [output.py](../pymodel/ref/output.py) | BF16拼接特征和独立`fc_weight[H,2H]` → BF16 `[...,H]` |
| 6 | `full_attention` / [attention.py](../pymodel/ref/attention.py) | 已做Decoder前置Norm的BF16 hidden、投影/Norm参数、位置和可选KV → out及新KV |
| 7 | `moe` / [moe.py](../pymodel/ref/moe.py) | BF16 hidden、Router、专家及shared参数 → out、路由及分桶索引 |
| 8 | `prepare_mtp_tokens`、`mtp_step` / [mtp.py](../pymodel/ref/mtp.py) | 显式对齐Token/hidden，执行一次MTP模型计算 → logits、下一步hidden和draft KV |

函数保留原输入。参数字典只用于集中传递各矩阵；具体内容如下。所有Linear按`x @ weight.T`，FP8矩阵配套scale作为乘数传入。

## 2. Full Attention

```python
result = full_attention(x, weights, scales, seq_pos, rope_pos=None, kv=None)
```

| 对象 | 字段和含义 |
| --- | --- |
| `x` | BF16 `[B,S,H]`，当前新输入 |
| `weights` | FP8 `q_gate`、`k`、`v`、`out`；BF16 `q_norm`、`k_norm` |
| `scales` | 与4个FP8权重同名的block scale，各为`[ceil(N/128),ceil(K/128)]` |
| `seq_pos` | int64 `[B,S]`，因果顺序位置 |
| `rope_pos` | int64 `[3,B,S]`，THW坐标；纯文本缺省时复用`seq_pos` |
| `kv` | 可选字典：`k/v[B,kv_heads,L,D]`、`seq_pos[B,L]`；全部历史行有效 |

head数和head_dim从权重推导。模型参数为Q/gate`[8192,2048]`、K/V`[512,2048]`、输出投影`[2048,4096]`、Q/K Norm`[256]`，即16个query head、2个KV head、D=256。

计算顺序：

1. FP8投影得到Q/gate、K、V；每个head的Q/gate按256+256拆分。
2. Q/K做普通RMSNorm，前64维做Partial MRoPE，保留其余192维。THW频率选择与half-split配对遵循固定来源。
3. 新K/V追加到传入历史；query仅看到因果允许的key。GQA按query/KV head比例对应。
4. BF16 score经FP32 Softmax后回BF16，与V相乘；再乘sigmoid(gate)，执行输出投影。

返回的命名结果为`out`、`kv_k_out`、`kv_v_out`、`kv_seq_pos_out`和`kv_valid_len_out`。等长输入的有效长度相同；连续调用将返回状态传给下一次：

```python
kv = {"k": result["kv_k_out"], "v": result["kv_v_out"],
      "seq_pos": result["kv_seq_pos_out"]}
next_result = full_attention(next_x, weights, scales, next_pos, kv=kv)
```

多请求变长和padding适配随后续集成补充。Decoder前置Norm和残差由外层组合。

## 3. MoE

```python
result = moe(x, weights, scales, top_k=8)
```

`x`为BF16 `[B,S,H]`或`[T,H]`；内部按原顺序展平，最终恢复原形。

| `weights`字段 | shape / dtype |
| --- | --- |
| `router`、`shared_gate` | BF16 `[E,H]`、`[1,H]` |
| `expert_gate`、`expert_up`、`expert_down` | FP8 `[E,I,H]`、`[E,I,H]`、`[E,H,I]` |
| `shared_ffn_gate`、`shared_ffn_up`、`shared_ffn_down` | FP8 `[I,H]`、`[I,H]`、`[H,I]` |

`scales`使用相同的FP8字段名；routed专家scale多一个E轴。模型为E=256、I=512、Top-8，所有专家参数相互独立。

Router在BF16投影后用FP32求概率、Top-8及选中权重归一化，再回BF16。按专家ID升序遍历各桶，对桶内Token执行Gate/Up、SiLU×Up、Down，再逐行乘对应路由权重并加回原Token。跨专家合并保留BF16边界；shared FFN独立计算，乘每Token的标量sigmoid gate后加入。

返回`out`、原Token顺序下的`expert_id[T,8]`、`route_weight[T,8]`，以及`bucket_offset[E+1]`、`bucket_token_id[8T]`、`bucket_slot[8T]`。专家e的记录位于两个相邻offset之间；`bucket_slot`指回原Token的Top-8列，便于后续Tile程序核对对应。

## 4. 最小MTP组合

```python
shifted_ids = prepare_mtp_tokens(target_ids, target_next_ids)
result = mtp_step(shifted_ids, target_hidden, seq_pos, weights,
                  attention_weights, attention_scales, moe_weights, moe_scales,
                  rope_pos=rope_pos, kv=draft_kv)
```

计算为：Embedding → 两路独立Norm → `[e;h]`拼接 → BF16 fc → 单层Attention Decoder（前置Norm、Attention、残差、Norm、MoE、残差）→ 输出Norm → LM Head。残差使用BF16 Tensor加法表达，K-04仍由A维护。

### 参数绑定

| `weights`字段 | 对应参数 |
| --- | --- |
| `embedding`、`lm_head` | 调用方传入target共享的Embedding和LM Head；两者各有独立权重 |
| `pre_fc_norm_embedding`、`pre_fc_norm_hidden` | `mtp.pre_fc_norm_embedding.weight`、`mtp.pre_fc_norm_hidden.weight` |
| `fc` | `mtp.fc.weight`，BF16 `[2048,4096]` |
| `input_layernorm`、`post_attention_layernorm` | `mtp.layers.0`对应的两路Decoder Norm |
| `final_norm` | `mtp.norm.weight` |

`attention_weights/scales`和`moe_weights/scales`按前两节格式传入MTP自己的层参数。所有普通Norm采用已核对的`1+weight`公式。

### Token、hidden和位置对应

SGLang固定来源的prefill逻辑把每请求的Token序列左移一位，并在末尾填入target产生的next Token，target hidden行保持原序：

| draft槽位 | Embedding输入 | hidden输入 |
| --- | --- | --- |
| 0 | x1 | target h0 |
| 1 | x2 | target h1 |
| 2 | x3（target产生的next Token） | target h2 |

这里target hidden取普通路径输出Norm后的值；每行在MTP中还会通过自己的`pre_fc_norm_hidden`。`seq_pos/rope_pos`标记draft槽位，与未移位的target行对齐，函数不根据Token ID猜测位置。

最后一行logits用于选择x4。下一次draft调用输入x4、上一行的`hidden_out`和递增的draft位置，使用独立的draft KV。`hidden_out`位于MTP输出Norm之后，应原样作为后续调用的previous_hidden；不要额外恢复为Norm前的值。

`mtp_step`返回BF16 logits `out[B,S,V]`、`hidden_out[B,S,H]`、更新后的4份KV字段，以及本次MoE的`expert_id/route_weight`。本入口计算全部传入行，调用方自行选取最后一行logits和hidden继续。采样、target验证、接受/拒绝及状态恢复由模型集成流程承担。

## 5. 测试和来源核对

```bash
python3 -m pytest tests/reference/test_output.py tests/reference/test_attention.py tests/reference/test_moe.py tests/reference/test_mtp.py -q
python3 -m pytest -q
```

本轮新增11项代表测试：3项简单算子、3项Attention、2项MoE、3项MTP。涵盖可手算数值、Prefill→Decode与KV延续、THW/旋转尾部、Token—专家对应、共享贡献、MTP移位和后续hidden传递。与原有133项合计144项通过，Ruff通过。

主干计算对照固定Transformers commit `f90e93c4f83974f634860e246cdfd2c68666d605`的原始Attention、RoPE、Norm、Router与Experts类。源码SHA256先与基线匹配；Attention只将4个投影接入已验证的v0 FP8 Linear，其余保留原实现，短Prefill→Decode输出与KV精确一致。MoE在无同分、可精确量化的小例子上，与原始BF16源码的输出、专家ID及路由权重精确一致。这些是限定输入的数学对照，原生FP8后端与真实权重尚未接入。

MTP结构与输入对应依据SGLang commit `2d93fbfe5115131dd443dcbccde4426270119d6b`：

- [MTP forward及参数映射](https://github.com/sgl-project/sglang/blob/2d93fbfe5115131dd443dcbccde4426270119d6b/python/sglang/srt/models/qwen3_5_mtp.py)。
- [Qwen模型输出Norm](https://github.com/sgl-project/sglang/blob/2d93fbfe5115131dd443dcbccde4426270119d6b/python/sglang/srt/models/qwen3_5.py)。
- [draft prefill移位及后续循环](https://github.com/sgl-project/sglang/blob/2d93fbfe5115131dd443dcbccde4426270119d6b/python/sglang/srt/speculative/eagle_worker_v2.py)。
- [hidden capture选择](https://github.com/sgl-project/sglang/blob/2d93fbfe5115131dd443dcbccde4426270119d6b/python/sglang/srt/layers/logits_processor.py)和[GemmaRMSNorm原生公式](https://github.com/sgl-project/sglang/blob/2d93fbfe5115131dd443dcbccde4426270119d6b/python/sglang/srt/layers/layernorm.py)。

本轮是CPU数学参考和最小组合测试，没有启动SGLang服务或完整模型。FP8量化、Top-K同分与logits的BF16输出按项目共同profile执行。源码核对脚本和本次测试XML保存在本地忽略目录`artifacts/golden-stage2-steps5-8/`。
