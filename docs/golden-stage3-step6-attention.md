# 阶段3步骤6：完整Full Attention

本步完成OP-05/K-05：输入已经归一化的hidden，输出Attention分支结果和追加后的KV。完整入口位于[kernels/pto/attention/full_attention.py](../kernels/pto/attention/full_attention.py)，Prefill和Decode使用同一入口。

## 入口接收什么

PyPTO入口使用显式Tensor参数。权重字典和KV字典在调用侧展开，测试中的`_tile_call`给出了完整参数顺序及输出分配方式。

| 参数组 | 名称与形状 | 含义 |
| --- | --- | --- |
| 当前输入 | BF16 `x[B,S,H]` | Decoder pre-norm之后的hidden |
| 投影权重 | FP8 E4M3FN `q_gate_weight[2*Q*D,H]`、`k_weight[K*D,H]`、`v_weight[K*D,H]`、`out_weight[H,Q*D]` | Q/gate、K、V和输出投影 |
| Norm权重 | BF16 `q_norm_weight[D]`、`k_norm_weight[D]` | 每个head的普通RMSNorm参数，使用`1+weight` |
| 投影scale | BF16 `q_gate_scale`、`k_scale`、`v_scale`、`out_scale` | 各投影的128×128权重块乘数；形状沿用公共FP8 Linear约定 |
| 位置 | INT64 `seq_pos[B,S]`、`rope_pos[3,B,S]` | 前者用于因果判断；后者按T/H/W轴用于旋转 |
| 历史KV | BF16 `old_k/old_v[B,K,L,D]`，INT64 `old_pos[B,L]` | 全部L行有效，各请求历史长度相同，只读 |
| 输出 | BF16 `out[B,S,H]`、`kv_k_out/kv_v_out[B,K,L+S,D]`；INT64 `kv_seq_pos_out[B,L+S]`、`kv_valid_len_out[B]` | 调用方分配，与输入分开存储 |

Q表示query head数，K表示KV head数。输入需符合当前模型配置：D=256，Q能被K整除；rotary_dim=64、theta=10,000,000、THW sections=(11,11,10)、Norm eps=`1e-6`。这些是本版使用的固定参数。投影scale当前采用checkpoint对应的BF16类型。

首次Prefill提供长度L=0的空KV Tensor；Decode把上次返回的K、V及位置作为历史输入。各请求当前S行、历史L行都有效，本版使用等长请求。文本位置可在调用侧把`seq_pos`扩展为三路相同的`rope_pos`；多模态输入提供各自THW坐标。

## 从hidden到输出的计算顺序

1. **投影并拆头。** 复用公共`fp8_linear`得到Q/gate、K、V的BF16结果。Q/gate按每个head的`[q256;gate256]`拆开。Q和K以每个head为一行调用普通RMSNorm。
2. **生成旋转表。** 生成32个半区频率，为每个Token选择T/H/W坐标。第j条频率选择`j%3`轴，正好对应11/11/10的交错分配。频率采用FP32的`exp(-ln(theta)*j/32)`表达，角度、sin和cos也在FP32中计算，再转BF16。
3. **旋转Q/K并排列head。** 前64维按前32/后32配对，分别计算`first*cos-second*sin`和`second*cos+first*sin`；两个乘积和加法保留BF16边界。后192维直接复制。数据由`[B,S,heads,D]`整理为按head连续的`[B,heads,S,D]`。
4. **追加KV。** 历史K/V复制到输出前缀，当前旋转后的K和投影后的V接在后面。键位置同步追加，有效长度写为L+S。历史输入不修改。
5. **按query head计算QK。** 以`kv_head=query_head//(Q/K)`选择对应KV head，通过共享BF16 Linear计算当前所有query与有效键的分数。当前query块使用Linear的16行分块，S不足16时按有效行处理；K方向部分积在FP32累加，完整score转BF16。
6. **缩放、mask和Softmax。** BF16 score乘`1/sqrt(256)=1/16`，再根据`key_pos>query_pos`将未来位置设为BF16最小有限值。公共Softmax用FP32完成整行统计，概率再转BF16。
7. **PV及门控。** 局部`_value_product`按`P[S,L+S] @ V[L+S,D]`直接读取原缓存布局，使用FP32部分和，结果转BF16。公共Sigmoid输出BF16 gate，context与gate逐元素相乘后仍为BF16，并整理回Token/head特征顺序。
8. **输出投影。** 使用公共FP8 Linear将`Q*D`特征映射回H维，得到BF16 `out`。

完整入口从hidden开始组织上述计算。Decoder pre-norm和残差由外部算子组合；本入口交付Attention分支结果。

## 暂存和当前实现范围

`@pl.jit`入口负责创建中间Tensor和组织公共/局部函数调用；运算节点使用显式Tile读取、运算和写回。本版保留投影、Norm、旋转、score和概率等中间Tensor，便于明确数值边界。每个query head访问完整有效KV，再做分块计算；没有实现online Softmax、分页KV或存储容量优化。

因果mask当前按键位置逐项判断，使用Tile中的标量写入表达屏蔽。GQA按需选择KV head视图，不创建重复的完整KV。BF16 Linear的权重形状为`[N,K]`，PV的V缓存形状为`[T,D]`，因此PV使用局部Tile乘法函数，按缓存现有顺序读取。

这些安排用于Python功能验证。LPU Buffer地址、容量、布局、跨单元流水及大序列性能在后续架构映射中确定。

## PyPTO执行适配

继续使用固定PyPTO `7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b`和步骤4的CPU代码生成路径。本步补充两项实际遇到的能力：

- `pymodel/isa/torch_codegen.py`为Tile序列生成`ci`以及FP32 sin/cos生成Torch基础运算。算子中的角度和旋转逻辑仍由PyPTO IR描述。
- `pymodel/isa/specialize.py`复用上游已有的按调用签名展开算法。当前上游只在含extern函数时启用它，普通调用会将不同形状的公共函数混用；Attention的Q、K/V、输出投影必须分别特化。局部入口让同一份公共实现按实际形状生成多个函数，已有相同签名则复用。

使用方式为`program = specialize(full_attention, *args)`，随后`torch_codegen(program)`并执行生成代码。输入参数及最小可运行组织见测试`_tile_call`。这些适配依赖所固定的PyPTO版本，升级时需复测；未修改上游源码、全局操作表或Golden。

## 本轮验证

```bash
.venv/bin/python -m pytest tests/kernels/test_attention_tile.py -q --basetemp=artifacts/golden-stage3-step6/pytest
.venv/bin/python -m pytest -q
```

| 用例 | 实际检查 |
| --- | --- |
| 两请求随机Prefill 3 Token接Decode 1 Token，Q=4/K=2/D=256 | 从hidden到主结果的全部步骤；不相同的THW坐标、分块尾部、两组KV head选择；Prefill和Decode的out/K/V分别按BF16容差比较，位置和有效长度精确比较 |
| GQA 8:1已知因果例子，Q=8/K=1/D=256 | Q/K/gate为零、V由输入构造，完整入口输出应依次为4、6、8；检查未来Token屏蔽和BF16门控 |
| RoPE定向例子 | 32条频率的THW选择、前后半区配对，以及192维尾部精确保留 |

三项专项均通过，本轮完整入口的主输出及K/V样本与Golden最大绝对误差均为0。历史输入及返回KV的历史前缀另按精确值检查，确认连续调用没有改写旧状态。数值验收继续使用`pto-functional-v0`，未放宽容差。

全套171项测试通过，项目Ruff及差异检查通过；全套结果保存在`artifacts/golden-stage3-step6/all-test-results.xml`。

测试为`tests/kernels/test_attention_tile.py`。Prefill、Decode和已知例子的IR/生成Torch程序，以及Case输入、两侧结果和报告，保存在`artifacts/golden-stage3-step6/pytest/`。实际后端为`pypto_torch_codegen_adapted`，设备CPU；本轮验证覆盖短序列功能，真实权重、长上下文和原生PTO/LPU后端仍在后续集成范围。

下一步为阶段3步骤7：完整MoE。
