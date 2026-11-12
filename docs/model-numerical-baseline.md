# Qwen3.6-35B-A3B-FP8 模型与数值基线

基线ID：`qwen3_6_35b_a3b_fp8-v1`；确认日期：2026-11-10。本文说明两件事：权重、激活与状态采用哪种精度保存，以及算子计算中在哪些位置转换精度。

模型固定为 [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/tree/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2)，revision 为 `c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`。计算规则依据 [Transformers 固定源码](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)及其 [FP8 集成](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/integrations/finegrained_fp8.py)。模型尺寸、版本和来源 hash 见 [baseline.json](../configs/models/qwen3_6_35b_a3b_fp8/baseline.json)，完整数学与精度规则见 [numerics.json](../configs/models/qwen3_6_35b_a3b_fp8/numerics.json)。

当前Python Tile与同profile参考程序采用已确认的 [PTO功能验证标准v0](functional-validation-v0.md)，配置为 [functional-v0.json](../configs/models/qwen3_6_35b_a3b_fp8/functional-v0.json)。该profile明确项目的量化、误差及TopK规则；本文继续记录模型来源和精度转换。

## 1. 权重、激活与状态的精度

### 1.1 模型权重与量化参数

下表的存储精度来自固定 checkpoint，即模型发布的权重文件。已核对全部 42 个 safetensors 分片的 header；张量数量、形状和代表张量记录在 [checkpoint_metadata.json](../configs/models/qwen3_6_35b_a3b_fp8/checkpoint_metadata.json)。本次核对未加载权重 payload。

| 对象 | checkpoint 存储精度 | 加载和使用规则 |
| --- | --- | --- |
| Full Attention 的 Q/gate、K、V、输出投影 | FP8 E4M3 | 通过 FP8 Linear 执行；每个权重配套 block scale |
| GDN 的 qkv、z、输出投影 | FP8 E4M3 | 同上 |
| MoE routed / shared 专家的 Gate、Up、Down 投影 | FP8 E4M3 | 同上；Router 与 shared 标量门控单列如下 |
| Token Embedding、LM Head、整个 Vision Encoder（含 Merger） | BF16 | 在基础 BF16 profile 中使用 BF16 权重 |
| Decoder / QK / Final Norm、GDN gated Norm 的权重 | BF16 | Norm 内部的 FP32 计算及回转位置见第 2 部分 |
| MoE Router、shared 标量门控投影 | BF16 | 保留 BF16 输入及输出边界；Router 概率计算转 FP32 |
| GDN 的 a/b 投影、Conv 权重、A_log、dt_bias | BF16 | g 的计算显式转 FP32；参数存储精度与计算精度分别处理 |
| FP8 权重的 `weight_scale_inv` | BF16 | 保留 checkpoint scale 值；固定 FP8Linear 在默认 `scale_fmt=float` 下声明 FP32 scale 参数，加载后记录实际 dtype |

FP8 权重按 `[N,K]` 的 `128×128` block 配 scale，scale 形状为 `[ceil(N/128),ceil(K/128)]`。恢复关系为：

```text
W_real[n,k] = float(W_fp8[n,k]) * float(weight_scale_inv[n//128,k//128])
```

这里的 `weight_scale_inv` 实际作为乘数使用，保留发布值。完整不量化清单及 loader 的模块名匹配规则以 [config.json](../configs/models/qwen3_6_35b_a3b_fp8/config.json)和固定源码为准。

MTP 权重中，输入 fc、两路前置 Norm 和输出 Norm 为 BF16，Attention / 专家投影含 FP8。固定 Transformers 主模型会忽略 `mtp.*` 权重，没有 MTP forward；MTP 的激活、Norm 解释及 draft 状态规则由独立的支持 MTP 的来源补充。

本分支另以固定 SGLang 来源补充了 [最小 MTP 功能参考](golden-stage2-steps5-8.md)：两路普通 Norm、独立 fc、单层 Decoder、输出 Norm 与共享输出头，并明确 Token 移位、输出 Norm 后 hidden 和 draft KV 的传递。FP8 数值采用项目 v0，完整服务运行及接受/拒绝流程在集成阶段验证。

### 1.2 中间激活与跨 Token 状态

基础 profile 使用 BF16 激活。**权重是 FP8，并不表示该层的输入接口、输出或持久状态也都是 FP8。** FP8 Linear 在内部动态量化输入，输出回到 BF16，具体转换见 2.1。

| 对象 | 基础 profile 的保存 / 输出精度 | 说明 |
| --- | --- | --- |
| 文本 hidden、残差分支、Norm 输出、Attention / FFN 投影输出 | BF16 | Decoder 层间边界与子算子输出保持源码 dtype；FP32 临时计算完成后按第 2 部分回转 |
| Vision hidden、视觉特征及多模态融合输出 | BF16 | 像素输入先在 Conv3D 内转投影权重 dtype；位置相加、Scatter 保持目标 hidden dtype |
| LM Head 输出 logits | BF16 | 基础 profile 下按投影输出 dtype 返回；模型 forward 没有统一转 FP32 |
| Full Attention 的 K/V cache | BF16 | 保持 K/V 投影路径的 dtype；K 在 QK Norm、RoPE 后写入，V 为投影结果 |
| GDN Conv 历史 | BF16 | 保存投影后的 qkv 历史，遵循 cache 实现的实际 dtype；与 recurrent State 分别管理 |
| GDN recurrent State | FP32 | 跨 Token / chunk 保留；形状 `[B,32,128,128]`，缺省初始值为 0 |
| GDN 核心的 q/k/v、beta、g 及临时递推结果 | FP32 | 属于核心内部计算；核心输出回转 BF16，State 继续保存为 FP32 |
| MoE Router Softmax 概率与选中权重重归一化 | FP32 | 属于 Router 临时计算；最终路由权重回转 BF16 |
| Token ID、Torch TopK 专家索引 | int64 | 后端采用 uint32 等形式时显式检查范围并转换，比较逻辑索引值 |

独立的 KV 量化或 FP8 激活存储方案使用单独的 profile / case 描述。FP32 排错 profile 也单独命名，reference 保留上述基础精度及源码中的转换位置。

## 2. 算子内部的精度转换

以下用 `BF16(x)` / `FP32(x)` 表示显式转换；表中的 BF16 对应基础 profile。函数名用于标明计算阶段，完整公式、维度和轴定义见 [numerics.json](../configs/models/qwen3_6_35b_a3b_fp8/numerics.json)。

### 2.1 FP8 Linear 的输入量化与输出回转

固定 FP8 集成接收 BF16 激活，在 GEMM 内部或前置步骤进行动态量化，计算结果回转 BF16：

```text
x_bf16
    → dynamic_quantize(x_bf16) → (x_fp8, activation_scale)
    → GEMM(x_fp8, W_fp8, activation_scale, weight_scale)
    → y_bf16
```

动态量化根据当前输入计算 scale，再把输入量化为 FP8；它包含数值缩放和舍入。固定集成描述按 Token / K 分组的动态激活量化，并将具体计算交给 DeepGEMM 或 finegrained-fp8 等 kernel。实际 scale 表示、分组、舍入、截断和 GEMM 累加精度须随所选 kernel 版本及设备记录，配置中的 FP8 dtype 不能单独确定这些行为。[来源：FP8 Linear 集成](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/integrations/finegrained_fp8.py#L179-L282)。

Python功能验证固定采用`pto-functional-v0`：每Token沿K轴每128个元素一组，FP32计算`amax`和`scale=amax/448`，转换为`float8_e4m3fn`；scale为反量化乘数。全零组`scale=1`；尾组只统计有效元素；FP32除以scale并截断到`[-448,448]`后按最近偶数舍入。输入含NaN/Inf判失败。GEMM按K组应用对应scale，FP32累加并输出BF16。完整规则见 [功能验证标准](functional-validation-v0.md#2-动态激活量化)。

下一步可在 LPUv2 中尝试由生产者直接输出 `FP8 数据 + scale`，供兼容的消费者复用；候选方案见 [FP8 激活存储、量化融合与复用](fp8-activation-storage.md)。当前 reference 继续保留上述 BF16 接口和动态量化位置。

### 2.2 Norm 的转换顺序

普通 RMSNorm 与 GDN gated Norm 的转换顺序不同，需要分别实现。普通 RMSNorm 用于 Decoder、Q/K 及 Final Norm，权重采用 `1 + weight`：

```text
x32 = FP32(x_bf16)
n32 = x32 * rsqrt(mean(x32 * x32, axis=-1) + 1e-6)
y_bf16 = BF16(n32 * (1 + FP32(weight_bf16)))
```

GDN gated Norm 使用直接权重，并在乘权重之前先发生一次 BF16 舍入：

```text
o32 = FP32(core_output_bf16)
n32 = o32 * rsqrt(mean(o32 * o32, axis=-1) + 1e-6)
u_bf16 = weight_bf16 * BF16(n32)
y_bf16 = BF16(u_bf16 * SiLU(FP32(z_bf16)))
```

两条路径中的 BF16 回转位置都是基线的一部分，融合 kernel 时保留对应舍入。Vision LayerNorm 按固定 `torch.nn.LayerNorm` 执行，输入、参数和输出为 BF16，具体归约累加实现随 Torch / device 后端记录。[来源：普通 RMSNorm](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L914-L931)、[GDN gated Norm](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L206-L222)。

### 2.3 Attention、GDN、MoE 与 Vision 的转换点

| 位置 | 输入 → 计算与转换 → 输出 / 状态 |
| --- | --- |
| Text MRoPE | position / 频率计算 → FP32 cos/sin → 转 Q/K dtype 后执行旋转；输出 Q/K 为 BF16 |
| Text / Vision eager Attention | BF16 Q/K → score 与 mask → `Softmax(score, dtype=FP32)` → 转 query dtype BF16 → 与 BF16 V 相乘；Softmax 的 FP32 转换发生在 score 已生成之后 |
| GDN Conv | qkv → 转 Conv 权重 dtype BF16 → Conv / SiLU → 转回输入 qkv dtype BF16 |
| GDN beta | `sigmoid(b_bf16)` 得到 BF16 beta → 进入 delta-rule 核心时转 FP32 |
| GDN g | `-exp(FP32(A_log)) * softplus(FP32(a) + dt_bias)` → FP32 g |
| GDN 核心与 State | q/k/v、beta、g 转 FP32 → Q/K L2Norm 及 recurrent / chunk 计算 → 核心输出转原 query dtype BF16；最终 State 保持 FP32 |
| MoE Router | BF16 输入 × BF16 Router 权重 → BF16 logits → FP32 Softmax / Top-8 权重重归一化 → 路由权重转 logits dtype BF16；索引为 int64 |
| MoE Gate / Up → Down | FP8 Gate/Up 投影各输出 BF16 → `SiLU(gate) * up`，保留各步 BF16 输出 → Down 输入动态量化为 FP8 → Down 输出 BF16 |
| MoE 专家结果合并 | 专家输出 × 路由权重 → 转输出 buffer dtype BF16 后 `index_add`；shared 分支的 BF16 标量门控与输出按源码相乘后合并 |
| Vision Patch / 位置 | BF16/FP32 像素张量在 Conv3D 内转投影权重 dtype BF16 → 卷积得到 BF16 hidden；位置插值结果转 hidden dtype BF16 后相加 |
| Vision axial RoPE | Q/K、cos/sin 显式转 FP32 → 旋转计算 → 分别转回 Q/K 原 dtype BF16 |
| Vision MLP / Merger | BF16 Linear → GELU → BF16 Linear；保留激活输出 dtype，block 使用 GELU(tanh)，Merger 使用 GELU exact |
| BF16 投影、Residual、Embedding / Scatter | 按源码保持 BF16 输出边界；底层 GEMM / 归约内部累加精度随实际后端记录 |

Attention 的数学 reference 使用固定 eager 路径，GDN 使用固定 Torch recurrent / chunk fallback。具体 FP8 kernel 版本、运行环境、比较输出与误差记录遵循 [验证约定](validation.md)；CPU 反量化到 BF16 的排错路径单独登记。当前已完成来源、元数据核对及[A组首批算子reference](reference-a.md)。功能验证v0的数值标准已确认；golden、kernel与模型的实际验证进度按对应条目和运行结果登记，完整模型与目标FP8后端容差随对应实现验证。
