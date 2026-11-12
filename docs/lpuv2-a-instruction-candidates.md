# A 组算子导出的 LPUv2 指令候选集

评审稿 v0.6。范围为 A 组负责的 LayerNorm、Residual、GDN、Vision 与多模态融合，共9个OP。目标是确定 LPUv2 需要表达的计算和数据操作，再结合架构确定执行模块、Buffer、操作数约束与编码。

当前整理为 **29项基础候选功能、12项融合候选和6项编程模型约定**。基础条目中的模式、标量形式和dtype变体可以共享编码；条目数不代表最终指令数。原有 `A-ISA-01…16` 保留为需求分类索引。

每个OP仍对应一个完整逻辑kernel。本文的矩阵乘、归约、量化等均为kernel内部操作。机器可读清单见 [a-instruction-candidates.json](../data/a-instruction-candidates.json)，执行与工具链见 [A组实现说明](pypto-a1.md)和[版本锁定](../data/upstream.lock.json)。

## 1. 从算子到基础操作

| 计算路径 / OP | kernel中的计算顺序 | 导出的主要候选 |
| --- | --- | --- |
| Vision encoder / OP-08 Patch Embedding | 像素 `[B,C,T,H,W]` → 三维窗口展开 → `BF16(columns @ W.T + bias)` → Merge Token与网格 | 窗口地址生成与Gather、BF16矩阵乘、bias广播、转换与行重排 |
| Vision encoder / OP-11 位置嵌入 | `grid → coords → floor/clamp → idx[4], weight[4] → gather(table,idx) → weighted_sum → BF16 → add(x)` | 序号/索引运算、floor、min/max、Gather行、FP32加权归约 |
| Vision encoder / OP-03 LayerNorm | `FP32(x) → mean → centered → variance → rsqrt → affine → BF16` | 行归约、广播、FP32算术、rsqrt、转换 |
| Vision encoder / OP-09 Attention | `QKV → half-split RoPE → 按片段QKᵀ → scale → Softmax → PV → output projection` | 矩阵乘、布局转换、子窗口、归约、exp、除法、转换 |
| Vision encoder / OP-04 Residual | `out[valid] = x[valid] + residual[valid]` | BF16加法、有效区域、部分写入 |
| Vision encoder / OP-10 Dense FFN | `Linear → BF16 → GELU(tanh) → BF16 → Linear` | 矩阵乘、逐元素多项式/exp/除法、bias、转换 |
| Vision encoder / OP-12 Merger | `LayerNorm → BF16 → 4个Token按特征拼接 → Linear → GELU(erf) → BF16 → Linear` | Norm基础操作、分组/reshape、矩阵乘、GELU组合 |
| Multimodal fusion / OP-13 | `out = copy(text); out[placeholder_rows] = visual_features` | 复制、索引行覆盖写入；保留其他行 |
| Decoder / OP-06 GDN | `mask → FP8 qkv/z + BF16 a/b → causal DWConv/SiLU → Q/K L2Norm → FP32 state scan → gated RMSNorm → FP8 out` | FP8量化/点积、scale广播、历史窗口、FP32矩阵向量乘与外积组合、状态读写 |

公共OP-03/04在后续调用点复用。GDN在同一K-06中提供recurrent与chunk两种核心路径，前后投影、Conv和gated Norm共用。Prefill/Decode是运行场景，两种核心路径的状态接口一致。对应源码入口集中记录在机器清单的 `source_index` 中。

源码：[公共计算](../kernels/pypto/common.py)、[GDN](../kernels/pypto/gdn.py)、[FP8投影](../kernels/pypto/fp8.py)、[Vision、位置与Scatter](../kernels/pypto/vision_ops.py)。

## 2. 候选的操作数怎样描述

- `Tile<T>`：具有逻辑shape、有效区域和布局的Tile值，后续映射到tile register。
- `Scalar<T>`：常量或标量参数；标量形式是否占独立opcode由架构决定。
- `MemRef<T>`：张量存储对象及其shape/stride描述；访问用逻辑offset表示。
- `Mask`、`IndexTile`：逻辑选择掩码、整数索引；物理打包格式后续确定。

例如 `LOAD(tDst, mem, offset, valid)` 表示按窗口读出Tile，`MMA(acc,a,b)` 表示矩阵累加。这里列出的是语义参数，尚未规定二进制字段、Buffer枚举、物理寄存器号和地址位宽。

## 3. 基础指令候选

### 3.1 读写与布局：7项

| 候选名 | 输入 → 输出与语义 | 本轮所需dtype/范围 | 使用位置与PTO对照 |
| --- | --- | --- | --- |
| `LOAD` | `MemRef, offset, valid → Tile`；按shape/stride读取有效窗口，尾部填充值明确 | BF16、FP32、FP8 payload、整数与Mask的存储访问；dtype按调用点支持 | 全部OP；`TLOAD`。FP8原始字节读入与算术支持分别验收 |
| `STORE` | `Tile, MemRef, offset, valid → 更新窗口`；有效区域外保留原值 | 同上；部分写入不能覆盖其他行/列 | 全部OP；`TSTORE` |
| `COPY` | `src → dst`；数值和dtype不变，结果有独立存储语义 | BF16/FP32及索引数据；要求精确复制 | OP-13保留文本行、状态/历史搬运；可由`TLOAD/TSTORE`等组合 |
| `EXTRACT` | `src Tile, offset, shape/valid → dst Tile`；复制源Tile中的指定子窗口 | 保持dtype与元素位模式；源保持不变，目的具有独立存储语义；窗口受源有效域约束 | GDN提取Q/K/V、head和chunk，FP8投影提取K分组，Vision Attention提取片段；PTO对照`TEXTRACT` |
| `INSERT` | `dst Tile, src Tile, offset, valid → 更新后的dst`；将源有效区域写入目标窗口 | 同dtype精确写入，目标窗口外保持原值，源保持不变；窗口须落在目标允许写入的有效域内 | GDN组装各head/chunk的`core`，分块投影组装输出，Vision按片段写回context；PTO对照`TINSERT` |
| `FILL` | `Scalar<T>, shape → Tile<T>`；支持0及归约所需的填充值 | BF16/FP32/整数；Softmax无效分数可用`-inf` | 累加器、零状态、量化尾块；`TEXPANDS/TFILLPAD`等组合 |
| `TRANSPOSE` | `Tile, axes/layout → Tile`；保持逻辑元素映射 | BF16/FP32；Attention的head/sequence重排和矩阵操作数准备 | OP-09及矩阵路径；`TTRANS`。能由布局/访问模式表达时可以消除搬运 |

`EXTRACT`与`INSERT`作为基础候选单列，先确定子窗口复制与局部更新的功能。`COPY`描述整个对象的复制，`LOAD/STORE`连接MemRef与Tile，`EXTRACT/INSERT`描述Tile之间的窗口操作。

窗口offset按逻辑元素坐标解释，shape与valid明确本次读写范围，布局和stride保留原有元素映射。两项基础形式保持dtype和位模式；本轮复制/写入形式要求源、目的存储不重叠，重叠调用视为非法。布局转换、dtype转换或重叠读写如需支持，在对应模式中另行约定。

Slice视图共享源存储；EXTRACT的复制形式产生独立结果。编译器在证明可观察行为等价时可以消除复制，或用带offset的COPY/读写实现。独立opcode、tile register子视图和实际搬运路径在架构映射阶段确定。

当前PyPTO切片和赋值给出相应功能需求；具体切片可能生成带offset的LOAD、片上搬运或视图，现有算子通过结果不表示已独立验证PTO同名指令。`reshape`、`slice`和`concat`的处理见第6节；拼接可通过多次INSERT或目标支持的`TCONCAT`实现。

### 3.2 逐元素、归约与广播：11项

| 候选名 | 语义 | 本轮要求 | 使用位置 / PTO对照 |
| --- | --- | --- | --- |
| `ADD` | `dst = a + b` | BF16/FP32；Tile-Tile及Tile-Scalar形式，允许规定轴的广播 | Residual、Norm、bias、Conv与State；`TADD/TADDS` |
| `SUB` | `dst = a - b` | 本轮为FP32；`0-x`可以表达取负 | 中心化、Softmax、delta、位置坐标；`TSUB/TSUBS` |
| `MUL` | `dst = a * b` | BF16/FP32；支持标量、行/列向量广播；BF16边界保留 | Norm、门控、缩放、插值、RoPE与State；`TMUL/TMULS` |
| `DIV` | `dst = a / b` | FP32为主；支持常量除法。reciprocal+mul实现按数值标准验收 | Norm、Softmax、SiLU、量化scale；`TDIV/TDIVS` |
| `ABS` | `dst = abs(a)` | FP32；量化amax和插值距离 | OP-06/11；`TABS` |
| `MIN/MAX` | 逐元素/标量`min/max` | FP32和索引整数；两个操作可用模式区分 | clamp、插值边界；`TMIN/TMAX`及标量形式 |
| `CMP` | `Mask = compare(a,b,relation)` | 至少`==`、`<`、`>=`、`>`；Tile-Scalar形式 | 零量化组、Softplus阈值、GELU符号；`TCMP/TCMPS` |
| `SELECT` | `dst = where(mask,a,b)` | BF16/FP32；Mask广播；只选择结果不改变输入 | padding mask、全零scale、Softplus/GELU分支；`TSEL/TSELS` |
| `REDUCE_SUM` | 沿指定逻辑轴求和 | FP32累加；行/列归约，保留或恢复轴；仅统计有效元素 | LayerNorm、Q/K L2Norm、Softmax、插值、State矩阵向量乘；`TROWSUM/TCOLSUM` |
| `REDUCE_MAX` | 沿指定逻辑轴求最大值 | FP32；amax由`ABS+REDUCE_MAX`组合；无效元素不改变结果 | Softmax、动态量化；`TROWMAX`，其他轴由布局/实现映射 |
| `BROADCAST` | 将标量、行/列向量沿指定轴复用 | 保留源窗口offset及Token/head对应关系；不要求显式物化整块副本 | bias、mean/variance、scale、beta、decay、cos/sin；`TROWEXPAND*/TCOLEXPAND*`或带广播的算术 |

本轮归约要求有效归约长度大于0。跨多个Tile时，部分sum/max要按同一逻辑轴合并。归约顺序和近似实现按共同数值profile检查；量化payload和scale的精确比较规则另行保留。

### 3.3 矩阵、转换和非线性：7项

| 候选名 | 输入 → 输出与语义 | 必须保留的数值信息 | 使用位置 / PTO对照 |
| --- | --- | --- | --- |
| `MMA_BF16` | `A[M,K], B[K,N], acc32 → acc32 + A@B` | BF16操作数、FP32累加；支持分K部分和和`M=1`场景；输出BF16可由后接转换表达 | Vision投影/Attention、GDN a/b；`TMATMUL/TMATMUL_ACC`，`TGEMV`为可选实现 |
| `MMA_FP8` | FP8 E4M3FN操作数的矩阵点积，产生FP32结果/部分和 | 明确E4M3FN格式及累加精度；block scale由第4节组合或融合 | GDN qkv/z/out；`TMATMUL`与FP32缩放组合。直接FP8指令尚未验证 |
| `CVT` | 同一逻辑元素的dtype转换 | BF16↔FP32、FP32→E4M3FN及索引格式转换；明确舍入与溢出策略 | OP-03/06/08/09/10/11/12；`TCVT`。FP8转换采用clamp后的RNE并保留负零 |
| `EXP` | `dst = exp(x)` | FP32；精度需要支持Softmax、门控与跨Token状态验收 | OP-06/09/10/12；`TEXP` |
| `LOG` | `dst = log(x)` | FP32，模型Softplus分支输入为正 | OP-06 Softplus；`TLOG` |
| `RSQRT` | `dst = 1/sqrt(x)` | FP32；eps在调用点加入，不由指令隐含设置 | LayerNorm、L2Norm、gated RMSNorm；`TRSQRT` |
| `FLOOR` | `dst = floor(x)`，本轮输入为非负坐标 | FP32坐标取整；与索引转换分清顺序 | OP-08窗口/组序、OP-11插值坐标；该输入域下截断转换也可组成等价结果 |

`MMA_BF16`的FP32 accumulator与最终输出dtype分开。GDN State本身为FP32，所需的 `State × key` 不能直接替换成先转BF16的GEMM；本轮由FP32乘法与归约完成。

### 3.4 索引与间接访问：4项

| 候选名 | 语义 | 本轮约定 | 使用位置 / PTO对照 |
| --- | --- | --- | --- |
| `IOTA` | 产生逻辑序号`start + i*step` | 整数序号；可以是指令、编译器展开或加载预生成序号 | OP-08窗口、OP-11位置、OP-06块内因果mask；`TCI` |
| `INDEX_ALU` | 整数加减乘、非负整除与余数 | 索引计算要求精确；整除/余数可组合实现，也可在已知shape时预计算 | OP-08窗口/组序、OP-11网格/merge顺序；整数算术及转换组合 |
| `GATHER_ROW` | `table[idx[r], :] → dst[r,:]` | 索引指向逻辑表的行；合法范围由接口校验；数据与4邻点权重保持顺序；像素可视为宽度1的行 | OP-08窗口读取与输出重排、OP-11；GM表到Tile对照`MGATHER(Row)`，Tile内部重排对照`TGATHER` |
| `SCATTER_ROW` | `dst[idx[r], :] = src[r,:]` | 本轮索引唯一、覆盖写入；其他行保留。OP-13先复制文本，再覆盖占位行 | OP-13；全局写入对照`MSCATTER(Row,None)`，Tile内部写入对照`TSCATTER` |

`GATHER_ROW/SCATTER_ROW`要求的是按逻辑行访问数据。可以设置独立指令，也可以由索引寻址和普通读写组合。当前A组没有提出重复索引原子累加要求；B组MoE汇聚另行补充。

位置 kernel 从网格及 Merge 顺序生成 Token 的 `(row,col)`，再计算四个邻点的表行索引与 FP32 权重。像素窗口按输出位置与窗口内坐标生成 NCTHW 线性地址。当前浮点坐标路径限制在精确整数范围内；LPU 的 `INDEX_ALU` 约定保持精确整数语义。

## 4. FP8量化和分块缩放的计算契约

模型激活边界为BF16，进入FP8投影时执行逐Token/K128量化：

```text
x32 = FP32(x_bf16)
for each Token m and valid K-group g:
    amax = max(abs(x32[m, g.valid]))
    sx[m,g] = 1 if amax == 0 else amax / 448
    x8[m,g] = E4M3FN_RNE(clamp(x32[m,g] / sx[m,g], -448, 448))
    fill invalid K elements with encoded zero
return x8, sx               # 两个关联的逻辑输出
```

权重为`W8[N,K]`，`sw[nblock,g]`是checkpoint的BF16 `weight_scale_inv`，对应`128×128`权重block。它作为反量化乘数使用：

```text
acc32 = zeros([M, Nblock])
for g in K-groups:
    dot32 = dot_FP32(x8[:,g], transpose(W8[nblock,g]))
    acc32 = acc32 + (dot32 * sx[:,g]) * FP32(sw[nblock,g])
y_bf16 = BF16(acc32)
```

由此需要明确：K分组边界、N/K scale索引、FP32部分和、scale乘法次序、尾块以及最终BF16转换位置。不能先把所有K组的未缩放dot相加，再乘一个统一scale。

当前 GDN 在 HW PyPTO 生成的 PTO C++ 模拟器中以BF16精确承载有限FP8数值，量化、解码、点积和scale计算都在原生模拟器内执行。`fp8.py`中用于该承载路径的INT32编码扩宽、exponent/mantissa解码、符号位及恢复二次幂等步骤是仿真适配；它们不直接变成LPU的FP8硬件指令需求。基础 `LOG` 来自模型 Softplus；`FLOOR` 用于位置插值和窗口内坐标分解。窗口地址的整除/取余也可由整数或标量地址生成表达，当前原型的 FP32 坐标组合方式不决定LPU索引精度。

PTO的`TMATMUL_MX`采用目标相关scale契约，不能仅因名称含有scale就认定它等于这里的BF16 `128×128` block scale。若映射到MX指令，需要另验scale格式、分组与乘法规则。

OP-08 的输入已扩展为 BF16/FP32 像素张量。kernel 源码显式分为 `im2col → pl.matmul → bias`，host 不预切 Patch，也不调用 `pypto.conv`。当前 im2col 由设备端窗口坐标和 Gather 组合实现；`IMG2COL_3D` 只完成数据组织，后续矩阵乘与 bias 完成卷积。PTO `TIMG2COL` 的目标布局和时间轴映射须另验，不能直接把二维窗口接口视为完整 Conv3D。接口、布局和范围见 [像素 Conv3D](vision-pixel-conv3d.md)。

## 5. 融合与专用指令候选：12项

这些项有明确模型收益来源，先保留基础组合，后续用指令数、搬运量、临时存储和精度评估独立指令的价值。

| 候选名 | 基础组合及结果 | 主要取舍 |
| --- | --- | --- |
| `DYN_QUANT_K128` | ABS、max、除法、比较/选择、clamp、CVT → `payload, scale` | 双输出关联、amax归约与GEMM输入能否流水衔接 |
| `SCALED_MMA_FP8` | FP8 dot、两种scale乘法、FP32部分和 | scale格式、广播轴、K-group累加边界；不等同于任意MX格式 |
| `NORM` | sum、中心化/平方、sum、rsqrt、广播仿射；按调用点转换 | LayerNorm、L2Norm、gated RMSNorm语义分别保留，不能统一丢掉中间转换 |
| `SOFTMAX` | max、减法、exp、sum、除法 → BF16概率 | 分段边界、有效长度与精度；在线分块融合可在后续kernel比较 |
| `GELU` | tanh型由exp/除法/多项式组合；erf型当前用多项式组合 | tanh与erf路径分别对照reference，专用实现保留各自数学定义 |
| `IMG2COL_3D` | 三维窗口地址与Gather → `[M,C*Kt*Kh*Kw]` | 专用窗口搬运与通用组合的成本；按需供数、时间轴组织及矩阵布局 |
| `CAUSAL_DWCONV1D` | 历史窗口、逐tap乘加 → BF16 → SiLU | 短核历史读写、FP32累加与转换位置；与Vision Patch的矩阵投影分别评估 |
| `FP32_GEMV_OUTER` | FP32乘法+归约实现`Sᵀk`，广播乘法实现`k⊗delta` | State保持FP32；评估专用GEMV/外积与向量组合的成本 |
| `DELTA_STATE_UPDATE` | decay、prediction、delta、外积、状态更新、读出 | 融合粒度、State驻留、Token顺序及状态提交接口 |

GDN递推的基础组合为：

```text
# 每个Token、每个value head独立取其g/beta；Q/K共享head按映射读取
S32 = exp(g32) * S32
prediction[V] = reduce_sum_K(S32[K,V] * k32[K,1])
delta[V] = beta32 * (v32[V] - prediction[V])
S32 = S32 + k32[K,1] * delta[1,V]
out_bf16[V] = BF16(reduce_sum_K(S32[K,V] * q_scaled32[K,1]))
```

序列中`S_t`由`S_(t-1)`产生；Conv历史保存原始qkv投影，而不是Conv/SiLU后的结果。两类状态覆盖同一个已消费Token前缀并分别返回，输入状态保持不变。

关键转换也约束融合：

```text
LayerNorm: FP32均值/中心化方差/仿射 → BF16
GDN gated Norm: FP32归一化 → BF16 → BF16乘weight → FP32乘SiLU(z) → BF16
Vision Attention: QKᵀ转BF16 → 缩放保持BF16边界 → FP32 Softmax → BF16概率 → PV
Dense FFN / Merger: Linear+bias → BF16 → 对应GELU → BF16 → Linear
```

单独的sigmoid、SiLU、tanh、RoPE指令可从上述基础组合继续评估；本轮先用EXP/LOG/算术、窗口和广播覆盖。当前RoPE的cos/sin作为输入，后续可以加载预计算表或另建角度生成程序，尚无设备端sin/cos kernel的验证结果。

### 5.1 chunk 路径新增的三项候选

chunk在K-06内部构造块内系数，求解两个单位下三角系统，再按块传递FP32 State。它与逐Token recurrent使用同一数学递推，计算组织不同。

| 候选名 | 语义与组合 | 需要明确的约束 |
| --- | --- | --- |
| `PREFIX_SUM` | 沿块内Token轴做FP32包含式前缀和；当前用原生标量前缀累加和 Tile 写入，基础上可由标量控制、Tile 读写与 ADD 组合 | 每个chunk重新起算；不把上个chunk的前缀和带入；尾块仅含有效Token |
| `TRIANGULAR_MASK` | 生成`j <= i`的因果mask；当前用 `pl.tri` 生成因果下三角，PTO 对照 `TTRI` | 对角线保留；先将上三角的指数输入置0，EXP 后再乘 mask，避免无效指数溢出 |
| `TRIANGULAR_SOLVE` | 解`(I + strict_lower(A)) X = RHS`；当前用FP32前代、MUL、REDUCE_SUM和SUB | 单位对角线，忽略存储矩阵的对角值与上三角；两个RHS分别为V宽度与K宽度；行间有依赖 |

```text
for each value head and valid chunk:
    Q, K = L2Norm_FP32(q), L2Norm_FP32(k); Q *= key_dim**-0.5
    G = inclusive_prefix_sum(g, token_axis)
    D[i,j] = exp(G[i] - G[j]) if j <= i else 0
    A = ((beta * K) @ K.T) * D
    P = (Q @ K.T) * D
    U = unit_lower_solve(A, beta * V)
    W = unit_lower_solve(A, beta * K * exp(G))
    delta = U - W @ State
    out = BF16((Q * exp(G)) @ State + P @ delta)
    State = exp(G[-1]) * State + (K * exp(G[-1] - G)).T @ delta
```

上述系数、两次求解、矩阵积和State更新全部为FP32，只有核心输出回转BF16。当前FP32矩阵积由广播乘法与归约完成；BF16 MMA不能替代这些计算。后续可以比较FP32矩阵指令与向量组合的成本。

三项均是可评估的融合/专用候选，没有新增必须采用的opcode。`chunk_size`是本轮算法分块参数，LPUv2的物理Tile容量、驻留策略与执行模块在架构映射阶段确定。

## 6. 编程模型需要同时约定：6项

| 编号 | 约定 | 算子给出的具体要求 |
| --- | --- | --- |
| PM-01 | Tile类型、shape、valid region和布局 | 区分物理容量与逻辑有效范围；尾块load/store、归约和矩阵乘采用一致的有效域 |
| PM-02 | View/Slice与实际搬运 | slice视图保留offset/stride并共享源存储；EXTRACT复制形式产生独立结果；窗口shape/valid与布局映射明确；reshape仅在逻辑顺序允许时共享存储 |
| PM-03 | 写入、累加与别名 | INSERT只覆盖指定窗口并保留其余元素；COPY/EXTRACT与别名绑定分别定义；复制/写入的源、目的本轮要求不重叠；输入State/历史只读 |
| PM-04 | 多输出与关联数据 | 量化payload/scale、插值index/weight及GDN输出/两类状态独立命名，保持轴和分组对应 |
| PM-05 | 依赖、完成与生命周期 | 先完成producer再读Tile；Tile复用前消费者完成；State按Token/chunk顺序提交，State与Conv历史对应同一已消费前缀。显式事件或scoreboard等实现由架构映射确定 |
| PM-06 | 标量控制、参数与函数作用域 | 本轮shape/网格/片段边界可特化，循环可展开；运行时循环由控制器还是指令表达后续决定。函数形参与 local 作用域独立；循环中缓冲区复用必须保持消费者完成依赖 |

参考接口使用int64逻辑索引，当前PyPTO部分路径在范围检查后转为int32。索引物理位宽在映射阶段确定，缩窄转换必须有范围依据。

Vision Merger的reshape依赖已约定的Token分组顺序，GDN多个value head可读取同一个Q/K head；这些并不自动要求复制数据。PTO的`TASSIGN`是地址绑定接口，也不等于复制、数值转换或LPU必须采用的寻址方式。

## 7. 与原16类需求的对应关系

| 原需求ID | 基础候选 / 融合候选 / 编程模型 |
| --- | --- |
| A-ISA-01 读写与有效域 | LOAD、STORE、COPY、FILL；PM-01/02/03 |
| A-ISA-02 视图、布局、窗口 | EXTRACT、INSERT、TRANSPOSE；PM-01/02/03 |
| A-ISA-03 算术与选择 | ADD、SUB、MUL、DIV、ABS、MIN/MAX、CMP、SELECT |
| A-ISA-04 归约与广播 | REDUCE_SUM、REDUCE_MAX、BROADCAST |
| A-ISA-05 BF16矩阵计算 | MMA_BF16（含M=1场景）；分K部分和由FP32累加表达 |
| A-ISA-06 FP8分块点积 | MMA_FP8、MUL、ADD、BROADCAST、CVT；SCALED_MMA_FP8 |
| A-ISA-07 动态量化 | ABS、REDUCE_MAX、DIV、CMP、SELECT、MIN/MAX、CVT；DYN_QUANT_K128 |
| A-ISA-08 类型与舍入 | CVT；PM-01/04 |
| A-ISA-09 非线性 | EXP、LOG、RSQRT及算术/选择；NORM、GELU |
| A-ISA-10 Softmax | REDUCE_MAX/SUM、SUB、EXP、DIV、CVT；SOFTMAX |
| A-ISA-11 half-split RoPE | 窗口、MUL、ADD、BROADCAST、CVT；PM-02 |
| A-ISA-12 位置与Gather | IOTA、INDEX_ALU、FLOOR、CVT、GATHER_ROW及加权归约 |
| A-ISA-13 行Scatter | COPY、SCATTER_ROW；PM-03 |
| A-ISA-14 因果DWConv | 窗口、MUL、ADD、CVT、EXP、DIV；CAUSAL_DWCONV1D |
| A-ISA-15 FP32状态递推 | FP32算术/归约/广播；FP32_GEMV_OUTER、DELTA_STATE_UPDATE |
| A-ISA-16 依赖、别名、生命周期 | PM-01…06；同步和寄存器分配规则在映射阶段定义 |

## 8. 验证依据与下一步评审

[原生验证记录](../data/native-validation-summary.json)登记9个完整 OP、FP8量化/Linear边界、公共算子窗口、GDN两种路径及状态交接、像素窗口与完整卷积，以及 Attention/FP8 连续重复执行。后端统一为 `hw_pto_cpp_sim`：HW PyPTO → PTO IR → PTOAS → C++ 共享库 → simpler a2a3sim。逐项通过范围、误差与源代码 SHA 随记录保存。

GDN 采用 `bf16_carrier` 验证 FP8 数值语义；直接 FP8 硬件指令单独验收。当前结果用于支持候选功能评审；LPU候选指令的独立功能模型、PTO逐指令dtype/shape变体测试、真实checkpoint和完整模型验证在对应阶段开展。

各基础条目在机器清单中记录：操作数、dtype、语义、使用OP、源码符号、PTO参考和待补的定向验收。后续评审依次确认：

1. 是否接受这29项功能作为A组基础候选，哪些功能由访问模式或编译器组合表达。
2. 12项融合候选中优先比较动态量化、带scale的FP8点积，以及FP32 State更新；用具体程序成本作取舍。
3. 将PM-01…06纳入共同programming model，与B组的TopK、路由、KV和MTP需求合并。
4. 架构映射时再填写执行模块、输入输出Buffer、布局限制、资源、同步形式和指令字段，形成可执行的ISA v0。

chunk路径导出PREFIX_SUM、TRIANGULAR_MASK与TRIANGULAR_SOLVE三项候选，详见5.1。它们由基础操作组合实现，也可以在架构映射时评估专用指令。

PTO参考均限定[固定版本目录](pto-isa/README.md)。特别是`TGATHER/TSCATTER`与`MGATHER/MSCATTER`的访问对象、`TCVT`的格式/舍入，以及`TMATMUL_MX`的scale契约，需要按对应源码分别核对。

关键定义：[TEXTRACT](../third_party/pto-isa/docs/isa/TEXTRACT.md)、[TINSERT](../third_party/pto-isa/docs/isa/TINSERT.md)、[TCVT](../third_party/pto-isa/docs/isa/TCVT.md)、[TMATMUL](../third_party/pto-isa/docs/isa/TMATMUL.md)、[TMATMUL_MX](../third_party/pto-isa/docs/isa/TMATMUL_MX.md)、[MGATHER](../third_party/pto-isa/docs/isa/tile/ops/memory-and-data-movement/mgather.md)、[MSCATTER](../third_party/pto-isa/docs/isa/tile/ops/memory-and-data-movement/mscatter.md)。
