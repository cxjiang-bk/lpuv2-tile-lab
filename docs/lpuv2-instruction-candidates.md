# LPUv2 指令候选清单

来自 A/B 两部分的16条 OP：31项基础候选（含NVFP4扩展）和17项融合/专用候选。名称表示计算语义，操作数、Buffer和编码在架构映射阶段确定。

A使用原生PTO C++功能仿真，B使用PyPTO CPU Torch生成；通过范围分别见[原生验证汇总](../data/native-validation-summary.json)与[B实现汇总](golden-stage3-step9-summary.md)。MMA_NVFP4为扩展需求。机器清单、16条OP覆盖关系及源码SHA见[instruction-candidates.json](../data/instruction-candidates.json)。INDEX_ALU沿用飞书的划除评审状态，整数控制/寻址能力随编程模型定义。

## 读写与布局（7项）

| 候选名 | 输入 → 输出与语义 | 数据类型与操作约束 | 使用位置 / PTO对照 |
| --- | --- | --- | --- |
| `LOAD` | MemRef, offset, valid → Tile；按shape/stride读取有效窗口，尾部填充值明确 | BF16、FP32、FP8 payload、INT32/INT64及Mask；支持Tile窗口和标量索引读取，明确stride、有效域与尾块填充值。 | 全部OP；Embedding ID、LM Head选行、KV位置、专家桶offset/Token/slot；TLOAD及标量读取组合。FP8字节读入与算术分别验收 |
| `STORE` | Tile, MemRef, offset, valid → 更新窗口；有效区域外保留原值 | 与LOAD对应；支持Tile窗口和标量元数据写回；只更新有效区域，其他行/列保持原值。 | 全部OP；主结果、KV、GDN/Conv状态及路由元数据；TSTORE及标量写入组合 |
| `COPY` | src → dst；数值和dtype不变，结果有独立存储语义 | BF16/FP32及索引数据；要求精确复制 | OP-05复制历史KV/位置，OP-06状态/Conv历史，OP-13保留文本行；LOAD/STORE组合 |
| `EXTRACT` | src Tile, offset, shape/valid → dst Tile；复制源Tile中的指定子窗口 | 保持dtype与元素位模式；源保持不变，目的具有独立存储语义；窗口受源有效域约束 | OP-05/06/09的Q/K/V、head、Token片段；OP-07专家权重/scale窗口；OP-14指定hidden行；PTO TEXTRACT。逻辑view可由寻址消除复制 |
| `INSERT` | dst Tile, src Tile, offset, valid → 更新后的dst；将源有效区域写入目标窗口 | 同dtype精确写入，目标窗口外保持原值，源保持不变；窗口须落在目标允许写入的有效域内 | OP-05追加KV及组装context，OP-06组装head/chunk，OP-15按列拼接；PTO TINSERT或偏移STORE组合 |
| `FILL` | Scalar<T>, shape → Tile<T>；支持0及归约所需的填充值 | BF16/FP32/整数；支持0及模型指定mask填充值，包含BF16最小有限值或FP32 -inf；无效lane按归约规则填充。 | Norm/Softmax/量化尾块、矩阵累加器、零状态、MoE合并结果；TEXPANDS/TFILLPAD组合 |
| `TRANSPOSE` | Tile, axes/layout → Tile；保持逻辑元素映射 | BF16/FP32；Attention的head/sequence重排和矩阵操作数准备 | OP-05/09的QK和PV、GQA/head/sequence布局及矩阵操作数准备；TTRANS。可用view/访问模式消除实际搬运 |

## 逐元素、归约与广播（11项）

| 候选名 | 输入 → 输出与语义 | 数据类型与操作约束 | 使用位置 / PTO对照 |
| --- | --- | --- | --- |
| `ADD` | dst = a + b | BF16/FP32；Tile-Tile及Tile-Scalar形式，允许规定轴的广播 | Residual、LayerNorm/RMSNorm、bias、RoPE、Conv、GDN State和MoE逐贡献合并；TADD/TADDS |
| `SUB` | dst = a - b | FP32；支持Tile-Tile/Tile-Scalar；0-x表达取负。 | 中心化、Softmax、GDN delta、插值坐标；TSUB/TSUBS |
| `MUL` | dst = a * b | BF16/FP32；支持标量、行/列向量广播；BF16边界保留 | 全部Norm、路由权重、门控、量化scale、RoPE/插值及GDN State；TMUL/TMULS |
| `DIV` | dst = a / b | FP32为主；支持常量除法。reciprocal+mul实现按数值标准验收 | Norm、Softmax、Top-8权重再归一化、Sigmoid/SiLU、K128 scale；TDIV/TDIVS |
| `ABS` | dst = abs(a) | FP32；量化amax和插值距离 | OP-05/06/07的量化amax，OP-11插值；TABS |
| `MIN/MAX` | 逐元素/标量min/max | FP32和索引整数；两个操作可用模式区分 | Softmax跨Tile最大值、量化clamp、插值边界及动态尾块；TMIN/TMAX和标量形式 |
| `CMP` | Mask = compare(a,b,relation) | FP32分数及INT32/INT64索引/计数按调用点支持；至少==、<、>=、>；支持Tile/标量比较，整数比较保持精确。 | MoE稳定Top-8、专家匹配与空桶分支；Attention位置mask；全零量化组、Softplus/GELU分支；TCMP/TCMPS或标量控制组合 |
| `SELECT` | dst = where(mask,a,b) | BF16/FP32；Mask广播；只选择结果不改变输入 | Attention/padding mask、Top-K已选屏蔽、全零scale、Softplus/GELU；TSEL/TSELS或分支组合 |
| `REDUCE_SUM` | 沿指定逻辑轴求和 | FP32累加；行/列归约，保留或恢复轴；仅统计有效元素 | LayerNorm/RMSNorm、Q/K L2Norm、Softmax、Top-8权重归一化、插值与State矩阵向量乘；TROWSUM/TCOLSUM |
| `REDUCE_MAX` | 沿指定逻辑轴求最大值 | FP32；amax由ABS+REDUCE_MAX组合；无效元素不改变结果 | OP-05/09 Softmax、OP-07 Router Softmax、所有FP8 Linear动态量化；TROWMAX等 |
| `BROADCAST` | 将标量、行/列向量沿指定轴复用 | 保留源窗口offset及Token/head对应关系；不要求显式物化整块副本 | Norm系数、bias、activation/weight scale、路由权重、beta/decay、cos/sin；TROWEXPAND*/TCOLEXPAND*或广播算术 |

## 矩阵、转换和非线性（9项）

| 候选名 | 输入 → 输出与语义 | 数据类型与操作约束 | 使用位置 / PTO对照 |
| --- | --- | --- | --- |
| `MMA_BF16` | A[M,K], B[K,N], acc32 → acc32 + A@B | BF16矩阵操作数，FP32累加器及部分和；支持分K累加、M=1和有效尾块；结果转BF16时由CVT规定舍入。 | Vision投影/Attention、GDN a/b，OP-05 QK/PV，MoE Router/shared门控，LM Head及MTP输入投影；TMATMUL/TMATMUL_ACC；M=1可用TGEMV |
| `MMA_FP8` | FP8 E4M3FN操作数的矩阵点积，产生FP32结果/部分和 | FP8 E4M3FN操作数、FP32累加；activation scale为FP32/K128，weight scale对应128×128 block、BF16存储后转FP32（也支持独立FP32 scale用例）；各K组先应用scale再合并。 | OP-05 Q/gate、K、V、out；OP-06 qkv/z/out；OP-07 routed/shared专家Gate/Up/Down。A用BF16精确承载FP8，B用CPU生成路径；直接FP8硬件指令单独验收 |
| `MMA_NVFP4` | A4[M,K], B4[K,N], sA, sB, gA, gB, acc32 → acc32 + Â@B̂；Â=A4×sA×gA，B̂=B4×sB×gB（scale按块广播）。 | 4-bit E2M1 payload；采用1D分块：沿K每16个元素共享FP8 E4M3 block scale，每个操作数另有FP32 tensor scale；FP32累加；明确打包顺序、scale布局及K尾块补零。 [格式参考](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) | 低精度Linear/GEMM扩展；当前Qwen FP8基线无该算子验收。PTO TMATMUL_MX的FP4/E8M0路径可作参考，NVFP4 E4M3/K16及FP32全局scale需独立映射 |
| `CVT` | 同一逻辑元素的dtype转换 | BF16↔FP32、FP32→E4M3FN及INT32/INT64索引转换；明确舍入与饱和规则。FP8采用clamp到[-448,448]后RNE并保留负零；整数缩窄前检查范围。 | 全部投影/Norm/Softmax/门控的BF16↔FP32边界，K128量化、权重scale扩宽、索引位宽转换；TCVT |
| `EXP` | dst = exp(x) | FP32输入/输出；允许负输入；明确精度以及溢出、下溢处理。 | Softmax、Sigmoid/SiLU、GDN decay/Softplus、GELU近似及MRoPE频率生成；TEXP |
| `LOG` | dst = log(x) | FP32输入/输出；有效输入x>0；明确精度及非法输入处理。 | OP-06 Softplus；TLOG |
| `RSQRT` | dst = 1/sqrt(x) | FP32输入/输出；有效输入x>0；eps由调用方显式相加；明确计算精度。 | OP-02普通RMSNorm、OP-03 LayerNorm、OP-05 Q/K Norm、OP-06 L2Norm与gated RMSNorm；TRSQRT |
| `FLOOR` | dst = floor(x) | FP32输入/输出；结果为向负无穷取整后的FP32数值，生成整数索引需接CVT；坐标输入要求非负。 | OP-11插值坐标；该输入域下截断转换也可组成等价结果 |
| `SIN/COS` | dst=sin(x)或cos(x)，输入角度单位为弧度 | FP32输入/输出；在模型要求的边界转BF16；定义范围缩减与计算精度。可用预计算表+Gather提供等价数据。 | OP-05 Partial MRoPE；B使用pl.sin/pl.cos的CPU生成适配。固定PTO参考目录没有同名独立指令，需组合或扩展 |

## 索引与间接访问（4项）

| 候选名 | 输入 → 输出与语义 | 数据类型与操作约束 | 使用位置 / PTO对照 |
| --- | --- | --- | --- |
| `IOTA` | 产生逻辑序号start + i*step | INT32/INT64序号Tile；start、step、长度与合法范围明确；索引向量存入Tile/Buffer，消费者接收其引用。 | OP-11位置/窗口坐标、OP-06因果mask、OP-05 MRoPE频率序号；PTO TCI。静态序列也可由host或编译器准备 |
| `GATHER_ROW` | table[idx[r], :] → dst[r,:] | idx是Tile/Buffer中的索引向量，由调用方准备或前序kernel生成；指令引用索引对象。BF16/FP32数据、INT32/INT64索引，0≤idx[r]<表行数；保持顺序和元素位模式。 | OP-01 Embedding、OP-11位置表、OP-14选hidden行、OP-07专家桶收集；当前B用标量read+动态load组合；PTO MGATHER(Row)/TGATHER |
| `SCATTER_ROW` | dst[idx[r], :] = src[r,:] | 数据为BF16/FP32，索引为INT32/INT64；索引唯一且在合法行范围内；同dtype覆盖写入，未选中行保持原值；源行与索引逐项对应。 | OP-13视觉特征覆盖写入、OP-05按唯一位置追加KV、OP-15按偏移组装；PTO MSCATTER(Row,None)/TSCATTER。重复Token累加见SCATTER_ADD_ROW |
| `GATHER_ELEM` | src_flat[idx[r,c]] → dst[r,c]；每个输出元素独立索引 | BF16/FP32数据、INT32/INT64索引Tile；0≤idx<源元素数；重复索引合法，保持元素位模式及输出顺序。 | OP-08原始像素三维im2col窗口；当前A以FP32暂存和pl.tile.mgather(coalesce="elem")实现；PTO MGATHER(Elem) |

## 融合与专用候选（17项）

| 候选名 | 输入 → 输出与语义 | 数据类型与操作约束 | 使用位置 / PTO对照 |
| --- | --- | --- | --- |
| `DYN_QUANT_K128` | BF16 x[M,K] → E4M3FN payload[M,K]与FP32 scale[M,ceil(K/128)] | 每Token/K128组amax；scale=amax/448，全零组取1；clamp后RNE并保留负零；尾组只计有效元素，padding为0。 | OP-05/06/07所有FP8 Linear；ABS+REDUCE_MAX+DIV+CVT组合 |
| `SCALED_MMA_FP8` | FP8点积 → 两种scale乘法 → FP32跨K组累加 | activation scale为FP32/K128；weight scale按128×128 block读取并转FP32。每K组先乘两种scale，再FP32累加，最后按模型边界转BF16。 | OP-05/06/07投影；MMA_FP8+广播MUL+ADD+CVT组合 |
| `NORM` | LayerNorm、L2Norm和RMSNorm的归约、归一化与仿射 | FP32归约与归一化；明确eps、有效维数、gamma或1+weight解释；保留GDN gated Norm等路径的中间BF16转换。 | OP-02/03/05/06/12及MTP；REDUCE_SUM/MUL/ADD/RSQRT/CVT组合 |
| `SOFTMAX` | max → 减法 → exp → sum → 除法 | FP32计算，支持跨Tile合并max和sum；Router保留FP32概率，Attention按模型转BF16；片段/有效长度和mask填充值明确。 | OP-05 Full Attention、OP-09 Vision Attention、OP-07 Router；REDUCE_MAX/SUB/EXP/REDUCE_SUM/DIV组合 |
| `GELU` | GELU(tanh)或GELU(erf) | 两种数学定义分别保留，转换位置与模型一致 | OP-10 Vision Dense FFN、OP-12 Merger；基础算术/EXP组合或专用近似 |
| `IMG2COL_3D` | 三维像素窗口 → [M,C*Kt*Kh*Kw] | 明确窗口、stride、时间/空间轴及输出顺序 | OP-08 Patch Embedding；GATHER_ELEM+坐标/寻址组合；PTO TIMG2COL的布局和时间轴映射独立评估 |
| `CAUSAL_DWCONV1D` | 历史窗口逐tap乘加 → BF16 → SiLU | FP32累加；历史保存卷积前数据并按Token顺序更新 | OP-06 GDN causal Conv；LOAD/MUL/ADD/CVT及SiLU组合 |
| `FP32_GEMV_OUTER` | Stateᵀ×key及key⊗delta | 矩阵向量乘与外积保持FP32 | OP-06 GDN recurrent/chunk；FP32 MUL+REDUCE_SUM/广播组合 |
| `DELTA_STATE_UPDATE` | 衰减 → prediction → delta → 外积更新 → 读出 | State保持FP32，Token依赖顺序与状态提交边界明确 | OP-06 GDN recurrent及状态交接；EXP/MUL/REDUCE_SUM/ADD组合 |
| `PREFIX_SUM` | 沿指定轴产生包含式前缀和，或以mode指定不含当前项的前缀偏移 | GDN为FP32且每chunk重新起算；专家计数/偏移为精确INT32/INT64，空桶允许，检查总数和位宽。 | OP-06 chunk gate前缀和；OP-07桶offset可由计数+整数scan构造，当前B以有序扫描写offset |
| `TRIANGULAR_MASK` | 按j≤i/j<i，或key_pos[j]≤query_pos[i]产生Mask | INT32/INT64位置比较精确；明确对角线、历史KV偏移和padding有效域；Mask由SELECT写入模型规定填充值。 | OP-06 chunk因果系数，OP-05 Full Attention含历史KV的因果mask；IOTA/CMP/SELECT或标量控制组合 |
| `TRIANGULAR_SOLVE` | 解(I+strict_lower(A))X=RHS | FP32前代求解；单位对角线；忽略上三角 | OP-06 GDN chunk；FP32前代MUL/REDUCE_SUM/SUB组合 |
| `TOPK` | scores[M,E],k → values[M,k],expert_id[M,k]，按分数降序选择 | 分数FP32；模型E=256、k=8；同分专家ID小者优先，ID不重复且在合法范围。values和ID分开输出；权重再归一化由REDUCE_SUM/DIV/CVT完成。 | OP-07 MoE路由；B以严格比较+升序扫描+已选屏蔽实现；可评估PTO TSORT32/TMRGSORT组合或专用Top-K |
| `EXPERT_BUCKET` | expert_id[M,k] → offset[E+1],token_id[M*k],slot[M*k]，按expert分桶 | INT32/INT64元数据；offset单调、首项0、末项M*k；同专家内按Token再slot顺序，允许空桶；Token/slot/route weight及量化payload/scale关联保留。 | OP-07 MoE Dispatch；B用标量比较/计数/写入，候选组合为计数+PREFIX_SUM+Gather/Scatter |
| `SCATTER_ADD_ROW` | 按输入顺序执行dst[idx[r],:]+=src[r,:]，允许重复目标行 | BF16/FP32；保持每次加法的模型精度边界。当前MoE按专家/桶顺序串行读-加-写；并发实现需另定冲突顺序和原子语义。路由权重乘法先独立执行。 | OP-07专家结果回到原Token；动态LOAD+ADD+STORE组合。当前结果不作为硬件atomic验收 |
| `SIGMOID/SILU` | sigmoid(x)=1/(1+exp(-x))；silu(x)=x*sigmoid(x) | FP32内部计算；输入/输出及中间BF16舍入位置按调用路径保留；GDN gate、MoE专家和shared标量门控分别定义。 | OP-05输出gate、OP-06 beta/Conv/gated Norm、OP-07 SwiGLU/shared gate；EXP/ADD/DIV/MUL/CVT组合 |
| `ROPE` | a′=a*cos-b*sin；b′=b*cos+a*sin；其余维度原样保留 | OP-05 Partial MRoPE固定256维head中前64维旋转、两半32维配对，THW交错频率及位置来源明确；BF16乘加边界保留。Vision使用自己的cos/sin和旋转范围。 | OP-05 Partial MRoPE、OP-09 Vision RoPE；SIN/COS或查表+EXTRACT/MUL/ADD/INSERT组合 |
