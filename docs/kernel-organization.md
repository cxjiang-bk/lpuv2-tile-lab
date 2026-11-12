# OP与逻辑kernel的一一对应

当前阶段按16条OP登记16个完整的逻辑kernel，实现与验证均以OP输入输出为边界。kernel表示Python/PyPTO的完整实现入口；实际设备调度和跨OP融合在后续架构映射阶段评估。

GDN的Conv、门控和三角求解，MoE的TopK、分桶和专家计算，均作为对应kernel的内部步骤。Prefill/Decode、recurrent/chunk是同一kernel的实现路径。公共GEMM、量化和归约可以通过函数复用，按各自语义测试，再随完整kernel验证。

| OP | Kernel | 计算功能 | 内部步骤 / 路径 |
| --- | --- | --- | --- |
| OP-01 | K-01 | Token Embedding | Token ID检查；Embedding行Gather；输出组织；路径：gather |
| OP-08 | K-08 | Vision Patch Embedding | 像素[B,C,T,H,W]；显式im2col与矩阵乘及bias；Merge Token顺序与grid_thw；路径：pixel_im2col_matmul |
| OP-11 | K-11 | 视觉位置嵌入 | 由网格生成插值索引和FP32权重；四邻居Gather；位置插值；BF16转换后与hidden相加；路径：bilinear_position |
| OP-03 | K-03 | LayerNorm | 均值与方差归约；归一化；weight/bias仿射；BF16输出；路径：layernorm |
| OP-09 | K-09 | Vision Attention | QKV投影；完整axial RoPE；分段非因果QK、Softmax和PV；输出投影；路径：packed_noncausal |
| OP-04 | K-04 | Residual Add | 检查同形同dtype输入；残差相加；结果返回；路径：residual_add |
| OP-10 | K-10 | Dense FFN（GELU MLP） | 第一层Linear及bias；GELU(tanh)；第二层Linear及bias；路径：gelu_mlp |
| OP-12 | K-12 | Spatial Merger / Projector | 每Patch LayerNorm；2×2行分组与拼接；Linear、GELU exact、Linear；2048维视觉特征输出；路径：spatial_merger |
| OP-13 | K-13 | Visual Feature Scatter | 视觉特征与占位行关联；转换为目标hidden dtype；Scatter写入并保留其余文本行；路径：visual_scatter |
| OP-02 | K-02 | RMSNorm | FP32均方归约；rsqrt归一化；1+weight仿射；转换为输入dtype；路径：rmsnorm |
| OP-06 | K-06 | Gated DeltaNet 线性注意力 | qkv/z/a/b投影；因果Conv与SiLU及历史更新；Q/K head组织、L2Norm与门控；recurrent递推或chunk系数、三角求解与状态传递；gated Norm与输出投影；路径：recurrent, chunk |
| OP-05 | K-05 | Full Attention | Q/gate、K/V投影；Q/K Norm与Partial MRoPE；KV读取与更新；QK、mask、Softmax、PV；输出门控与输出投影；路径：prefill, decode |
| OP-07 | K-07 | MoE FFN | Router投影与Softmax；Top-8与权重归一化；按专家分桶与Gather；专家Gate/Up、SwiGLU和Down；按原Token合并贡献；shared专家和标量门控合并；路径：expert_bucketed |
| OP-14 | K-14 | LM Head | 选择输出位置；词表Linear投影；返回对应logits；路径：selected_positions |
| OP-15 | K-15 | Feature Concat | 检查两路输入位置和形状；按特征维拼接；返回4096维特征；路径：feature_concat |
| OP-16 | K-16 | MTP 输入投影 | 接收拼接特征；使用独立fc参数投影；返回2048维hidden；路径：input_projection |

## 验证与任务边界

每项kernel交付一个完整入口、独立OP golden、必要的中间检查点和端到端结果。内部阶段可分批实现、导出中间值和单独排错；完成状态按完整OP范围登记。模型子图由这些OP/kernel组合，MTP的Norm、Concat、输入投影和输出头按相应OP复用。

公共数值规则采用 [pto-functional-v0](functional-validation-v0.md)。逻辑shape、dtype、索引、有效区域、输出与状态沿用OP定义；此处的一一对应不改变数学计算或精度边界。

## 编号与旧记录

当前编号为`OP-01↔K-01`至`OP-16↔K-16`。[operators.json](../data/operators.json)与[kernels.json](../data/kernels.json)分别记录OP与完整kernel，A负责9项，B负责7项。

旧版阶段编号仅用于查阅历史case与结果，迁移关系保存在 [legacy-kernel-map.json](../data/legacy-kernel-map.json)。例如旧`K-GDN-04`是当前`K-06`的recurrent内部阶段，旧`K-NRM-03`对应当前`K-04`，旧`K-QNT-01`归入FP8 GEMM输入量化的公共函数。新用例使用当前编号，已有运行产物保留原始元数据。
