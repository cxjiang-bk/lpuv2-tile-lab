# 阶段3步骤9：B路线实现与操作使用汇总

本文件汇总阶段3已经完成的B路线7个逻辑kernel，并把结果交给阶段4做Tile指令候选集整理。这里记录的是**Python/PyPTO功能验证证据**：程序使用固定版本的PyPTO生成CPU Torch代码，与独立PyTorch Golden比较。它不等同于PTO CPU/NPU后端验证，也不等同于LPUv2硬件指令已经确定。

## 1. 统一执行基线

| 项目 | 本阶段使用 |
| --- | --- |
| PyPTO | `7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b`，来源为官方main |
| PTO ISA | `90a3796597a5fabc2ce5066e73f785c2e85822f7`，与PyPTO分开登记 |
| 执行路径 | `pymodel.isa.specialize` + `pymodel.isa.torch_codegen` |
| 实际设备 | CPU；后端标签为`pypto_torch_codegen`或`pypto_torch_codegen_adapted` |
| 数值标准 | `pto-functional-v0`；BF16使用`atol=1e-2, rtol=1e-2`，FP32使用`atol=1e-5, rtol=1e-4` |
| 精确检查 | shape/dtype、搬运、索引、计数、mask、Top-K索引、FP8 payload及激活scale |
| 固定量化 | 激活沿K轴每128个元素一组，FP8为E4M3FN，scale为FP32反量化乘数 |

本阶段的程序仍使用小尺寸和随机权重验证功能边界；真实checkpoint权重、片上容量、物理Tile宽度、Bank/Buffer映射、性能和原生设备后端留到后续阶段。

登记表中的状态含义为：`available`表示独立Golden已经存在，`implemented`表示对应Python/PyPTO入口已经实现，`python_cpu_functional`表示已通过当前CPU生成功能路径与Golden比较，`partial_mtp`表示只完成最小MTP组合验证，不代表完整模型验证。

## 2. 七个完整kernel的交付情况

| OP/K | 入口与场景 | Tile组织和主要中间量 | 实际使用的能力 | 验证证据 |
| --- | --- | --- | --- | --- |
| OP-01/K-01 Embedding | `kernels/pto/common/embedding.py`；文本 Prefill/Decode，MTP复用 | 将`[B,S]`视为连续Token行；按行读取ID，按32列读取Embedding，尾块用有效区域写回 | `reshape`、标量`read`、带动态行偏移的`load`、`store`、循环和尾块计算 | [步骤2](golden-stage3-step2-embedding.md)；H=64/67两例，搬运精确一致；`pypto_torch_codegen` |
| OP-02/K-02 RMSNorm | `kernels/pto/common/rmsnorm.py`；Decoder、Q/K、Final Norm、MTP | 每行按128列分块；第一遍累加FP32平方和，第二遍重读输入和weight，广播归一化系数后转BF16 | BF16/FP32 `cast`、`fillpad`、`row_sum`、FP32累加、`rsqrt`、广播乘、有效区域slice/store | [步骤3](golden-stage3-step3-rmsnorm.md)；D=256/2048/259三例；全套150项时通过 |
| OP-05/K-05 Full Attention | `kernels/pto/attention/full_attention.py`；文本因果Prefill/Decode，历史KV和GQA | 复用FP8投影；按head组织Q/K/V；Partial MRoPE；追加KV；分块QK、mask、Softmax、PV和gate | FP8/BF16 Linear、head切片、三角函数、Tile matmul/FP32累加、归约Softmax、KV复制和位置更新 | [步骤6](golden-stage3-step6-attention.md)；Prefill→Decode、GQA8已知例、RoPE定向例；全套171项时通过 |
| OP-07/K-07 MoE FFN | `kernels/pto/moe/moe.py`；文本Prefill/Decode，空专家和动态桶 | Router→Softmax→Top-8；保留Token/slot；专家桶Gather；按专家复用工作区计算Gate/Up、SwiGLU、Down；按原Token合并并加入shared expert | BF16/FP8 Linear、Softmax、动态索引、循环选择、分桶计数、Gather、按路由权重缩放、累加写回 | [步骤7](golden-stage3-step7-moe.md)；随机、全同分、量化尾组、跨32专家边界4项专项；全套175项时通过 |
| OP-14/K-14 LM Head | `kernels/pto/output/lm_head.py`；Prefill/Decode和MTP输出头 | 读取请求内指定位置的hidden行，按词表列分块调用BF16 Linear，写回对应logits行 | 位置标量读取、slice/view、BF16矩阵乘和尾块写回 | [步骤5](golden-stage3-step5-output-mtp.md)；重复位置及词表尾块；6项专项中的LM Head用例 |
| OP-15/K-15 Feature Concat | `kernels/pto/mtp/input.py`；MTP两路已归一化特征 | 对每行按128列读取两路输入，依次写入`[e_norm;h_norm]` | 两路Tile load、偏移store、行/列尾块和逻辑视图 | [步骤5](golden-stage3-step5-output-mtp.md)；标记输入检查拼接顺序；6项专项中的Concat用例 |
| OP-16/K-16 MTP输入投影 | `kernels/pto/mtp/input.py`；MTP输入 | 调用已有BF16 Linear，将`2H`特征投影回`H`；输出由调用方分配 | 复用BF16 GEMM、M/N/K尾块和独立fc权重 | [步骤5](golden-stage3-step5-output-mtp.md)；独立fc方向和输出检查；6项专项中的投影用例 |

步骤8的[最小MTP组合](golden-stage3-step8-mtp.md)把以上7个入口串起来，并使用本地BF16 Tile加法作为A的OP-04尚未接入时的集成辅助。两项组合用例通过，完整回归为177项；这证明了kernel间的Token、输出Norm后hidden和draft KV交接，不代表A的Residual Add已完成。

## 3. 操作能力归并

下面按“解决什么问题”归并实际代码中出现的操作。表中的“PyPTO表达”是当前程序看到的API或组合；“阶段4关注点”是待决定的LPUv2候选能力，不是已经确定的指令名。

| 能力组 | PyPTO表达 | 已覆盖的kernel | 阶段4关注点 |
| --- | --- | --- | --- |
| 连续Tile读写 | `pl.load`、`pl.store`、`pl.write`、`pl.slice`、`pl.reshape` | 全部7个 | 连续读写、有效区域、视图/偏移和输出可见性 |
| 索引读取与重排 | `pl.read`配合动态行偏移、`slice`和循环；Embedding、LM Head、MoE桶和KV位置均使用 | K-01、K-05、K-07、K-14 | 标量索引、Gather、按位置选择、Scatter/重排是否需要专用能力 |
| 尾块与填充 | `pl.min`、`valid_shape`、`pl.tile.fillpad`、`pl.tile.slice` | 全部7个，尤其K-02/K-05/K-07 | 有效行列掩码、填充不参与归约、尾块写回 |
| 逐元素算术 | `pl.add`、`pl.mul`、`pl.div`、`pl.abs`、`pl.maximums/minimums` | Norm、Linear缩放、Attention、MoE、残差 | BF16/FP32算术、标量/Tile广播、舍入和格式转换边界 |
| 矩阵乘和累加 | `pl.tile.matmul`、`pl.tile.matmul_acc`、Left/Right/Acc/Vec移动 | 公共Linear、Attention QK/PV、LM Head、MoE专家 | 矩阵块尺寸、FP32累加、权重布局、累加器和搬运通路 |
| 行归约与广播 | `pl.row_sum`、`pl.row_max`、`pl.row_expand_mul/div` | RMSNorm、量化、Softmax、MoE路由 | 归约状态保存、跨Tile合并、广播、多生产者/消费者关系 |
| 类型转换与量化 | `pl.cast`、FP8 E4M3FN、按128元素产生activation scale | FP8 Linear、Attention、MoE、公共量化 | BF16/FP32/FP8转换，scale格式及与payload的成套搬运 |
| 特殊函数 | `pl.rsqrt`、`pl.exp`、`pl.sin`、`pl.cos`及Sigmoid/SiLU组合 | RMSNorm、Attention MRoPE/Softmax/gate、MoE | 是否提供独立特殊函数，或用基础指数/乘加组合 |
| 控制和动态范围 | `pl.range`、`if`、`pl.const`、`pl.yield_`、`pl.constexpr` | 尾块、Top-8选择、动态桶、空专家和形状特化 | 循环/分支表达、动态Ne、空工作项处理，哪些属于软件控制 |
| 状态和多输出 | 显式`Out`/`InOut` Tensor；KV、位置、长度和路由元数据独立返回 | K-05、K-07、MTP组合 | 写回完成、状态生命周期、别名/原地更新和提交边界 |

## 4. 由基础能力组合出的复杂步骤

以下功能已经在kernel中验证，但当前不是单一PyPTO操作，也不应直接登记为一条LPU指令：

- **Top-8**：用概率读取、稳定比较、已选标记和8轮循环实现；同分按专家ID升序。后续可评估排序/选择指令。
- **专家分桶**：扫描专家ID、写入`bucket_offset`、`bucket_token_id`和`bucket_slot`；后续可评估计数、前缀和、Gather/Scatter组合。
- **MoE合并**：按桶记录读取原Token和路由权重，逐块读—乘—加—写回；当前按专家顺序串行，没有验收硬件Atomic Add。
- **Partial MRoPE与因果Attention**：位置读取、sin/cos、半区配对、QK、位置mask、Softmax、PV和gate共同完成；后续评估是否需要位置/旋转专用能力。
- **KV追加**：复制历史KV和位置，再写入当前K/V；当前使用独立输出Tensor，未验证物理cache容量和原地更新。
- **最小MTP**：Python只负责Tensor分配、逻辑视图和调用顺序；数值计算仍由7个kernel及显式Tile残差加法完成。

## 5. PyPTO适配与真实缺口

步骤4以后使用的`pymodel/isa/torch_codegen.py`补充了几项CPU生成功能：嵌套语句分派、FP32点积语义、FP8 E4M3FN转换、标量clamp以及序列/三角函数生成。`specialize.py`按调用形状展开函数，避免公共函数在不同形状间错误复用。

这些适配只保证当前Python代码能执行，不代表对应能力已经存在于PTO CPU/NPU或LPU硬件。阶段4需要分别核对：

1. PyPTO API如何映射到现有PTO ISA语义；
2. 哪些组合可由基础Tile指令稳定表达；
3. 哪些能力值得加入LPUv2候选指令集；
4. Buffer、Tile寄存器、地址计算、端口冲突和状态生命周期如何落地；
5. 原生后端和真实模型规模下的性能、容量与反压。

## 6. 阶段4交付清单

阶段4可以直接使用以下材料：

| 材料 | 用途 |
| --- | --- |
| 本文第2节 | 作为7个kernel的实际入口、场景、Tile组织和证据索引 |
| 本文第3节 | 合并重复的Tile能力，形成候选指令分类 |
| 本文第4节 | 区分“已验证的组合行为”和“值得单独评估的硬件能力” |
| `docs/isa-requirements.md` REQ-B-01～07 | 保留每个OP/K的需求、当前表达和待确认边界 |
| `data/operators.json`、`data/kernels.json` | 查询责任归属、实现状态和验证证据 |
| `docs/golden-stage3-step1-execution.md`～`step8-mtp.md` | 追溯具体程序、case、命令和运行产物 |

阶段3步骤9至此完成。阶段4的第一项工作是从操作能力归并表中去重，并为每项候选能力补充输入输出、状态、资源和替代组合；不应把本文件中的PyPTO函数名直接当作最终LPUv2指令名。
