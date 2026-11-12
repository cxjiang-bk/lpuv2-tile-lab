# LPUv2 ISA与programming model需求

每条需求来自具体OP/K及可复现case，随程序验证逐步确认。当前尚无已接受的LPUv2指令或编码。

A组已根据9个完整kernel整理[指令候选集评审稿](lpuv2-a-instruction-candidates.md)：29项基础候选功能、12项融合候选及6项编程模型约定，保留原`A-ISA-01…16`分类索引。[机器清单](../data/a-instruction-candidates.json)记录操作数、dtype、源代码入口、PTO对照和待补验收。条目数不代表最终opcode数量；当前证据是算子级功能对照，LPU指令级模型与测试在后续阶段建立。

| 字段 | 内容 |
| --- | --- |
| 需求编号 / 状态 | REQ编号；candidate/accepted及评审结果 |
| 计算来源 | OP/K编号、场景、公式与case |
| 数据对象 | Tensor/Tile/Slice、dtype、布局、有效区域及多输出 |
| 行为 | 新建/部分更新/累加/别名、状态可见性与数值规则 |
| 依赖 | producer/consumer、完成条件、同步与生命周期 |
| PTO实现 | 所用指令与后端约束、组合或适配方式 |
| LPU选择 | 候选能力、替代组合、资源/搬运取舍及证据 |
| 验证 | 功能模型、程序路径、误差与回归结果 |

逻辑Tile与硬件tile register的映射、指令编码和buffer分配在候选规范中明确后再实现后端。

A/B合并后的完整清单见 [LPUv2指令候选集](lpuv2-instruction-candidates.md)，机器清单及16条OP覆盖关系见 [instruction-candidates.json](../data/instruction-candidates.json)。保留原A/B需求编号用于追溯。

## B路线已验证的候选需求

| 编号 / 状态 | 来源及需求 | 当前表达与证据 | 待确定内容 |
| --- | --- | --- | --- |
| REQ-B-01 / candidate | OP-01、K-01 Embedding：根据Token ID读取BF16表行，按原Token位置输出，列尾块只读写有效元素 | `pl.read`读取ID，`pl.load`用该ID作行偏移，`pl.store`写回；H=64/67两例经`pypto_torch_codegen`在CPU与Golden精确一致，见[步骤2记录](golden-stage3-step2-embedding.md) | LPU的索引读取、地址计算与Tile搬运如何分工；32列只是本例分块选择，未固定指令宽度或Buffer位置 |
| REQ-B-02 / candidate | OP-02、K-02普通RMSNorm：跨列Tile保留FP32整行平方和，统计完成后广播，乘`1+weight`后转BF16；尾块不扩大归约分母 | `row_sum/add`累加，`rsqrt/row_expand_mul`归一化，`cast`控制精度；D=256/2048/259经`pypto_torch_codegen`与Golden比较通过，见[步骤3记录](golden-stage3-step3-rmsnorm.md) | 归约状态及scratch如何映射、原输入重读或暂存的取舍；功能Tile宽128未固定硬件规格；生成器有效区适配单独记录 |
| REQ-B-03 / candidate | 步骤4公共数值能力：BF16/FP8 Linear、128分组量化、稳定Softmax、Sigmoid/SiLU；保留FP32归约与累加、scale关联及最终格式转换 | 六个PyPTO Tile函数经`pypto_torch_codegen_adapted`在CPU通过12项专项验证，量化payload字节与scale精确一致，见[步骤4记录](golden-stage3-step4-numerics.md) | LPU量化/矩阵乘法/特殊函数的指令映射和资源取舍；原生后端精度、布局与存储约束待验证；生成器适配与kernel能力分开登记 |
| REQ-B-04 / candidate | OP-14/K-14、OP-15/K-15、OP-16/K-16：按请求局部位置选行、按特征列偏移拼接、复用BF16投影，保留输出行及特征顺序 | 三个完整入口经`pypto_torch_codegen_adapted`通过6项CPU专项验证，含重复位置及尾块，见[步骤5记录](golden-stage3-step5-output-mtp.md) | 索引和视图的地址表达、拼接与投影的数据交接、选中多行投影的资源复用；不因新OP入口而直接新增指令 |
| REQ-B-05 / candidate | OP-05/K-05：Partial MRoPE的THW索引/半区配对、按head整理数据、GQA选择KV、按键位置屏蔽及KV追加；保留QK/概率/PV/gate的BF16边界 | 完整hidden到out/KV的Prefill→Decode、GQA8因果例子及RoPE定向检查经适配生成路径通过，见[步骤6记录](golden-stage3-step6-attention.md) | 旋转/三角函数和mask的指令组合，跨块score/概率存储，KV布局、容量与访问复用；软件多形状特化与硬件能力分开处理 |
| REQ-B-06 / candidate | OP-07/K-07：同分按专家ID升序的Top-8、保持Token/slot/权重关联的专家分桶、按Ne Gather及FFN、原Token合并和shared标量门控 | 完整PyPTO入口经适配路径通过随机、全同分、量化组尾部及跨32专家选择4项专项；基础比较循环组合Top-8，索引/Tile读写组合分桶搬运，见[步骤7记录](golden-stage3-step7-moe.md) | 高效排序/选择、计数和桶前缀和的指令组合；Gather及scale复用、并发Scatter原子性与数值边界；不将当前串行合并当作硬件atomic验收 |
| REQ-B-07 / candidate | 最小MTP组合：连接7类完整kernel，保留两处BF16残差边界、输出Norm后hidden及draft独立KV的跨调用交接 | Prefill→draft与已知投影路径经适配执行通过，全部具名输出与Golden比较；残差使用本地基础Tile加法，A的OP-04未集成，见[步骤8记录](golden-stage3-step8-mtp.md) | kernel间结果及KV的存储/生命周期映射、A残差入口接入；组合调用本身不直接增加硬件指令或B kernel编号 |

以上为Python功能证据，不表示已完成PTO CPU/NPU后端验证或已接受LPU指令。
