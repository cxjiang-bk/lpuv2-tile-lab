# B路线阶段3：Tile程序功能验证实施步骤

目标：用官方PyPTO的Python API和PTO操作语义表达B负责的7个完整算子，在CPU上实际执行，与现有Golden比较，记录用到的操作及表达缺口，为阶段4整理指令候选集提供依据。

本阶段按下面9步推进。每步是一批可独立交付的工作；完成一批，再进入下一批。开发方式沿用前一阶段的精简安排：正常输入先跑通，再加一个与分块、索引或状态直接相关的代表用例。

## 执行进展

- 步骤1已完成：见[最小CPU执行路径](golden-stage3-step1-execution.md)。
- 步骤2已完成：见[Embedding Tile程序](golden-stage3-step2-embedding.md)。
- 步骤3已完成：见[RMSNorm Tile程序](golden-stage3-step3-rmsnorm.md)。
- 步骤4已完成：见[公共Tile数值函数](golden-stage3-step4-numerics.md)。
- 步骤5已完成：见[LM Head、Concat与MTP输入投影](golden-stage3-step5-output-mtp.md)。
- 步骤6已完成：见[完整Full Attention](golden-stage3-step6-attention.md)。
- 步骤7已完成：见[完整MoE](golden-stage3-step7-moe.md)。
- 步骤8已完成：见[最小MTP Tile组合](golden-stage3-step8-mtp.md)。
- 步骤9已完成：见[B路线实现与操作使用汇总](golden-stage3-step9-summary.md)。阶段3交付完成，下一步进入阶段4的候选能力归并。

## 计划起点（执行前）

- Embedding、RMSNorm、Full Attention、MoE、LM Head、Concat和MTP输入投影的ref已具备，另有最小MTP组合。当前144项测试属于Golden与框架检查。
- `kernels/pto/`、`pymodel/isa/`等目录尚无实际Tile程序，阶段3从执行路径开始。
- `data/upstream.lock.json`固定的是`pto-isa`提交`90a3796597a5fabc2ce5066e73f785c2e85822f7`，PyPTO版本尚未登记；PTO子模块尚未初始化。
- “官方main”表示从主线选取来源，本轮实施仍记录具体commit，保证两人使用相同接口。2026-11-11只读观察到PyPTO main为`7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b`，尚未安装或验证该提交。

## 代码各部分负责什么

| 位置 | 职责 |
| --- | --- |
| `pymodel/ref/` | 已有标准答案，阶段3直接调用作为预期结果 |
| `kernels/pto/` | 实际被测Tile程序及其公共计算函数 |
| `pymodel/isa/` | 仅在需要时适配实际使用的PyPTO操作或补充其Python语义；按用到的能力增加 |
| `pymodel/ops/` | 调用入口需要统一时接入，避免每个算子重复建立包装层 |
| `tests/kernels/` | 同一输入执行Golden和Tile程序，比较全部实际输出 |
| `tests/models/` | 已通过kernel的最小MTP等组合验证 |
| `pymodel/task_gen/validate.py` | 已有`run_case`，负责输入隔离、命名输出比较和产物保存 |
| `docs/isa-requirements.md` | 随开发记录操作用途、组合方法和缺口，阶段4再集中归纳 |

Tile程序自己完成计算，不能调用`pymodel/ref`中的Linear、量化、Norm或完整算子函数得到被测结果。Golden一侧可以继续复用这些函数。输入、参数、case及比较工具可以共享。

PyPTO的Python操作、PTO ISA条目和未来LPU指令是三层不同对象。本阶段记录它们之间的对应，不把一个Python函数自动视为一条LPU指令，也不把高层Tensor计算直接登记为Tile实现。

## 实施顺序总表

| 步骤 | 主要工作 | 交付 |
| --- | --- | --- |
| 1 | 固定PyPTO版本，跑通Python执行路径 | 一个能在CPU实际运行的最小Tile例子 |
| 2 | 完成Embedding Tile程序并接入Golden比较 | 第一个完整kernel及端到端测试 |
| 3 | 完成RMSNorm Tile程序 | 分块归约、广播和转换的可复用写法 |
| 4 | 完成Tile公共Linear、量化和概率计算 | 后续投影与路由可以直接复用的公共函数 |
| 5 | 完成LM Head、Concat、MTP输入投影 | 再完成3个完整kernel |
| 6 | 完成Full Attention | Prefill/Decode输出与KV对齐 |
| 7 | 完成MoE | 完整路由、分桶、专家和合并流程对齐 |
| 8 | 组合最小MTP Tile流程 | 各kernel连接正确，hidden与draft KV连续 |
| 9 | 汇总实现及操作使用记录 | 交给阶段4的7个kernel和能力清单 |

## 步骤1：固定版本，跑通Python执行路径

职责：解决“使用什么API编写、怎样在CPU执行”，让后续每个算子使用相同方式运行。

执行内容：

1. 读取所选PyPTO提交的安装说明、Tile API及Torch codegen示例，记录具体commit、Python/Torch版本。PyPTO和pto-isa分别登记，不把两个仓库的main混为同一版本。
2. 只准备前端Python库和CPU Torch执行所需依赖。PyPTO自身的Python扩展可能需要构建；本阶段不要求PTOAS、设备编译和NPU runtime。
3. 用显式Tile读取、逐元素加法、写回来做最小例子，查看程序或IR，确认能看见Tile操作。
4. 优先用官方`torch_codegen`从程序IR生成Torch代码并执行，记录为`pypto_torch_codegen`。先用直接IR codegen，不启动完整硬件优化流水。
5. 所选版本如不能执行某项Tile操作，先查官方支持范围；仅为实际缺少的操作补Python语义，记录为`python_tile_semantic`及具体缺口。两人采用同一适配，不回退成调用Golden。

最小验证：一个小Tile的加法，结果与直接Torch加法一致。只有import成功或生成IR不算完成，必须得到实际输出。

交付：一段最小程序、测试命令和几行版本/执行方式说明。建议新增`tests/kernels/test_tile_execution.py`；适配需要代码时放`pymodel/isa/`，不另建通用runtime。

此步由B先打通，A参与确认公共调用方式。后续继续沿用，不为每个算子重新搭环境。

## 步骤2：完成Embedding，打通第一个完整kernel

职责：把Token ID对应的Embedding行按Tile读取和输出，并接通已有比较框架。

执行内容：

1. 在`kernels/pto/common/embedding.py`建立完整K-01入口，输入输出与`pymodel/ref/embedding.py`一致。
2. 按Token行和hidden列分块，依据Token ID读取BF16表行，写入原请求/Token位置。API中用行偏移读取还是Gather，以步骤1核实的接口为准。
3. 用现有`Case`和`run_case`连接独立Golden与Tile入口。适配只放在测试侧；尺寸和输入直接用Python构造。
4. 在`tests/kernels/test_embedding_tile.py`比较主输出，记录实际PyPTO版本与执行路径。

最小验证：一个多Token正常例子，包含重复ID；再用一个hidden列或Token行不整除Tile尺寸的例子检查尾部。纯搬运采用精确比较。

交付：完整K-01和一个可复用的“Golden—Tile—比较”测试例子。候选操作从读取、行索引、存储开始记录，不涉及DDR/UB/LMB地址映射。

## 步骤3：完成RMSNorm，解决分块归约

职责：正确处理“输入分成多块，但统计范围仍是完整一行”的计算。

执行内容：

1. 在`kernels/pto/common/rmsnorm.py`建立K-02，展开读取、转FP32、平方、求和、均值、eps、rsqrt、广播、乘`1+weight`及BF16回转。
2. 一行跨多个列Tile时，先合并所有有效列的平方和，再计算最终均方。尾部填充不计入分母。
3. 统计量产生后重新读取原输入或使用明确的局部暂存，完成逐元素缩放。先采用容易理解的重新读取方式。
4. 用`tests/kernels/test_rmsnorm_tile.py`与现有普通RMSNorm ref比较。D=256和2048可作为两个代表尺寸；其中至少一例跨列Tile。

最小验证：主结果、BF16输出和跨Tile归约正确即可；发生差异时再导出平方和定位。

交付：完整K-02及归约/广播模式。GDN gated Norm仍按A的定义单独实现；本步骤不改变其权重含义和转换顺序。

## 步骤4：实现Tile公共数值函数

职责：提供后续LM Head、Attention、MoE共用的Tile计算主体。这些函数独立于Golden实现，不增加模型kernel编号。

建议集中在`kernels/pto/common/`，依次完成：

| 子任务 | 具体职责 | 最小验证 |
| --- | --- | --- |
| BF16 Linear | 按M/N/K分块，使用逻辑`weight[N,K]`，合并K方向部分和，按profile返回BF16 | 一个矩形矩阵；一个K或N尾块；对照`bf16_linear` |
| 动态激活量化 | 每Token、完整K轴每128元素分组；FP32 amax/scale、限幅、FP8转换；返回payload和scale | 一个跨128边界且两组幅度不同的例子，含零组；payload和scale分别精确比较 |
| FP8 Linear | 使用对应激活scale和权重128×128块scale，按K组合并FP32结果，最后回BF16 | 不同N/K块使用不同scale的小矩阵；对照`fp8_linear` |
| 稳定Softmax | 展开最大值、减法、exp、求和、除法；跨Tile统计范围完整，返回FP32 | 一个多列Tile概率例子，与公共Softmax ref比较 |
| 必要激活 | 按实际算子需要组合Sigmoid/SiLU；记录支持的基础操作及转换边界 | 随Attention/MoE的代表用例验证 |

步骤4已用PyPTO显式Tile程序实现上述能力，并通过项目内补充基础语义适配的Torch codegen执行。适配范围、公共接口和实际结果见[步骤4记录](golden-stage3-step4-numerics.md)，后端标签为`pypto_torch_codegen_adapted`。

关键点：

- Tile切分不能移动128元素量化组的起点。尾块保留有效列数和其对应scale。
- FP32累加是数值语义；底层如何表达和执行以选定API为准。接口缺失时明确记录补充能力。
- 比较器仍用项目v0。BF16结果有差异时先核对累加、缩放及舍入位置，不能通过放宽容差代替定位。

本轮比较集中在`tests/kernels/test_numeric_tile.py`。公共定义及最小调用例子交给A复用，双方直接引用同一份代码。

## 步骤5：完成LM Head、Concat与MTP输入投影

职责：用已有Tile公共函数补齐3个完整OP，避免再写一套投影主体。

| Kernel | 程序负责什么 | 主要检查 |
| --- | --- | --- |
| K-14 LM Head | 选择指定hidden行，按词表列分块投影，重组完整logits | 多位置含重复行；一个词表尾块；BF16输出 |
| K-15 Concat | 从两路已归一化输入读取Tile，按`[e_norm;h_norm]`写入完整特征 | 两路标记值不同，检查拼接顺序和行对应 |
| K-16输入投影 | 用独立BF16 fc参数，把完整拼接特征投影回H维 | 一个小矩阵，检查权重方向和输出 |

入口分别位于`kernels/pto/output/lm_head.py`和`kernels/pto/mtp/input.py`，比较集中在`tests/kernels/test_output_tile.py`。

LM Head内部不重复Final Norm；Concat内部不执行两路Norm；MTP投影不改为FP8。核对标准直接使用现有`pymodel/ref/output.py`。

交付后K-01、K-02、K-14、K-15、K-16均已有完整Tile入口，可以开始对应能力归纳。

## 步骤6：实现完整Full Attention

职责：从hidden出发，完整计算Attention并正确交接历史KV；复用已通过的Tile投影、Norm及Softmax函数。

执行内容：

1. Q/gate、K/V投影；按head拆Q/gate，Q/K按head做普通Norm。
2. 实现64维Partial MRoPE，保留192维尾部；因果位置和THW坐标分别处理。
3. 读取原KV并追加当前K/V，按GQA关系计算QK，使用因果mask，再做Softmax和PV。
4. context乘对应sigmoid(gate)，完成输出投影，返回主结果及新KV。

初版建议：沿query行和head分块；每块query先保留完整有效key范围，再计算Softmax。内存允许时用普通分块计算，便于与Golden对齐。key进一步分块或online Softmax作为后续优化，不将其性能优化列为首版必要条件。

最小验证：短Prefill接一个Decode，对比`out`、K/V、有效长度和键位置；用一个非整除query块检查有效区。先沿用当前Golden支持的等长有效输入。

实现与测试：`kernels/pto/attention/full_attention.py`、`tests/kernels/test_attention_tile.py`。Prefill/Decode共用K-05；仅有QK或预先提供Q/K/V的测试属于内部步骤，不表示完整入口完成。

## 步骤7：实现完整MoE

职责：维持Token、专家和路由权重的对应，从hidden算到最终合并结果。

执行内容：

1. BF16 Router投影→FP32 Softmax→Top-8→选中概率重归一化→BF16路由权重。
2. 用选定API的排序/选择能力实现Top-K。采用TSORT32/TMRGSORT等候选操作前，核对真实Python接口、值/索引布局及同分规则；保证同分按专家ID升序。
3. 按专家统计桶边界，保留原Token和slot，Gather专家输入。若Gather的是预量化输入，payload和scale同步重排。
4. 遍历有效专家桶，用Tile投影和激活完成Gate/Up、SiLU×Up、Down，逐行乘路由权重。
5. 按原Token合并全部贡献，再加入独立shared FFN及其标量门控贡献，保留当前BF16合并边界。

初版允许Python循环遍历专家和组织调用；涉及张量数据的排序、分桶、Gather与Scatter等操作，需用可追溯的Tile操作/补充语义表达。不能用一次整Tensor的Torch MoE代替被测主体，也不能把未映射的Python数据处理静默算作已有PTO支持。

最小验证：少量Token、至少8个专家，包含共享命中和一个空桶；另用一次同分输入检查Top-8顺序。比较out、expert_id、route_weight及基本分桶对应，不系统枚举所有路由组合。

实现与测试：`kernels/pto/moe/moe.py`、`tests/kernels/test_moe_tile.py`。当前Top-8由基础比较循环组合，分桶/Gather/合并由PyPTO索引与Tile读写表达，见[步骤7记录](golden-stage3-step7-moe.md)。Scatter合并按专家顺序执行；硬件atomic及UB冲突在架构映射时讨论。

## 步骤8：组合最小MTP Tile流程

职责：检查已通过kernel能否按真实模型关系连接；这里只组织调用和状态，避免复制kernel主体。

执行内容：

1. 复用Token移位规则，准备已对齐的Embedding输入和上一阶段hidden。
2. 用Tile kernel执行两路Norm、Concat、BF16输入投影、单层Attention/MoE及残差、输出Norm、LM Head。
3. target与draft参数和KV分开传递。返回输出Norm后的hidden，供下一次draft调用。
4. 与`pymodel/ref/mtp.py::mtp_step`比较一次prefill及接续draft，检查logits、hidden和draft KV。

两路Norm复用K-02，Concat/投影/Attention/MoE/Head复用各自入口，Residual调用A的K-04。A的Tile入口未到位时可显式提供等价的基础Tile加法，但标记其集成状态，不冒充A的实现验收。

建议文件：`pymodel/models/qwen36/mtp_tile.py`、`tests/models/test_mtp_tile.py`。MTP属于子图组合，不新增第8个B路线kernel。候选树、接受/拒绝及target GDN/Conv恢复继续由模型集成阶段负责。

## 步骤9：整理结果，交给阶段4

职责：把实际程序证明了什么、需要哪些操作整理清楚，开始收敛指令候选集。

每个完整kernel只需记录：

| 内容 | 说明 |
| --- | --- |
| 入口及场景 | K/OP编号、函数、实际支持的路径 |
| Tile组织 | 主要分块轴、有效区域和需要保留的中间量 |
| 操作对应 | PyPTO操作、对应PTO ISA语义、实际用途 |
| 执行与结果 | 实际commit/路径、测试命令、具名输出比较结果 |
| 表达缺口 | 用基础操作组合解决，或增加了什么补充Python语义 |

记录到现有OP/K及需求说明中；Python功能结果与实际PTO CPU/NPU字段分别维护。阶段4再合并重复能力、判断必要子集和组合取舍。某个kernel通过即可先整理，不等全部完成才开始记录。

## 推进和检查的统一规则

- 一个OP一个完整kernel入口，内部可循环、复用函数；只有实际使用到的公共能力才实现。
- 新kernel先用正常小例子跑通，再增加一个最相关的尾块、位置或状态用例；避免广泛枚举报错分支。
- 基线不变：BF16计算按`atol=1e-2,rtol=1e-2`，FP32按`atol=1e-5,rtol=1e-4`；搬运/索引/FP8 payload及激活scale精确比较。
- Tensor和Tile的逻辑形状、有效区域、dtype、位置及状态在阶段3明确；LPUv2 Buffer容量、Bank、物理地址、乱序调度和流水优化在架构映射阶段明确。
- 被测程序要能显示实际Tile分块和基础操作。PyPTO生成Torch也是被测路径，Golden继续独立。
- 测试使用现有`Case/run_case`，不另建框架；缺实现时报错，失败时按需保留输入和报告。README仅链接实施说明。
- 代表用例通过后继续下一步；出现实际差异再增加针对性的检查。无需先完成完整模型、设备后端或性能分析。

步骤1至步骤9现已完成，B路线的实现、操作归并和验证证据见[步骤9汇总](golden-stage3-step9-summary.md)。后续工作进入阶段4：从实际操作归并中筛选必要的Tile指令候选集。

## 参考

- [原B路线阶段3](https://tju-opentpu.feishu.cn/docx/J8uQdwJhRobt7BxL6USc7kv1nxh#doxcn8odwwWMqr2sp39HxH5NEjm)。
- [已完成的Golden与MTP调用](golden-stage2-steps5-8.md)、[公共数值函数](golden-stage2-step4-numerics.md)、[验证框架](validation.md)。
- [PyPTO Torch codegen说明](https://github.com/hw-native-sys/pypto/blob/7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b/docs/en/user/tools/01-torch-codegen.md)、[操作命名空间](https://github.com/hw-native-sys/pypto/blob/7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b/docs/en/user/ops/index.md)。链接为本次核对的固定版本，实际安装和覆盖在步骤1登记。
