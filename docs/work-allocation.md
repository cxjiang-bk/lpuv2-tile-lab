# 两人分工

A主责GDN与视觉分支，B主责Full Attention、MoE、输入输出及MTP。当前按一个OP对应一个逻辑kernel登记：A为9条OP/9个kernel，B为7条OP/7个kernel。工作量按复合算子的计算、状态与验证复杂度评估，编号数不直接代表工作量。

A由jcx认领，使用 `operator_dev_jcx`；B由Starset-Zheng认领，使用 `operator_dev_zhengtk`。下表记录合入后的功能验证路径。

A的接口见[reference说明](reference-a.md)，原生仿真范围见[实现说明](pypto-a1.md)；B的范围见[实现汇总](golden-stage3-step9-summary.md)。完整模型验证独立登记。

| 主责 | OP | 算子 | 当前状态 |
| --- | --- | --- | --- |
| B | OP-01 | Token Embedding | Python Tile功能验证通过 |
| A | OP-08 | Vision Patch Embedding | 原生PTO C++功能仿真通过 |
| A | OP-11 | 视觉位置嵌入 | 原生PTO C++功能仿真通过 |
| A | OP-03 | LayerNorm | 原生PTO C++功能仿真通过 |
| A | OP-09 | Vision Attention | 原生PTO C++功能仿真通过 |
| A | OP-04 | Residual Add | 原生PTO C++功能仿真通过 |
| A | OP-10 | Dense FFN（GELU MLP） | 原生PTO C++功能仿真通过 |
| A | OP-12 | Spatial Merger / Projector | 原生PTO C++功能仿真通过 |
| A | OP-13 | Visual Feature Scatter | 原生PTO C++功能仿真通过 |
| B | OP-02 | RMSNorm | Python Tile功能验证通过 |
| A | OP-06 | Gated DeltaNet 线性注意力 | 原生PTO C++功能仿真通过 |
| B | OP-05 | Full Attention | Python Tile功能验证通过 |
| B | OP-07 | MoE FFN | Python Tile功能验证通过 |
| B | OP-14 | LM Head | Python Tile功能验证通过 |
| B | OP-15 | Feature Concat | Python Tile功能验证通过 |
| B | OP-16 | MTP 输入投影 | Python Tile功能验证通过 |

## 公共接口与集成

A负责GDN/Conv状态、Vision/位置元数据及视觉接入；B负责KV、路由与MTP接口并牵头文本主干集成。MTP接受前缀与拒绝恢复由B组织，A提供GDN/Conv状态配合。

RMSNorm由B、LayerNorm/Residual由A维护；GEMM与量化接口由B牵头，参考版本及比较入口由A牵头。公共接口变更说明受影响的调用点，另一人交叉评审。

完整逐kernel归属见 [kernels.json](../data/kernels.json)，编号与内部步骤见 [kernel组织](kernel-organization.md)，状态接口见 [interfaces.json](../data/interfaces.json)。首轮完整算子原型完成后按实际用时调整任务安排。GDN的Conv/门控和MoE的路由/分桶等作为内部步骤，公共GEMM/量化以函数复用。

## 公共能力的维护与交付

主责负责公共定义、独立reference和实现；使用方提供实际调用用例并交叉评审。公共变更按能力拆成小PR，先进入主线，再由依赖它的算子接入。

| 公共能力 | 主责 | 使用方与交付内容 |
| --- | --- | --- |
| BF16 GEMM、FP8 Linear与激活量化 | B | A的GDN和B的投影路径共用；交付输入、权重、scale、输出精度的调用说明、独立reference与最小数值用例 |
| RMSNorm | B | Decoder及相关调用点；说明轴、eps、权重解释与转换位置 |
| LayerNorm、Residual Add | A | Vision及需要复用的算子；交付reference、输入更新约束与独立用例 |
| Case、命名输出比较与运行记录 | A | 两人共用现有框架；B提供Attention、MoE与索引场景，框架改动保持已有调用可用 |
| 模型与数值profile | 两人共同维护 | 模型来源以固定基线为准；量化、容差和同分按已确认的[功能验证v0](functional-validation-v0.md)实现并补充case |
| GDN State与Conv历史 | A | B的模型集成和后续状态处理使用；说明状态含义、初始化、顺序与更新结果 |
| Vision与位置元数据 | A | B的文本接入使用；说明Patch、特征、位置、片段和占位行的对应关系 |
| KV、MoE路由与MTP | B | A参与交叉检查；MTP涉及GDN/Conv状态时，由A提供相应连续执行及恢复用例 |

## 使用公共程序前写清什么

每项公共能力随代码维护一份简短说明，包含以下内容；一个双方可运行的调用例子作为交付入口。

- 输入和所有输出的名称、逻辑shape、dtype及轴含义；数值转换、归约和量化规则引用共同profile。
- 数据行顺序、有效区域和offset；索引指向哪个逻辑对象，重排后数据、scale和权重怎样保持对应。
- 是否修改输入、输出是否与输入共享存储，以及更新结果怎样返回。用同一输入分别运行golden与被测程序，检查双方使用的初始状态一致。
- 跨Token或chunk的状态含义、缺省初值、历史顺序、更新时机与连续执行方式；主输出和状态分别检查。
- 对应OP/K、维护入口、最小调用及测试命令、实际验证范围。尚未确定的函数签名或字段标为待统一，随公共实现落地。

公共说明优先写在实现旁或对应开发文档，状态/索引定义关联 `data/interfaces.json`，数值规则关联模型配置与case。本文件维护归属和交付约束，避免复制整套算子数学定义。

## 交付顺序与并行方式

1. 两人复用 `main` 已有的Case、`run_case`、比较器和模型基线，按已确认的`pto-functional-v0`实现和比较，记录结果命名及实际执行方式。
2. A先交付公共LayerNorm/Residual reference及测试；B先交付投影和量化的共同调用方式及最小实现。双方的其余算子在各自分支继续推进。
3. GDN核心先直接使用Q/K/V、门控与初始状态验证；目标FP8投影就绪后再验收完整GDN。Vision按既定BF16路径推进。
4. 状态和视觉元数据先交付能够独立运行的部分，分别登记覆盖范围；完整模型、MTP接入时再验证端到端交接。

调用缺失的公共实现时明确报错；功能用例可直接提供阶段输入，但需写明该用例实际覆盖的范围。

## 当前阶段怎样验证和登记

当前使用hw-native-sys PyPTO language/Tile API，构造基础计算与 Tile 数据流，并在 PTO C++ 模拟器验证功能；具体发行版本记录在 `data/upstream.lock.json`。两人记录每次运行实际使用的commit、Python/Torch版本、随机种子、数值profile及执行方式。单算子先使用随机输入和权重，按小尺寸、模型相关真实维度、定向边界推进；真实权重在集成阶段接入。

独立golden表达完整模型计算，Tile程序展开基础操作、分块、有效区域和状态顺序。两者使用同一输入，分别执行自己的计算主体；所需补充能力附独立语义测试和kernel组合用例。

| 执行结果 | 记录方式 |
| --- | --- |
| Reference与固定模型来源核对 | 标明 `pytorch_reference_check` 及覆盖范围 |
| HW PyPTO 原生 C++ 功能仿真 | 使用 `hw_pto_cpp_sim`，保存 PTO/PTOAS 生成的 C++、共享库、case 和逐输出结果 |
| 实际PTO CPU/NPU | 记录实际后端、构建/运行信息和结果，更新对应PTO验证状态 |
| LPU功能模型 | 按架构映射后的程序和约束独立验证 |

`backend`字段用于记录实际路径，完成依据还包括程序、版本和结果。现有登记表的PTO验证字段保留其后端含义；Python功能结果先在PR和运行记录中明确登记，新增结构化状态时同步修改登记说明。具体比较与产物格式见 [验证约定](validation.md)。

A 组使用真实 HW PyPTO/PTOAS/C++ 仿真 执行，9 个完整算子已有小尺寸原生通过结果。GDN 的量化、FP8解码和分块乘法在 PTO C++ 模拟器内执行，以BF16精确承载FP8数值；A5直接FP8路径另行验证。安装、公共入口、用例与候选需求见 [A 组说明](pypto-a1.md)；字段说明见 [登记表说明](../data/README.md)。这些变更位于 A 开发分支，公共 FP8 Linear 与量化部分在合入前由使用方交叉评审。

MoE按已确认的专家分桶策略组织路由记录：保留原Token、路由权重及FP8数据与scale的关联，按原Token合并结果。具体规则见 [功能标准第5节](functional-validation-v0.md#5-moe分桶与合并)；字段布局随B的Dispatch/Combine实现共同细化。

## 公共变更怎样同步

PR列出公共定义的变化、关联OP/K及受影响调用点，附至少一个实际使用方的调用例子和检查结果。合入后使用方更新依赖，并重跑受影响的算子与状态用例。改变数学、转换或索引规则时同步更新golden、profile和case；改变归属时同步更新本表与对应登记项。

整体阶段目标和验收方式由 [流程文档](https://tju-opentpu.feishu.cn/wiki/IxrNwnbariyrlYkBBJnc8CQHnGJ) 管理，本文件维护两人协作的具体约束。
