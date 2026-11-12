# LPUv2 Tile Lab

面向 Qwen3.6-35B-A3B 的算子协作工程：建立独立 PyTorch golden，先在 Python/PyPTO 层实现 Tile kernel 并验证功能，整理 LPUv2 指令候选集，再结合架构完善 Tile ISA 与 programming model。

**16条 OP 的独立 golden 与 Tile kernel 已汇合。** A 的9个 OP 使用原生 PTO C++ 功能仿真，见 [验证汇总](data/native-validation-summary.json)和[运行说明](docs/pypto-a1.md)；B 的7个 OP 及最小MTP组合使用 PyPTO CPU Torch生成，见[实现与操作汇总](docs/golden-stage3-step9-summary.md)。两条执行路径的版本和通过范围分别登记。

## 从这里开始

1. 阅读 [目录分层](docs/architecture.md)，确定自己修改的层。
2. 在 [两人分工与公共协作约束](docs/work-allocation.md) 中确认OP/kernel归属、公共依赖及交付顺序。
3. 按 [开发流程](CONTRIBUTING.md) 先确认计算与接口，再提交golden和Tile实现。
4. 查阅 [PTO参考入口](docs/pto-isa/README.md) 和 [机器可读指令目录](data/pto_isa_catalog.json)。
5. 按 [验证框架使用方法](docs/validation.md) 接入reference、实际后端和命名输出。
6. A组的接口、覆盖范围和依赖见 [reference说明](docs/reference-a.md)。
7. A 组 kernel 的安装、调用与结果见 [实现说明](docs/pypto-a1.md)，功能需求见 [A组候选来源](docs/lpuv2-a-instruction-candidates.md)。完整16条OP的去重清单见 [LPUv2 指令候选集](docs/lpuv2-instruction-candidates.md)。

当前采用 [一个OP对应一个逻辑kernel](docs/kernel-organization.md)，编号为`OP-01↔K-01`至`OP-16↔K-16`。Conv、TopK、量化和状态更新等在相应kernel内部组织；Prefill/Decode与recurrent/chunk作为实现路径记录。

```bash
git clone --recurse-submodules https://github.com/cxjiang-bk/lpuv2-tile-lab.git
cd lpuv2-tile-lab
# 已经普通clone时：
git submodule update --init --recursive
```

本地框架与reference检查：

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/framework tests/operators tests/reference -q
```

框架检查覆盖BF16/FP32、索引、状态隔离和错误报告。算子检查对照固定来源验证A组reference，包含实际Vision维度和GDN State维度。OP-04的[示例case](cases/examples/op04-residual-bf16.json)与[GDN核心case](cases/operators/gdn-recurrent-bf16.json)已接入reference检查；A 的原生 PTO C++ 后端已经接入，B 的原生后端按后续任务推进。

执行原生 Tile kernel 验证：

```bash
bash scripts/setup-hw-pypto.sh
source scripts/hw-env.sh
python -m pymodel.task_gen.validate_a --case 09
python -m pymodel.task_gen.validate_a
```

kernel 采用 `hw-native-sys/pypto` 的 Tensor/Tile 接口，经 PTOAS 生成 C++，由 GCC 编译并在 simpler `a2a3sim` 中执行，后端标记 `hw_pto_cpp_sim`。GDN 保留 K128 量化与 BF16 精确承载 FP8 数值。未加载工具链环境时 kernel pytest 明确跳过；116项框架/reference检查可独立运行。

PTO固定在 `90a3796597a5fabc2ce5066e73f785c2e85822f7`，来源为公开 [hw-native-sys/pto-isa](https://github.com/hw-native-sys/pto-isa)。当前PTO目录覆盖139个core参考条目（136个core公开API名）及12个通信接口。语义变体、兼容名和命名空间重名在目录中分别记录。

## 模型与数值基线

模型固定为 **Qwen/Qwen3.6-35B-A3B-FP8**。共同基线放在 [configs/models/qwen3_6_35b_a3b_fp8/](configs/models/qwen3_6_35b_a3b_fp8/)，包含官方配置快照、模型/源码revision、checkpoint dtype元数据和数值规则。两人实现reference前阅读 [模型与数值基线](docs/model-numerical-baseline.md)。逐算子reference进度见登记表；完整模型和目标FP8端到端结果在相应任务中验证。

当前采用已确认的 [PTO功能验证标准v0](docs/functional-validation-v0.md)：动态激活量化、BF16/FP32误差阈值、TopK同分与MoE分桶。共同配置为 [functional-v0.json](configs/models/qwen3_6_35b_a3b_fp8/functional-v0.json)。

## 目录

```text
pymodel/             Python侧职责分层，参考pymodel的组织方式
  common/            shape、dtype、layout、状态等公共契约
  isa/               PTO调用适配；后续接入候选LPU ISA
  semantic/          后续候选LPU指令的独立语义定义
  ops/               算子入口与kernel选择
  ref/               独立PyTorch golden
  models/qwen36/     子图、Decoder layer和模型集成
  task_gen/          可复现case及实验产物组织
  utils/             输出、索引、状态比较等公共工具
kernels/pypto/       HW PyPTO Tensor/Tile程序；由 PTOAS/GCC 编译、C++ 功能仿真
kernels/pto/         B组 HW PyPTO Tile程序；CPU Torch生成功能验证
tests/               指令、kernel、算子、子图和模型验证
cases/               测试场景说明与小型case定义
configs/             模型基线和后续运行配置
data/                OP、kernel、接口及PTO指令登记
docs/                架构、协作、验证和ISA需求记录
third_party/pto-isa/  固定版本上游子模块
artifacts/           本地运行产物，默认不入库
```

## 后续闭环

模型需求 → PyTorch golden → PyPTO kernel 功能验证 → LPUv2指令候选集 → 架构映射 → LPU程序与编程模型验证。公共比较和运行记录已可使用；实际PTO CPU/NPU后端及对应构建、CI随实现接入，执行环境和精度按具体kernel记录。
