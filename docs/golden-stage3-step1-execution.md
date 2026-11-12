# 阶段3步骤1：PyPTO的最小CPU执行路径

本步只建立Tile程序的Python执行方式。示例是2×16的BF16加法，入口为[kernels/pto/common/tile_add.py](../kernels/pto/common/tile_add.py)，测试为[tests/kernels/test_tile_execution.py](../tests/kernels/test_tile_execution.py)。

## 怎样运行

在仓库根目录使用已准备的项目环境：

```bash
.venv/bin/python -m pytest tests/kernels/test_tile_execution.py -q
```

测试用PyPTO实际生成的Torch代码计算结果，与直接`a+b`精确比较。此示例用于检查执行路径，不表示模型算子K-01或K-04已经实现。

## 这条路径做了什么

1. `@pl.jit.incore`描述计算：两次Tile读取、一次加法、一次写回。
2. 测试调用`tile_add.specialize(a,b,out)`，取得未执行硬件变换的Program IR。
3. 官方`pypto.debug.torch_codegen(program, check_shapes=True)`将该IR转换为可运行的Torch代码。
4. 执行生成的`tile_add`函数，检查BF16结果和CPU设备。生成代码和IR可从测试临时目录查看。

这里只使用JIT的前端特化，不直接调用其设备执行入口；没有调用PTOAS或NPU dispatch，也没有用Golden替代被测计算。Tile load/add/store对应PTO的TLOAD/TADD/TSTORE计算语义，实际后端标签为`pypto_torch_codegen`。

## 固定版本与环境

| 对象 | 本轮使用 |
| --- | --- |
| PyPTO | 官方main提交`7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b`，独立登记于`data/upstream.lock.json` |
| 本地源码 | `artifacts/upstream/pypto/`，保持上游内容 |
| Python / Torch | Python 3.13.9；Torch 2.12.0+cu130，实际执行CPU |
| Python环境 | `.venv`，通过`--system-site-packages`复用已有Torch和pytest |
| 构建依赖 | nanobind 2.15.0、scikit-build-core 1.0.3、Ninja 1.13.0 |
| 本机编译工具 | CMake 3.28.3、GCC 13.3.0，Release构建；使用上游资源配置的2个编译任务 |

PyPTO与原有pto-isa分开固定。此次前端从源码构建，需要上游固定的libbacktrace、msgpack-c及simpler源码中的通信头文件；没有构建或运行simpler设备runtime。

### 重新准备环境

需要在新机器重建时，先建立环境并取得固定源码：

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install nanobind==2.15.0 scikit-build-core==1.0.3 ninja==1.13.0
git clone --filter=blob:none --no-checkout https://github.com/hw-native-sys/pypto.git artifacts/upstream/pypto
git -C artifacts/upstream/pypto checkout --detach 7302cd7ffeb552b3bec7c0e5bd92f164ba9fbd4b
git -C artifacts/upstream/pypto submodule update --init --depth 1 -- 3rdparty/libbacktrace 3rdparty/msgpack-c runtime
.venv/bin/python -m pip install --no-deps --no-build-isolation --config-settings=cmake.build-type=Release --config-settings=build.tool-args=-j2 ./artifacts/upstream/pypto
```

构建需要满足上游要求的CMake与C++17编译器。本轮PyPTO CI固定的CMake 4.4.2包未能取得，因此使用满足上游最低要求的本机3.28.3，实际结果随本轮记录。网络问题仅影响源码获取方式，不修改程序语义或上游源码。

运行产物、构建日志和版本检查保存在`artifacts/golden-stage3-step1/`，不进入Git。下一步可基于这条执行路径实现Embedding Tile程序。

## 本轮结果

Tile专项测试1项通过，BF16结果与直接Torch加法精确一致；连同原有144项测试，全套145项通过。项目代码的Ruff和差异检查通过。

可查看本轮保存的[Program IR](../artifacts/golden-stage3-step1/pytest/test_tile_add_executes_generat0/tile_add_ir.py)、[生成的Torch代码](../artifacts/golden-stage3-step1/pytest/test_tile_add_executes_generat0/tile_add_torch.py)及[测试XML](../artifacts/golden-stage3-step1/all-test-results.xml)。生成代码含上游通用辅助函数，加法入口`tile_add`位于文件末尾。
