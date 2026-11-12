# A组reference：jcx

工作分支：`operator_dev_jcx`。本轮实现A组的9条OP计算定义，固定BF16激活与Vision权重、FP32 GDN核心与State；数值依据[模型基线](model-numerical-baseline.md)。

## 覆盖范围

| OP | Python入口 | 已验证内容 |
| --- | --- | --- |
| OP-03 LayerNorm | `pymodel.ref.common.layer_norm` | 最后一维仿射LayerNorm；eps=1e-6；BF16与1152实际宽度 |
| OP-04 Residual | `pymodel.ref.common.residual_add` | 同形同dtype相加；输入保留；通过公共run_case入口比较 |
| OP-08 Patch Embedding | `pymodel.ref.vision.vision_patch_embedding` | BF16/FP32 像素[B,C,T,H,W]，完整 Conv3D、BF16输入转换及 Merge Token顺序；见[像素入口](vision-pixel-conv3d.md) |
| OP-11 视觉位置嵌入 | `pymodel.ref.vision.vision_position_interpolation` → `vision_position_embedding` | 由Patch网格生成[N,4]索引和FP32系数，按2×2 merge顺序组织；位置结果先转BF16再加hidden |
| OP-09 Vision Attention | `pymodel.ref.vision.vision_attention` | 16 heads、72维完整axial RoPE、分段非因果MHA、FP32 Softmax回转BF16 |
| OP-10 Dense FFN | `pymodel.ref.vision.dense_ffn` | 含bias的Linear→GELU(tanh)→Linear；含1152→4304→1152尺寸 |
| OP-12 Spatial Merger | `pymodel.ref.vision.spatial_merger` | 先做每Patch的LayerNorm，再按已排序的2×2行分组；GELU exact |
| OP-13 Visual Scatter | `pymodel.ref.multimodal.visual_feature_scatter` | 跨batch占位行对应、特征转目标dtype、保留文本行、空视觉输入与非法索引 |
| OP-06 GDN | `pymodel.ref.gdn.gdn_reference`及各阶段函数，`pymodel.ref.fp8` | Conv历史、beta/g、L2Norm、recurrent/chunk、gated Norm；完整图接入v0 FP8投影参考 |

`golden_ready`表示对应BF16算子reference已通过当前PyTorch检查；完整模型权重、PTO后端、模型级结果分别登记。OP-06保持`in_progress`，已就绪部分与投影依赖分别记录。

## 运行与来源

```bash
python -m pytest tests/framework tests/operators -q
```

`tests/operators/conftest.py`校验固定Transformers源码快照的SHA，再提取所需函数或类，执行其Torch fallback。关闭加速器/offload装饰器，Attention固定eager。测试在单CPU线程下运行，并在结束后恢复线程设置。其他平台或后端精度通过对应case验证。

Vision Patch、位置插值元数据、axial RoPE、Attention、MLP、Merger以及GDN核心/计算图均与固定源码对照；LayerNorm按其调用的`nn.LayerNorm`对照。位置相加、Residual和Scatter用独立契约用例检查舍入、行映射及输入保留。另验证了一个Vision block的组合顺序，以及生成位置元数据后查表相加的组合路径。

同路径、同Torch环境的上游对照使用零容差。GDN recurrent/chunk的FP32跨算法一致性检查在65 Token、小型head的测试输入上使用`atol=2e-6, rtol=1e-5`；这项阈值仅用于该数学路径检查。FP8算子与目标硬件容差继续随实际执行校准。

## GDN接口与数值边界

```python
from pymodel.ref.gdn import gdn_recurrent

# Q/K: [B,S,H,K]，V: [B,S,H,V]，输入激活通常为BF16。
# 同一个V head对应的Q/K已由调用者repeat_interleave。
result = gdn_recurrent(q, k, v, beta, g, initial_state)
out = result["out"]               # 原query dtype，目标BF16
state_out = result["state_out"]   # [B,H,K,V]，FP32
```

核心接收原始Q/K，默认在内部转FP32做L2Norm，并对Q缩放一次。`normalize_qk=False`只关闭L2Norm；Q缩放仍在核心内执行。调用者不能再提前缩放Q。

`gdn_conv`接收qkv `[B,S,C]`，返回`out`与`conv_history_out`；逻辑历史为 `[B,K-1,C]`，按最旧到最新排列，保存SiLU之前的原始投影结果。模型中K=4、C=8192。输入history和FP32 State均不原地修改。

完整图`gdn_reference`通过`GDNProjections`显式接收qkv、z、a、b与out五个独立参考投影。`pymodel.ref.fp8`提供v0动态激活量化与分块Linear，qkv/z/out使用FP8权重及checkpoint scale，a/b使用BF16。原生GDN用例通过这些独立参考投影比较完整输出与状态；原有BF16 Linear fixture保留为计算图连接测试。

图输出为`out`、`state_out`、`conv_history_out`。Prefill/Decode、分段执行、非零初始State、实际32×128×128 State，以及64 Token chunk的63/64/65/129长度边界已覆盖。

## IF-04输入约定

OP-08 接收像素张量，返回按空间 Merge 组排列的 Token 和 `grid_thw[M,3]`，其中每行为一个图像/视频卷积输出的时间、高、宽方向 Patch 数量。不同尺寸媒体分别执行后，Token 和网格按同一媒体顺序拼接。`vision_position_interpolation`根据该网格生成位置插值所需的两个张量：

```python
import torch

from pymodel.ref.vision import vision_position_interpolation, vision_position_embedding

grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
interp_indices, interp_weights = vision_position_interpolation(grid_thw)
# interp_indices: [16,4]，int64；interp_weights: [16,4]，FP32
y = vision_position_embedding(x, position_table, interp_indices, interp_weights)
```

默认`table_side=48`、`spatial_merge_size=2`，使用bilinear/align_corners=True及border clamp。坐标计算和四邻居系数均为FP32，索引为int64；输出与`grid_thw`在同一个device。行顺序按空间2×2块内排列，视频各时间片重复对应空间网格；支持打包多个不同网格的输入。空输入返回两个`[0,4]`张量。

该计算作为OP-11的独立前置reference，计划放在LPU执行；当前完成PyTorch功能验证，后续接入实际Tile kernel。空间`position_ids[N,2]`、累计`cu_seqlens`及文本占位行继续由各自的IF-04输入提供。

测试覆盖4×4示例、48×48同尺寸映射、50×52输入、单轴退化（merge_size=1）、多图/视频、边界重复索引和Patch顺序。还通过独立`F.interpolate`结果验证“元数据生成→位置查表相加”的组合行为。公共验证case见 [vision-position-interpolation.json](../cases/operators/vision-position-interpolation.json)。

`vision_axial_cos_sin`与`vision_axial_rope`提供Vision专用频率与旋转。它们使用完整head_dim和half-split配对，文本Partial MRoPE由B组另行实现。
