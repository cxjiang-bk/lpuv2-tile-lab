"""A-owned Vision operator entries backed only by native PTO C++ simulation."""

from pathlib import Path

import torch

from kernels.pypto import vision_ops as tile_kernels
from pymodel.isa.hw_pypto import TensorProgram


def patch_embedding(pixels, weight, bias, *, stride=None, spatial_merge_size=2,
                    artifact_dir: str | Path | None = None):
    """Raw pixel Conv3D and merge-order tokens; no host pixel preprocessing."""
    if pixels.ndim != 5 or weight.ndim != 5 or pixels.shape[1] != weight.shape[1]:
        raise ValueError("Expected pixels[B,C,T,H,W] and weight[D,C,Kt,Kh,Kw]")
    if pixels.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Pixels must be BF16 or FP32")
    hidden = weight.shape[0]
    if bias.shape != (hidden,) or weight.dtype != torch.bfloat16 or bias.dtype != weight.dtype:
        raise ValueError("Conv3D requires BF16 weight and bias[D]")
    if any(t.device.type != "cpu" for t in (pixels, weight, bias)):
        raise ValueError("HW PTO C++ simulation takes CPU tensors")
    kernel_size = tuple(weight.shape[2:])
    stride = kernel_size if stride is None else tuple(stride)
    if len(stride) != 3 or any(type(s) is not int or s < 1 for s in stride):
        raise ValueError("stride must contain three positive integers")
    if (any(d < 1 for d in (*pixels.shape, *weight.shape))
            or any(d < k for d, k in zip(pixels.shape[2:], kernel_size))):
        raise ValueError("Pixels must contain at least one complete Conv3D window")
    if type(spatial_merge_size) is not int or spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be a positive integer")
    ot, oh, ow = ((d - k) // s + 1 for d, k, s in zip(pixels.shape[2:], kernel_size, stride))
    if oh % spatial_merge_size or ow % spatial_merge_size:
        raise ValueError("Conv3D output height and width must form complete spatial merge groups")
    rows = pixels.shape[0] * ot * oh * ow
    if max(rows * weight[0].numel(), pixels.numel()) >= 2**24:
        raise ValueError("This composed im2col path requires exact FP32 coordinates below 2**24")
    program = TensorProgram.compile(tile_kernels.patch_embedding(tuple(pixels.shape), hidden, kernel_size,
                                    stride=stride, spatial_merge_size=spatial_merge_size,
                                    input_dtype=str(pixels.dtype).removeprefix("torch.")))
    program.details.update(pixel_input_layout="B,C,T,H,W", pixel_input_dtype=str(pixels.dtype),
                           conv3d_kernel=list(kernel_size), conv3d_stride=list(stride),
                           conv3d_padding=0, conv3d_dilation=1, conv3d_groups=1,
                           window_extraction="im2col3d_composed_gather",
                           dedicated_im2col_instruction=False,
                           convolution_lowering="im2col -> pl.matmul -> FP32 bias -> BF16", host_patchify=False,
                           token_order="B,Ot,block_h,block_w,intra_h,intra_w",
                           spatial_merge_size=spatial_merge_size)
    out = torch.full((rows, hidden), float("nan"), dtype=torch.bfloat16)
    grid_thw = torch.full((pixels.shape[0], 3), -1, dtype=torch.int64)
    # 展平仅改变连续权重的 shape；像素窗口仍由设备端 im2col 读取。
    program.execute(pixels.contiguous(), weight.reshape(hidden, -1).contiguous(),
                    bias.reshape(1, hidden).contiguous(), out, grid_thw)
    if artifact_dir is not None:
        program.save(artifact_dir)
    return {"out": out, "grid_thw": grid_thw}


def dense_ffn(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias, *, artifact_dir: str | Path | None = None):
    hidden = x.shape[-1]
    intermediate = fc1_weight.shape[0]
    if (fc1_weight.shape != (intermediate, hidden) or fc1_bias.shape != (intermediate,)
            or fc2_weight.shape != (hidden, intermediate) or fc2_bias.shape != (hidden,)):
        raise ValueError("Dense FFN requires D->I->D weight/bias shapes")
    if any(t.dtype != torch.bfloat16 or t.device.type != "cpu"
           for t in (x, fc1_weight, fc1_bias, fc2_weight, fc2_bias)):
        raise ValueError("Vision FFN requires CPU BF16 tensors")
    rows = x.numel() // hidden
    program = TensorProgram.compile(tile_kernels.dense_ffn(rows, hidden, intermediate))
    out = torch.full(x.shape, float("nan"), dtype=torch.bfloat16)
    program.execute(x.reshape(rows, hidden).contiguous(), fc1_weight.contiguous(), fc1_bias.reshape(1, intermediate).contiguous(),
                    fc2_weight.contiguous(), fc2_bias.reshape(1, hidden).contiguous(), out.reshape(rows, hidden))
    if artifact_dir is not None:
        program.save(artifact_dir)
    return {"out": out}


def spatial_merger(x, norm_weight, norm_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, *,
                   spatial_merge_size=2, eps=1e-6, artifact_dir: str | Path | None = None):
    if x.ndim != 2 or spatial_merge_size < 1 or x.shape[0] % (spatial_merge_size ** 2):
        raise ValueError("Merger input must contain complete spatial merge groups")
    rows, hidden = x.shape
    intermediate, width = fc1_weight.shape[0], fc2_weight.shape[0]
    if (norm_weight.shape != (hidden,) or norm_bias.shape != (hidden,)
            or fc1_weight.shape[1] != spatial_merge_size ** 2 * hidden or fc1_bias.shape != (intermediate,)
            or fc2_weight.shape[1] != intermediate or fc2_bias.shape != (width,)):
        raise ValueError("Merger parameter shapes disagree")
    if any(t.dtype != torch.bfloat16 or t.device.type != "cpu"
           for t in (x, norm_weight, norm_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias)):
        raise ValueError("Vision Merger requires CPU BF16 tensors")
    program = TensorProgram.compile(tile_kernels.spatial_merger(rows, hidden, intermediate, width,
                                                          merge_size=spatial_merge_size, eps=eps))
    out = torch.full((rows // spatial_merge_size ** 2, width), float("nan"), dtype=torch.bfloat16)
    program.execute(x, norm_weight.reshape(1, hidden), norm_bias.reshape(1, hidden),
                    fc1_weight, fc1_bias.reshape(1, intermediate), fc2_weight, fc2_bias.reshape(1, width), out)
    if artifact_dir is not None:
        program.save(artifact_dir)
    return {"out": out}


def vision_attention(x, qkv_weight, qkv_bias, proj_weight, proj_bias, *, cos, sin, cu_seqlens,
                     num_heads=16, artifact_dir: str | Path | None = None):
    if x.ndim != 2 or num_heads < 1 or x.shape[1] % num_heads:
        raise ValueError("Vision hidden width must be divisible by heads")
    rows, hidden = x.shape
    head_dim = hidden // num_heads
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64) or cu_seqlens.device.type != "cpu":
        raise ValueError("Packed segment boundaries must be a CPU INT32/INT64 vector")
    if rows > torch.iinfo(torch.int32).max:
        raise ValueError("Packed segment offsets must fit the native INT32 metadata")
    boundaries = tuple(int(v) for v in cu_seqlens.tolist())
    if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != rows
            or any(a >= b for a, b in zip(boundaries, boundaries[1:])) or head_dim % 2):
        raise ValueError("Positive packed segments and even head_dim are required")
    if (qkv_weight.shape != (3 * hidden, hidden) or qkv_bias.shape != (3 * hidden,)
            or proj_weight.shape != (hidden, hidden) or proj_bias.shape != (hidden,)
            or cos.shape != (rows, head_dim) or sin.shape != cos.shape):
        raise ValueError("Vision projection/position shapes disagree")
    if any(t.dtype != torch.bfloat16 or t.device.type != "cpu"
           for t in (x, qkv_weight, qkv_bias, proj_weight, proj_bias)):
        raise ValueError("Vision projections require CPU BF16 tensors")
    if any(t.dtype != torch.float32 or t.device.type != "cpu" for t in (cos, sin)):
        raise ValueError("Vision cos/sin must retain CPU FP32")
    program = TensorProgram.compile(tile_kernels.vision_attention(rows, hidden, num_heads, boundaries))
    out = torch.full_like(x, float("nan"))
    program.execute(x, qkv_weight, qkv_bias.reshape(1, 3 * hidden), proj_weight,
                    proj_bias.reshape(1, hidden), cos, sin, cu_seqlens.to(torch.int32).reshape(1, -1).contiguous(), out)
    if artifact_dir is not None:
        program.save(artifact_dir)
    return {"out": out}


def visual_feature_scatter(text_embedding, visual_features, placeholder_rows, *,
                           artifact_dir: str | Path | None = None):
    if text_embedding.ndim != 3 or visual_features.ndim != 2:
        raise ValueError("Expected text [B,S,D], features [N,D]")
    hidden = text_embedding.shape[-1]
    if visual_features.shape[1] != hidden or placeholder_rows.shape != (len(visual_features),):
        raise ValueError("Feature width/placeholder count mismatch")
    if placeholder_rows.dtype not in (torch.int32, torch.int64):
        raise TypeError("Placeholder rows must contain logical integer indices")
    rows = text_embedding.numel() // hidden
    if placeholder_rows.numel() and (int(placeholder_rows.min()) < 0 or int(placeholder_rows.max()) >= rows):
        raise ValueError("Placeholder row outside text")
    if placeholder_rows.unique().numel() != placeholder_rows.numel():
        raise ValueError("Placeholder rows must be unique")
    if text_embedding.dtype != torch.bfloat16 or any(t.device.type != "cpu"
                                                   for t in (text_embedding, visual_features, placeholder_rows)):
        raise ValueError("Text must be BF16; native PTO C++ simulation takes CPU tensors")
    program = TensorProgram.compile(tile_kernels.visual_feature_scatter(rows, hidden, len(visual_features)))
    out = torch.full(text_embedding.shape, float("nan"), dtype=torch.bfloat16)
    program.execute(text_embedding.reshape(rows, hidden), visual_features.bfloat16(),
                    placeholder_rows.int().reshape(1, -1), out.reshape(rows, hidden))
    if artifact_dir is not None:
        program.save(artifact_dir)
    return {"out": out}
