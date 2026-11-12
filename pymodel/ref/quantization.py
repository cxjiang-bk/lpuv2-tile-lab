"""Functional-v0 activation quantization; scales are dequantization multipliers."""

import torch
from torch.nn import functional as F


@torch.inference_mode()
def dynamic_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 x[...,K] -> FP8 payload[...,K], FP32 scale[...,ceil(K/128)].

    Each Token/K group has its own amax/448 scale; zero groups use 1. Temporary
    tail padding is zero and trimmed from the returned logical payload. Inputs
    are preserved. The caller supplies the complete logical K axis, not a Tile
    fragment whose start could shift the 128-element grouping.
    """
    if x.dtype != torch.bfloat16:
        raise TypeError("x must be bfloat16")
    if x.ndim < 1 or x.shape[-1] <= 0:
        raise ValueError("Expected x[...,K] with positive K")
    if not torch.isfinite(x).all():
        raise ValueError("x must contain only finite values")
    width = x.shape[-1]
    groups = (width + 127) // 128
    blocks = F.pad(x.float(), (0, groups * 128 - width)).reshape(*x.shape[:-1], groups, 128)
    amax = blocks.abs().amax(dim=-1)
    scale = torch.where(amax == 0, 1.0, amax / 448.0)
    normalized = (blocks / scale.unsqueeze(-1)).clamp(-448, 448)
    payload = normalized.to(torch.float8_e4m3fn).reshape(*x.shape[:-1], groups * 128)[..., :width]
    return payload, scale
