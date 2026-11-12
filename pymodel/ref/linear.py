"""Common projections with logical weight[N,K] and explicit dtype boundaries."""

import torch
from torch.nn import functional as F

from pymodel.ref.quantization import dynamic_quantize


@torch.inference_mode()
def bf16_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """BF16 [...,K] @ BF16 [N,K].T, with optional BF16 bias[N], returns BF16.

    Uses PyTorch's BF16 Linear path; internal accumulation follows that backend.
    Inputs are preserved. No layout, Tile, or physical capacity is modelled.
    """
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("x and weight must be bfloat16")
    if x.ndim < 1 or weight.ndim != 2 or min(weight.shape) <= 0 or x.shape[-1] != weight.shape[1]:
        raise ValueError("Expected x[...,K] and positive weight[N,K] with matching K")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if bias is not None:
        if bias.dtype != torch.bfloat16:
            raise TypeError("bias must be bfloat16")
        if bias.shape != (weight.shape[0],) or bias.device != x.device:
            raise ValueError("bias must have shape [N] and be on the input device")
    return F.linear(x, weight, bias)


@torch.inference_mode()
def fp8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Functional-v0 BF16 x[...,K] -> BF16 [...,N] using FP8 weight[N,K].

    weight_scale[ceil(N/128),ceil(K/128)] is BF16 or FP32 and multiplies
    the weight block; checkpoint values are kept, not inverted. Each K-group
    dot product is FP32, scaled by its activation then weight scale, and added
    in increasing K-group order. This is Python math, not a native FP8 backend.
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError("weight must be float8_e4m3fn")
    if weight_scale.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("weight_scale must be bfloat16 or float32")
    if x.ndim < 1 or weight.ndim != 2 or min(weight.shape) <= 0 or x.shape[-1] != weight.shape[1]:
        raise ValueError("Expected x[...,K] and positive weight[N,K] with matching K")
    n, k = weight.shape
    if weight_scale.shape != ((n + 127) // 128, (k + 127) // 128):
        raise ValueError("weight_scale must have shape [ceil(N/128),ceil(K/128)]")
    if x.device != weight.device or x.device != weight_scale.device:
        raise ValueError("x, weight and weight_scale must be on the same device")
    if not torch.isfinite(weight_scale).all() or not (weight_scale > 0).all():
        raise ValueError("weight_scale must contain finite positive multipliers")
    weight32 = weight.float()
    if not torch.isfinite(weight32).all():
        raise ValueError("weight must contain only finite values")
    payload, activation_scale = dynamic_quantize(x)
    x32 = payload.reshape(-1, k).float()
    activation_scale = activation_scale.reshape(x32.shape[0], (k + 127) // 128)
    row_scale = weight_scale.float().repeat_interleave(128, dim=0)[:n]
    result = torch.zeros(x32.shape[0], n, dtype=torch.float32, device=x.device)
    for group, start in enumerate(range(0, k, 128)):
        partial = x32[:, start:start + 128] @ weight32[:, start:start + 128].T
        result += partial * activation_scale[:, group:group + 1] * row_scale[:, group]
    return result.reshape(*x.shape[:-1], n).to(torch.bfloat16)
