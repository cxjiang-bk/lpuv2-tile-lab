"""Ordinary Qwen RMSNorm and FP32 Softmax; caller controls later casts."""

import math

import torch


@torch.inference_mode()
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """BF16 x[...,D], weight[D] -> BF16 output, using FP32 and 1+weight.

    This is ordinary RMSNorm, not GDN's separately defined gated Norm.
    Normalization and the weight product stay FP32 until the final cast.
    """
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("x and weight must be bfloat16")
    if x.ndim < 1 or x.shape[-1] <= 0 or weight.shape != (x.shape[-1],):
        raise ValueError("Expected x[...,D] and weight[D] with positive D")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    x32 = x.float()
    normalized = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * (1 + weight.float())).to(x.dtype)


@torch.inference_mode()
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """BF16/FP32 logits -> FP32 probabilities using PyTorch's stable Softmax.

    -Inf can mask individual positions; each row must have a finite position.
    NaN, +Inf and all-masked rows fail. Attention explicitly casts the returned
    probabilities to BF16 at its source boundary; Router keeps them FP32.
    """
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("x must be bfloat16 or float32")
    if x.ndim < 1 or x.shape[dim] <= 0:
        raise ValueError("Softmax needs a nonempty reduction axis")
    if torch.isnan(x).any() or torch.isposinf(x).any() or not torch.isfinite(x).any(dim=dim).all():
        raise ValueError("Softmax rows need a finite logit and may not contain NaN or +Inf")
    return torch.softmax(x, dim=dim, dtype=torch.float32)
