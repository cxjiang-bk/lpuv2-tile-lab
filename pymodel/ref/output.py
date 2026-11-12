"""K-14/15/16: selected logits, normalized feature concat and MTP projection."""

import torch

from pymodel.ref.linear import bf16_linear


@torch.inference_mode()
def lm_head(h: torch.Tensor, weight: torch.Tensor, output_positions: torch.Tensor) -> torch.Tensor:
    """Normalized BF16 h[B,S,H], weight[V,H], local int64 positions[B,P] -> logits[B,P,V]."""
    if output_positions.dtype != torch.int64 or output_positions.shape[0] != h.shape[0]:
        raise ValueError("output_positions must be int64 [B,P] local row indices")
    if ((output_positions < 0) | (output_positions >= h.shape[1])).any():
        raise ValueError("output_positions must be in [0,S)")
    rows = torch.arange(h.shape[0], device=h.device)[:, None]
    return bf16_linear(h[rows, output_positions], weight)


@torch.inference_mode()
def feature_concat(e_norm: torch.Tensor, h_norm: torch.Tensor) -> torch.Tensor:
    """Already normalized, aligned BF16 features[...,H] -> [e_norm;h_norm][...,2H]."""
    if e_norm.shape != h_norm.shape:
        raise ValueError("Embedding and hidden rows must have the same logical shape")
    return torch.cat((e_norm, h_norm), dim=-1)


@torch.inference_mode()
def mtp_input_projection(eh: torch.Tensor, fc_weight: torch.Tensor) -> torch.Tensor:
    """BF16 concatenated features[...,2H] and independent fc[H,2H] -> BF16 [...,H]."""
    return bf16_linear(eh, fc_weight)
