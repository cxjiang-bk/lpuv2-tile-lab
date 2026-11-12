"""OP-01 BF16 row lookup for the fixed model's Token Embedding."""

import torch
from torch.nn import functional as F


@torch.inference_mode()
def embedding(token_ids: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
    """Return fresh BF16 [B,S,H] rows from weight[V,H], without modifying inputs.

    IDs are int64 [B,S] in [0,V). The supplied tensors describe the complete
    valid logical region, including noncontiguous views. Empty B/S is allowed;
    V/H must be positive. No padding-ID convention or hidden state is applied.
    """
    if token_ids.dtype != torch.int64:
        raise TypeError("token_ids must have dtype int64")
    if embedding_weight.dtype != torch.bfloat16:
        raise TypeError("embedding_weight must have dtype bfloat16")
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [B, S]")
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must have shape [V, H]")
    if min(embedding_weight.shape) <= 0:
        raise ValueError("V and H must be positive")
    if token_ids.device != embedding_weight.device:
        raise ValueError("token_ids and embedding_weight must be on the same device")
    if token_ids.numel() and (token_ids.min().item() < 0 or token_ids.max().item() >= embedding_weight.shape[0]):
        raise ValueError("token_ids must be in range [0, V)")
    return F.embedding(token_ids, embedding_weight)
