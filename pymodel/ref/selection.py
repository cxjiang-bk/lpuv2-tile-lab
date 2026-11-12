"""Small routing references using the project's explicit Top-K tie policy."""

import torch


@torch.inference_mode()
def stable_topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the last-axis Top-K; return original-dtype values and int64 IDs.

    Columns must already be ordered by ascending expert ID. Stable descending
    sort then keeps that ID order for exact ties. Does not normalize or cast
    selected weights; the Router caller owns those model-specific operations.
    """
    if scores.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("scores must be bfloat16 or float32")
    if scores.ndim < 1 or type(k) is not int or not 1 <= k <= scores.shape[-1]:
        raise ValueError("k must be an integer in [1, number of experts]")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    values, indices = torch.sort(scores, dim=-1, descending=True, stable=True)
    return values[..., :k], indices[..., :k]
