"""K-07 routed and shared MoE; BF16 merge boundaries follow the model source."""

import torch
from torch.nn import functional as F

from pymodel.ref.linear import bf16_linear, fp8_linear
from pymodel.ref.normalization import softmax
from pymodel.ref.selection import stable_topk


def _ffn(x, weights, scales):
    gate = fp8_linear(x, weights["gate"], scales["gate"])
    up = fp8_linear(x, weights["up"], scales["up"])
    return fp8_linear(F.silu(gate) * up, weights["down"], scales["down"])


@torch.inference_mode()
def moe(x: torch.Tensor, weights: dict[str, torch.Tensor], scales: dict[str, torch.Tensor], *,
        top_k: int = 8) -> dict[str, torch.Tensor]:
    """BF16 x[...,H] -> out, original-Token routes and expert bucket metadata.

    BF16 weights: router[E,H], shared_gate[1,H]. FP8 weights: expert_gate/up
    [E,I,H], expert_down[E,H,I], shared_ffn_gate/up[I,H], shared_ffn_down[H,I].
    scales has matching FP8 names and per-matrix block multipliers. Each call
    has T=product(leading axes) valid Tokens; expert IDs are processed ascending.
    Inputs are preserved; no cross-call state or physical buffering is hidden.
    """
    tokens = x.reshape(-1, x.shape[-1])
    logits = bf16_linear(tokens, weights["router"])
    selected, expert_id = stable_topk(softmax(logits), top_k)
    route_weight = (selected / selected.sum(dim=-1, keepdim=True)).to(logits.dtype)
    merged = torch.zeros_like(tokens)
    bucket_offset, bucket_tokens, bucket_slots = [0], [], []
    for expert in range(weights["router"].shape[0]):
        token_id, slot = torch.where(expert_id == expert)
        bucket_tokens.append(token_id)
        bucket_slots.append(slot)
        bucket_offset.append(bucket_offset[-1] + token_id.numel())
        if token_id.numel() == 0:
            continue
        expert_weights = {name: weights[f"expert_{name}"][expert] for name in ("gate", "up", "down")}
        expert_scales = {name: scales[f"expert_{name}"][expert] for name in ("gate", "up", "down")}
        contribution = _ffn(tokens[token_id], expert_weights, expert_scales)
        contribution = contribution * route_weight[token_id, slot, None]
        merged.index_add_(0, token_id, contribution.to(merged.dtype))

    shared_weights = {name: weights[f"shared_ffn_{name}"] for name in ("gate", "up", "down")}
    shared_scales = {name: scales[f"shared_ffn_{name}"] for name in ("gate", "up", "down")}
    shared = _ffn(tokens, shared_weights, shared_scales)
    shared = shared * torch.sigmoid(bf16_linear(tokens, weights["shared_gate"]))
    return {"out": (merged + shared).reshape_as(x), "expert_id": expert_id, "route_weight": route_weight,
            "bucket_offset": torch.tensor(bucket_offset, dtype=torch.int64, device=x.device),
            "bucket_token_id": torch.cat(bucket_tokens), "bucket_slot": torch.cat(bucket_slots)}
