"""K-05 Full Attention with functional-v0 FP8 projections and explicit KV."""

import torch

from pymodel.ref.linear import fp8_linear
from pymodel.ref.normalization import rmsnorm, softmax


@torch.inference_mode()
def partial_mrope(q: torch.Tensor, k: torch.Tensor, rope_pos: torch.Tensor, *,
                  rotary_dim: int = 64, mrope_section: tuple[int, int, int] = (11, 11, 10),
                  theta: float = 10_000_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate BF16 q/k[B,heads,S,D] with THW positions[3,B,S]; preserve D tail.

    Match the pinned interleaved THW frequency selection and half-split pairs.
    Frequencies/trig are FP32, cast before the BF16 rotation operations.
    """
    inv_freq = 1 / (theta ** (torch.arange(0, rotary_dim, 2, device=q.device).float() / rotary_dim))
    angles = rope_pos[..., None].float() * inv_freq
    tables = []
    for values in (angles.cos(), angles.sin()):
        selected = values[0].clone()
        for axis in (1, 2):
            selected[..., axis:mrope_section[axis] * 3:3] = values[axis, ..., axis:mrope_section[axis] * 3:3]
        tables.append(torch.cat((selected, selected), dim=-1).to(q.dtype).unsqueeze(1))
    cos, sin = tables

    def rotate(x):
        first, second = x[..., :rotary_dim].chunk(2, dim=-1)
        rotated = x[..., :rotary_dim] * cos + torch.cat((-second, first), dim=-1) * sin
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)

    return rotate(q), rotate(k)


@torch.inference_mode()
def full_attention(x: torch.Tensor, weights: dict[str, torch.Tensor], scales: dict[str, torch.Tensor],
                   seq_pos: torch.Tensor, rope_pos: torch.Tensor | None = None,
                   kv: dict[str, torch.Tensor] | None = None, *, rotary_dim: int = 64,
                   mrope_section: tuple[int, int, int] = (11, 11, 10), theta: float = 10_000_000,
                   eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """Already normalized BF16 x[B,S,H] -> out and new KV, without input mutation.

    weights: FP8 q_gate/k/v/out, BF16 q_norm/k_norm. scales: corresponding
    projection block multipliers. Heads and D are inferred from those weights.
    seq_pos[B,S] is causal order; rope_pos[3,B,S] defaults to text positions.
    Optional kv has BF16 k/v[B,kv_heads,L,D] and int64 seq_pos[B,L]. All rows
    are valid, equal length per request. Decoder pre-norm/residual are external.
    """
    batch, length, _ = x.shape
    head_dim = weights["q_norm"].numel()
    query_heads = weights["q_gate"].shape[0] // (2 * head_dim)
    kv_heads = weights["k"].shape[0] // head_dim
    if query_heads % kv_heads or rotary_dim > head_dim:
        raise ValueError("Query heads must group evenly over KV heads and rotary_dim must fit D")
    if rope_pos is None:
        rope_pos = seq_pos.unsqueeze(0).expand(3, -1, -1)

    q_gate = fp8_linear(x, weights["q_gate"], scales["q_gate"]).reshape(batch, length, query_heads, 2 * head_dim)
    q, gate = q_gate.chunk(2, dim=-1)
    k = fp8_linear(x, weights["k"], scales["k"]).reshape(batch, length, kv_heads, head_dim)
    v = fp8_linear(x, weights["v"], scales["v"]).reshape(batch, length, kv_heads, head_dim).transpose(1, 2)
    q = rmsnorm(q, weights["q_norm"], eps=eps).transpose(1, 2)
    k = rmsnorm(k, weights["k_norm"], eps=eps).transpose(1, 2)
    q, k = partial_mrope(q, k, rope_pos, rotary_dim=rotary_dim, mrope_section=mrope_section, theta=theta)

    key_pos = seq_pos
    if kv is not None:
        k, v = torch.cat((kv["k"], k), dim=2), torch.cat((kv["v"], v), dim=2)
        key_pos = torch.cat((kv["seq_pos"], seq_pos), dim=1)
    repeats = query_heads // kv_heads
    scores = (q @ k.repeat_interleave(repeats, dim=1).transpose(-2, -1)) * head_dim**-0.5
    future = key_pos[:, None, None, :] > seq_pos[:, None, :, None]
    scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
    probabilities = softmax(scores).to(q.dtype)
    context = probabilities @ v.repeat_interleave(repeats, dim=1)
    context = context.transpose(1, 2).reshape(batch, length, query_heads * head_dim)
    gated = context * torch.sigmoid(gate.reshape(batch, length, -1))
    out = fp8_linear(gated, weights["out"], scales["out"])
    return {"out": out, "kv_k_out": k, "kv_v_out": v, "kv_seq_pos_out": key_pos.clone(),
            "kv_valid_len_out": torch.full((batch,), k.shape[2], dtype=torch.int64, device=x.device)}
