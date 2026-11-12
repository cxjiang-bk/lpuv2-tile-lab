"""Minimal Qwen MTP draft model, following the pinned SGLang call structure."""

import torch

from pymodel.ref.attention import full_attention
from pymodel.ref.embedding import embedding
from pymodel.ref.moe import moe
from pymodel.ref.normalization import rmsnorm
from pymodel.ref.output import feature_concat, lm_head, mtp_input_projection


def prepare_mtp_tokens(target_token_ids: torch.Tensor, next_token_ids: torch.Tensor) -> torch.Tensor:
    """Prefill: [x0,x1,...,xt] -> [x1,...,xt,next], per request.

    Keep target hidden rows and draft cache/RoPE positions in their original
    order: row t pairs embedding(x[t+1]) with target hidden(t). No hidden shift.
    """
    return torch.cat((target_token_ids[:, 1:], next_token_ids.reshape(-1, 1)), dim=1)


@torch.inference_mode()
def mtp_step(token_ids: torch.Tensor, previous_hidden: torch.Tensor, seq_pos: torch.Tensor,
             weights: dict[str, torch.Tensor], attention_weights: dict[str, torch.Tensor],
             attention_scales: dict[str, torch.Tensor], moe_weights: dict[str, torch.Tensor],
             moe_scales: dict[str, torch.Tensor], *, rope_pos: torch.Tensor | None = None,
             kv: dict[str, torch.Tensor] | None = None, top_k: int = 8) -> dict[str, torch.Tensor]:
    """Aligned Token IDs[B,S] and normalized previous_hidden[B,S,H] -> one draft.

    weights: shared target embedding/lm_head, independent MTP fc and BF16 Norm
    weights pre_fc_norm_embedding/pre_fc_norm_hidden/input_layernorm/
    post_attention_layernorm/final_norm. Attention/MoE parameters are the
    independent MTP layer, not a target layer. seq_pos addresses draft slots.

    Prefill uses prepare_mtp_tokens with unshifted target output-Norm hidden.
    Subsequent calls pair a selected Token with the prior hidden_out row and
    increment draft position; hidden_out is AFTER the MTP final Norm. kv is
    draft-only history. Sampling, target verification and rollback are external.
    """
    e_norm = rmsnorm(embedding(token_ids, weights["embedding"]), weights["pre_fc_norm_embedding"])
    h_norm = rmsnorm(previous_hidden, weights["pre_fc_norm_hidden"])
    hidden = mtp_input_projection(feature_concat(e_norm, h_norm), weights["fc"])
    attention = full_attention(rmsnorm(hidden, weights["input_layernorm"]), attention_weights,
                               attention_scales, seq_pos, rope_pos, kv)
    # Residual Add is expressed directly in BF16; OP-04 remains A's entry.
    hidden = hidden + attention["out"]
    experts = moe(rmsnorm(hidden, weights["post_attention_layernorm"]), moe_weights, moe_scales, top_k=top_k)
    hidden = rmsnorm(hidden + experts["out"], weights["final_norm"])
    output_positions = torch.arange(token_ids.shape[1], device=token_ids.device).expand(token_ids.shape[0], -1)
    logits = lm_head(hidden, weights["lm_head"], output_positions)
    return {**{name: value for name, value in attention.items() if name != "out"},
            "out": logits, "hidden_out": hidden,
            "expert_id": experts["expert_id"], "route_weight": experts["route_weight"]}
