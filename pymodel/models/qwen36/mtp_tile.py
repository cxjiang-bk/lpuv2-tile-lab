"""Minimal MTP composition; numerical work executes the existing Tile kernels."""

from pathlib import Path

import pypto.language as pl
import torch

from kernels.pto.attention.full_attention import full_attention
from kernels.pto.common.embedding import embedding
from kernels.pto.common.rmsnorm import rmsnorm
from kernels.pto.moe.moe import moe
from kernels.pto.mtp.input import feature_concat, mtp_input_projection
from kernels.pto.output.lm_head import lm_head
from pymodel.isa.specialize import specialize
from pymodel.isa.torch_codegen import torch_codegen


@pl.jit.incore
def _residual_add(lhs: pl.Tensor, rhs: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Local BF16 integration fallback until A's OP-04 Tile entry is available."""
    for row in pl.range(lhs.shape[0]):
        for col in pl.range(0, lhs.shape[1], 128):
            valid = pl.min(128, lhs.shape[1] - col)
            left = pl.load(lhs, [row, col], [1, 128], valid_shape=[1, valid])
            right = pl.load(rhs, [row, col], [1, 128], valid_shape=[1, valid])
            result = pl.add(left, right)
            result = pl.tile.slice(result, [1, 128], [0, 0], valid_shape=[1, valid])
            pl.store(result, [row, col], out)
    return out


@torch.inference_mode()
def mtp_step(token_ids: torch.Tensor, previous_hidden: torch.Tensor, seq_pos: torch.Tensor,
             weights: dict[str, torch.Tensor], attention_weights: dict[str, torch.Tensor],
             attention_scales: dict[str, torch.Tensor], moe_weights: dict[str, torch.Tensor],
             moe_scales: dict[str, torch.Tensor], *, rope_pos: torch.Tensor | None = None,
             kv: dict[str, torch.Tensor] | None = None, top_k: int = 8,
             artifact_dir: str | Path | None = None) -> dict[str, torch.Tensor]:
    """Aligned IDs[B,S] and previous hidden[B,S,H] -> logits, normalized hidden, draft KV.

    Python allocates tensors and organizes calls. All numerical operations run
    PyPTO IR through the project's CPU Torch codegen adapter. Embedding/head
    may be shared with the target; fc/Norm/Attention/MoE parameters and ``kv``
    belong to the draft. Token shifting, selection and target verification are
    supplied by the caller. This entry currently supports Top-8 on CPU.
    """
    if top_k != 8:
        raise ValueError("The current MoE Tile entry requires top_k=8")
    batch, sequence = token_ids.shape
    hidden = previous_hidden.shape[-1]
    rows = batch * sequence
    directory = Path(artifact_dir) if artifact_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)

    def run(stage, kernel, *args):
        program = specialize(kernel, *args)
        code = torch_codegen(program, check_shapes=True)
        if directory is not None:
            (directory / f"{stage}_ir.py").write_text(program.as_python())
            (directory / f"{stage}_torch.py").write_text(code)
        namespace = {}
        exec(compile(code, f"<mtp_{stage}>", "exec"), namespace)
        namespace[kernel.__name__](*args)

    def norm(stage, value, weight):
        result = torch.empty_like(value)
        run(stage, rmsnorm, value.reshape(-1, hidden), weight, result.view(-1, hidden), 1e-6)
        return result

    embedded = torch.empty(batch, sequence, hidden, dtype=torch.bfloat16)
    run("embedding", embedding, token_ids, weights["embedding"], embedded)
    e_norm = norm("embedding_norm", embedded, weights["pre_fc_norm_embedding"])
    h_norm = norm("previous_hidden_norm", previous_hidden, weights["pre_fc_norm_hidden"])
    concatenated = torch.empty(rows, 2 * hidden, dtype=torch.bfloat16)
    run("concat", feature_concat, e_norm.reshape(rows, hidden), h_norm.reshape(rows, hidden), concatenated)
    projected = torch.empty(batch, sequence, hidden, dtype=torch.bfloat16)
    run("input_projection", mtp_input_projection, concatenated, weights["fc"], projected.view(rows, hidden))
    attention_input = norm("attention_input_norm", projected, weights["input_layernorm"])

    dim = attention_weights["q_norm"].numel()
    kv_heads = attention_weights["k"].shape[0] // dim
    if kv is None:
        kv = {"k": torch.empty(batch, kv_heads, 0, dim, dtype=torch.bfloat16),
              "v": torch.empty(batch, kv_heads, 0, dim, dtype=torch.bfloat16),
              "seq_pos": torch.empty(batch, 0, dtype=torch.int64)}
    if rope_pos is None:
        rope_pos = seq_pos.unsqueeze(0).expand(3, -1, -1)
    total = kv["k"].shape[2] + sequence
    attention_out = torch.empty_like(projected)
    cache = {
        "kv_k_out": torch.empty(batch, kv_heads, total, dim, dtype=torch.bfloat16),
        "kv_v_out": torch.empty(batch, kv_heads, total, dim, dtype=torch.bfloat16),
        "kv_seq_pos_out": torch.empty(batch, total, dtype=torch.int64),
        "kv_valid_len_out": torch.empty(batch, dtype=torch.int64),
    }
    run("attention", full_attention, attention_input,
        *(attention_weights[name] for name in ("q_gate", "k", "v", "out", "q_norm", "k_norm")),
        *(attention_scales[name] for name in ("q_gate", "k", "v", "out")),
        seq_pos, rope_pos, kv["k"], kv["v"], kv["seq_pos"], attention_out, *cache.values())
    after_attention = torch.empty_like(projected)
    run("attention_residual", _residual_add, projected.view(rows, hidden), attention_out.view(rows, hidden),
        after_attention.view(rows, hidden))
    moe_input = norm("moe_input_norm", after_attention, weights["post_attention_layernorm"])

    expert_count = moe_weights["router"].shape[0]
    moe_out = torch.empty_like(projected)
    routes = {
        "expert_id": torch.empty(rows, 8, dtype=torch.int64),
        "route_weight": torch.empty(rows, 8, dtype=torch.bfloat16),
        "bucket_offset": torch.empty(expert_count + 1, dtype=torch.int64),
        "bucket_token_id": torch.empty(rows * 8, dtype=torch.int64),
        "bucket_slot": torch.empty(rows * 8, dtype=torch.int64),
    }
    names = ("router", "shared_gate", "expert_gate", "expert_up", "expert_down",
             "shared_ffn_gate", "shared_ffn_up", "shared_ffn_down")
    run("moe", moe, moe_input.view(rows, hidden), *(moe_weights[name] for name in names),
        *(moe_scales[name] for name in names[2:]), moe_out.view(rows, hidden), *routes.values())
    after_moe = torch.empty_like(projected)
    run("moe_residual", _residual_add, after_attention.view(rows, hidden), moe_out.view(rows, hidden),
        after_moe.view(rows, hidden))
    hidden_out = norm("output_norm", after_moe, weights["final_norm"])
    output_positions = torch.arange(sequence, dtype=torch.int64).expand(batch, -1)
    logits = torch.empty(batch, sequence, weights["lm_head"].shape[0], dtype=torch.bfloat16)
    run("lm_head", lm_head, hidden_out, weights["lm_head"], output_positions, logits)
    return {**cache, "out": logits, "hidden_out": hidden_out,
            "expert_id": routes["expert_id"], "route_weight": routes["route_weight"]}
