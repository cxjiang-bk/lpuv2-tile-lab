"""One draft-model computation with explicit shifted Tokens and hidden carry."""

import torch
from torch.nn import functional as F

from pymodel.ref.mtp import mtp_step, prepare_mtp_tokens


def make_mtp_weights():
    generator = torch.Generator().manual_seed(59)
    weights = {"embedding": torch.randn(13, 8, generator=generator, dtype=torch.bfloat16),
               "lm_head": torch.randn(13, 8, generator=generator, dtype=torch.bfloat16),
               "fc": torch.cat((torch.eye(8), 2 * torch.eye(8)), dim=-1).bfloat16()}
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "input_layernorm",
                 "post_attention_layernorm", "final_norm"):
        weights[name] = torch.randn(8, generator=generator, dtype=torch.bfloat16) * 0.125
    return weights


def test_mtp_prefill_shifts_tokens_per_request_but_hidden_rows_stay_in_place():
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    tail = torch.tensor([7, 8], dtype=torch.int64)
    expected = torch.tensor([[2, 3, 7], [5, 6, 8]], dtype=torch.int64)
    assert torch.equal(prepare_mtp_tokens(tokens, tail), expected)
    assert torch.equal(tokens, torch.tensor([[1, 2, 3], [4, 5, 6]]))


def test_mtp_known_concat_projection_and_final_norm(attention_case, moe_case):
    x, attention_weights, attention_scales, positions = attention_case
    moe_weights, moe_scales = moe_case
    weights = make_mtp_weights()
    attention_weights["out"] = torch.zeros(attention_weights["out"].shape).to(torch.float8_e4m3fn)
    for name in ("expert_down", "shared_ffn_down"):
        moe_weights[name] = torch.zeros(moe_weights[name].shape).to(torch.float8_e4m3fn)
    tokens = torch.tensor([[2, 3, 7], [5, 6, 8]], dtype=torch.int64)

    def norm(value, weight):
        return F.rms_norm(value.float(), (8,), 1 + weight.float(), eps=1e-6).bfloat16()

    e_norm = norm(weights["embedding"][tokens], weights["pre_fc_norm_embedding"])
    h_norm = norm(x, weights["pre_fc_norm_hidden"])
    projected = F.linear(torch.cat((e_norm, h_norm), dim=-1), weights["fc"])
    expected_hidden = norm(projected, weights["final_norm"])
    expected_logits = F.linear(expected_hidden, weights["lm_head"])
    result = mtp_step(tokens, x, positions, weights, attention_weights, attention_scales, moe_weights, moe_scales)
    torch.testing.assert_close(result["hidden_out"], expected_hidden, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(result["out"], expected_logits, atol=1e-2, rtol=1e-2)
    assert result["expert_id"].shape == (6, 8)
    assert result["kv_k_out"].shape == (2, 1, 3, 256)


def test_mtp_prefill_then_next_draft_carries_normalized_hidden_and_own_kv(attention_case, moe_case):
    x, attention_weights, attention_scales, positions = attention_case
    moe_weights, moe_scales = moe_case
    weights = make_mtp_weights()
    target_tokens = torch.tensor([[1, 2], [4, 5]], dtype=torch.int64)
    shifted = prepare_mtp_tokens(target_tokens, torch.tensor([3, 6]))
    args = (weights, attention_weights, attention_scales, moe_weights, moe_scales)
    first = mtp_step(shifted, x[:, :2], positions[:, :2], *args)
    next_token = first["out"][:, -1:].argmax(-1)
    next_hidden = first["hidden_out"][:, -1:]
    kv = {"k": first["kv_k_out"], "v": first["kv_v_out"], "seq_pos": first["kv_seq_pos_out"]}
    second = mtp_step(next_token, next_hidden, positions[:, 2:], *args, kv=kv)
    whole = mtp_step(torch.cat((shifted, next_token), dim=1), torch.cat((x[:, :2], next_hidden), dim=1),
                     positions, *args)
    torch.testing.assert_close(second["out"], whole["out"][:, -1:], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(second["hidden_out"], whole["hidden_out"][:, -1:], atol=1e-2, rtol=1e-2)
    for name in ("kv_k_out", "kv_v_out"):
        torch.testing.assert_close(second[name], whole[name], atol=1e-2, rtol=1e-2)
    assert torch.equal(second["kv_seq_pos_out"], positions)
