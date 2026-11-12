"""Complete Attention: causal output, MRoPE coordinates and explicit KV carry."""

import torch

from pymodel.ref.attention import full_attention, partial_mrope


def test_attention_uniform_scores_have_known_causal_output(attention_case):
    _, weights, scales, _ = attention_case
    # Q/K/gate=0. V is sum(x), so causal means are [8,12,16], gate=1/2.
    for name in ("q_gate", "k"):
        weights[name] = torch.zeros(weights[name].shape).to(torch.float8_e4m3fn)
    weights["v"] = torch.ones(weights["v"].shape).to(torch.float8_e4m3fn)
    out_weight = torch.zeros(weights["out"].shape)
    out_weight[:, 0] = 1
    weights["out"] = out_weight.to(torch.float8_e4m3fn)
    scales = {name: torch.ones_like(scale) for name, scale in scales.items()}
    x = torch.tensor([1, 2, 3], dtype=torch.bfloat16)[None, :, None].expand(1, 3, 8)
    positions = torch.arange(3, dtype=torch.int64)[None]
    result = full_attention(x, weights, scales, positions)
    expected = torch.tensor([4, 6, 8], dtype=torch.bfloat16)[None, :, None].expand(1, 3, 8)
    torch.testing.assert_close(result["out"], expected, atol=0, rtol=0)
    assert result["kv_k_out"].shape == (1, 1, 3, 256)
    assert torch.equal(result["kv_seq_pos_out"], positions)
    assert torch.equal(result["kv_valid_len_out"], torch.tensor([3]))


def test_attention_prefill_then_decode_matches_one_call(attention_case):
    x, weights, scales, positions = attention_case
    rope_pos = torch.stack((positions, positions + 2, positions + 4))
    whole = full_attention(x, weights, scales, positions, rope_pos)
    prefill = full_attention(x[:, :2], weights, scales, positions[:, :2], rope_pos[:, :, :2])
    saved_k = prefill["kv_k_out"].clone()
    kv = {"k": prefill["kv_k_out"], "v": prefill["kv_v_out"], "seq_pos": prefill["kv_seq_pos_out"]}
    decode = full_attention(x[:, 2:], weights, scales, positions[:, 2:], rope_pos[:, :, 2:], kv)
    torch.testing.assert_close(torch.cat((prefill["out"], decode["out"]), dim=1), whole["out"],
                               atol=1e-2, rtol=1e-2)
    for name in ("kv_k_out", "kv_v_out"):
        torch.testing.assert_close(decode[name], whole[name], atol=1e-2, rtol=1e-2)
    assert torch.equal(prefill["kv_k_out"], saved_k)
    torch.testing.assert_close(decode["kv_k_out"][:, :, :2], saved_k, atol=0, rtol=0)


def test_partial_mrope_uses_thw_frequencies_and_leaves_tail():
    q = torch.ones(1, 2, 1, 256, dtype=torch.bfloat16)
    k = torch.ones(1, 1, 1, 256, dtype=torch.bfloat16)
    positions = torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.int64)
    rotated_q, rotated_k = partial_mrope(q, k, positions)
    inv_freq = 1 / (10_000_000 ** (torch.arange(0, 6, 2).float() / 64))
    angles = torch.tensor([1, 2, 3]) * inv_freq
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    torch.testing.assert_close(rotated_q[0, 0, 0, :3], cos - sin, atol=0, rtol=0)
    torch.testing.assert_close(rotated_q[0, 0, 0, 32:35], cos + sin, atol=0, rtol=0)
    torch.testing.assert_close(rotated_k[0, 0], rotated_q[0, 0], atol=0, rtol=0)
    torch.testing.assert_close(rotated_q[..., 64:], q[..., 64:], atol=0, rtol=0)
