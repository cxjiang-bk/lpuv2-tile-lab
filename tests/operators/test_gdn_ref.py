"""GDN fallback/core checks; projection fixtures explicitly use BF16 weights."""

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from pymodel.common.case import load_case
from pymodel.ref.gdn import (
    GDNProjections,
    gdn_chunk,
    gdn_conv,
    gdn_gated_norm,
    gdn_gates,
    gdn_recurrent,
    gdn_reference,
)
from pymodel.task_gen.validate import run_case


def core_inputs(length, *, batch=2, heads=2, key_dim=4, value_dim=3, dtype=torch.bfloat16, seed=31):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, length, heads, key_dim, generator=generator, dtype=dtype)
    k = torch.randn(q.shape, generator=generator, dtype=dtype)
    v = torch.randn(batch, length, heads, value_dim, generator=generator, dtype=dtype)
    beta = torch.rand(batch, length, heads, generator=generator).to(dtype)
    g = -torch.rand(batch, length, heads, generator=generator) * 0.2
    state = torch.randn(batch, heads, key_dim, value_dim, generator=generator) * 0.1
    return q, k, v, beta, g, state


@pytest.mark.parametrize("length", [1, 7, 65])
def test_recurrent_matches_pinned_fallback_with_nonzero_state(upstream, length):
    source = upstream("l2norm", "torch_recurrent_gated_delta_rule")
    q, k, v, beta, g, state = core_inputs(length)
    original = state.clone()
    actual = gdn_recurrent(q, k, v, beta, g, state)
    out, final = source.torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=state,
                                                       output_final_state=True, use_qk_l2norm_in_kernel=True)
    torch.testing.assert_close(actual["out"], out, atol=0, rtol=0)
    torch.testing.assert_close(actual["state_out"], final, atol=0, rtol=0)
    assert actual["out"].dtype == torch.bfloat16
    assert actual["state_out"].dtype == torch.float32
    assert torch.equal(state, original)


@pytest.mark.parametrize("length", [1, 63, 64, 65, 129])
def test_chunk_matches_pinned_triangular_solve_and_tail_padding(upstream, length):
    source = upstream("l2norm", "torch_chunk_gated_delta_rule")
    q, k, v, beta, g, state = core_inputs(length)
    actual = gdn_chunk(q, k, v, beta, g, state)
    out, final = source.torch_chunk_gated_delta_rule(q, k, v, g, beta, initial_state=state,
                                                   output_final_state=True, use_qk_l2norm_in_kernel=True)
    torch.testing.assert_close(actual["out"], out, atol=0, rtol=0)
    torch.testing.assert_close(actual["state_out"], final, atol=0, rtol=0)


@pytest.mark.parametrize("chunk_size", [1, 4, 64])
def test_recurrent_and_chunk_agree_in_fp32_arithmetic(chunk_size):
    q, k, v, beta, g, state = core_inputs(65, dtype=torch.float32)
    recurrent = gdn_recurrent(q, k, v, beta, g, state)
    chunk = gdn_chunk(q, k, v, beta, g, state, chunk_size=chunk_size)
    for name in ("out", "state_out"):
        torch.testing.assert_close(chunk[name], recurrent[name], atol=2e-6, rtol=1e-5)


def test_recurrent_continuous_decode_matches_prefill_without_changing_initial_state():
    q, k, v, beta, g, state = core_inputs(7)
    full = gdn_recurrent(q, k, v, beta, g, state)
    current, outputs = state, []
    for t in range(7):
        result = gdn_recurrent(q[:, t:t+1], k[:, t:t+1], v[:, t:t+1], beta[:, t:t+1], g[:, t:t+1], current)
        current = result["state_out"]
        outputs.append(result["out"])
    torch.testing.assert_close(torch.cat(outputs, dim=1), full["out"], atol=0, rtol=0)
    torch.testing.assert_close(current, full["state_out"], atol=0, rtol=0)


def test_actual_model_head_and_state_dimensions(upstream):
    source = upstream("l2norm", "torch_recurrent_gated_delta_rule")
    q, k, v, beta, g, state = core_inputs(2, heads=32, key_dim=128, value_dim=128)
    actual = gdn_recurrent(q, k, v, beta, g, state)
    expected, expected_state = source.torch_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(actual["out"], expected, atol=0, rtol=0)
    torch.testing.assert_close(actual["state_out"], expected_state, atol=0, rtol=0)
    assert actual["state_out"].shape == (2, 32, 128, 128)


def test_zero_initial_state_and_zero_beta_preserve_zero_state():
    q, k, v, beta, g, _ = core_inputs(3)
    result = gdn_recurrent(q, k, v, torch.zeros_like(beta), g)
    assert not torch.count_nonzero(result["out"])
    assert not torch.count_nonzero(result["state_out"])


def test_conv_prefill_and_token_updates_match_upstream_and_keep_raw_history(upstream):
    source = upstream("causal_conv1d_fn", "causal_conv1d_update")
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(2, 7, 8, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(8, 1, 4, generator=generator, dtype=torch.bfloat16)
    full = gdn_conv(x, weight)
    expected = source.causal_conv1d_fn(x.transpose(1, 2), weight[:, 0], activation="silu").transpose(1, 2)
    torch.testing.assert_close(full["out"], expected, atol=0, rtol=0)
    history = torch.zeros(2, 3, 8, dtype=torch.bfloat16)
    outputs = []
    for t in range(7):
        step = gdn_conv(x[:, t:t+1], weight, history)
        oracle_history = history.transpose(1, 2).clone()
        expected = source.causal_conv1d_update(x[:, t:t+1].transpose(1, 2), oracle_history,
                                              weight[:, 0], activation="silu").transpose(1, 2)
        torch.testing.assert_close(step["out"], expected, atol=0, rtol=0)
        torch.testing.assert_close(step["conv_history_out"].transpose(1, 2), oracle_history, atol=0, rtol=0)
        history = step["conv_history_out"]
        outputs.append(step["out"])
    torch.testing.assert_close(torch.cat(outputs, dim=1), full["out"], atol=0, rtol=0)
    torch.testing.assert_close(history, x[:, -3:], atol=0, rtol=0)


def test_gates_retain_the_beta_bf16_and_decay_fp32_boundary():
    a = torch.tensor([[[0.25, -1.5]]], dtype=torch.bfloat16)
    b = torch.tensor([[[1.5, -0.25]]], dtype=torch.bfloat16)
    A_log = torch.tensor([0.5, 1.0], dtype=torch.bfloat16)
    bias = torch.tensor([0.125, -0.75], dtype=torch.bfloat16)
    result = gdn_gates(a, b, A_log, bias)
    assert result["beta"].dtype == torch.bfloat16 and result["g"].dtype == torch.float32
    assert (result["g"] < 0).all()
    torch.testing.assert_close(result["beta"], b.sigmoid(), atol=0, rtol=0)


def test_gated_norm_matches_direct_weight_and_source_cast_order(upstream):
    source = upstream("Qwen3_5MoeRMSNormGated")
    generator = torch.Generator().manual_seed(13)
    module = source.Qwen3_5MoeRMSNormGated(128).to(torch.bfloat16).requires_grad_(False)
    module.weight.copy_(torch.randn(128, generator=generator, dtype=torch.bfloat16))
    x = torch.randn(2, 3, 128, generator=generator, dtype=torch.bfloat16)
    z = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    actual = gdn_gated_norm(x, z, module.weight)
    torch.testing.assert_close(actual, module(x, z), atol=0, rtol=0)
    assert not torch.count_nonzero(gdn_gated_norm(x, z, torch.zeros_like(module.weight)))
    u = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    wrong_late_cast = (u * module.weight.float() * torch.nn.functional.silu(z.float())).to(x.dtype)
    assert not torch.equal(actual, wrong_late_cast)


def full_gdn_module(upstream):
    source = upstream("l2norm", "torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule",
                      "causal_conv1d_fn", "causal_conv1d_update", "Qwen3_5MoeRMSNormGated",
                      "apply_mask_to_padding_states", "Qwen3_5MoeGatedDeltaNet")
    config = SimpleNamespace(hidden_size=8, linear_num_value_heads=2, linear_num_key_heads=1,
                             linear_key_head_dim=4, linear_value_head_dim=3, linear_conv_kernel_dim=4,
                             hidden_act="silu", rms_norm_eps=1e-6, layer_types=["linear_attention"])
    with torch.random.fork_rng():
        torch.manual_seed(23)
        module = source.Qwen3_5MoeGatedDeltaNet(config, 0).to(torch.bfloat16).eval().requires_grad_(False)
    return module


def run_full_gdn(x, module, **kwargs):
    project = GDNProjections(module.in_proj_qkv, module.in_proj_z, module.in_proj_a, module.in_proj_b,
                             module.out_proj)
    return gdn_reference(x, project, conv_weight=module.conv1d.weight, A_log=module.A_log,
                         dt_bias=module.dt_bias, norm_weight=module.norm.weight,
                         qk_heads=1, value_heads=2, key_head_dim=4, value_head_dim=3, **kwargs)


def test_gdn_functional_graph_matches_upstream_with_declared_bf16_projection_fixture(upstream):
    module = full_gdn_module(upstream)
    x = torch.randn(2, 7, 8, generator=torch.Generator().manual_seed(17), dtype=torch.bfloat16)
    mask = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    expected = module(x, attention_mask=mask)
    actual = run_full_gdn(x, module, path="chunk", attention_mask=mask)
    torch.testing.assert_close(actual["out"], expected, atol=0, rtol=0)


def test_gdn_decode_returns_both_states_and_matches_upstream_with_history(upstream):
    module = full_gdn_module(upstream)
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(2, 1, 8, generator=generator, dtype=torch.bfloat16)
    history = torch.randn(2, 3, 14, generator=generator, dtype=torch.bfloat16)
    state = torch.randn(2, 2, 4, 3, generator=generator) * 0.1

    class Cache:
        def get_conv_state(self, layer_idx, state_idx):
            return self.conv

        def get_recurrent_state(self, layer_idx, state_idx):
            return self.state

        def is_recording_past(self, layer_idx):
            return False

        def update_recurrent_state(self, value, layer_idx):
            self.state = value

    cache = Cache()
    cache.conv, cache.state = history.transpose(1, 2).clone(), state.clone()
    expected = module(x, cache_params=cache)
    actual = run_full_gdn(x, module, initial_state=state, conv_history=history)
    torch.testing.assert_close(actual["out"], expected, atol=0, rtol=0)
    torch.testing.assert_close(actual["state_out"], cache.state, atol=0, rtol=0)
    torch.testing.assert_close(actual["conv_history_out"].transpose(1, 2), cache.conv, atol=0, rtol=0)


def test_core_rejects_bf16_recurrent_state_and_missing_head_repeat():
    q, k, v, beta, g, state = core_inputs(1)
    with pytest.raises(ValueError, match="FP32"):
        gdn_recurrent(q, k, v, beta, g, state.bfloat16())
    with pytest.raises(ValueError, match="repeat Q/K heads"):
        gdn_recurrent(q[:, :, :1], k[:, :, :1], v, beta, g, state)


def test_gdn_core_records_bf16_output_and_fp32_state_through_the_shared_runner(upstream, tmp_path):
    root = Path(__file__).resolve().parents[2]
    case = load_case(root / "cases/operators/gdn-recurrent-bf16.json")
    p = case.parameters
    q, k, v, beta, g, state = core_inputs(p["seq_len"], batch=p["batch"], heads=p["heads"],
                                        key_dim=p["key_head_dim"], value_dim=p["value_head_dim"], seed=case.seed)
    source = upstream("l2norm", "torch_recurrent_gated_delta_rule")

    def reference(tensors, parameters):
        out, final = source.torch_recurrent_gated_delta_rule(
            tensors["q"], tensors["k"], tensors["v"], tensors["g"], tensors["beta"],
            initial_state=tensors["state"], output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        return {"out": out, "state_out": final}

    def actual(tensors, parameters):
        return gdn_recurrent(tensors["q"], tensors["k"], tensors["v"], tensors["beta"], tensors["g"], tensors["state"])

    result = run_case(case, inputs=dict(q=q, k=k, v=v, beta=beta, g=g, state=state),
                      reference=reference, implementation=actual, backend="pytorch_reference_check",
                      artifact_root=tmp_path)
    assert result.passed, result.report
    assert result.report["outputs"]["out"]["actual"]["dtype"] == "bfloat16"
    assert result.report["outputs"]["state_out"]["actual"]["dtype"] == "float32"


def test_missing_projection_is_not_replaced_with_a_bf16_fallback():
    with pytest.raises(TypeError, match="All five"):
        GDNProjections(None, None, None, None, None)
