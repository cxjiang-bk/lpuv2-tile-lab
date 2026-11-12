"""Minimal MTP integration: all Tile operators, normalized hidden and draft KV."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from pymodel.common.case import Case, OutputCheck
from pymodel.models.qwen36.mtp_tile import mtp_step as mtp_tile
from pymodel.ref.mtp import mtp_step as mtp_ref, prepare_mtp_tokens
from pymodel.task_gen.validate import run_case


def _inputs():
    generator = torch.Generator().manual_seed(91)
    hidden, dim, query_heads, kv_heads, vocab, experts, intermediate = 8, 256, 2, 1, 13, 10, 4
    values = {
        "tokens": torch.tensor([[1, 2], [4, 5]], dtype=torch.int64),
        "tail": torch.tensor([3, 6], dtype=torch.int64),
        "previous_hidden": torch.randn(2, 2, hidden, generator=generator, dtype=torch.bfloat16),
        "positions": torch.tensor([[5, 6, 7], [10, 11, 12]], dtype=torch.int64),
        # Target history is deliberately unrelated to the draft's empty start.
        "target_k": torch.randn(2, 1, 5, dim, generator=generator, dtype=torch.bfloat16),
        "target_v": torch.randn(2, 1, 5, dim, generator=generator, dtype=torch.bfloat16),
    }
    for name in ("embedding", "lm_head"):
        values[f"model_{name}"] = torch.randn(vocab, hidden, generator=generator, dtype=torch.bfloat16)
    values["model_fc"] = torch.cat((torch.eye(hidden), 2 * torch.eye(hidden)), -1).bfloat16()
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "input_layernorm",
                 "post_attention_layernorm", "final_norm"):
        values[f"model_{name}"] = torch.randn(hidden, generator=generator, dtype=torch.bfloat16) / 8
    shapes = {"q_gate": (2 * query_heads * dim, hidden), "k": (kv_heads * dim, hidden),
              "v": (kv_heads * dim, hidden), "out": (hidden, query_heads * dim)}
    for name, (n, k) in shapes.items():
        values[f"attn_{name}"] = torch.randint(-2, 3, (n, k), generator=generator).to(torch.float8_e4m3fn)
        values[f"attn_scale_{name}"] = torch.full(((n + 127) // 128, (k + 127) // 128),
                                                   0.02, dtype=torch.bfloat16)
    for name in ("q_norm", "k_norm"):
        values[f"attn_{name}"] = torch.randn(dim, generator=generator, dtype=torch.bfloat16) / 8
    values["moe_router"] = torch.randn(experts, hidden, generator=generator, dtype=torch.bfloat16)
    values["moe_shared_gate"] = torch.randn(1, hidden, generator=generator, dtype=torch.bfloat16)
    for name, n, k in (("gate", intermediate, hidden), ("up", intermediate, hidden),
                       ("down", hidden, intermediate)):
        for prefix, leading in (("expert", (experts,)), ("shared_ffn", ())):
            key = f"{prefix}_{name}"
            values[f"moe_{key}"] = torch.randint(-2, 3, (*leading, n, k), generator=generator).to(
                torch.float8_e4m3fn)
            values[f"moe_scale_{key}"] = torch.full((*leading, 1, 1), 0.02, dtype=torch.bfloat16)
    return values


def _parameters(values):
    def group(prefix, excluded=""):
        return {key.removeprefix(prefix): value for key, value in values.items()
                if key.startswith(prefix) and not (excluded and key.startswith(excluded))}

    return (group("model_"), group("attn_", "attn_scale_"), group("attn_scale_"),
            group("moe_", "moe_scale_"), group("moe_scale_"))


def _checks():
    return {name: OutputCheck(atol=1e-2, rtol=1e-2) for name in
            ("out", "hidden_out", "kv_k_out", "kv_v_out", "route_weight")} | {
        name: OutputCheck() for name in ("expert_id", "kv_seq_pos_out", "kv_valid_len_out")}


def _details(tmp_path):
    root = Path(__file__).resolve().parents[2]
    return {"pypto_commit": json.loads((root / "data/upstream.lock.json").read_text())["pypto"]["commit"],
            "device": "cpu", "programs": str(tmp_path),
            "adapter": "pymodel.isa.specialize + pymodel.isa.torch_codegen",
            "residual": "local Tile add for MTP integration; A OP-04 not integrated"}


def test_mtp_prefill_then_draft_carries_own_hidden_and_kv(tmp_path):
    case = Case(case_id="mtp_prefill_draft", seed=91, numerical_profile="pto-functional-v0",
                outputs={f"prefill_{name}": check for name, check in _checks().items()} | _checks() | {
                    "aligned_tokens": OutputCheck(), "next_token": OutputCheck()})

    def execute(values, tile):
        parameters = _parameters(values)
        aligned = prepare_mtp_tokens(values["tokens"], values["tail"])
        torch.testing.assert_close(aligned, torch.tensor([[2, 3], [5, 6]]), atol=0, rtol=0)
        target_k, target_v = values["target_k"].clone(), values["target_v"].clone()

        def call(tokens, hidden, positions, kv, stage):
            kwargs = {"kv": kv}
            if tile:
                kwargs["artifact_dir"] = tmp_path / stage
            return (mtp_tile if tile else mtp_ref)(tokens, hidden, positions, *parameters, **kwargs)

        first = call(aligned, values["previous_hidden"], values["positions"][:, :2], None, "prefill")
        # Token choice is test orchestration outside the MTP numerical program.
        next_token = first["out"][:, -1:].argmax(-1)
        next_hidden = first["hidden_out"][:, -1:]
        kv = {"k": first["kv_k_out"], "v": first["kv_v_out"], "seq_pos": first["kv_seq_pos_out"]}
        saved_k, saved_v = kv["k"].clone(), kv["v"].clone()
        second = call(next_token, next_hidden, values["positions"][:, 2:], kv, "draft")
        torch.testing.assert_close(kv["k"], saved_k, atol=0, rtol=0)
        torch.testing.assert_close(kv["v"], saved_v, atol=0, rtol=0)
        torch.testing.assert_close(second["kv_k_out"][:, :, :2], saved_k, atol=0, rtol=0)
        torch.testing.assert_close(second["kv_v_out"][:, :, :2], saved_v, atol=0, rtol=0)
        torch.testing.assert_close(second["kv_seq_pos_out"], values["positions"], atol=0, rtol=0)
        torch.testing.assert_close(second["kv_valid_len_out"], torch.full((2,), 3), atol=0, rtol=0)
        torch.testing.assert_close(values["target_k"], target_k, atol=0, rtol=0)
        torch.testing.assert_close(values["target_v"], target_v, atol=0, rtol=0)
        return {f"prefill_{name}": value for name, value in first.items()} | second | {
            "aligned_tokens": aligned, "next_token": next_token}

    result = run_case(case, inputs=_inputs(), reference=lambda values, params: execute(values, False),
                      implementation=lambda values, params: execute(values, True),
                      backend="pypto_torch_codegen_adapted", backend_details=_details(tmp_path),
                      artifact_root=tmp_path / "runs")
    assert result.passed, result.report


def test_mtp_known_projection_and_output_norm(tmp_path):
    values = _inputs()
    values["attn_out"] = torch.zeros(values["attn_out"].shape).to(torch.float8_e4m3fn)
    for name in ("expert_down", "shared_ffn_down"):
        values[f"moe_{name}"] = torch.zeros(values[f"moe_{name}"].shape).to(torch.float8_e4m3fn)
    case = Case(case_id="mtp_known_projection", seed=91, numerical_profile="pto-functional-v0", outputs=_checks())

    def call(inputs, tile):
        kwargs = {"artifact_dir": tmp_path / "known"} if tile else {}
        aligned = prepare_mtp_tokens(inputs["tokens"], inputs["tail"])
        return (mtp_tile if tile else mtp_ref)(aligned, inputs["previous_hidden"], inputs["positions"][:, :2],
                                              *_parameters(inputs), **kwargs)

    result = run_case(case, inputs=values, reference=lambda inputs, params: call(inputs, False),
                      implementation=lambda inputs, params: call(inputs, True),
                      backend="pypto_torch_codegen_adapted", backend_details=_details(tmp_path),
                      artifact_root=tmp_path / "runs")
    assert result.passed, result.report
    model, *_ = _parameters(values)

    def norm(x, weight):
        return F.rms_norm(x.float(), (8,), 1 + weight.float(), eps=1e-6).bfloat16()

    aligned = prepare_mtp_tokens(values["tokens"], values["tail"])
    e_norm = norm(model["embedding"][aligned], model["pre_fc_norm_embedding"])
    h_norm = norm(values["previous_hidden"], model["pre_fc_norm_hidden"])
    expected_hidden = norm(F.linear(torch.cat((e_norm, h_norm), -1), model["fc"]), model["final_norm"])
    actual = torch.load(result.run_dir / "actual.pt", weights_only=True)
    torch.testing.assert_close(actual["hidden_out"], expected_hidden, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual["out"], F.linear(expected_hidden, model["lm_head"]), atol=1e-2, rtol=1e-2)
