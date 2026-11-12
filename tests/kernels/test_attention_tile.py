"""Complete K-05: hidden-to-output Attention and KV across Prefill/Decode."""

import json
from pathlib import Path

import torch

from kernels.pto.attention.full_attention import _mrope_tables, _rotate_heads, full_attention
from pymodel.common.case import Case, OutputCheck
from pymodel.isa.specialize import specialize
from pymodel.isa.torch_codegen import torch_codegen
from pymodel.ref.attention import full_attention as attention_ref
from pymodel.task_gen.validate import run_case


def _parameters(query_heads=4, kv_heads=2, batch=2, length=4):
    generator = torch.Generator().manual_seed(71)
    hidden, dim = 8, 256
    shapes = {"q_gate": (2 * query_heads * dim, hidden), "k": (kv_heads * dim, hidden),
              "v": (kv_heads * dim, hidden), "out": (hidden, query_heads * dim)}
    inputs = {f"w_{name}": torch.randint(-2, 3, shape, generator=generator).to(torch.float8_e4m3fn)
              for name, shape in shapes.items()}
    for name, (n, k) in shapes.items():
        # Vary scale by output block as well as K group.
        scales = torch.arange(((n + 127) // 128) * ((k + 127) // 128)).reshape(
            (n + 127) // 128, (k + 127) // 128)
        inputs[f"s_{name}"] = ((scales % 3 + 1) / 64).to(torch.bfloat16)
    inputs["q_norm"] = torch.randn(dim, generator=generator, dtype=torch.bfloat16) / 8
    inputs["k_norm"] = torch.randn(dim, generator=generator, dtype=torch.bfloat16) / 8
    inputs["x"] = torch.randn(batch, length, hidden, generator=generator, dtype=torch.bfloat16)
    inputs["seq_pos"] = torch.arange(length, dtype=torch.int64)[None].expand(batch, -1).clone()
    inputs["seq_pos"] += torch.arange(batch)[:, None] * 10 + 5
    inputs["rope_pos"] = torch.stack((inputs["seq_pos"], inputs["seq_pos"] + 2, inputs["seq_pos"] + 4))
    return inputs


def _weights(inputs):
    weights = {name: inputs[f"w_{name}"] for name in ("q_gate", "k", "v", "out")}
    weights.update(q_norm=inputs["q_norm"], k_norm=inputs["k_norm"])
    scales = {name: inputs[f"s_{name}"] for name in ("q_gate", "k", "v", "out")}
    return weights, scales


def _tile_call(inputs, x, positions, rope, kv, path):
    batch, length, hidden = x.shape
    dim = inputs["q_norm"].numel()
    kv_heads = inputs["w_k"].shape[0] // dim
    if kv is None:
        kv = {"k": torch.empty(batch, kv_heads, 0, dim, dtype=torch.bfloat16),
              "v": torch.empty(batch, kv_heads, 0, dim, dtype=torch.bfloat16),
              "seq_pos": torch.empty(batch, 0, dtype=torch.int64)}
    total = kv["k"].shape[2] + length
    outputs = {
        "out": torch.full((batch, length, hidden), torch.nan, dtype=torch.bfloat16),
        "kv_k_out": torch.full((batch, kv_heads, total, dim), torch.nan, dtype=torch.bfloat16),
        "kv_v_out": torch.full((batch, kv_heads, total, dim), torch.nan, dtype=torch.bfloat16),
        "kv_seq_pos_out": torch.full((batch, total), -1, dtype=torch.int64),
        "kv_valid_len_out": torch.full((batch,), -1, dtype=torch.int64),
    }
    args = (x, inputs["w_q_gate"], inputs["w_k"], inputs["w_v"], inputs["w_out"],
            inputs["q_norm"], inputs["k_norm"], inputs["s_q_gate"], inputs["s_k"],
            inputs["s_v"], inputs["s_out"], positions, rope,
            kv["k"], kv["v"], kv["seq_pos"], *outputs.values())
    program = specialize(full_attention, *args)
    path.mkdir(parents=True, exist_ok=True)
    (path / "attention_ir.py").write_text(program.as_python())
    code = torch_codegen(program, check_shapes=True)
    code_path = path / "attention_torch.py"
    code_path.write_text(code)
    namespace = {}
    exec(compile(code, str(code_path), "exec"), namespace)
    namespace["full_attention"](*args)
    return outputs


def _checks():
    return {name: OutputCheck(atol=1e-2, rtol=1e-2) for name in ("out", "kv_k_out", "kv_v_out")} | {
        "kv_seq_pos_out": OutputCheck(), "kv_valid_len_out": OutputCheck()}


def _details(path):
    root = Path(__file__).resolve().parents[2]
    return {"pypto_commit": json.loads((root / "data/upstream.lock.json").read_text())["pypto"]["commit"],
            "device": "cpu", "adapter": "pymodel.isa.torch_codegen",
            "specialization": "pymodel.isa.specialize", "programs": str(path)}


def test_full_attention_prefill_then_decode(tmp_path):
    inputs = _parameters()
    checks = {f"prefill_{name}": check for name, check in _checks().items()} | _checks()
    case = Case(case_id="attention_prefill_decode", seed=71, numerical_profile="pto-functional-v0",
                op_id="OP-05", kernel_id="K-05", outputs=checks)

    def execute(values, tile):
        weights, scales = _weights(values)

        def call(x, seq, rope, kv, stage):
            if tile:
                return _tile_call(values, x, seq, rope, kv, tmp_path / stage)
            return attention_ref(x, weights, scales, seq, rope, kv)

        prefill = call(values["x"][:, :3], values["seq_pos"][:, :3], values["rope_pos"][:, :, :3],
                       None, "prefill")
        kv = {"k": prefill["kv_k_out"], "v": prefill["kv_v_out"], "seq_pos": prefill["kv_seq_pos_out"]}
        saved = {name: value.clone() for name, value in kv.items()}
        decode = call(values["x"][:, 3:], values["seq_pos"][:, 3:], values["rope_pos"][:, :, 3:],
                      kv, "decode")
        for name in kv:
            torch.testing.assert_close(kv[name], saved[name], atol=0, rtol=0)
        torch.testing.assert_close(decode["kv_k_out"][:, :, :3], saved["k"], atol=0, rtol=0)
        torch.testing.assert_close(decode["kv_v_out"][:, :, :3], saved["v"], atol=0, rtol=0)
        return {f"prefill_{name}": value for name, value in prefill.items()} | decode

    result = run_case(case, inputs=inputs, reference=lambda values, params: execute(values, False),
                      implementation=lambda values, params: execute(values, True),
                      backend="pypto_torch_codegen_adapted", artifact_root=tmp_path / "runs",
                      backend_details=_details(tmp_path))
    assert result.passed, result.report


def test_full_attention_known_causal_output_with_gqa8(tmp_path):
    inputs = _parameters(query_heads=8, kv_heads=1, batch=1, length=3)
    inputs["x"] = torch.tensor([1, 2, 3], dtype=torch.bfloat16)[None, :, None].expand(1, 3, 8).clone()
    for name in ("q_gate", "k"):
        inputs[f"w_{name}"] = torch.zeros(inputs[f"w_{name}"].shape).to(torch.float8_e4m3fn)
    inputs["w_v"] = torch.ones(inputs["w_v"].shape).to(torch.float8_e4m3fn)
    out_weight = torch.zeros(inputs["w_out"].shape)
    out_weight[:, 0] = 1
    inputs["w_out"] = out_weight.to(torch.float8_e4m3fn)
    for name in ("q_gate", "k", "v", "out"):
        inputs[f"s_{name}"] = torch.ones_like(inputs[f"s_{name}"])
    case = Case(case_id="attention_known_causal", seed=71, numerical_profile="pto-functional-v0",
                op_id="OP-05", kernel_id="K-05", outputs=_checks())

    def reference(values, parameters):
        weights, scales = _weights(values)
        return attention_ref(values["x"], weights, scales, values["seq_pos"], values["rope_pos"])

    def implementation(values, parameters):
        return _tile_call(values, values["x"], values["seq_pos"], values["rope_pos"], None, tmp_path / "known")

    result = run_case(case, inputs=inputs, reference=reference, implementation=implementation,
                      backend="pypto_torch_codegen_adapted", artifact_root=tmp_path / "runs",
                      backend_details=_details(tmp_path))
    assert result.passed, result.report
    actual = torch.load(result.run_dir / "actual.pt", weights_only=True)
    expected = torch.tensor([4, 6, 8], dtype=torch.bfloat16)[None, :, None].expand(1, 3, 8)
    torch.testing.assert_close(actual["out"], expected, atol=0, rtol=0)


def test_mrope_thw_pairing_and_unrotated_tail(tmp_path):
    positions = torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.int64)
    cosine = torch.empty(1, 32, dtype=torch.bfloat16)
    sine = torch.empty_like(cosine)
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    x[:, 64:] = torch.arange(192, dtype=torch.bfloat16)
    out = torch.full_like(x, torch.nan)
    for kernel, args in ((_mrope_tables, (positions, cosine, sine)),
                         (_rotate_heads, (x, cosine, sine, positions, out))):
        program = specialize(kernel, *args)
        code = torch_codegen(program, check_shapes=True)
        (tmp_path / f"{kernel.__name__}_ir.py").write_text(program.as_python())
        (tmp_path / f"{kernel.__name__}_torch.py").write_text(code)
        namespace = {}
        exec(compile(code, f"<{kernel.__name__}>", "exec"), namespace)
        namespace[kernel.__name__](*args)
    frequency = 1 / (10_000_000 ** (torch.arange(0, 64, 2).float() / 64))
    angles = torch.tensor([1, 2, 3])[torch.arange(32) % 3] * frequency
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    torch.testing.assert_close(out[:, :32], (cos - sin).expand(2, -1), atol=0, rtol=0)
    torch.testing.assert_close(out[:, 32:64], (cos + sin).expand(2, -1), atol=0, rtol=0)
    torch.testing.assert_close(out[:, 64:], x[:, 64:], atol=0, rtol=0)
