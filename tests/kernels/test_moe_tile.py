"""Complete K-07 with independent output, routes and expert-bucket checks."""

import json
from pathlib import Path

import pytest
import torch

from kernels.pto.moe.moe import _select_routes, moe
from pymodel.common.case import Case, OutputCheck
from pymodel.isa.specialize import specialize
from pymodel.isa.torch_codegen import torch_codegen
from pymodel.ref.moe import moe as moe_ref
from pymodel.task_gen.validate import run_case


def _inputs(hidden=17, intermediate=19, experts=12, tokens=4):
    generator = torch.Generator().manual_seed(81)
    x = torch.randn(2, tokens // 2, hidden, generator=generator, dtype=torch.bfloat16)
    x[..., 0] = 1
    weights = {
        "router": torch.randn(experts, hidden, generator=generator, dtype=torch.bfloat16),
        "shared_gate": torch.randn(1, hidden, generator=generator, dtype=torch.bfloat16),
    }
    weights["router"][-2:] = 0
    weights["router"][-2:, 0] = -64
    scales = {}
    for name, n, k in (("gate", intermediate, hidden), ("up", intermediate, hidden),
                       ("down", hidden, intermediate)):
        weights[f"expert_{name}"] = torch.randint(-2, 3, (experts, n, k), generator=generator).to(
            torch.float8_e4m3fn)
        weights[f"shared_ffn_{name}"] = torch.randint(-2, 3, (n, k), generator=generator).to(
            torch.float8_e4m3fn)
        nblocks, kblocks = (n + 127) // 128, (k + 127) // 128
        scales[f"expert_{name}"] = ((torch.arange(experts * nblocks * kblocks) % 3 + 1) / 32).reshape(
            experts, nblocks, kblocks).to(torch.bfloat16)
        scales[f"shared_ffn_{name}"] = ((torch.arange(nblocks * kblocks) % 3 + 1) / 32).reshape(
            nblocks, kblocks).to(torch.bfloat16)
    return {"x": x} | {f"w_{key}": value for key, value in weights.items()} | {
        f"s_{key}": value for key, value in scales.items()}


def _reference(values, parameters):
    weights = {key.removeprefix("w_"): value for key, value in values.items() if key.startswith("w_")}
    scales = {key.removeprefix("s_"): value for key, value in values.items() if key.startswith("s_")}
    return moe_ref(values["x"], weights, scales)


def _tile_call(values, path):
    x = values["x"]
    hidden = x.shape[-1]
    rows = x.view(-1, hidden)
    tokens = rows.shape[0]
    experts = values["w_router"].shape[0]
    outputs = {
        "out": torch.full_like(x, torch.nan),
        "expert_id": torch.full((tokens, 8), -1, dtype=torch.int64),
        "route_weight": torch.full((tokens, 8), torch.nan, dtype=torch.bfloat16),
        "bucket_offset": torch.full((experts + 1,), -1, dtype=torch.int64),
        "bucket_token_id": torch.full((tokens * 8,), -1, dtype=torch.int64),
        "bucket_slot": torch.full((tokens * 8,), -1, dtype=torch.int64),
    }
    names = ("router", "shared_gate", "expert_gate", "expert_up", "expert_down",
             "shared_ffn_gate", "shared_ffn_up", "shared_ffn_down")
    fp8_names = names[2:]
    args = (rows, *(values[f"w_{name}"] for name in names),
            *(values[f"s_{name}"] for name in fp8_names), outputs["out"].view(-1, hidden),
            *(outputs[name] for name in tuple(outputs)[1:]))
    program = specialize(moe, *args)
    (path / "moe_ir.py").write_text(program.as_python())
    code = torch_codegen(program, check_shapes=True)
    (path / "moe_torch.py").write_text(code)
    namespace = {}
    exec(compile(code, str(path / "moe_torch.py"), "exec"), namespace)
    namespace["moe"](*args)
    return outputs


@pytest.mark.parametrize("mode", ["routed", "ties", "group_tail"])
def test_moe_complete_routes_and_buckets(mode, tmp_path):
    values = _inputs(hidden=129, intermediate=131, experts=10, tokens=2) if mode == "group_tail" else _inputs()
    if mode == "ties":
        values["w_router"].zero_()
    tokens = values["x"].numel() // values["x"].shape[-1]
    case = Case(
        case_id=f"moe_{mode}", seed=81, numerical_profile="pto-functional-v0",
        op_id="OP-07", kernel_id="K-07",
        outputs={"out": OutputCheck(atol=1e-2, rtol=1e-2),
                 "route_weight": OutputCheck(atol=1e-2, rtol=1e-2),
                 **{name: OutputCheck() for name in ("expert_id", "bucket_offset", "bucket_token_id", "bucket_slot")}},
    )
    root = Path(__file__).resolve().parents[2]
    upstream = json.loads((root / "data/upstream.lock.json").read_text())
    result = run_case(
        case, inputs=values, reference=_reference,
        implementation=lambda inputs, parameters: _tile_call(inputs, tmp_path),
        backend="pypto_torch_codegen_adapted", artifact_root=tmp_path / "runs",
        backend_details={"pypto_commit": upstream["pypto"]["commit"], "device": "cpu",
                         "specialization": "pymodel.isa.specialize", "programs": str(tmp_path)},
    )
    assert result.passed, result.report
    actual = torch.load(result.run_dir / "actual.pt", weights_only=True)
    offsets = actual["bucket_offset"]
    assert offsets[-1] == tokens * 8
    assert (offsets[1:] == offsets[:-1]).any()  # At least one empty expert.
    assert (offsets[1:] - offsets[:-1]).max() > 1  # A bucket contains multiple Tokens.
    pairs = actual["bucket_token_id"] * 8 + actual["bucket_slot"]
    assert torch.equal(pairs.sort().values, torch.arange(tokens * 8))  # Every contribution exactly once.
    if mode == "ties":
        assert torch.equal(actual["expert_id"], torch.arange(8).expand(tokens, -1))
        torch.testing.assert_close(actual["route_weight"], torch.full((tokens, 8), 0.125, dtype=torch.bfloat16),
                                   atol=0, rtol=0)


def test_top8_threshold_ties_across_32_experts(tmp_path):
    probabilities = torch.full((1, 36), 0.01, dtype=torch.float32)
    probabilities[0, 31], probabilities[0, 35] = 0.2, 0.3
    before = probabilities.clone()
    expert_id = torch.full((1, 8), -1, dtype=torch.int64)
    route_weight = torch.full((1, 8), torch.nan, dtype=torch.bfloat16)
    program = specialize(_select_routes, probabilities, expert_id, route_weight)
    (tmp_path / "top8_ir.py").write_text(program.as_python())
    code = torch_codegen(program, check_shapes=True)
    (tmp_path / "top8_torch.py").write_text(code)
    namespace = {}
    exec(compile(code, "<top8>", "exec"), namespace)
    namespace["_select_routes"](probabilities, expert_id, route_weight)
    expected_ids = torch.tensor([[35, 31, 0, 1, 2, 3, 4, 5]])
    assert torch.equal(expert_id, expected_ids)
    selected = probabilities.gather(1, expected_ids)
    torch.testing.assert_close(route_weight, (selected / selected.sum(-1, keepdim=True)).bfloat16(),
                               atol=0, rtol=0)
    torch.testing.assert_close(probabilities, before, atol=0, rtol=0)
