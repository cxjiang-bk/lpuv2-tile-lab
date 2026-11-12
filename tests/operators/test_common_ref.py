"""OP-03/04 contracts and a real reference check through the shared runner."""

from pathlib import Path

import pytest
import torch

from pymodel.common.case import load_case
from pymodel.ref.common import layer_norm, residual_add
from pymodel.task_gen.validate import run_case


@pytest.mark.parametrize("shape", [(1, 17), (3, 1152), (2, 2, 1152)])
def test_layer_norm_matches_affine_torch_module_at_model_width(shape):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
    module = torch.nn.LayerNorm(shape[-1], eps=1e-6, dtype=torch.bfloat16).requires_grad_(False)
    module.weight.copy_(torch.randn(shape[-1], generator=generator, dtype=torch.bfloat16))
    module.bias.copy_(torch.randn(shape[-1], generator=generator, dtype=torch.bfloat16))
    actual = layer_norm(x, module.weight, module.bias)
    torch.testing.assert_close(actual, module(x), atol=0, rtol=0)
    assert actual.dtype == torch.bfloat16


def test_layer_norm_constant_rows_return_bias_and_do_not_mutate_input():
    x = torch.full((3, 1152), 2.0, dtype=torch.bfloat16)
    original = x.clone()
    weight, bias = torch.ones(1152, dtype=x.dtype), torch.full((1152,), 0.25, dtype=x.dtype)
    torch.testing.assert_close(layer_norm(x, weight, bias), bias.expand_as(x), atol=0, rtol=0)
    assert torch.equal(x, original)


def test_residual_reference_uses_the_existing_case_and_comparison_runner(tmp_path):
    root = Path(__file__).resolve().parents[2]
    case = load_case(root / "cases/examples/op04-residual-bf16.json")
    generator = case.generator()
    shape = (case.parameters["rows"], case.parameters["cols"])
    inputs = {"x": torch.randn(shape, generator=generator, dtype=torch.bfloat16),
              "residual": torch.randn(shape, generator=generator, dtype=torch.bfloat16)}
    result = run_case(
        case, inputs=inputs,
        reference=lambda t, p: {"out": torch.add(t["residual"], t["x"])},
        implementation=lambda t, p: {"out": residual_add(t["residual"], t["x"])},
        backend="pytorch_reference_check", artifact_root=tmp_path,
    )
    assert result.passed, result.report


def test_residual_rejects_broadcast_and_dtype_promotion():
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="same shape"):
        residual_add(x, x[:1])
    with pytest.raises(ValueError, match="dtype"):
        residual_add(x, x.float())


def test_layer_norm_rejects_wrong_axis_parameters():
    with pytest.raises(ValueError, match="last input dimension"):
        layer_norm(torch.ones(2, 4, dtype=torch.bfloat16), torch.ones(2), torch.zeros(2))
