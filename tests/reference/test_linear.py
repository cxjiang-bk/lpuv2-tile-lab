"""Common BF16 projections: logical [N,K] weights and source dtype boundaries."""

import pytest
import torch

from pymodel.ref.linear import bf16_linear, fp8_linear
from pymodel.ref.quantization import dynamic_quantize


def test_bf16_linear_selects_features_and_adds_bias():
    x = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.bfloat16)
    weight = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.bfloat16)
    bias = torch.tensor([2, -1], dtype=torch.bfloat16)
    expected = torch.tensor([[[5, 0], [8, 3]]], dtype=torch.bfloat16)
    torch.testing.assert_close(bf16_linear(x, weight, bias), expected, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 3, 5)])
def test_bf16_linear_matches_explicit_inner_products(shape):
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(4, 5, generator=generator, dtype=torch.bfloat16)
    expected = (x.float().unsqueeze(-2) * weight.float()).sum(-1).to(torch.bfloat16)
    actual = bf16_linear(x, weight)
    assert actual.shape == (*shape[:-1], 4)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("zero_operand", ["x", "weight"])
def test_bf16_linear_zero_operand(zero_operand):
    x, weight = torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(4, 3, dtype=torch.bfloat16)
    (x if zero_operand == "x" else weight).zero_()
    torch.testing.assert_close(bf16_linear(x, weight), torch.zeros(2, 4, dtype=torch.bfloat16), atol=0, rtol=0)


def test_bf16_linear_noncontiguous_views_preserve_inputs():
    x = torch.arange(24, dtype=torch.bfloat16).reshape(2, 12)[:, 1:7:2]
    weight = torch.eye(3, dtype=torch.bfloat16).T
    before = x.clone()
    torch.testing.assert_close(bf16_linear(x, weight), before, atol=0, rtol=0)
    assert torch.equal(x, before)


@pytest.mark.parametrize("x,weight,bias,error", [
    (torch.ones(2, 3), torch.ones(4, 3, dtype=torch.bfloat16), None, TypeError),
    (torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(4, 3), None, TypeError),
    (torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(4, 2, dtype=torch.bfloat16), None, ValueError),
    (torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(4, 3, dtype=torch.bfloat16), torch.ones(4), TypeError),
    (torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(4, 3, dtype=torch.bfloat16),
     torch.ones(1, 4, dtype=torch.bfloat16), ValueError),
])
def test_bf16_linear_rejects_wrong_contract(x, weight, bias, error):
    with pytest.raises(error):
        bf16_linear(x, weight, bias)


def test_shared_projection_example_for_a():
    """A's GDN can call the shared projections before adding its core/state."""
    x = torch.stack([torch.ones(130), torch.full((130,), 2.)]).to(torch.bfloat16)
    qkv_weight = torch.ones(192, 130).to(torch.float8_e4m3fn)
    qkv_scale = torch.tensor([[0.5, 0.5], [0.25, 0.25]], dtype=torch.bfloat16)
    payload, activation_scale = dynamic_quantize(x)
    assert payload.shape == (2, 130) and activation_scale.shape == (2, 2)
    qkv = fp8_linear(x, qkv_weight, qkv_scale)
    expected = torch.empty(2, 192, dtype=torch.bfloat16)
    expected[0, :128], expected[0, 128:] = 65, 32.5
    expected[1, :128], expected[1, 128:] = 130, 65
    torch.testing.assert_close(qkv, expected, atol=0, rtol=0)

    a_b_weight = torch.zeros(4, 130, dtype=torch.bfloat16)
    a_b_weight[:, 0] = 1
    expected_a_b = torch.tensor([[1] * 4, [2] * 4], dtype=torch.bfloat16)
    torch.testing.assert_close(bf16_linear(x, a_b_weight), expected_a_b, atol=0, rtol=0)
