"""Ordinary Qwen RMSNorm and explicit FP32 Softmax boundaries."""

import math

import pytest
import torch
from torch.nn import functional as F

from pymodel.ref.normalization import rmsnorm, softmax


def test_rmsnorm_known_weight_and_epsilon():
    x = torch.tensor([[3, 4]], dtype=torch.bfloat16)
    weight = torch.tensor([1, 0.5], dtype=torch.bfloat16)
    expected = torch.full((1, 2), 6 / math.sqrt(13), dtype=torch.bfloat16)
    torch.testing.assert_close(rmsnorm(x, weight, eps=0.5), expected, atol=0, rtol=0)


def test_rmsnorm_keeps_normalization_and_weight_product_fp32_until_final_cast():
    x = torch.tensor([[1, 2, 3]], dtype=torch.bfloat16)
    weight = torch.tensor([0.1, 0.25, -0.5], dtype=torch.bfloat16)
    expected = torch.tensor([[0.5078125, 1.15625, 0.6953125]], dtype=torch.bfloat16)
    torch.testing.assert_close(rmsnorm(x, weight), expected, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(2, 256), (2, 3, 2048), (1, 2, 16, 256)])
def test_rmsnorm_matches_torch_normalization_with_zero_centered_weight(shape):
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], generator=generator, dtype=torch.bfloat16)
    expected = F.rms_norm(x.float(), (shape[-1],), weight=1 + weight.float(), eps=1e-6).to(torch.bfloat16)
    torch.testing.assert_close(rmsnorm(x, weight), expected, atol=1e-2, rtol=1e-2)


def test_rmsnorm_zero_rows_and_zero_stored_weight():
    x = torch.tensor([[0, 0], [2, 2], [-2, -2]], dtype=torch.bfloat16)
    expected = torch.tensor([[0, 0], [1, 1], [-1, -1]], dtype=torch.bfloat16)
    torch.testing.assert_close(rmsnorm(x, torch.zeros(2, dtype=torch.bfloat16)), expected, atol=0, rtol=0)


def test_rmsnorm_view_and_inputs_are_preserved():
    x = torch.arange(12, dtype=torch.bfloat16).reshape(2, 6)[:, 1:5:2]
    weight = torch.full((2,), -1, dtype=torch.bfloat16)
    before = x.clone()
    torch.testing.assert_close(rmsnorm(x, weight), torch.zeros_like(x), atol=0, rtol=0)
    assert torch.equal(x, before)
    assert torch.equal(weight, torch.full((2,), -1, dtype=torch.bfloat16))


@pytest.mark.parametrize("eps", [0, -1, math.nan, math.inf])
def test_rmsnorm_rejects_invalid_epsilon(eps):
    with pytest.raises(ValueError, match="eps"):
        rmsnorm(torch.ones(2, 3, dtype=torch.bfloat16), torch.zeros(3, dtype=torch.bfloat16), eps=eps)


@pytest.mark.parametrize("x,weight,error", [
    (torch.ones(2, 3), torch.zeros(3, dtype=torch.bfloat16), TypeError),
    (torch.ones(2, 3, dtype=torch.bfloat16), torch.zeros(3), TypeError),
    (torch.ones(2, 3, dtype=torch.bfloat16), torch.zeros(1, 3, dtype=torch.bfloat16), ValueError),
])
def test_rmsnorm_rejects_wrong_contract(x, weight, error):
    with pytest.raises(error):
        rmsnorm(x, weight)


def test_softmax_large_logits_stay_finite_and_fp32():
    x = torch.tensor([[10000, 10001, 9999], [0, 0, 0]], dtype=torch.float32)
    expected = torch.tensor([[1, math.e, 1 / math.e], [1, 1, 1]], dtype=torch.float64)
    expected /= expected.sum(dim=-1, keepdim=True)
    actual = softmax(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float(), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(softmax(x + 1000), actual, atol=1e-5, rtol=1e-4)


def test_softmax_bf16_input_and_negative_infinity_mask():
    x = torch.tensor([[1, -torch.inf, 1]], dtype=torch.bfloat16)
    expected = torch.tensor([[0.5, 0, 0.5]], dtype=torch.float32)
    torch.testing.assert_close(softmax(x), expected, atol=0, rtol=0)


def test_softmax_can_reduce_a_specified_axis_without_modifying_input():
    x = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    torch.testing.assert_close(softmax(x, dim=1), torch.full((2, 3, 4), 1 / 3), atol=1e-5, rtol=1e-4)
    assert not torch.count_nonzero(x)


@pytest.mark.parametrize("row", [[-torch.inf, -torch.inf], [0, torch.nan], [0, torch.inf]])
def test_softmax_rejects_rows_without_defined_probabilities(row):
    with pytest.raises(ValueError):
        softmax(torch.tensor([row], dtype=torch.float32))
