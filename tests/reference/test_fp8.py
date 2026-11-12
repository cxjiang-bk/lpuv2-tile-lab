"""Functional-v0 FP8 payload, scale, grouping and projection checks."""

import pytest
import torch

from pymodel.ref.linear import fp8_linear
from pymodel.ref.quantization import dynamic_quantize


@pytest.mark.parametrize("width", [127, 128, 129, 255, 256, 257])
def test_quantize_group_boundaries_and_per_token_scales(width):
    groups = (width + 127) // 128
    x = torch.empty(2, width, dtype=torch.bfloat16)
    for group in range(groups):
        x[0, group * 128:(group + 1) * 128] = 448 * (group + 1)
        x[1, group * 128:(group + 1) * 128] = -896 * (group + 1)
    before = x.clone()
    payload, scale = dynamic_quantize(x)
    expected_scale = torch.tensor([[g + 1 for g in range(groups)], [2 * (g + 1) for g in range(groups)]],
                                  dtype=torch.float32)
    assert payload.shape == x.shape and payload.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, groups) and scale.dtype == torch.float32
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
    torch.testing.assert_close(payload.float(), torch.tensor([[448.] * width, [-448.] * width]), atol=0, rtol=0)
    assert torch.equal(x, before)


def test_quantize_zero_groups_and_zero_tail():
    x = torch.zeros(2, 257, dtype=torch.bfloat16)
    x[0, 0] = 224
    x[1, 128] = -896
    payload, scale = dynamic_quantize(x)
    torch.testing.assert_close(scale, torch.tensor([[0.5, 1, 1], [1, 2, 1.]]), atol=0, rtol=0)
    expected = torch.zeros(2, 257)
    expected[0, 0], expected[1, 128] = 448, -448
    torch.testing.assert_close(payload.float(), expected, atol=0, rtol=0)


def test_quantize_known_nearest_even_bytes_including_subnormals():
    # amax=448 fixes scale=1. These are exact BF16 midpoints between FP8 values.
    x = torch.tensor([[448, 1.0625, 1.1875, 1.3125, -1.0625, 2**-10, 3 * 2**-10, 0]],
                     dtype=torch.bfloat16)
    payload, scale = dynamic_quantize(x)
    expected_bytes = torch.tensor([[0x7E, 0x38, 0x3A, 0x3A, 0xB8, 0, 2, 0]], dtype=torch.uint8)
    assert torch.equal(payload.view(torch.uint8), expected_bytes)
    torch.testing.assert_close(scale, torch.ones(1, 1), atol=0, rtol=0)


def test_quantize_noncontiguous_logical_region_excludes_poison_padding():
    parent = torch.full((2, 132), torch.nan, dtype=torch.bfloat16)
    x = parent[:, 1:131:2]
    x.fill_(224)
    payload, scale = dynamic_quantize(x)
    torch.testing.assert_close(scale, torch.full((2, 1), 0.5), atol=0, rtol=0)
    torch.testing.assert_close(payload.float(), torch.full((2, 65), 448.), atol=0, rtol=0)


def test_quantize_preserves_leading_dimensions():
    payload, scale = dynamic_quantize(torch.zeros(2, 3, 129, dtype=torch.bfloat16))
    assert payload.shape == (2, 3, 129)
    torch.testing.assert_close(scale, torch.ones(2, 3, 2), atol=0, rtol=0)


@pytest.mark.parametrize("value", [torch.nan, torch.inf, -torch.inf])
def test_quantize_rejects_nonfinite_input(value):
    with pytest.raises(ValueError, match="finite"):
        dynamic_quantize(torch.tensor([[value]], dtype=torch.bfloat16))


@pytest.mark.parametrize("x,error", [
    (torch.ones(2, 3), TypeError),
    (torch.tensor(1, dtype=torch.bfloat16), ValueError),
    (torch.empty(2, 0, dtype=torch.bfloat16), ValueError),
])
def test_quantize_rejects_wrong_contract(x, error):
    with pytest.raises(error):
        dynamic_quantize(x)


def test_quantize_random_payload_and_scales_match_explicit_groups():
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(2, 257, generator=generator, dtype=torch.bfloat16)
    expected_payload = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    expected_scale = torch.empty(2, 3, dtype=torch.float32)
    for token in range(2):
        for group, start in enumerate(range(0, 257, 128)):
            values = x[token, start:start + 128].float()
            scale = values.abs().max() / 448
            expected_scale[token, group] = scale
            expected_payload[token, start:start + 128] = (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    payload, scale = dynamic_quantize(x)
    assert torch.equal(payload.view(torch.uint8), expected_payload.view(torch.uint8))
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)


def test_fp8_linear_known_block_scales_k_tail_n_tail_and_cancellation():
    x = torch.empty(2, 257, dtype=torch.bfloat16)
    x[0, :128], x[0, 128:256], x[0, 256] = 1, 2, 4
    x[1, :128], x[1, 128:256], x[1, 256] = -2, 1, -1
    weight = torch.ones(129, 257).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.25, 0.5, 2], [1, 2, 0.25]], dtype=torch.bfloat16)
    expected = torch.empty(2, 129, dtype=torch.bfloat16)
    expected[0, :128], expected[0, 128] = 168, 641
    expected[1, :128], expected[1, 128] = -2, -0.25
    torch.testing.assert_close(fp8_linear(x, weight, scale), expected, atol=0, rtol=0)


@pytest.mark.parametrize("n,k", [(5, 7), (129, 129), (130, 257)])
@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32])
def test_fp8_linear_matches_independent_dequantized_math(n, k, scale_dtype):
    generator = torch.Generator().manual_seed(37)
    x = torch.randn(2, 2, k, generator=generator, dtype=torch.bfloat16)
    weight = torch.randint(-3, 4, (n, k), generator=generator).to(torch.float8_e4m3fn)
    scale = (torch.arange(((n + 127) // 128) * ((k + 127) // 128), dtype=torch.float32) + 1)
    scale = (scale / 32).reshape((n + 127) // 128, (k + 127) // 128).to(scale_dtype)
    before_x, before_weight, before_scale = x.clone(), weight.view(torch.uint8).clone(), scale.clone()

    # A different expression: quantize each explicit Token/group, dequantize
    # both complete matrices in FP64, then take one full mathematical product.
    dequant_x = torch.empty(4, k, dtype=torch.float64)
    for token, row in enumerate(x.reshape(4, k)):
        for start in range(0, k, 128):
            values = row[start:start + 128].float()
            amax = values.abs().max()
            activation_scale = amax / 448 if amax else torch.tensor(1.)
            q = (values / activation_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            dequant_x[token, start:start + 128] = q.double() * activation_scale.double()
    scale_per_element = scale.double()[torch.arange(n)[:, None] // 128, torch.arange(k)[None, :] // 128]
    dequant_weight = weight.double() * scale_per_element
    expected = (dequant_x @ dequant_weight.T).reshape(2, 2, n).to(torch.bfloat16)
    actual = fp8_linear(x, weight, scale)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    assert torch.equal(x, before_x)
    assert torch.equal(weight.view(torch.uint8), before_weight)
    assert torch.equal(scale, before_scale)


@pytest.mark.parametrize("zero_operand", ["x", "weight"])
def test_fp8_linear_zero_operand(zero_operand):
    x = torch.ones(2, 129, dtype=torch.bfloat16)
    weight_values = torch.ones(3, 129)
    (x if zero_operand == "x" else weight_values).zero_()
    actual = fp8_linear(x, weight_values.to(torch.float8_e4m3fn), torch.tensor([[0.5, 2.]]))
    torch.testing.assert_close(actual, torch.zeros(2, 3, dtype=torch.bfloat16), atol=0, rtol=0)


@pytest.mark.parametrize("x,weight,scale,error", [
    (torch.ones(2, 3), torch.ones(4, 3).to(torch.float8_e4m3fn), torch.ones(1, 1), TypeError),
    (torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(4, 3), torch.ones(1, 1), TypeError),
    (torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(4, 2).to(torch.float8_e4m3fn),
     torch.ones(1, 1), ValueError),
    (torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(4, 3).to(torch.float8_e4m3fn),
     torch.ones(4, 1), ValueError),
    (torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(4, 3).to(torch.float8_e4m3fn),
     torch.ones(1, 1, dtype=torch.int64), TypeError),
])
def test_fp8_linear_rejects_wrong_contract(x, weight, scale, error):
    with pytest.raises(error):
        fp8_linear(x, weight, scale)


@pytest.mark.parametrize("scale_value", [0, -1, torch.nan, torch.inf])
def test_fp8_linear_rejects_invalid_scales(scale_value):
    with pytest.raises(ValueError, match="scale"):
        fp8_linear(torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(4, 3).to(torch.float8_e4m3fn),
                   torch.tensor([[scale_value]], dtype=torch.float32))


def test_fp8_linear_rejects_nan_weight():
    with pytest.raises(ValueError, match="weight"):
        fp8_linear(torch.ones(2, 3, dtype=torch.bfloat16), torch.full((4, 3), torch.nan).to(torch.float8_e4m3fn),
                   torch.ones(1, 1))


@pytest.mark.parametrize("shape", [(129,), (0, 129)])
def test_fp8_linear_vector_and_empty_batch_shapes(shape):
    x = torch.ones(shape, dtype=torch.bfloat16)
    weight = torch.ones(3, 129).to(torch.float8_e4m3fn)
    expected = torch.full((*shape[:-1], 3), 129, dtype=torch.bfloat16)
    torch.testing.assert_close(fp8_linear(x, weight, torch.ones(1, 2)), expected, atol=0, rtol=0)


def test_fp8_linear_noncontiguous_views():
    # [1,2] and [2,4] are exact after quantization; outside columns are ignored.
    x = torch.tensor([[99, 1, 99, 2, 99, 99], [99, 2, 99, 4, 99, 99]], dtype=torch.bfloat16)[:, 1:5:2]
    weight = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1.]]).to(torch.float8_e4m3fn)[:, 1::2]
    expected = x.clone()
    torch.testing.assert_close(fp8_linear(x, weight, torch.ones(1, 1)), expected, atol=0, rtol=0)


def test_fp8_linear_retains_dynamic_quantization_error():
    x = torch.tensor([[1, 3]], dtype=torch.bfloat16)
    weight = torch.eye(2).to(torch.float8_e4m3fn)
    # scale=3/448; 1/scale rounds to FP8 144, then returns BF16 0.96484375.
    expected = torch.tensor([[0.96484375, 3]], dtype=torch.bfloat16)
    actual = fp8_linear(x, weight, torch.ones(1, 1))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(actual, x)
