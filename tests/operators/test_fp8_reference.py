"""Known E4M3FN payloads and per-K-block scaling, independent of kernels."""

import pytest
import torch

from pymodel.ref.fp8 import activation_quantize, linear


def test_zero_tail_signed_zero_and_midpoint_payloads():
    x = torch.zeros((2, 129), dtype=torch.bfloat16)
    x[1, :8] = torch.tensor([448., -448., 432., -432., 400., -400., 0., -0.])
    actual = activation_quantize(x)
    assert torch.equal(actual['scale'], torch.ones(2, 2))
    assert actual['payload'].shape == x.shape
    assert actual['payload'][1, :8].view(torch.uint8).tolist() == [126, 254, 126, 254, 124, 252, 0, 128]
    assert not bool(actual['payload'][0].view(torch.uint8).any())


def test_linear_scales_each_k_group_and_each_n_block():
    x = torch.ones((2, 129), dtype=torch.bfloat16)
    x[:, -1] = 2
    weight = torch.ones((129, 129)).to(torch.float8_e4m3fn)
    scales = torch.tensor([[.5, .25], [.125, .0625]], dtype=torch.bfloat16)
    out = linear(x, weight, scales)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out[:, :128], torch.full((2, 128), 64.5, dtype=torch.bfloat16))
    assert torch.equal(out[:, 128:], torch.full((2, 1), 16.125, dtype=torch.bfloat16))


def test_quantization_rejects_nonfinite_input():
    with pytest.raises(ValueError, match='finite BF16'):
        activation_quantize(torch.tensor([[float('nan')]], dtype=torch.bfloat16))
