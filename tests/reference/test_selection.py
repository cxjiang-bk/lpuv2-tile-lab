"""Functional-v0 Top-K: descending score, ascending expert ID for exact ties."""

import pytest
import torch

from pymodel.ref.selection import stable_topk


def test_topk_ties_keep_ascending_original_expert_ids():
    scores = torch.tensor([[0.2, 0.8, 0.8, 0.8, 0.1], [1, 1, 1, 1, 1]], dtype=torch.float32)
    values, indices = stable_topk(scores, 3)
    assert indices.dtype == torch.int64
    assert torch.equal(indices, torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.int64))
    torch.testing.assert_close(values, torch.tensor([[0.8, 0.8, 0.8], [1, 1, 1.]]), atol=0, rtol=0)


def test_topk_bf16_rounding_can_create_a_tie():
    scores = torch.tensor([[1.001, 1.002, 0.5]], dtype=torch.bfloat16)
    values, indices = stable_topk(scores, 2)
    assert values.dtype == torch.bfloat16
    assert torch.equal(indices, torch.tensor([[0, 1]], dtype=torch.int64))


def test_top8_of_256_tied_experts():
    values, indices = stable_topk(torch.ones(2, 256, dtype=torch.float32), 8)
    assert torch.equal(indices, torch.arange(8).expand(2, 8))
    torch.testing.assert_close(values, torch.ones(2, 8), atol=0, rtol=0)


def test_topk_random_values_and_ids_match_python_sort():
    generator = torch.Generator().manual_seed(43)
    scores = torch.randint(-2, 3, (2, 3, 11), generator=generator).float()
    before = scores.clone()
    expected_ids = torch.tensor([
        sorted(range(11), key=lambda expert: (-row[expert].item(), expert))[:4]
        for row in scores.reshape(-1, 11)
    ], dtype=torch.int64).reshape(2, 3, 4)
    values, indices = stable_topk(scores, 4)
    assert torch.equal(indices, expected_ids)
    torch.testing.assert_close(values, torch.gather(scores, -1, expected_ids), atol=0, rtol=0)
    assert torch.equal(scores, before)


@pytest.mark.parametrize("k", [0, -1, 4, 1.5, True])
def test_topk_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="k"):
        stable_topk(torch.ones(2, 3), k)


@pytest.mark.parametrize("value", [torch.nan, torch.inf, -torch.inf])
def test_topk_rejects_nonfinite_scores(value):
    with pytest.raises(ValueError, match="finite"):
        stable_topk(torch.tensor([[value, 1.]]), 1)
