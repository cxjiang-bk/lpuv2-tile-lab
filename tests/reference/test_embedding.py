"""OP-01 Golden: direct Tensor calls and exact row-lookup checks."""

import pytest
import torch

from pymodel.ref.embedding import embedding


def test_known_rows_keep_request_order_and_repeated_ids():
    weight = torch.tensor([[1, -2, 3], [4, 5, -6], [-7, 8, 9]], dtype=torch.bfloat16)
    token_ids = torch.tensor([[2, 0, 2], [1, 2, 0]], dtype=torch.int64)
    before_weight, before_ids = weight.clone(), token_ids.clone()
    out = embedding(token_ids, weight)
    expected = torch.tensor([
        [[-7, 8, 9], [1, -2, 3], [-7, 8, 9]],
        [[4, 5, -6], [-7, 8, 9], [1, -2, 3]],
    ], dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert torch.equal(weight, before_weight)
    assert torch.equal(token_ids, before_ids)
    with torch.inference_mode():
        out.fill_(0)
    assert torch.equal(weight, before_weight)


def test_logical_slices_use_only_supplied_rows_and_columns():
    weight_parent = torch.tensor([
        [900, 900, 900, 900, 900],
        [901, 1, 2, 3, 901],
        [902, 4, 5, 6, 902],
        [903, 7, 8, 9, 903],
        [904, 904, 904, 904, 904],
    ], dtype=torch.bfloat16)
    weight = weight_parent[1:4, 1:4]
    ids_parent = torch.tensor([[99, 2, 0, 1, 99], [99, 1, 2, 0, 99]], dtype=torch.int64)
    token_ids = ids_parent[:, 1:4]
    expected = torch.tensor([
        [[7, 8, 9], [1, 2, 3], [4, 5, 6]],
        [[4, 5, 6], [7, 8, 9], [1, 2, 3]],
    ], dtype=torch.bfloat16)
    assert not weight.is_contiguous() and not token_ids.is_contiguous()
    torch.testing.assert_close(embedding(token_ids, weight), expected, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(2, 0), (0, 3)])
def test_empty_logical_input_has_empty_output(shape):
    out = embedding(torch.empty(shape, dtype=torch.int64), torch.zeros(3, 4, dtype=torch.bfloat16))
    assert out.shape == (*shape, 4)
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("bad_id", [-1, 3])
def test_invalid_token_id_fails_instead_of_wrapping(bad_id):
    with pytest.raises(ValueError, match="token_ids.*range"):
        embedding(torch.tensor([[bad_id]], dtype=torch.int64), torch.zeros(3, 2, dtype=torch.bfloat16))


@pytest.mark.parametrize("ids,weight,error,match", [
    (torch.zeros(1, 1, dtype=torch.int32), torch.zeros(3, 2, dtype=torch.bfloat16),
     TypeError, "token_ids.*int64"),
    (torch.zeros(1, 1, dtype=torch.int64), torch.zeros(3, 2, dtype=torch.float32),
     TypeError, "embedding_weight.*bfloat16"),
    (torch.zeros(1, dtype=torch.int64), torch.zeros(3, 2, dtype=torch.bfloat16),
     ValueError, "token_ids.*\\[B, S\\]"),
    (torch.zeros(1, 1, dtype=torch.int64), torch.zeros(3, dtype=torch.bfloat16),
     ValueError, "embedding_weight.*\\[V, H\\]"),
    (torch.zeros(1, 1, dtype=torch.int64), torch.zeros(0, 2, dtype=torch.bfloat16),
     ValueError, "V and H.*positive"),
    (torch.zeros(1, 1, dtype=torch.int64), torch.zeros(3, 0, dtype=torch.bfloat16),
     ValueError, "V and H.*positive"),
])
def test_invalid_input_contract_fails(ids, weight, error, match):
    with pytest.raises(error, match=match):
        embedding(ids, weight)


@pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (2, 4)])
def test_random_rows_match_direct_indexing(batch_size, seq_len):
    generator = torch.Generator().manual_seed(17)
    token_ids = torch.randint(7, (batch_size, seq_len), generator=generator, dtype=torch.int64)
    weight = torch.randn(7, 5, generator=generator, dtype=torch.bfloat16)
    expected = weight[token_ids]
    actual = embedding(token_ids, weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
