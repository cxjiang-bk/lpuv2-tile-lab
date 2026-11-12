"""Small direct checks for K-14, K-15 and K-16."""

import torch

from pymodel.ref.output import feature_concat, lm_head, mtp_input_projection


def test_lm_head_selects_per_request_rows_in_given_order():
    hidden = torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=torch.bfloat16)
    weight = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.bfloat16)
    positions = torch.tensor([[2, 0, 2], [0, 1, 0]], dtype=torch.int64)
    expected = torch.tensor([[[5, 6, 11], [1, 2, 3], [5, 6, 11]],
                             [[7, 8, 15], [9, 10, 19], [7, 8, 15]]], dtype=torch.bfloat16)
    torch.testing.assert_close(lm_head(hidden, weight, positions), expected, atol=0, rtol=0)


def test_feature_concat_places_embedding_before_hidden():
    e_norm = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.bfloat16)
    h_norm = torch.tensor([[[10, 20], [30, 40]]], dtype=torch.bfloat16)
    expected = torch.tensor([[[1, 2, 10, 20], [3, 4, 30, 40]]], dtype=torch.bfloat16)
    torch.testing.assert_close(feature_concat(e_norm, h_norm), expected, atol=0, rtol=0)


def test_mtp_input_projection_uses_independent_fc_weight():
    eh = torch.tensor([[1, 2, 10, 20], [3, 4, 30, 40]], dtype=torch.bfloat16)
    fc_weight = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bfloat16)
    expected = torch.tensor([[11, 22], [33, 44]], dtype=torch.bfloat16)
    torch.testing.assert_close(mtp_input_projection(eh, fc_weight), expected, atol=0, rtol=0)
