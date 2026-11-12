"""Placeholder association, batch flattening and preservation of text rows."""

import pytest
import torch

from pymodel.ref.multimodal import visual_feature_scatter


def test_scatter_matches_masked_scatter_for_image_and_video_rows_across_batches():
    x = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    original = x.clone()
    features = torch.tensor([[100., 101., 102., 103.], [200., 201., 202., 203.], [300., 301., 302., 303.]])
    rows = torch.tensor([1, 3, 5])
    mask = torch.zeros(6, dtype=torch.bool)
    mask[rows] = True
    mask = mask.reshape(2, 3, 1).expand_as(x)
    expected = x.masked_scatter(mask, features.to(x.dtype))
    actual = visual_feature_scatter(x, features, rows)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(x, original)


def test_no_visual_tokens_preserves_the_text_without_aliasing():
    x = torch.ones(2, 3, 4, dtype=torch.bfloat16)
    actual = visual_feature_scatter(x, x.new_empty(0, 4), torch.empty(0, dtype=torch.int64))
    assert torch.equal(actual, x) and actual.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("rows", [[1, 1], [-1, 2], [0, 6]])
def test_invalid_placeholder_associations_are_rejected(rows):
    with pytest.raises(ValueError):
        visual_feature_scatter(torch.zeros(2, 3, 4, dtype=torch.bfloat16),
                               torch.ones(2, 4, dtype=torch.bfloat16), torch.tensor(rows))
