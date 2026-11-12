"""Position metadata generation against pinned vision_utils and resized tables."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from pymodel.common.case import load_case
from pymodel.ref.vision import vision_position_embedding, vision_position_interpolation
from pymodel.task_gen.validate import run_case


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("grids,merge", [
    ([[1, 4, 4]], 2),
    ([[2, 4, 6], [1, 2, 8]], 2),
    ([[1, 48, 48]], 2),
    ([[1, 50, 52]], 2),
    ([[1, 2, 2]], 2),
    ([[1, 1, 5], [2, 3, 1]], 1),
    ([[1, 6, 6]], 3),
])
def test_interpolation_matches_upstream_fp32_and_merge_order(upstream, grids, merge, dtype):
    source = upstream("_interpolation_axis_taps_weights", "get_vision_interpolation_indices_and_weights",
                      source_file="vision_utils.py")
    grid = torch.tensor(grids, dtype=dtype)
    indices, weights = vision_position_interpolation(grid, spatial_merge_size=merge)
    expected_indices, expected_weights = source.get_vision_interpolation_indices_and_weights(
        grid, 48, mode="bilinear", align_corners=True, spatial_merge_size=merge,
    )
    torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)
    torch.testing.assert_close(weights, expected_weights, atol=0, rtol=0)
    assert indices.dtype == torch.int64 and weights.dtype == torch.float32
    assert indices.device == weights.device == grid.device
    assert indices.shape == (sum(t*h*w for t, h, w in grids), 4)
    torch.testing.assert_close(weights.sum(-1), torch.ones(len(weights)), atol=1e-7, rtol=0)


def test_four_by_four_example_includes_merge_groups_and_clamped_border():
    indices, weights = vision_position_interpolation(torch.tensor([[1, 4, 4]]))
    # Processor order starts (0,0), (0,1), (1,0), (1,1), then the next 2x2 block.
    assert indices[3].tolist() == [735, 736, 783, 784]
    torch.testing.assert_close(weights[3], torch.tensor([1/9, 2/9, 2/9, 4/9]), atol=1e-6, rtol=0)
    assert indices[0].tolist() == [0, 1, 48, 49]
    assert indices[-1].tolist() == [2303, 2303, 2303, 2303]
    torch.testing.assert_close(weights[[0, -1]], torch.tensor([[1., 0., 0., 0.]]).expand(2, 4), atol=0, rtol=0)


def test_equal_input_and_table_grids_select_each_position_in_merge_order():
    indices, weights = vision_position_interpolation(torch.tensor([[1, 4, 4]]), table_side=4)
    assert indices[:, 0].tolist() == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    torch.testing.assert_close(weights, torch.tensor([[1., 0., 0., 0.]]).expand(16, 4), atol=0, rtol=0)


def test_video_slices_and_equal_image_grids_reuse_the_same_metadata():
    indices, weights = vision_position_interpolation(torch.tensor([[2, 4, 6], [1, 4, 6]]))
    for start in (24, 48):
        assert torch.equal(indices[:24], indices[start:start+24])
        assert torch.equal(weights[:24], weights[start:start+24])


def test_generated_metadata_connects_to_position_embedding_like_resizing_the_learned_table():
    generator = torch.Generator().manual_seed(17)
    table = torch.randn(48*48, 8, generator=generator, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 4, 6]])
    indices, weights = vision_position_interpolation(grid)
    hidden = torch.randn(24, 8, generator=generator, dtype=torch.bfloat16)
    actual = vision_position_embedding(hidden, table, indices, weights)
    resized = F.interpolate(table.T.float().reshape(1, 8, 48, 48), size=(4, 6),
                             mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
    # Convert raster order to the processor's block-row/block-col/in-row/in-col order.
    expected_position = resized.reshape(2, 2, 3, 2, 8).permute(0, 2, 1, 3, 4).reshape(24, 8)
    torch.testing.assert_close(actual, hidden + expected_position.to(hidden.dtype), atol=0, rtol=0)


def test_interpolation_case_checks_both_named_outputs_through_the_shared_runner(upstream, tmp_path):
    root = Path(__file__).resolve().parents[2]
    case = load_case(root / "cases/operators/vision-position-interpolation.json")
    source = upstream("_interpolation_axis_taps_weights", "get_vision_interpolation_indices_and_weights",
                      source_file="vision_utils.py")

    def reference(tensors, parameters):
        indices, weights = source.get_vision_interpolation_indices_and_weights(
            tensors["grid_thw"], parameters["table_side"], mode="bilinear", align_corners=True,
            spatial_merge_size=parameters["spatial_merge_size"],
        )
        return {"interp_indices": indices, "interp_weights": weights}

    def implementation(tensors, parameters):
        indices, weights = vision_position_interpolation(
            tensors["grid_thw"], parameters["table_side"], parameters["spatial_merge_size"],
        )
        return {"interp_indices": indices, "interp_weights": weights}

    result = run_case(case, inputs={"grid_thw": torch.tensor(case.parameters["grids"])},
                      reference=reference, implementation=implementation, backend="pytorch_reference_check",
                      artifact_root=tmp_path)
    assert result.passed, result.report


def test_empty_input_and_single_position_table_have_defined_outputs():
    indices, weights = vision_position_interpolation(torch.empty((0, 3), dtype=torch.int64))
    assert indices.shape == weights.shape == (0, 4)
    assert indices.dtype == torch.int64 and weights.dtype == torch.float32
    indices, weights = vision_position_interpolation(torch.tensor([[1, 2, 2]]), table_side=1)
    assert not torch.count_nonzero(indices)
    torch.testing.assert_close(weights, torch.tensor([[1., 0., 0., 0.]]).expand(4, 4), atol=0, rtol=0)


def test_interpolation_weights_stay_fp32_with_a_different_default_float_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        indices, weights = vision_position_interpolation(torch.tensor([[1, 4, 4]]))
    finally:
        torch.set_default_dtype(previous)
    assert indices.dtype == torch.int64 and weights.dtype == torch.float32


@pytest.mark.parametrize("grids", [[[0, 4, 4]], [[1, -2, 4]], [[1, 3, 4]], [[1, 4, 5]]])
def test_invalid_patch_counts_or_incomplete_merge_groups_are_rejected(grids):
    with pytest.raises(ValueError):
        vision_position_interpolation(torch.tensor(grids))


def test_grid_shape_and_integral_dtype_are_required():
    with pytest.raises(ValueError, match="shape"):
        vision_position_interpolation(torch.tensor([1, 4, 4]))
    with pytest.raises(TypeError, match="int32 or int64"):
        vision_position_interpolation(torch.tensor([[1., 4., 4.]]))
