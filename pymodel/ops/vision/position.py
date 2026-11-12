"""Complete K-11: grid metadata through final visual position addition."""

from pathlib import Path

import torch

from kernels.pypto import vision_ops as tile_kernels
from pymodel.isa.hw_pypto import TensorProgram


def position_embedding(grid_thw: torch.Tensor, x: torch.Tensor, position_table: torch.Tensor, *,
                       table_side: int = 48, spatial_merge_size: int = 2, tile_cols: int = 128,
                       artifact_dir: str | Path | None = None) -> dict[str, torch.Tensor]:
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or grid_thw.dtype not in (torch.int32, torch.int64):
        raise ValueError("grid_thw must be an integer [M,3] tensor")
    if any(type(v) is not int or v < 1 for v in (table_side, spatial_merge_size, tile_cols)):
        raise ValueError("Table side, merge size and Tile width must be positive integers")
    if bool((grid_thw <= 0).any()) or bool((grid_thw[:, 1:] % spatial_merge_size != 0).any()):
        raise ValueError("Grid counts must be positive and contain complete spatial merge groups")
    if x.ndim != 2 or x.shape[1] < 1 or position_table.shape != (table_side * table_side, x.shape[1]):
        raise ValueError("Position table must match the square learned grid and hidden width")
    if x.shape[0] != int(grid_thw.to(torch.int64).prod(-1).sum()):
        raise ValueError("Packed Patch count must equal sum(t*h*w)")
    if x.dtype != torch.bfloat16 or position_table.dtype != torch.bfloat16:
        raise TypeError("The current Vision profile requires BF16 activations and table")
    if any(t.device.type != "cpu" for t in (grid_thw, x, position_table)):
        raise ValueError("HW PTO C++ simulation requires CPU inputs")

    grids = tuple(tuple(row) for row in grid_thw.tolist())
    if any(t * h * w > 2 ** 24 for t, h, w in grids):
        raise ValueError("This native coordinate path requires each media grid to fit exact FP32 integers")
    if table_side * table_side > 2 ** 31:
        raise ValueError("Position table indices must fit the native INT32 representation")
    program = tile_kernels.position_embedding(grids, x.shape[1], table_side=table_side,
                               merge_size=spatial_merge_size, tile_cols=tile_cols)
    executable = TensorProgram.compile(program)
    if artifact_dir is not None:
        executable.save(artifact_dir)
    indices = torch.empty((x.shape[0], 4), dtype=torch.int32)
    weights = torch.empty((x.shape[0], 4), dtype=torch.float32)
    out = torch.full(x.shape, float("nan"), dtype=x.dtype)
    with torch.inference_mode():
        executable.execute(grid_thw.to(torch.int32).contiguous(), x.contiguous(), position_table.contiguous(), indices, weights, out)
    if artifact_dir is not None:
        executable.save(artifact_dir)
    # 只转换索引存储位宽，坐标、索引和插值计算全部在 kernel 中完成。
    return {"out": out, "interp_indices": indices.to(torch.int64), "interp_weights": weights}
