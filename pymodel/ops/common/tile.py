"""Complete K-04/K-03 entries for the hw-native-sys PyPTO Tile execution path."""

from pathlib import Path

import torch

from kernels.pypto import common as tile_kernels
from pymodel.isa.hw_pypto import TensorProgram


def residual_add(x: torch.Tensor, residual: torch.Tensor, *, tile_rows: int = 1, tile_cols: int = 128,
                 row_offset: int = 0, col_offset: int = 0, valid_rows: int | None = None,
                 valid_cols: int | None = None, output_seed: torch.Tensor | None = None,
                 artifact_dir: str | Path | None = None) -> dict[str, torch.Tensor]:
    if x.shape != residual.shape or x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("Residual requires same-shaped inputs and a nonempty hidden dimension")
    if x.dtype != torch.bfloat16 or residual.dtype != x.dtype:
        raise TypeError("The current K-04 profile requires BF16 inputs")
    if x.device.type != "cpu" or residual.device != x.device:
        raise ValueError("HW PTO C++ simulation requires CPU inputs")
    rows, cols = x.numel() // x.shape[-1], x.shape[-1]
    valid_rows = rows - row_offset if valid_rows is None else valid_rows
    valid_cols = cols - col_offset if valid_cols is None else valid_cols
    if any(type(v) is not int or v < 0 for v in (row_offset, col_offset, valid_rows, valid_cols)):
        raise ValueError("Offsets and valid extents must be nonnegative integers")
    if row_offset + valid_rows > rows or col_offset + valid_cols > cols:
        raise ValueError("The valid window must fit inside the logical tensor")
    if any(type(v) is not int or v < 1 for v in (tile_rows, tile_cols)):
        raise ValueError("Tile dimensions must be positive integers")
    if output_seed is not None and (output_seed.shape != x.shape or output_seed.dtype != x.dtype
                                    or output_seed.device != x.device):
        raise ValueError("Output seed must match the input shape, dtype and device")

    program = tile_kernels.residual_add(rows, cols, tile_rows=tile_rows, tile_cols=tile_cols,
                               row_offset=row_offset, col_offset=col_offset,
                               valid_rows=valid_rows, valid_cols=valid_cols)
    executable = TensorProgram.compile(program)
    if artifact_dir is not None:
        executable.save(artifact_dir)
    complete = row_offset == 0 and valid_rows == rows and col_offset == 0 and valid_cols == cols
    out = torch.full(x.shape, float("nan") if complete else 0.0, dtype=x.dtype) if output_seed is None else output_seed.clone(
        memory_format=torch.contiguous_format)
    with torch.inference_mode():
        executable.execute(x.reshape(rows, cols).contiguous(), residual.reshape(rows, cols).contiguous(), out.reshape(rows, cols))
    if artifact_dir is not None:
        executable.save(artifact_dir)
    return {"out": out}


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, eps: float = 1e-6,
               tile_rows: int = 8, tile_cols: int = 128, row_offset: int = 0,
               valid_rows: int | None = None, output_seed: torch.Tensor | None = None,
               artifact_dir: str | Path | None = None) -> dict[str, torch.Tensor]:
    if x.ndim < 1 or x.shape[-1] < 1 or weight.shape != (x.shape[-1],) or bias.shape != weight.shape:
        raise ValueError("LayerNorm parameters must match the nonempty last axis")
    if any(t.dtype != torch.bfloat16 for t in (x, weight, bias)):
        raise TypeError("The current K-03 profile requires BF16 input and parameters")
    if x.device.type != "cpu" or any(t.device != x.device for t in (weight, bias)):
        raise ValueError("HW PTO C++ simulation requires CPU inputs")
    rows, cols = x.numel() // x.shape[-1], x.shape[-1]
    valid_rows = rows - row_offset if valid_rows is None else valid_rows
    if type(row_offset) is not int or type(valid_rows) is not int or row_offset < 0 or valid_rows < 0:
        raise ValueError("Row offset and extent must be nonnegative integers")
    if row_offset + valid_rows > rows:
        raise ValueError("The valid rows must fit inside the logical tensor")
    if any(type(v) is not int or v < 1 for v in (tile_rows, tile_cols)) or eps <= 0:
        raise ValueError("Tile dimensions and epsilon must be positive")
    if output_seed is not None and (output_seed.shape != x.shape or output_seed.dtype != x.dtype
                                    or output_seed.device != x.device):
        raise ValueError("Output seed must match the input shape, dtype and device")

    program = tile_kernels.layer_norm(rows, cols, tile_rows=tile_rows, tile_cols=tile_cols,
                                 row_offset=row_offset, valid_rows=valid_rows, eps=eps)
    executable = TensorProgram.compile(program)
    if artifact_dir is not None:
        executable.save(artifact_dir)
    complete = row_offset == 0 and valid_rows == rows
    out = torch.full(x.shape, float("nan") if complete else 0.0, dtype=x.dtype) if output_seed is None else output_seed.clone(
        memory_format=torch.contiguous_format)
    with torch.inference_mode():
        executable.execute(x.reshape(rows, cols).contiguous(), weight.reshape(1, cols).contiguous(), bias.reshape(1, cols).contiguous(),
                           out.reshape(rows, cols))
    if artifact_dir is not None:
        executable.save(artifact_dir)
    return {"out": out}
