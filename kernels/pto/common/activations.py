"""Shared BF16/FP32 activations with FP32 intermediate arithmetic."""

import pypto.language as pl


@pl.jit.incore
def sigmoid(x: pl.Tensor, out: pl.Out[pl.Tensor],
            input_bf16: pl.constexpr = True, output_bf16: pl.constexpr = True) -> pl.Tensor:
    """Compute sigmoid for x[M,D]; the caller chooses BF16/FP32 out."""
    rows = x.shape[0]
    width = x.shape[1]
    for row in pl.range(rows):
        for c0 in pl.range(0, width, 128):
            valid_cols = pl.min(128, width - c0)
            tile = pl.load(x, [row, c0], [1, 128], valid_shape=[1, valid_cols])
            if input_bf16:
                tile = pl.cast(tile, pl.FP32)
            ones = pl.tile.full([1, 128], pl.FP32, 1.0)
            denom = pl.tile.add(ones, pl.exp(pl.tile.muls(tile, -1.0)))
            result = pl.tile.div(ones, denom)
            if output_bf16:
                result = pl.cast(result, pl.BF16)
            result = pl.tile.slice(result, [1, 128], [0, 0], valid_shape=[1, valid_cols])
            pl.store(result, [row, c0], out)
    return out


@pl.jit.incore
def silu(x: pl.Tensor, out: pl.Out[pl.Tensor],
         input_bf16: pl.constexpr = True, output_bf16: pl.constexpr = True) -> pl.Tensor:
    """Compute SiLU for x[M,D]; the caller chooses BF16/FP32 out."""
    rows = x.shape[0]
    width = x.shape[1]
    for row in pl.range(rows):
        for c0 in pl.range(0, width, 128):
            valid_cols = pl.min(128, width - c0)
            tile = pl.load(x, [row, c0], [1, 128], valid_shape=[1, valid_cols])
            if input_bf16:
                tile = pl.cast(tile, pl.FP32)
            ones = pl.tile.full([1, 128], pl.FP32, 1.0)
            denom = pl.tile.add(ones, pl.exp(pl.tile.muls(tile, -1.0)))
            gate = pl.tile.div(ones, denom)
            result = pl.tile.mul(tile, gate)
            if output_bf16:
                result = pl.cast(result, pl.BF16)
            result = pl.tile.slice(result, [1, 128], [0, 0], valid_shape=[1, valid_cols])
            pl.store(result, [row, c0], out)
    return out
