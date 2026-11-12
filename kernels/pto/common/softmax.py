"""Shared numerically stable row-wise Softmax."""

import pypto.language as pl


@pl.jit.incore
def softmax(x: pl.Tensor, out: pl.Out[pl.Tensor], input_bf16: pl.constexpr = False) -> pl.Tensor:
    """BF16/FP32 logits[M,D] -> FP32 probabilities; each row has a finite logit."""
    rows = x.shape[0]
    width = x.shape[1]
    max_scratch = pl.create_tile([1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    sum_scratch = pl.create_tile([1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    for row in pl.range(rows):
        first_valid = pl.min(128, width)
        first = pl.load(x, [row, 0], [1, 128], valid_shape=[1, first_valid])
        first32 = first
        if input_bf16:
            first32 = pl.cast(first, pl.FP32)
        first_view = pl.tile.slice(first32, [1, 128], [0, 0], valid_shape=[1, first_valid])
        row_max = pl.row_max(pl.tile.fillpad(first_view, pad_value=pl.PadValue.min), max_scratch)
        for col in pl.range(128, width, 128):
            valid_cols = pl.min(128, width - col)
            tile = pl.load(x, [row, col], [1, 128], valid_shape=[1, valid_cols])
            x32 = tile
            if input_bf16:
                x32 = pl.cast(tile, pl.FP32)
            valid_x = pl.tile.slice(x32, [1, 128], [0, 0], valid_shape=[1, valid_cols])
            padded = pl.tile.fillpad(valid_x, pad_value=pl.PadValue.min)
            current = pl.row_max(padded, max_scratch)
            row_max = pl.maximum(row_max, current)
        row_sum = pl.tile.full([1, 1], pl.FP32, 0.0)
        for col in pl.range(0, width, 128):
            valid_cols = pl.min(128, width - col)
            tile = pl.load(x, [row, col], [1, 128], valid_shape=[1, valid_cols])
            x32 = tile
            if input_bf16:
                x32 = pl.cast(tile, pl.FP32)
            valid_x = pl.tile.slice(x32, [1, 128], [0, 0], valid_shape=[1, valid_cols])
            padded = pl.tile.fillpad(valid_x, pad_value=pl.PadValue.min)
            shifted = pl.row_expand_sub(padded, row_max)
            exp_tile = pl.exp(shifted)
            current = pl.row_sum(exp_tile, sum_scratch)
            row_sum = pl.add(row_sum, current)
        for col in pl.range(0, width, 128):
            valid_cols = pl.min(128, width - col)
            tile = pl.load(x, [row, col], [1, 128], valid_shape=[1, valid_cols])
            x32 = tile
            if input_bf16:
                x32 = pl.cast(tile, pl.FP32)
            result = pl.row_expand_div(pl.exp(pl.row_expand_sub(x32, row_max)), row_sum)
            result = pl.tile.slice(result, [1, 128], [0, 0], valid_shape=[1, valid_cols])
            pl.store(result, [row, col], out)
    return out
