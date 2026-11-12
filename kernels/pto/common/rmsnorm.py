"""K-02: BF16 x[M,D], weight[D] -> out[M,D]; FP32 RMSNorm with 1+weight."""

import pypto.language as pl


@pl.jit.incore
def rmsnorm(
    x: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
    eps: pl.FP32 = 1e-6,
) -> pl.Tensor:
    hidden = weight.shape[0]
    weight_row = pl.reshape(weight, [1, hidden])
    scratch = pl.create_tile([1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    for row in pl.range(x.shape[0]):
        square_sum = pl.tile.full([1, 1], pl.FP32, 0.0)
        for col in pl.range(0, hidden, 128):
            valid_cols = pl.min(128, hidden - col)
            values = pl.load(x, [row, col], [1, 128], valid_shape=[1, valid_cols])
            x32 = pl.cast(values, pl.FP32)
            padded = pl.tile.fillpad(x32)
            partial_sum = pl.row_sum(pl.mul(padded, padded), scratch)
            square_sum = pl.add(square_sum, partial_sum)
        inv_rms = pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / hidden), eps))

        # Re-read each input block only after the complete row statistic exists.
        for col in pl.range(0, hidden, 128):
            valid_cols = pl.min(128, hidden - col)
            values = pl.load(x, [row, col], [1, 128], valid_shape=[1, valid_cols])
            weights = pl.load(weight_row, [0, col], [1, 128], valid_shape=[1, valid_cols])
            x32 = pl.cast(values, pl.FP32)
            gamma = pl.add(pl.cast(weights, pl.FP32), 1.0)
            normalized = pl.row_expand_mul(x32, inv_rms)
            result = pl.cast(pl.mul(normalized, gamma), pl.BF16)
            # Keep the valid extent explicit across Torch arithmetic/casts.
            result = pl.tile.slice(result, [1, 128], [0, 0], valid_shape=[1, valid_cols])
            pl.store(result, [row, col], out)
    return out
