"""K-15/16: concatenate normalized MTP features, then project with BF16 fc."""

import pypto.language as pl

from kernels.pto.common.linear import bf16_linear


@pl.jit.incore
def feature_concat(e_norm: pl.Tensor, h_norm: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Aligned BF16 e_norm[M,H], h_norm[M,H] -> [e_norm;h_norm][M,2H]."""
    hidden = e_norm.shape[1]
    for row in pl.range(e_norm.shape[0]):
        for col in pl.range(0, hidden, 128):
            valid_cols = pl.min(128, hidden - col)
            embedding = pl.load(e_norm, [row, col], [1, 128], valid_shape=[1, valid_cols])
            previous = pl.load(h_norm, [row, col], [1, 128], valid_shape=[1, valid_cols])
            pl.store(embedding, [row, col], out)
            pl.store(previous, [row, hidden + col], out)
    return out


@pl.jit
def mtp_input_projection(eh: pl.Tensor, fc_weight: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """BF16 eh[M,2H], independent fc_weight[H,2H] -> BF16 out[M,H]."""
    out = bf16_linear(eh, fc_weight, out)
    return out
