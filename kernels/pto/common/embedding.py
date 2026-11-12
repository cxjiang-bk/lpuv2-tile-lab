"""K-01: BF16 Embedding through Token-indexed Tile loads and stores."""

import pypto.language as pl


@pl.jit.incore
def embedding(
    token_ids: pl.Tensor,
    embedding_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
) -> pl.Tensor:
    token_count = token_ids.shape[0] * token_ids.shape[1]
    hidden = embedding_weight.shape[1]
    ids = pl.reshape(token_ids, [token_count])
    rows = pl.reshape(out, [token_count, hidden])
    for row in pl.range(token_count):
        token_id = pl.read(ids, [row])
        for col in pl.range(0, hidden, 32):
            valid_cols = pl.min(32, hidden - col)
            values = pl.load(
                embedding_weight,
                [token_id, col],
                [1, 32],
                valid_shape=[1, valid_cols],
            )
            pl.store(values, [row, col], rows)
    return out
