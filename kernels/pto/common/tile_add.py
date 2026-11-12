"""Stage 3 execution example: explicit BF16 Tile load, add and store."""

import pypto.language as pl


@pl.jit.incore
def tile_add(
    a: pl.Tensor[[2, 16], pl.BF16],
    b: pl.Tensor[[2, 16], pl.BF16],
    out: pl.Out[pl.Tensor[[2, 16], pl.BF16]],
) -> pl.Tensor[[2, 16], pl.BF16]:
    tile_a = pl.load(a, [0, 0], [2, 16])
    tile_b = pl.load(b, [0, 0], [2, 16])
    tile_out = pl.add(tile_a, tile_b)
    pl.store(tile_out, [0, 0], out)
    return out
