"""Per-Token, 128-element FP8 E4M3FN quantization with FP32 scales."""

import pypto.language as pl


@pl.jit.inline
def _quantize_group(values: pl.Tile[[1, 128], pl.BF16]):
    """Quantize one zero-padded [1,128] group; retain its exact FP32 scale."""
    scratch = pl.create_tile([1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    x32 = pl.cast(values, pl.FP32)
    amax = pl.row_max(pl.abs(x32), scratch)
    maximum = pl.tile.read(amax, [0, 0])
    if maximum == 0.0:
        scale = pl.yield_(pl.const(1.0, pl.FP32))
    else:
        scale = pl.yield_(maximum / 448.0)
    normalized = pl.tile.divs(x32, scale)
    clipped = pl.tile.minimums(pl.tile.maximums(normalized, -448.0), 448.0)
    payload = pl.cast(clipped, pl.FP8E4M3FN, mode="round")
    scale_tile = pl.mul(pl.tile.full([1, 1], pl.FP32, 1.0), scale)
    return payload, scale_tile


@pl.jit.incore
def dynamic_quantize(x: pl.Tensor, payload: pl.Out[pl.Tensor], scales: pl.Out[pl.Tensor]):
    """BF16 x[M,K] -> FP8 payload[M,K], FP32 scales[M,ceil(K/128)]."""
    width = x.shape[1]
    for row in pl.range(x.shape[0]):
        for col in pl.range(0, width, 128):
            valid_cols = pl.min(128, width - col)
            values = pl.load(x, [row, col], [1, 128], valid_shape=[1, valid_cols])
            padded = pl.tile.fillpad(values)
            quantized, scale = _quantize_group(padded)
            valid_payload = pl.tile.slice(quantized, [1, 128], [0, 0], valid_shape=[1, valid_cols])
            pl.store(valid_payload, [row, col], payload)
            pl.store(scale, [row, col // 128], scales)
    return payload, scales
