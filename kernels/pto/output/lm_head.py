"""K-14: project selected request-local hidden rows to BF16 logits."""

import pypto.language as pl

from kernels.pto.common.linear import bf16_linear


@pl.jit
def lm_head(
    h: pl.Tensor,
    weight: pl.Tensor,
    output_positions: pl.Tensor,
    out: pl.Out[pl.Tensor],
) -> pl.Tensor:
    """Normalized h[B,S,H], weight[V,H], positions[B,P] -> out[B,P,V]."""
    batch = h.shape[0]
    sequence = h.shape[1]
    hidden = h.shape[2]
    count = output_positions.shape[1]
    vocab = weight.shape[0]
    hidden_rows = pl.reshape(h, [batch * sequence, hidden])
    logits_rows = pl.reshape(out, [batch * count, vocab])
    for request in pl.range(batch):
        for position in pl.range(count):
            token_row = pl.read(output_positions, [request, position])
            selected = pl.slice(hidden_rows, [1, hidden], [request * sequence + token_row, 0])
            destination = pl.slice(logits_rows, [1, vocab], [request * count + position, 0])
            bf16_linear(selected, weight, destination)
    return out
