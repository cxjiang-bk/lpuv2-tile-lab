"""Shared BF16/FP8 projections with tiled dot products and FP32 accumulation."""

import pypto.language as pl

from kernels.pto.common.quantization import _quantize_group


@pl.jit.incore
def bf16_linear(
    x: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
) -> pl.Tensor:
    """Compute ``x[M,K] @ weight[N,K].T`` and write a BF16 ``[M,N]`` result."""
    rows = x.shape[0]
    outputs = weight.shape[0]
    width = x.shape[1]
    for m0 in pl.range(0, rows, 16):
        for n_block in pl.range(0, outputs, 16):
            valid_m = pl.min(16, rows - m0)
            valid_n = pl.min(16, outputs - n_block)
            acc = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
            for k0 in pl.range(0, width, 64):
                valid_k = pl.min(64, width - k0)
                lhs = pl.load(x, [m0, k0], [16, 64], valid_shape=[valid_m, valid_k])
                rhs = pl.load(weight, [n_block, k0], [16, 64], valid_shape=[valid_n, valid_k])
                lhs = pl.tile.fillpad(lhs)
                rhs = pl.tile.fillpad(rhs)
                lhs = pl.move(lhs, target_memory=pl.MemorySpace.Left)
                rhs = pl.move(pl.tile.transpose_view(rhs), target_memory=pl.MemorySpace.Right)
                acc = pl.tile.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            result = pl.move(acc, target_memory=pl.MemorySpace.Vec)
            result = pl.cast(result, pl.BF16)
            result = pl.tile.slice(result, [16, 16], [0, 0], valid_shape=[valid_m, valid_n])
            pl.store(result, [m0, n_block], out)
    return out


@pl.jit.incore
def fp8_linear(x: pl.Tensor, weight: pl.Tensor, weight_scale: pl.Tensor, out: pl.Out[pl.Tensor],
               scale_bf16: pl.constexpr = True):
    """BF16 x[M,K], FP8 weight[N,K], block scales -> BF16 out[M,N]."""
    outputs = weight.shape[0]
    width = x.shape[1]
    for row in pl.range(x.shape[0]):
        for n0 in pl.range(0, outputs, 16):
            valid_n = pl.min(16, outputs - n0)
            acc = pl.tile.full([1, 16], pl.FP32, 0.0)
            for k0 in pl.range(0, width, 128):
                valid_k = pl.min(128, width - k0)
                values = pl.load(x, [row, k0], [1, 128], valid_shape=[1, valid_k])
                payload, activation_scale = _quantize_group(pl.tile.fillpad(values))
                weights = pl.load(weight, [n0, k0], [16, 128], valid_shape=[valid_n, valid_k])
                weights = pl.tile.fillpad(weights)
                lhs = pl.move(payload, target_memory=pl.MemorySpace.Left)
                rhs = pl.move(pl.tile.transpose_view(weights), target_memory=pl.MemorySpace.Right)
                partial = pl.move(pl.matmul(lhs, rhs), target_memory=pl.MemorySpace.Vec)
                scale = pl.load(weight_scale, [n0 // 128, k0 // 128], [1, 1])
                if scale_bf16:
                    scale = pl.cast(scale, pl.FP32)
                partial = pl.row_expand_mul(partial, activation_scale)
                partial = pl.row_expand_mul(partial, scale)
                acc = pl.add(acc, partial)
            result = pl.cast(acc, pl.BF16)
            result = pl.tile.slice(result, [1, 16], [0, 0], valid_shape=[1, valid_n])
            pl.store(result, [row, n0], out)
    return out
