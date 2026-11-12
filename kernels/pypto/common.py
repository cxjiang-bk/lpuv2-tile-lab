"""K-03/K-04 in the hw-native-sys PyPTO Tile DSL."""

from functools import lru_cache

import pypto.language as pl


@lru_cache(maxsize=64)
def residual_add(rows: int, cols: int, *, tile_rows: int, tile_cols: int,
                 row_offset: int, col_offset: int, valid_rows: int, valid_cols: int):
    row_blocks = (valid_rows + tile_rows - 1) // tile_rows
    col_blocks = (valid_cols + tile_cols - 1) // tile_cols
    physical_cols = (tile_cols + 15) // 16 * 16

    @pl.jit
    def kernel(x: pl.Tensor, residual: pl.Tensor, out: pl.InOut[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            for rb in pl.unroll(row_blocks):
                r = row_offset + rb * tile_rows
                nr = pl.min(tile_rows, row_offset + valid_rows - r)
                for cb in pl.unroll(col_blocks):
                    c = col_offset + cb * tile_cols
                    nc = pl.min(tile_cols, col_offset + valid_cols - c)
                    # OP-04：只读/写有效子窗口，保持输出窗口外原有数据。
                    a = pl.load(x, [r, c], [tile_rows, physical_cols], valid_shape=[nr, nc])
                    b = pl.load(residual, [r, c], [tile_rows, physical_cols], valid_shape=[nr, nc])
                    # A2/A3 PTO 的向量 ADD 用 FP32 操作数；只在相加后舍入到 BF16。
                    result = pl.add(pl.cast(a, pl.FP32), pl.cast(b, pl.FP32))
                    pl.store(pl.cast(result, pl.BF16), [r, c], out)
        return out

    return kernel


@lru_cache(maxsize=64)
def layer_norm(rows: int, cols: int, *, tile_rows: int, tile_cols: int,
               row_offset: int, valid_rows: int, eps: float):
    # 完整特征行参与归约；物理宽度对齐，valid 保留逻辑 D=cols。
    physical_cols = (cols + 31) // 32 * 32
    # 列主序归约结果[M,1]至少分配8个FP32物理行，满足32B；valid仍按窗口取值。
    physical_rows = (tile_rows + 7) // 8 * 8
    row_blocks = (valid_rows + tile_rows - 1) // tile_rows
    inv_cols = 1.0 / cols

    @pl.jit
    def kernel(x: pl.Tensor, weight: pl.Tensor, bias: pl.Tensor, out: pl.InOut[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            wf = pl.cast(pl.load(weight, [0, 0], [1, physical_cols], valid_shape=[1, cols]), pl.FP32)
            bf = pl.cast(pl.load(bias, [0, 0], [1, physical_cols], valid_shape=[1, cols]), pl.FP32)
            for rb in pl.unroll(row_blocks):
                r = row_offset + rb * tile_rows
                nr = pl.min(tile_rows, row_offset + valid_rows - r)
                xf = pl.cast(pl.load(x, [r, 0], [physical_rows, physical_cols], valid_shape=[nr, cols]), pl.FP32)
                # OP-03：mu=sum_D(x)/D；先中心化，再计算总体方差。
                sum_tmp = pl.create_tile([physical_rows, physical_cols], dtype=pl.FP32)
                mean = pl.mul(pl.row_sum(xf, sum_tmp), inv_cols)
                centered = pl.row_expand_sub(xf, mean)
                square = pl.mul(centered, centered)
                var_tmp = pl.create_tile([physical_rows, physical_cols], dtype=pl.FP32)
                variance = pl.mul(pl.row_sum(square, var_tmp), inv_cols)
                inv_std = pl.rsqrt(pl.add(variance, eps))
                normalized = pl.row_expand_mul(centered, inv_std)
                # y=BF16((x-mu)*rsqrt(var+eps)*gamma+bias)，仿射保持 FP32。
                affine = pl.col_expand_add(pl.col_expand_mul(normalized, wf), bf)
                pl.store(pl.cast(affine, pl.BF16), [r, 0], out)
        return out

    return kernel
