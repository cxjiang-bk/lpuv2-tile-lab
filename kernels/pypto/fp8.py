"""K128 E4M3FN quantization and block-scaled Linear in native PyPTO.

The A2/A3 path carries exactly representable FP8 values in BF16. Checkpoint
bytes, activation rounding, signs and both scales are processed by PTO kernels.
"""

from functools import lru_cache

import pypto.language as pl


@pl.jit.incore
def quantize_e4m3_bf16(x: pl.Tensor, decoded: pl.Out[pl.Tensor],
                        scales: pl.Out[pl.Tensor], codes: pl.Out[pl.Tensor],
                        rows: pl.constexpr, width: pl.constexpr):
    """Quantize each Token/K128 group, including RNE ties and negative zero."""
    groups = (width + 127) // 128
    for row in pl.range(rows):
        for group in pl.range(groups):
            begin = group * 128
            count = pl.min(128, width - begin)
            value = pl.load(x, [row, begin], [8, 128], valid_shape=[1, count])
            value = pl.cast(value, pl.FP32, mode="rint")
            # Reduce valid elements before making tail lanes explicit zeroes.
            scratch = pl.create_tile([8, 128], dtype=pl.FP32)
            amax = pl.row_max(pl.abs(value), scratch)
            scale = pl.div(amax, pl.add(pl.mul(amax, 0.0), 448.0), high_precision=True)
            nonzero = pl.cast(pl.cast(pl.minimum(amax, 1.0), pl.INT32, mode="ceil"), pl.FP32)
            scale = pl.add(scale, pl.sub(pl.mul(nonzero, -1.0), -1.0))
            value = pl.tile.fillpad(value, pad_value=pl.PadValue.zero)
            value = pl.set_validshape(value, 1, 128)
            normalized = pl.div(value, pl.row_expand(value, scale), high_precision=True)
            normalized = pl.minimum(pl.maximum(normalized, -448.0), 448.0)
            # Exponent bits construct an exact power-of-two quantizer unit.
            # No approximate log/exp can move an FP8 midpoint.
            bits = pl.reinterpret_view(normalized, pl.INT32)
            magnitude_bits = pl.and_(bits, pl.set_validshape(pl.tile.full([8, 128], dtype=pl.INT32, value=2147483647), 1, 128))
            exponent = pl.sub(pl.shrs(magnitude_bits, 23), 127)
            exponent = pl.maximum(exponent, -6)
            unit_bits = pl.shls(pl.add(exponent, 124), 23)
            unit = pl.reinterpret_view(unit_bits, pl.FP32)
            rounded = pl.cast(pl.div(pl.abs(normalized), unit, high_precision=True), pl.INT32, mode="rint")
            positive = pl.mul(pl.cast(rounded, pl.FP32), unit)
            sign_bits = pl.and_(bits, pl.set_validshape(pl.tile.full([8, 128], dtype=pl.INT32, value=-2147483648), 1, 128))
            rounded_bits = pl.or_(pl.reinterpret_view(positive, pl.INT32), sign_bits)
            rounded_value = pl.reinterpret_view(rounded_bits, pl.FP32)
            quantized = pl.cast(rounded_value, pl.BF16, mode="rint")
            positive_code = pl.add(rounded, pl.mul(pl.add(exponent, 6), 8))
            sign_code = pl.and_(pl.shrs(bits, 24), pl.set_validshape(pl.tile.full([8, 128], dtype=pl.INT32, value=128), 1, 128))
            encoded = pl.add(positive_code, sign_code)
            pl.store(quantized, [row, begin], decoded)
            pl.store(encoded, [row, begin], codes)
            pl.store(scale, [row, group], scales)
    return decoded, scales, codes


@pl.jit.incore
def decode_e4m3_bf16(weight_codes: pl.Tensor, decoded: pl.Out[pl.Tensor],
                      rows: pl.constexpr, width: pl.constexpr):
    """Decode padded checkpoint bytes without changing their FP8 scale."""
    for row in pl.range(rows):
        for begin in pl.range(0, width, 128):
            bits = pl.load(weight_codes, [row, begin], [1, 128])
            magnitude = pl.and_(bits, pl.tile.full([1, 128], dtype=pl.INT32, value=127))
            exponent = pl.shrs(magnitude, 3)
            mantissa = pl.and_(magnitude, pl.tile.full([1, 128], dtype=pl.INT32, value=7))
            # subnormal: mantissa*2^-9; normal: (8+mantissa)*2^(e-10).
            unit_bits = pl.shls(pl.add(pl.maximum(exponent, 1), 117), 23)
            unit = pl.reinterpret_view(unit_bits, pl.FP32)
            significand = pl.add(mantissa, pl.mul(pl.minimum(exponent, 1), 8))
            positive = pl.mul(pl.cast(significand, pl.FP32), unit)
            sign_bits = pl.shls(pl.and_(bits, pl.tile.full([1, 128], dtype=pl.INT32, value=128)), 24)
            value_bits = pl.or_(pl.reinterpret_view(positive, pl.INT32), sign_bits)
            value = pl.cast(pl.reinterpret_view(value_bits, pl.FP32), pl.BF16, mode="rint")
            pl.store(value, [row, begin], decoded)
    return decoded


@pl.jit.incore
def k128_dot(x: pl.Tensor, weight: pl.Tensor, dot: pl.Out[pl.Tensor],
             row: pl.Scalar[pl.INDEX], nb: pl.Scalar[pl.INDEX], kb: pl.Scalar[pl.INDEX]):
    lhs = pl.load(x, [row, kb], [16, 128], valid_shape=[1, 128], target_memory=pl.MemorySpace.Mat)
    rhs = pl.load(weight, [nb, kb], [128, 128], target_memory=pl.MemorySpace.Mat)
    rhs = pl.tile.transpose_view(rhs)
    lhs = pl.move(lhs, target_memory=pl.MemorySpace.Left)
    rhs = pl.move(rhs, target_memory=pl.MemorySpace.Right)
    product = pl.matmul(lhs, rhs)
    pl.store(product, [0, 0], dot)
    return dot


@lru_cache(maxsize=64)
def scale_accumulator(rows: int, output_width: int, groups: int):
    @pl.jit.incore
    def accumulate(dot: pl.Tensor, activation_scales: pl.Tensor, weight_scale_inv: pl.Tensor,
                          accumulator: pl.InOut[pl.Tensor], out: pl.Out[pl.Tensor],
                          row: pl.Scalar[pl.INDEX], nb: pl.Scalar[pl.INDEX], kb: pl.Scalar[pl.INDEX]):
        sx = pl.read(activation_scales, [row, kb // 128])
        sw_tile = pl.cast(pl.load(weight_scale_inv, [nb // 128, kb // 128], [1, 16],
                                 valid_shape=[1, 1]), pl.FP32, mode="rint")
        sw = pl.read(sw_tile, [0, 0])
        partial = pl.load(dot, [0, 0], [1, 128])
        contribution = pl.mul(pl.mul(partial, sx), sw)
        if kb == 0:
            updated = contribution
        else:
            previous = pl.load(accumulator, [0, 0], [1, 128])
            updated = pl.add(previous, contribution)
        pl.store(updated, [0, 0], accumulator)
        valid = pl.min(128, out.shape[1] - nb)
        result = pl.cast(pl.set_validshape(updated, 1, valid), pl.BF16, mode="rint")
        pl.store(result, [row, nb], out)
        return accumulator, out

    return accumulate

@lru_cache(maxsize=64)
def scaled_k128_projection(rows: int, padded_k: int, output_width: int):
    accumulate = scale_accumulator(rows, output_width, padded_k // 128)

    @pl.jit.inline(auto_scope=False)
    def project(x: pl.Tensor, weight: pl.Tensor, activation_scales: pl.Tensor,
                                 weight_scale_inv: pl.Tensor, out: pl.Out[pl.Tensor]):
        """A GM FP32 dot boundary separates cube and vector scale application."""
        for row in pl.range(rows):
            for nb in pl.range(0, output_width, 128):
                accumulator = pl.create_tensor([1, 128], dtype=pl.FP32)
                for kb in pl.range(0, padded_k, 128):
                    dot = pl.create_tensor([1, 128], dtype=pl.FP32)
                    dot = k128_dot(x, weight, dot, row, nb, kb)
                    accumulator, out = accumulate(dot, activation_scales, weight_scale_inv,
                                                         accumulator, out, row, nb, kb)
        return out

    return project

@lru_cache(maxsize=64)
def fp8_linear_bf16(rows: int, width: int, output_width: int):
    project = scaled_k128_projection(rows, (width + 127) // 128 * 128, output_width)

    @pl.jit.inline(auto_scope=False)
    def linear(x: pl.Tensor, weight_codes: pl.Tensor, weight_scale_inv: pl.Tensor,
                         out: pl.Out[pl.Tensor]):
        """Native orchestration of quantization, byte decode and scaled GEMM."""
        decoded = pl.create_tensor([rows, (width + 127) // 128 * 128], dtype=pl.BF16)
        scales = pl.create_tensor([rows, (width + 127) // 128], dtype=pl.FP32)
        codes = pl.create_tensor([rows, (width + 127) // 128 * 128], dtype=pl.INT32)
        decoded, scales, codes = quantize_e4m3_bf16(x, decoded, scales, codes, rows, width)
        weight = pl.create_tensor([(output_width + 127) // 128 * 128, (width + 127) // 128 * 128], dtype=pl.BF16)
        weight = decode_e4m3_bf16(weight_codes, weight, (output_width + 127) // 128 * 128, (width + 127) // 128 * 128)
        out = project(decoded, weight, scales, weight_scale_inv, out)
        return out

    return linear

@lru_cache(maxsize=32)
def activation_quantize_bf16(rows: int, width: int):
    @pl.jit
    def kernel(x: pl.Tensor, decoded: pl.Out[pl.Tensor], scales: pl.Out[pl.Tensor], codes: pl.Out[pl.Tensor]):
        decoded, scales, codes = quantize_e4m3_bf16(x, decoded, scales, codes, rows, width)
        return decoded, scales, codes

    return kernel


@lru_cache(maxsize=32)
def fp8_linear_bf16_sim(rows: int, width: int, output_width: int):
    linear = fp8_linear_bf16(rows, width, output_width)
    @pl.jit
    def kernel(x: pl.Tensor, weight: pl.Tensor, scales: pl.Tensor, out: pl.Out[pl.Tensor]):
        out = linear(x, weight, scales, out)
        return out

    return kernel

