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
            # 1. amax=max(abs(x_valid))；尾组先按valid归约，再把padding lane置0。
            scratch = pl.create_tile([8, 128], dtype=pl.FP32)
            amax = pl.row_max(pl.abs(value), scratch)
            # 反量化乘数scale=amax/448；全零组取1，ceil(min(amax,1))生成非零标志。
            scale = pl.div(amax, pl.add(pl.mul(amax, 0.0), 448.0), high_precision=True)
            nonzero = pl.cast(pl.cast(pl.minimum(amax, 1.0), pl.INT32, mode="ceil"), pl.FP32)
            scale = pl.add(scale, pl.sub(pl.mul(nonzero, -1.0), -1.0))
            value = pl.tile.fillpad(value, pad_value=pl.PadValue.zero)
            value = pl.set_validshape(value, 1, 128)
            normalized = pl.div(value, pl.row_expand(value, scale), high_precision=True)
            normalized = pl.minimum(pl.maximum(normalized, -448.0), 448.0)
            # 2. E4M3的量化步长=2^(max(floor(log2(abs(x_scaled))),-6)-3)。
            # 直接读取FP32指数位构造二次幂；最小步长2^-9覆盖subnormal。
            bits = pl.reinterpret_view(normalized, pl.INT32)
            magnitude_bits = pl.and_(bits, pl.set_validshape(pl.tile.full([8, 128], dtype=pl.INT32, value=2147483647), 1, 128))
            exponent = pl.sub(pl.shrs(magnitude_bits, 23), 127)
            exponent = pl.maximum(exponent, -6)
            unit_bits = pl.shls(pl.add(exponent, 124), 23)
            unit = pl.reinterpret_view(unit_bits, pl.FP32)
            # 对abs(x_scaled)/步长做最近偶数舍入，再还原数值。
            rounded = pl.cast(pl.div(pl.abs(normalized), unit, high_precision=True), pl.INT32, mode="rint")
            positive = pl.mul(pl.cast(rounded, pl.FP32), unit)
            # 3. 直接恢复原符号位，包括负零；编码只作为数值profile的仿真表示。
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
            # FP8数值：subnormal=mantissa*2^-9；normal=(8+mantissa)*2^(exponent-10)。
            # 此处不应用weight_scale_inv；block scale在点积后逐K组应用。
            unit_bits = pl.shls(pl.add(pl.maximum(exponent, 1), 117), 23)
            unit = pl.reinterpret_view(unit_bits, pl.FP32)
            significand = pl.add(mantissa, pl.mul(pl.minimum(exponent, 1), 8))
            positive = pl.mul(pl.cast(significand, pl.FP32), unit)
            sign_bits = pl.shls(pl.and_(bits, pl.tile.full([1, 128], dtype=pl.INT32, value=128)), 24)
            value_bits = pl.or_(pl.reinterpret_view(positive, pl.INT32), sign_bits)
            value = pl.cast(pl.reinterpret_view(value_bits, pl.FP32), pl.BF16, mode="rint")
            pl.store(value, [row, begin], decoded)
    return decoded


@lru_cache(maxsize=64)
def fp8_linear_bf16(rows: int, width: int, output_width: int):
    """K128量化、权重解码、逐组点积及缩放累加；BF16精确承载FP8数值。"""
    padded_k = (width + 127) // 128 * 128
    padded_n = (output_width + 127) // 128 * 128

    @pl.jit.inline(auto_scope=False)
    def linear(x: pl.Tensor, weight_codes: pl.Tensor, weight_scale_inv: pl.Tensor,
               out: pl.Out[pl.Tensor]):
        decoded = pl.create_tensor([rows, padded_k], dtype=pl.BF16)
        scales = pl.create_tensor([rows, padded_k // 128], dtype=pl.FP32)
        codes = pl.create_tensor([rows, padded_k], dtype=pl.INT32)
        # x8=E4M3FN_RNE(clamp(x/scale))；量化与字节解码是独立共享数值原语。
        decoded, scales, codes = quantize_e4m3_bf16(x, decoded, scales, codes, rows, width)
        weight = pl.create_tensor([padded_n, padded_k], dtype=pl.BF16)
        weight = decode_e4m3_bf16(weight_codes, weight, padded_n, padded_k)
        for row in pl.range(rows):
            for nb in pl.range(0, output_width, 128):
                # 每个Token/N块独立分配部分和，消费者完成前不复用其他块的存储。
                accumulator = pl.create_tensor([1, 128], dtype=pl.FP32)
                for kb in pl.range(0, padded_k, 128):
                    dot = pl.create_tensor([1, 128], dtype=pl.FP32)
                    # dot32 = decoded[row,K128] @ decoded_weight[N128,K128].T。
                    with pl.at(level=pl.Level.CORE_GROUP):
                        lhs = pl.load(decoded, [row, kb], [16, 128], valid_shape=[1, 128], target_memory=pl.MemorySpace.Mat)
                        rhs = pl.load(weight, [nb, kb], [128, 128], target_memory=pl.MemorySpace.Mat)
                        rhs = pl.tile.transpose_view(rhs)
                        lhs = pl.move(lhs, target_memory=pl.MemorySpace.Left)
                        rhs = pl.move(rhs, target_memory=pl.MemorySpace.Right)
                        product = pl.matmul(lhs, rhs)
                        pl.store(product, [0, 0], dot)

                    # acc32 += dot32 * activation_scale[row,kgroup] * weight_scale[ngroup,kgroup]。
                    # scale乘法次序不变；输出回转BF16，部分和本身仍保留FP32。
                    with pl.at(level=pl.Level.CORE_GROUP):
                        sx = pl.read(scales, [row, kb // 128])
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
        return out

    return linear


@lru_cache(maxsize=32)
def activation_quantize_bf16(rows: int, width: int):
    """独立校验共享量化原语的入口；模型kernel直接调用上方同一量化计算。"""
    @pl.jit
    def kernel(x: pl.Tensor, decoded: pl.Out[pl.Tensor], scales: pl.Out[pl.Tensor], codes: pl.Out[pl.Tensor]):
        decoded, scales, codes = quantize_e4m3_bf16(x, decoded, scales, codes, rows, width)
        return decoded, scales, codes

    return kernel


@lru_cache(maxsize=32)
def fp8_linear_bf16_sim(rows: int, width: int, output_width: int):
    """独立校验共享FP8 Linear的入口，复用GDN投影使用的同一程序。"""
    linear = fp8_linear_bf16(rows, width, output_width)

    @pl.jit
    def kernel(x: pl.Tensor, weight: pl.Tensor, scales: pl.Tensor, out: pl.Out[pl.Tensor]):
        out = linear(x, weight, scales, out)
        return out

    return kernel
