"""Complete Vision kernels written with hw-native-sys/PyPTO.

Tensor expressions are lowered to explicit PTO Tile operations by the compiler.
Window addressing, cast boundaries, and token order stay visible in this source.
"""

from functools import lru_cache

import pypto.language as pl


@lru_cache(maxsize=64)
def im2col(image_shape: tuple[int, ...], kernel_size: tuple[int, int, int],
           stride: tuple[int, int, int], *, input_dtype: str = "bfloat16"):
    """像素NCTHW -> 窗口矩阵；坐标、Gather与BF16边界按执行顺序展开。"""
    batch, channels, time, height, width = image_shape
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    ot, oh, ow = (time - kt) // st + 1, (height - kh) // sh + 1, (width - kw) // sw + 1
    rows, patch_width = batch * ot * oh * ow, channels * kt * kh * kw
    image_size = batch * channels * time * height * width

    @pl.jit
    def kernel(pixels: pl.Tensor, columns: pl.Out[pl.Tensor]):
        flat_pixels = pl.reshape(pixels, [1, image_size])
        image = pl.create_tensor([1, image_size], dtype=pl.FP32)
        # 1. 精确扩宽像素到FP32暂存，供元素Gather读取。
        with pl.at(level=pl.Level.CORE_GROUP):
            for pixel_begin in pl.range(0, image_size, 1024):
                pixel_count = pl.min(1024, image_size - pixel_begin)
                pixel_values = pl.load(flat_pixels, [0, pixel_begin], [1, 1024], valid_shape=[1, pixel_count])
                if input_dtype == "float32":
                    image = pl.store(pixel_values, [0, pixel_begin], image)
                else:
                    image = pl.store(pl.cast(pixel_values, pl.FP32), [0, pixel_begin], image)

        # 2. columns[row,k] = BF16(pixels[b,c,t*st+dt,h*sh+dh,w*sw+dw])。
        with pl.at(level=pl.Level.CORE_GROUP):
            for patch_row in pl.range(rows):
                batch_index = patch_row // (ot * oh * ow)
                output_t = (patch_row // (oh * ow)) % ot
                output_h = (patch_row // ow) % oh
                output_w = patch_row % ow
                for window_begin in pl.range(0, patch_width, 128):
                    window_count = pl.min(128, patch_width - window_begin)
                    window_index = pl.add(pl.cast(pl.tile.ci(0, [1, 128]), pl.FP32), pl.cast(pl.cast(window_begin, pl.INT32), pl.FP32))
                    channel = pl.cast(pl.cast(pl.div(window_index, kt * kh * kw), pl.INT32, mode="floor"), pl.FP32)
                    kernel_t = pl.sub(pl.cast(pl.cast(pl.div(window_index, kh * kw), pl.INT32, mode="floor"), pl.FP32), pl.mul(channel, kt))
                    channel_time = pl.add(pl.mul(channel, kt), kernel_t)
                    kernel_h = pl.sub(pl.cast(pl.cast(pl.div(window_index, kw), pl.INT32, mode="floor"), pl.FP32), pl.mul(channel_time, kh))
                    channel_time_height = pl.add(pl.mul(channel_time, kh), kernel_h)
                    kernel_w = pl.sub(window_index, pl.mul(channel_time_height, kw))
                    # NCTHW 窗口地址；重叠窗口直接重复 Gather，不由 host 切 Patch。
                    pixel_index = pl.add(channel, pl.cast(pl.cast(batch_index * channels, pl.INT32), pl.FP32))
                    pixel_index = pl.add(pl.add(pl.mul(pixel_index, time), pl.cast(pl.cast(output_t * st, pl.INT32), pl.FP32)), kernel_t)
                    pixel_index = pl.add(pl.add(pl.mul(pixel_index, height), pl.cast(pl.cast(output_h * sh, pl.INT32), pl.FP32)), kernel_h)
                    pixel_index = pl.add(pl.add(pl.mul(pixel_index, width), pl.cast(pl.cast(output_w * sw, pl.INT32), pl.FP32)), kernel_w)
                    pixel_indices = pl.set_validshape(pl.cast(pixel_index, pl.INT32), 1, window_count)
                    window_values = pl.tile.mgather(image, pixel_indices, coalesce="elem")
                    columns = pl.store(pl.cast(window_values, pl.BF16), [patch_row, window_begin], columns)
        return columns

    return kernel


@lru_cache(maxsize=64)
def patch_embedding(image_shape: tuple[int, ...], hidden: int, kernel_size: tuple[int, int, int], *,
                    stride: tuple[int, int, int], spatial_merge_size: int = 2, input_dtype: str = "bfloat16"):
    """OP-08：像素窗口 -> FP32累加的Linear+bias -> BF16 -> Merge Token重排。"""
    batch, channels, time, height, width = image_shape
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    ot, oh, ow = (time - kt) // st + 1, (height - kh) // sh + 1, (width - kw) // sw + 1
    rows, patch_width = batch * ot * oh * ow, channels * kt * kh * kw
    image_size = batch * channels * time * height * width
    m = spatial_merge_size

    @pl.jit
    def kernel(pixels: pl.Tensor, weight: pl.Tensor, bias: pl.Tensor,
               out: pl.Out[pl.Tensor], grid_thw: pl.Out[pl.Tensor]):
        flat_pixels = pl.reshape(pixels, [1, image_size])
        image = pl.create_tensor([1, image_size], dtype=pl.FP32)
        columns = pl.create_tensor([rows, patch_width], dtype=pl.BF16)
        raster = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        # 1. 原始像素 -> FP32暂存；host不预先切Patch。
        with pl.at(level=pl.Level.CORE_GROUP):
            for pixel_begin in pl.range(0, image_size, 1024):
                pixel_count = pl.min(1024, image_size - pixel_begin)
                pixel_values = pl.load(flat_pixels, [0, pixel_begin], [1, 1024], valid_shape=[1, pixel_count])
                if input_dtype == "float32":
                    image = pl.store(pixel_values, [0, pixel_begin], image)
                else:
                    image = pl.store(pl.cast(pixel_values, pl.FP32), [0, pixel_begin], image)

        # 2. im2col：每行是一个输出窗口，列顺序为C,Kt,Kh,Kw。
        with pl.at(level=pl.Level.CORE_GROUP):
            for patch_row in pl.range(rows):
                batch_index = patch_row // (ot * oh * ow)
                output_t = (patch_row // (oh * ow)) % ot
                output_h = (patch_row // ow) % oh
                output_w = patch_row % ow
                for window_begin in pl.range(0, patch_width, 128):
                    window_count = pl.min(128, patch_width - window_begin)
                    window_index = pl.add(pl.cast(pl.tile.ci(0, [1, 128]), pl.FP32), pl.cast(pl.cast(window_begin, pl.INT32), pl.FP32))
                    channel = pl.cast(pl.cast(pl.div(window_index, kt * kh * kw), pl.INT32, mode="floor"), pl.FP32)
                    kernel_t = pl.sub(pl.cast(pl.cast(pl.div(window_index, kh * kw), pl.INT32, mode="floor"), pl.FP32), pl.mul(channel, kt))
                    channel_time = pl.add(pl.mul(channel, kt), kernel_t)
                    kernel_h = pl.sub(pl.cast(pl.cast(pl.div(window_index, kw), pl.INT32, mode="floor"), pl.FP32), pl.mul(channel_time, kh))
                    channel_time_height = pl.add(pl.mul(channel_time, kh), kernel_h)
                    kernel_w = pl.sub(window_index, pl.mul(channel_time_height, kw))
                    # NCTHW 窗口地址；重叠窗口直接重复 Gather，不由 host 切 Patch。
                    pixel_index = pl.add(channel, pl.cast(pl.cast(batch_index * channels, pl.INT32), pl.FP32))
                    pixel_index = pl.add(pl.add(pl.mul(pixel_index, time), pl.cast(pl.cast(output_t * st, pl.INT32), pl.FP32)), kernel_t)
                    pixel_index = pl.add(pl.add(pl.mul(pixel_index, height), pl.cast(pl.cast(output_h * sh, pl.INT32), pl.FP32)), kernel_h)
                    pixel_index = pl.add(pl.add(pl.mul(pixel_index, width), pl.cast(pl.cast(output_w * sw, pl.INT32), pl.FP32)), kernel_w)
                    pixel_indices = pl.set_validshape(pl.cast(pixel_index, pl.INT32), 1, window_count)
                    window_values = pl.tile.mgather(image, pixel_indices, coalesce="elem")
                    columns = pl.store(pl.cast(window_values, pl.BF16), [patch_row, window_begin], columns)

        # 3. raster = BF16(sum_K(columns @ weight.T) + FP32(bias))。
        #    分K累加器始终保持FP32，最终bias后才转换。
        with pl.at(level=pl.Level.CORE_GROUP):
            for row_begin in pl.range(0, rows, 16):
                for feature_begin in pl.range(0, hidden, 128):
                    valid_rows = pl.min(16, rows - row_begin)
                    valid_features = pl.min(128, hidden - feature_begin)
                    valid_k0 = pl.min(128, patch_width)
                    lhs_first = pl.load(columns, [row_begin, 0], [16, 128], valid_shape=[valid_rows, valid_k0], target_memory=pl.Mem.Mat)
                    rhs_first = pl.load(weight, [feature_begin, 0], [128, 128], valid_shape=[valid_features, valid_k0], target_memory=pl.Mem.Mat)
                    first_accumulator = pl.matmul(pl.move(lhs_first, target_memory=pl.Mem.Left),
                        pl.move(pl.tile.transpose_view(rhs_first), target_memory=pl.Mem.Right))
                    for k_begin, (partial_sum,) in pl.range(128, (patch_width + 127) // 128 * 128, 128, init_values=(first_accumulator,)):
                        valid_k = pl.min(128, patch_width - k_begin)
                        lhs = pl.load(columns, [row_begin, k_begin], [16, 128], valid_shape=[valid_rows, valid_k], target_memory=pl.Mem.Mat)
                        rhs = pl.load(weight, [feature_begin, k_begin], [128, 128], valid_shape=[valid_features, valid_k],
                            target_memory=pl.Mem.Mat)
                        accumulated = pl.matmul_acc(partial_sum, pl.move(lhs, target_memory=pl.Mem.Left),
                            pl.move(pl.tile.transpose_view(rhs), target_memory=pl.Mem.Right))
                        accumulator = pl.yield_(accumulated)
                    # FP32 部分和跨 K 块累加，bias 后才舍入 BF16。
                    dot32 = pl.set_validshape(accumulator, valid_rows, valid_features)
                    bias32 = pl.cast(pl.load(bias, [0, feature_begin], [1, 128], valid_shape=[1, valid_features]), pl.FP32)
                    raster = pl.store(pl.cast(pl.col_expand_add(dot32, bias32), pl.BF16), [row_begin, feature_begin], raster)

        # 4. 将空间栅格顺序改为Merge组顺序：B,Ot,block_h,block_w,intra_h,intra_w。
        with pl.at(level=pl.Level.CORE_GROUP):
            for token_index in pl.range(rows):
                frame_index = token_index // (oh * ow)
                spatial_index = token_index % (oh * ow)
                merge_group = spatial_index // (m * m)
                merge_lane = spatial_index % (m * m)
                source_h = (merge_group // (ow // m)) * m + merge_lane // m
                source_w = (merge_group % (ow // m)) * m + merge_lane % m
                source_row = frame_index * oh * ow + source_h * ow + source_w
                for col_begin in pl.range(0, hidden, 128):
                    reordered_values = pl.load(raster, [source_row, col_begin], [1, 128], valid_shape=[1, pl.min(128, hidden - col_begin)])
                    out = pl.store(reordered_values, [token_index, col_begin], out)
            for media_index in pl.range(batch):
                pl.write(grid_thw, [media_index, 0], pl.const(ot, pl.INT64))
                pl.write(grid_thw, [media_index, 1], pl.const(oh, pl.INT64))
                pl.write(grid_thw, [media_index, 2], pl.const(ow, pl.INT64))
        return out, grid_thw

    return kernel


@lru_cache(maxsize=64)
def position_embedding(grids: tuple[tuple[int, int, int], ...], hidden: int, *, table_side: int,
                       merge_size: int, tile_cols: int):
    """OP-11：网格 -> Token坐标 -> 四邻点/权重 -> 位置插值 -> BF16位置相加。"""
    media_count = len(grids)
    physical_count = (max(t * h * w for t, h, w in grids) + 7) // 8 * 8

    @pl.jit
    def kernel(grid: pl.Tensor, x: pl.Tensor, table: pl.Tensor, indices: pl.Out[pl.Tensor],
               weights: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            offset_zero = pl.const(0, pl.INDEX)
            for media, (start,) in pl.range(media_count, init_values=(offset_zero,)):
                time = pl.read(grid, [media, 0])
                height = pl.read(grid, [media, 1])
                width = pl.read(grid, [media, 2])
                count = time * height * width
                valid_count = pl.cast(count, pl.INDEX)
                frame_size = pl.cast(height * width, pl.FP32)
                merge_columns = pl.cast(pl.cast(width // merge_size, pl.INT32), pl.FP32)
                row_divisor = pl.cast(pl.cast(pl.max(height - 1, 1), pl.INT32), pl.FP32)
                col_divisor = pl.cast(pl.cast(pl.max(width - 1, 1), pl.INT32), pl.FP32)
                # 1. IOTA产生媒体内Token编号；padding lane复用最后有效编号，Gather始终合法。
                serial = pl.reshape(pl.cast(pl.tile.ci(0, [1, physical_count]), pl.FP32), [physical_count, 1])
                serial = pl.minimum(serial, pl.cast(pl.cast(count - 1, pl.INT32), pl.FP32))
                local = pl.sub(serial, pl.mul(pl.cast(pl.cast(pl.div(serial, frame_size), pl.INT32, mode="floor"), pl.FP32), frame_size))
                group = pl.cast(pl.cast(pl.div(local, merge_size * merge_size), pl.INT32, mode="floor"), pl.FP32)
                inside = pl.sub(local, pl.mul(group, merge_size * merge_size))
                block_row = pl.cast(pl.cast(pl.div(group, merge_columns), pl.INT32, mode="floor"), pl.FP32)
                inside_row = pl.cast(pl.cast(pl.div(inside, merge_size), pl.INT32, mode="floor"), pl.FP32)
                inside_col = pl.sub(inside, pl.mul(inside_row, merge_size))
                row = pl.add(pl.mul(block_row, merge_size), inside_row)
                block_col = pl.sub(group, pl.mul(block_row, merge_columns))
                col = pl.add(pl.mul(block_col, merge_size), inside_col)
                # 2. align_corners=True：将空间坐标映射到learned position table的二维位置。
                row_pos = pl.div(pl.mul(row, table_side - 1), row_divisor)
                col_pos = pl.div(pl.mul(col, table_side - 1), col_divisor)
                iy0 = pl.cast(row_pos, pl.INT32, mode="floor")
                ix0 = pl.cast(col_pos, pl.INT32, mode="floor")
                iy1 = pl.minimum(pl.add(iy0, 1), table_side - 1)
                ix1 = pl.minimum(pl.add(ix0, 1), table_side - 1)
                i00 = pl.add(pl.mul(iy0, table_side), ix0)
                i01 = pl.add(pl.mul(iy0, table_side), ix1)
                i10 = pl.add(pl.mul(iy1, table_side), ix0)
                i11 = pl.add(pl.mul(iy1, table_side), ix1)
                # 3. 四邻点的双线性权重，顺序为y0x0,y0x1,y1x0,y1x1。
                dy = pl.sub(row_pos, pl.cast(iy0, pl.FP32))
                dx = pl.sub(col_pos, pl.cast(ix0, pl.FP32))
                wy0 = pl.maximum(pl.add(pl.mul(pl.abs(dy), -1.0), 1.0), 0.0)
                wy1 = pl.maximum(pl.add(pl.mul(pl.abs(pl.sub(dy, 1.0)), -1.0), 1.0), 0.0)
                wx0 = pl.maximum(pl.add(pl.mul(pl.abs(dx), -1.0), 1.0), 0.0)
                wx1 = pl.maximum(pl.add(pl.mul(pl.abs(pl.sub(dx, 1.0)), -1.0), 1.0), 0.0)
                w00 = pl.mul(wy0, wx0)
                w01 = pl.mul(wy0, wx1)
                w10 = pl.mul(wy1, wx0)
                w11 = pl.mul(wy1, wx1)
                # 4. 按四个索引读取表行，FP32加权；位置向量先转BF16再与hidden相加。
                n00 = pl.cast(pl.tile.mgather(table, pl.reshape(i00, [1, physical_count]), coalesce="row"), pl.FP32)
                n01 = pl.cast(pl.tile.mgather(table, pl.reshape(i01, [1, physical_count]), coalesce="row"), pl.FP32)
                n10 = pl.cast(pl.tile.mgather(table, pl.reshape(i10, [1, physical_count]), coalesce="row"), pl.FP32)
                n11 = pl.cast(pl.tile.mgather(table, pl.reshape(i11, [1, physical_count]), coalesce="row"), pl.FP32)
                position = pl.row_expand_mul(n00, w00)
                position = pl.add(position, pl.row_expand_mul(n01, w01))
                position = pl.add(position, pl.row_expand_mul(n10, w10))
                position = pl.add(position, pl.row_expand_mul(n11, w11))
                position = pl.set_validshape(position, valid_count, hidden)
                position_rounded = pl.cast(pl.cast(position, pl.BF16), pl.FP32)
                x_tile = pl.cast(pl.load(x, [start, 0], [physical_count, hidden], valid_shape=[valid_count, hidden]), pl.FP32)
                output_tile = pl.cast(pl.add(x_tile, position_rounded), pl.BF16)
                out = pl.store(output_tile, [start, 0], out)
                # 5. 行主序打包四列元数据；只写回当前媒体有效Token。
                index_rows = pl.tile.full([physical_count, 8], dtype=pl.INT32, value=0)
                weight_rows = pl.tile.full([physical_count, 8], dtype=pl.FP32, value=0.0)
                for token in pl.range(valid_count):
                    pl.tile.write(index_rows, [token, 0], pl.read(i00, [token, 0]))
                    pl.tile.write(index_rows, [token, 1], pl.read(i01, [token, 0]))
                    pl.tile.write(index_rows, [token, 2], pl.read(i10, [token, 0]))
                    pl.tile.write(index_rows, [token, 3], pl.read(i11, [token, 0]))
                    pl.tile.write(weight_rows, [token, 0], pl.read(w00, [token, 0]))
                    pl.tile.write(weight_rows, [token, 1], pl.read(w01, [token, 0]))
                    pl.tile.write(weight_rows, [token, 2], pl.read(w10, [token, 0]))
                    pl.tile.write(weight_rows, [token, 3], pl.read(w11, [token, 0]))
                indices = pl.store(pl.set_validshape(index_rows, valid_count, 4), [start, 0], indices)
                weights = pl.store(pl.set_validshape(weight_rows, valid_count, 4), [start, 0], weights)
                _packed_end = pl.yield_(start + valid_count)
        return indices, weights, out

    return kernel


@lru_cache(maxsize=64)
def dense_ffn(rows: int, hidden: int, intermediate: int):
    @pl.jit
    def kernel(x: pl.Tensor, w1: pl.Tensor, b1: pl.Tensor, w2: pl.Tensor, b2: pl.Tensor,
               out: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            # OP-10：Linear -> BF16 -> GELU(tanh) -> BF16 -> Linear。
            first = pl.matmul(x, w1, out_dtype=pl.FP32, b_trans=True)
            first_bf16 = pl.cast(pl.col_expand_add(first, pl.cast(b1, pl.FP32)), pl.BF16)
            u = pl.cast(first_bf16, pl.FP32)
            cubic = pl.mul(pl.mul(pl.mul(u, 0.044715), u), u)
            argument = pl.mul(pl.add(u, cubic), 0.7978845608028654)
            tanh_value = pl.sub(pl.mul(pl.recip(pl.add(pl.exp(pl.mul(argument, -2.0)), 1.0)), 2.0), 1.0)
            activated = pl.cast(pl.mul(pl.mul(u, 0.5), pl.add(tanh_value, 1.0)), pl.BF16)
            second = pl.matmul(activated, w2, out_dtype=pl.FP32, b_trans=True)
            out[:] = pl.cast(pl.col_expand_add(second, pl.cast(b2, pl.FP32)), pl.BF16)
        return out

    return kernel


@lru_cache(maxsize=64)
def spatial_merger(rows: int, hidden: int, intermediate: int, output_width: int, *,
                   merge_size: int = 2, eps: float = 1e-6):
    group = merge_size * merge_size
    groups = rows // group

    @pl.jit
    def kernel(x: pl.Tensor, norm_weight: pl.Tensor, norm_bias: pl.Tensor, w1: pl.Tensor,
               b1: pl.Tensor, w2: pl.Tensor, b2: pl.Tensor, out: pl.Out[pl.Tensor]):
        grouped_buffer = pl.create_tensor([groups, group * hidden], dtype=pl.BF16)
        gelu_input = pl.create_tensor([groups, intermediate], dtype=pl.BF16)
        gelu_output = pl.create_tensor([groups, intermediate], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP):
            # OP-12：逐 Patch LayerNorm，FP32 均值、中心化方差及仿射。
            xf = pl.cast(x, pl.FP32)
            mean = pl.div(pl.row_sum(xf), hidden)
            centered = pl.row_expand_sub(xf, mean)
            variance = pl.div(pl.row_sum(pl.mul(centered, centered)), hidden)
            normalized = pl.row_expand_mul(centered, pl.rsqrt(pl.add(variance, eps)))
            affine = pl.col_expand_add(pl.col_expand_mul(normalized, pl.cast(norm_weight, pl.FP32)), pl.cast(norm_bias, pl.FP32))
            grouped_buffer[:] = pl.reshape(pl.cast(affine, pl.BF16), [groups, group * hidden])
        with pl.at(level=pl.Level.CORE_GROUP):
            first = pl.matmul(grouped_buffer, w1, out_dtype=pl.FP32, b_trans=True)
            gelu_input[:] = pl.cast(pl.col_expand_add(first, pl.cast(b1, pl.FP32)), pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP):
            u = pl.cast(pl.load(gelu_input, [0, 0], [groups, intermediate]), pl.FP32)
            # GELU(erf)：多项式近似 Phi(-abs(u))；独立 reference 使用 F.gelu。
            z = pl.mul(pl.abs(u), 0.7071067811865476)
            t = pl.recip(pl.add(pl.mul(z, 0.3275911), 1.0))
            polynomial = pl.sub(pl.mul(t, 1.061405429), 1.453152027)
            polynomial = pl.add(pl.mul(polynomial, t), 1.421413741)
            polynomial = pl.sub(pl.mul(polynomial, t), 0.284496736)
            polynomial = pl.add(pl.mul(polynomial, t), 0.254829592)
            tail = pl.mul(pl.mul(pl.mul(polynomial, t), 0.5), pl.exp(pl.mul(pl.mul(u, -0.5), u)))
            # 避免依赖高层 where：符号分支写成原生比较和 Tile SELECT。
            cdf = pl.tile.select(pl.tile.cmps(u, 0.0, cmp_type=2), tail, pl.add(pl.mul(tail, -1.0), 1.0))
            activated = pl.cast(pl.mul(u, cdf), pl.BF16)
            pl.store(activated, [0, 0], gelu_output)
        with pl.at(level=pl.Level.CORE_GROUP):
            second = pl.matmul(gelu_output, w2, out_dtype=pl.FP32, b_trans=True)
            out[:] = pl.cast(pl.col_expand_add(second, pl.cast(b2, pl.FP32)), pl.BF16)
        return out

    return kernel


@lru_cache(maxsize=64)
def vision_attention(rows: int, hidden: int, heads: int, boundaries: tuple[int, ...]):
    """OP-09：QKV -> half-split RoPE -> 分段QK/Softmax/PV -> 输出投影。"""
    head_dim, half = hidden // heads, hidden // heads // 2
    segment_count = len(boundaries) - 1
    token_tile = (max(b - a for a, b in zip(boundaries, boundaries[1:])) + 15) // 16 * 16
    feature_tile = (head_dim + 15) // 16 * 16
    physical_rows = (rows + 15) // 16 * 16
    physical_hidden = (hidden + 15) // 16 * 16

    @pl.jit
    def kernel(x: pl.Tensor, qkv_weight: pl.Tensor, qkv_bias: pl.Tensor,
               proj_weight: pl.Tensor, proj_bias: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor,
               segments: pl.Tensor, out: pl.Out[pl.Tensor]):
        context = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        qkv_buffer = pl.create_tensor([rows, 3 * hidden], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP):
            # 1. QKV = BF16(FP32_accumulate(x @ W_qkv.T) + FP32(bias))。
            projected = pl.matmul(x, qkv_weight, out_dtype=pl.FP32, b_trans=True)
            qkv_buffer[:] = pl.cast(pl.col_expand_add(projected, pl.cast(qkv_bias, pl.FP32)), pl.BF16)
        for head in pl.unroll(heads):
            # 每个head独立存储，避免前一head的消费者尚未完成就复用缓冲。
            q_buffer = pl.create_tensor([rows, head_dim], dtype=pl.BF16)
            k_buffer = pl.create_tensor([rows, head_dim], dtype=pl.BF16)
            v_buffer = pl.create_tensor([rows, head_dim], dtype=pl.BF16)
            head_context = pl.create_tensor([rows, head_dim], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP):
                # 2. half-split RoPE：[a,b] -> [-b,a]；FP32旋转后舍入BF16。
                begin_q = head * head_dim
                begin_k = hidden + begin_q
                begin_v = 2 * hidden + begin_q
                q32 = pl.cast(qkv_buffer[:, begin_q:begin_q + head_dim], pl.FP32)
                k32 = pl.cast(qkv_buffer[:, begin_k:begin_k + head_dim], pl.FP32)
                q_half = pl.concat(pl.mul(q32[:, half:], -1.0), q32[:, :half])
                k_half = pl.concat(pl.mul(k32[:, half:], -1.0), k32[:, :half])
                q_buffer[:] = pl.cast(pl.add(pl.mul(q32, cos), pl.mul(q_half, sin)), pl.BF16)
                k_buffer[:] = pl.cast(pl.add(pl.mul(k32, cos), pl.mul(k_half, sin)), pl.BF16)
                v_buffer[:] = qkv_buffer[:, begin_v:begin_v + head_dim]
            with pl.at(level=pl.Level.CORE_GROUP):
                # 3. 每个图像/时间片段独立做QK、Softmax、PV，不跨片段混合Token。
                for segment in pl.range(segment_count):
                    begin = pl.read(segments, [0, segment])
                    end = pl.read(segments, [0, segment + 1])
                    length = pl.cast(end - begin, pl.INDEX)
                    q_mat = pl.load(q_buffer, [begin, 0], [token_tile, feature_tile], valid_shape=[length, head_dim],
                        target_memory=pl.Mem.Mat, clamp=True)
                    k_mat = pl.load(k_buffer, [begin, 0], [token_tile, feature_tile], valid_shape=[length, head_dim],
                        target_memory=pl.Mem.Mat, clamp=True)
                    q_left = pl.move(q_mat, target_memory=pl.Mem.Left)
                    k_right = pl.move(pl.tile.transpose_view(k_mat), target_memory=pl.Mem.Right)
                    score_acc = pl.set_validshape(pl.matmul(q_left, k_right), length, length)
                    scores = pl.cast(score_acc, pl.BF16)
                    # QK点积与1/sqrt(K)缩放各自保留BF16边界，再进行FP32 Softmax。
                    scaled = pl.mul(pl.cast(scores, pl.FP32), head_dim ** -0.5)
                    scores32 = pl.cast(pl.cast(scaled, pl.BF16), pl.FP32)
                    reduce_tmp = pl.create_tile([token_tile, token_tile], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    maximum = pl.row_max(scores32, reduce_tmp)
                    exponent = pl.exp(pl.row_expand_sub(scores32, maximum))
                    denominator = pl.row_sum(exponent, reduce_tmp)
                    probabilities = pl.cast(pl.row_expand_div(exponent, denominator), pl.BF16)
                    probabilities = pl.set_validshape(pl.fillpad(probabilities, pad_value=pl.PadValue.zero), length, length)
                    p_left = pl.move(pl.move(probabilities, target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left)
                    v_mat = pl.load(v_buffer, [begin, 0], [token_tile, feature_tile], valid_shape=[length, head_dim],
                        target_memory=pl.Mem.Mat, clamp=True)
                    v_right = pl.move(v_mat, target_memory=pl.Mem.Right)
                    attended = pl.set_validshape(pl.matmul(p_left, v_right), length, head_dim)
                    head_context = pl.store(pl.cast(attended, pl.BF16), [begin, 0], head_context)
            with pl.at(level=pl.Level.CORE_GROUP):
                # 4. 将各head的context写入对应特征段。
                head_tile = pl.load(head_context, [0, 0], [rows, head_dim])
                context = pl.store(head_tile, [0, head * head_dim], context)
        projected_out = pl.create_tensor([rows, hidden], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP):
            # 5. 输出投影：FP32累加，独立bias阶段后回转BF16。
            output_lhs = pl.load(context, [0, 0], [physical_rows, physical_hidden], valid_shape=[rows, hidden], target_memory=pl.Mem.Mat)
            output_rhs = pl.load(proj_weight, [0, 0], [physical_hidden, physical_hidden], valid_shape=[hidden, hidden], target_memory=pl.Mem.Mat)
            output_left = pl.move(output_lhs, target_memory=pl.Mem.Left)
            output_right = pl.move(pl.tile.transpose_view(output_rhs), target_memory=pl.Mem.Right)
            output_acc = pl.set_validshape(pl.matmul(output_left, output_right), rows, hidden)
            projected_out = pl.store(output_acc, [0, 0], projected_out)
        with pl.at(level=pl.Level.CORE_GROUP):
            dot = pl.load(projected_out, [0, 0], [rows, hidden])
            bias = pl.cast(pl.load(proj_bias, [0, 0], [1, hidden]), pl.FP32)
            out = pl.store(pl.cast(pl.col_expand_add(dot, bias), pl.BF16), [0, 0], out)
        return out

    return kernel


@lru_cache(maxsize=64)
def visual_feature_scatter(text_rows: int, hidden: int, feature_rows: int):
    @pl.jit
    def kernel(text: pl.Tensor, features: pl.Tensor, indices: pl.Tensor, out: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            # OP-13：先精确复制文本；唯一占位行由视觉特征覆盖，其余行保持不变。
            for row in pl.range(text_rows):
                for begin in pl.range(0, hidden, 128):
                    valid = pl.min(128, hidden - begin)
                    original = pl.load(text, [row, begin], [1, 128], valid_shape=[1, valid])
                    out = pl.store(original, [row, begin], out)
            for feature in pl.range(feature_rows):
                destination = pl.read(indices, [0, feature])
                for column in pl.range(0, hidden, 128):
                    count = pl.min(128, hidden - column)
                    values = pl.load(features, [feature, column], [1, 128], valid_shape=[1, count])
                    out = pl.store(values, [destination, column], out)
        return out

    return kernel
