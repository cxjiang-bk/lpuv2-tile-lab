"""Complete Vision kernels written with hw-native-sys/PyPTO.

Tensor expressions are lowered to explicit PTO Tile operations by the compiler.
Window addressing, cast boundaries, and token order stay visible in this source.
"""

from functools import lru_cache

import pypto.language as pl


@lru_cache(maxsize=64)
def image_windows(image_shape: tuple[int, ...], kernel_size: tuple[int, int, int], stride: tuple[int, int, int], input_dtype: str):
    """Native im2col stages; only bounded coordinate Tiles are materialized."""
    batch, channels, time, height, width = image_shape
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    ot, oh, ow = (time - kt) // st + 1, (height - kh) // sh + 1, (width - kw) // sw + 1
    rows, patch_width = batch * ot * oh * ow, channels * kt * kh * kw
    image_size = batch * channels * time * height * width

    @pl.jit.incore
    def widen(pixels: pl.Tensor, image: pl.Out[pl.Tensor]):
        for begin in pl.range(0, image_size, 1024):
            valid = pl.min(1024, image_size - begin)
            values = pl.load(pixels, [0, begin], [1, 1024], valid_shape=[1, valid])
            if input_dtype == "float32":
                image = pl.store(values, [0, begin], image)
            else:
                image = pl.store(pl.cast(values, pl.FP32), [0, begin], image)
        return image

    @pl.jit.incore
    def extract(image: pl.Tensor, columns: pl.Out[pl.Tensor]):
        for row in pl.range(rows):
            b = row // (ot * oh * ow)
            t = (row // (oh * ow)) % ot
            h = (row // ow) % oh
            w = row % ow
            for begin in pl.range(0, patch_width, 128):
                valid = pl.min(128, patch_width - begin)
                index = pl.add(pl.cast(pl.tile.ci(0, [1, 128]), pl.FP32), pl.cast(pl.cast(begin, pl.INT32), pl.FP32))
                c = pl.cast(pl.cast(pl.div(index, kt * kh * kw), pl.INT32, mode="floor"), pl.FP32)
                dt = pl.sub(pl.cast(pl.cast(pl.div(index, kh * kw), pl.INT32, mode="floor"), pl.FP32), pl.mul(c, kt))
                dh = pl.sub(pl.cast(pl.cast(pl.div(index, kw), pl.INT32, mode="floor"), pl.FP32), pl.mul(pl.add(pl.mul(c, kt), dt), kh))
                dw = pl.sub(index, pl.mul(pl.add(pl.mul(pl.add(pl.mul(c, kt), dt), kh), dh), kw))
                # NCTHW 窗口地址；重叠窗口直接重复 Gather，不由 host 切 Patch。
                address = pl.add(c, pl.cast(pl.cast(b * channels, pl.INT32), pl.FP32))
                address = pl.add(pl.add(pl.mul(address, time), pl.cast(pl.cast(t * st, pl.INT32), pl.FP32)), dt)
                address = pl.add(pl.add(pl.mul(address, height), pl.cast(pl.cast(h * sh, pl.INT32), pl.FP32)), dh)
                address = pl.add(pl.add(pl.mul(address, width), pl.cast(pl.cast(w * sw, pl.INT32), pl.FP32)), dw)
                indices = pl.set_validshape(pl.cast(address, pl.INT32), 1, valid)
                windows = pl.tile.mgather(image, indices, coalesce="elem")
                columns = pl.store(pl.cast(windows, pl.BF16), [row, begin], columns)
        return columns

    @pl.jit.inline(auto_scope=False)
    def windows(pixels: pl.Tensor, columns: pl.Out[pl.Tensor]):
        flat = pl.reshape(pixels, [1, image_size])
        image = pl.create_tensor([1, image_size], dtype=pl.FP32)
        image = widen(flat, image)
        columns = extract(image, columns)
        return columns

    return windows


@lru_cache(maxsize=64)
def im2col(image_shape: tuple[int, ...], kernel_size: tuple[int, int, int],
           stride: tuple[int, int, int], *, input_dtype: str = "bfloat16"):
    windows = image_windows(image_shape, kernel_size, stride, input_dtype)

    @pl.jit
    def kernel(pixels: pl.Tensor, columns: pl.Out[pl.Tensor]):
        columns = windows(pixels, columns)
        return columns

    return kernel


@lru_cache(maxsize=64)
def patch_embedding(image_shape: tuple[int, ...], hidden: int, kernel_size: tuple[int, int, int], *,
                    stride: tuple[int, int, int], spatial_merge_size: int = 2, input_dtype: str = "bfloat16"):
    batch, channels, time, height, width = image_shape
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    ot, oh, ow = (time - kt) // st + 1, (height - kh) // sh + 1, (width - kw) // sw + 1
    rows, patch_width = batch * ot * oh * ow, channels * kt * kh * kw
    m = spatial_merge_size
    windows = image_windows(image_shape, kernel_size, stride, input_dtype)

    @pl.jit.incore
    def project(columns: pl.Tensor, weight: pl.Tensor, bias: pl.Tensor, raster: pl.Out[pl.Tensor]):
        for row in pl.range(0, rows, 16):
            for feature in pl.range(0, hidden, 128):
                mr = pl.min(16, rows - row)
                nr = pl.min(128, hidden - feature)
                kr = pl.min(128, patch_width)
                a0 = pl.load(columns, [row, 0], [16, 128], valid_shape=[mr, kr], target_memory=pl.Mem.Mat)
                b0 = pl.load(weight, [feature, 0], [128, 128], valid_shape=[nr, kr], target_memory=pl.Mem.Mat)
                initial = pl.matmul(pl.move(a0, target_memory=pl.Mem.Left), pl.move(pl.tile.transpose_view(b0), target_memory=pl.Mem.Right))
                for kbegin, (acc,) in pl.range(128, (patch_width + 127) // 128 * 128, 128, init_values=(initial,)):
                    kval = pl.min(128, patch_width - kbegin)
                    a = pl.load(columns, [row, kbegin], [16, 128], valid_shape=[mr, kval], target_memory=pl.Mem.Mat)
                    b = pl.load(weight, [feature, kbegin], [128, 128], valid_shape=[nr, kval], target_memory=pl.Mem.Mat)
                    updated = pl.matmul_acc(acc, pl.move(a, target_memory=pl.Mem.Left), pl.move(pl.tile.transpose_view(b), target_memory=pl.Mem.Right))
                    total = pl.yield_(updated)
                # FP32 部分和跨 K 块累加，bias 后才舍入 BF16。
                dot = pl.set_validshape(total, mr, nr)
                bt = pl.cast(pl.load(bias, [0, feature], [1, 128], valid_shape=[1, nr]), pl.FP32)
                raster = pl.store(pl.cast(pl.col_expand_add(dot, bt), pl.BF16), [row, feature], raster)
        return raster

    @pl.jit.incore
    def reorder(raster: pl.Tensor, out: pl.Out[pl.Tensor], grid_thw: pl.Out[pl.Tensor]):
        for serial in pl.range(rows):
            frame = serial // (oh * ow)
            local = serial % (oh * ow)
            group = local // (m * m)
            inside = local % (m * m)
            row = (group // (ow // m)) * m + inside // m
            col = (group % (ow // m)) * m + inside % m
            source = frame * oh * ow + row * ow + col
            for feature in pl.range(0, hidden, 128):
                values = pl.load(raster, [source, feature], [1, 128], valid_shape=[1, pl.min(128, hidden - feature)])
                out = pl.store(values, [serial, feature], out)
        for b in pl.range(batch):
            pl.write(grid_thw, [b, 0], pl.const(ot, pl.INT64))
            pl.write(grid_thw, [b, 1], pl.const(oh, pl.INT64))
            pl.write(grid_thw, [b, 2], pl.const(ow, pl.INT64))
        return out, grid_thw

    @pl.jit
    def kernel(pixels: pl.Tensor, weight: pl.Tensor, bias: pl.Tensor,
               out: pl.Out[pl.Tensor], grid_thw: pl.Out[pl.Tensor]):
        columns = pl.create_tensor([rows, patch_width], dtype=pl.BF16)
        raster = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        columns = windows(pixels, columns)
        raster = project(columns, weight, bias, raster)
        out, grid_thw = reorder(raster, out, grid_thw)
        return out, grid_thw

    return kernel


@lru_cache(maxsize=64)
def position_interpolation(grids: tuple[tuple[int, int, int], ...], hidden: int, *,
                           table_side: int, merge_size: int, start: int = 0):
    """One media's interpolation; remaining media are specialized as inline stages."""
    t, h, w = grids[0]
    count, end = t * h * w, start + t * h * w
    physical_count = (count + 7) // 8 * 8
    row_divisor, col_divisor = max(h - 1, 1), max(w - 1, 1)
    remaining = len(grids) > 1
    if remaining:
        rest = position_interpolation(grids[1:], hidden, table_side=table_side, merge_size=merge_size, start=end)

    @pl.jit.inline(auto_scope=False)
    def kernel(x: pl.Tensor, table: pl.Tensor, indices: pl.Out[pl.Tensor],
               weights: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
        # OP-11：按连续 m*m Merge 组还原二维空间位置，各时间片复用位置表。
        serial = pl.reshape(pl.set_validshape(pl.cast(pl.tile.ci(0, [1, physical_count]), pl.FP32), 1, count), [physical_count, 1])
        local = pl.sub(serial, pl.mul(pl.cast(pl.cast(pl.div(serial, h * w), pl.INT32, mode="floor"), pl.FP32), h * w))
        group = pl.cast(pl.cast(pl.div(local, merge_size * merge_size), pl.INT32, mode="floor"), pl.FP32)
        inside = pl.sub(local, pl.mul(group, merge_size * merge_size))
        block_row = pl.cast(pl.cast(pl.div(group, w // merge_size), pl.INT32, mode="floor"), pl.FP32)
        row = pl.add(pl.mul(block_row, merge_size), pl.cast(pl.cast(pl.div(inside, merge_size), pl.INT32, mode="floor"), pl.FP32))
        block_col = pl.sub(group, pl.mul(block_row, w // merge_size))
        col = pl.add(pl.mul(block_col, merge_size), pl.sub(inside, pl.mul(pl.cast(pl.cast(pl.div(inside, merge_size), pl.INT32, mode="floor"), pl.FP32), merge_size)))
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
        # align_corners=True；四邻点权重按 y0x0,y0x1,y1x0,y1x1 排列。
        dy = pl.sub(row_pos, pl.cast(iy0, pl.FP32))
        dx = pl.sub(col_pos, pl.cast(ix0, pl.FP32))
        wy0 = pl.maximum(pl.add(pl.mul(pl.abs(dy), -1.0), 1.0), 0.0)
        wy1 = pl.maximum(pl.add(pl.mul(pl.abs(pl.sub(dy, 1.0)), -1.0), 1.0), 0.0)
        wx0 = pl.maximum(pl.add(pl.mul(pl.abs(dx), -1.0), 1.0), 0.0)
        wx1 = pl.maximum(pl.add(pl.mul(pl.abs(pl.sub(dx, 1.0)), -1.0), 1.0), 0.0)
        # 四次按行 MGATHER，直接从 GM 读取 BF16 位置表，不物化整表。
        n00 = pl.cast(pl.tile.mgather(table, pl.reshape(i00, [1, physical_count]), coalesce="row"), pl.FP32)
        n01 = pl.cast(pl.tile.mgather(table, pl.reshape(i01, [1, physical_count]), coalesce="row"), pl.FP32)
        n10 = pl.cast(pl.tile.mgather(table, pl.reshape(i10, [1, physical_count]), coalesce="row"), pl.FP32)
        n11 = pl.cast(pl.tile.mgather(table, pl.reshape(i11, [1, physical_count]), coalesce="row"), pl.FP32)
        position = pl.row_expand_mul(n00, pl.mul(wy0, wx0))
        position = pl.add(position, pl.row_expand_mul(n01, pl.mul(wy0, wx1)))
        position = pl.add(position, pl.row_expand_mul(n10, pl.mul(wy1, wx0)))
        position = pl.add(position, pl.row_expand_mul(n11, pl.mul(wy1, wx1)))
        # 先舍入位置向量，再与 BF16 hidden 相加；索引和权重分别返回。
        x_tile = pl.cast(pl.load(x, [start, 0], [physical_count, hidden], valid_shape=[count, hidden]), pl.FP32)
        position_rounded = pl.cast(pl.cast(position, pl.BF16), pl.FP32)
        output_tile = pl.cast(pl.add(x_tile, position_rounded), pl.BF16)
        out = pl.store(output_tile, [start, 0], out)
        # 先在32B对齐的行主序 Tile 中组织4列元数据，再一次写回窄 GM 行。
        # 逐列 TSTORE 会使列主序源与行主序目标的步长约定不一致。
        index_rows = pl.tile.full([physical_count, 8], dtype=pl.INT32, value=0)
        weight_rows = pl.tile.full([physical_count, 8], dtype=pl.FP32, value=0.0)
        w00 = pl.mul(wy0, wx0)
        w01 = pl.mul(wy0, wx1)
        w10 = pl.mul(wy1, wx0)
        w11 = pl.mul(wy1, wx1)
        for token in pl.range(count):
            pl.tile.write(index_rows, [token, 0], pl.read(i00, [token, 0]))
            pl.tile.write(index_rows, [token, 1], pl.read(i01, [token, 0]))
            pl.tile.write(index_rows, [token, 2], pl.read(i10, [token, 0]))
            pl.tile.write(index_rows, [token, 3], pl.read(i11, [token, 0]))
            pl.tile.write(weight_rows, [token, 0], pl.read(w00, [token, 0]))
            pl.tile.write(weight_rows, [token, 1], pl.read(w01, [token, 0]))
            pl.tile.write(weight_rows, [token, 2], pl.read(w10, [token, 0]))
            pl.tile.write(weight_rows, [token, 3], pl.read(w11, [token, 0]))
        indices = pl.store(pl.set_validshape(index_rows, count, 4), [start, 0], indices)
        weights = pl.store(pl.set_validshape(weight_rows, count, 4), [start, 0], weights)
        if remaining:
            indices, weights, out = rest(x, table, indices, weights, out)
        return indices, weights, out

    return kernel


@lru_cache(maxsize=64)
def position_embedding(grids: tuple[tuple[int, int, int], ...], hidden: int, *, table_side: int,
                       merge_size: int, tile_cols: int):
    interpolate = position_interpolation(grids, hidden, table_side=table_side, merge_size=merge_size)

    @pl.jit
    def kernel(x: pl.Tensor, table: pl.Tensor, indices: pl.Out[pl.Tensor],
               weights: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            indices, weights, out = interpolate(x, table, indices, weights, out)
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
def attention_segments(boundaries: tuple[int, ...], rows: int, head_dim: int):
    """One packed segment: padded physical Tile geometry, exact logical extents."""
    begin, end = boundaries[:2]
    length = end - begin
    token_tile = (length + 15) // 16 * 16
    feature_tile = (head_dim + 15) // 16 * 16
    remaining = len(boundaries) > 2
    if remaining:
        rest = attention_segments(boundaries[1:], rows, head_dim)

    @pl.jit.inline(auto_scope=False)
    def kernel(q: pl.Tensor[[rows, head_dim], pl.BF16], k: pl.Tensor[[rows, head_dim], pl.BF16],
               v: pl.Tensor[[rows, head_dim], pl.BF16], context: pl.Out[pl.Tensor[[rows, head_dim], pl.BF16]]):
        # Q/K 的物理 Tile 均补齐 Cube 的 16 元素块，有效区域只含当前片段。
        q_mat = pl.load(q, [begin, 0], [token_tile, feature_tile],
                        valid_shape=[length, head_dim], target_memory=pl.Mem.Mat, clamp=True)
        k_mat = pl.load(k, [begin, 0], [token_tile, feature_tile],
                        valid_shape=[length, head_dim], target_memory=pl.Mem.Mat, clamp=True)
        q_left = pl.move(q_mat, target_memory=pl.Mem.Left)
        k_right = pl.move(pl.tile.transpose_view(k_mat), target_memory=pl.Mem.Right)
        score_acc = pl.set_validshape(pl.matmul(q_left, k_right), length, length)
        scores = pl.cast(score_acc, pl.BF16)
        # QK 与缩放各自保留 BF16 舍入，再执行 FP32 Softmax。
        scaled = pl.mul(pl.cast(scores, pl.FP32), head_dim ** -0.5)
        scores32 = pl.cast(pl.cast(scaled, pl.BF16), pl.FP32)
        reduce_tmp = pl.create_tile([token_tile, token_tile], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        maximum = pl.row_max(scores32, reduce_tmp)
        exponent = pl.exp(pl.row_expand_sub(scores32, maximum))
        denominator = pl.row_sum(exponent, reduce_tmp)
        probabilities = pl.cast(pl.row_expand_div(exponent, denominator), pl.BF16)
        probabilities = pl.set_validshape(pl.fillpad(probabilities, pad_value=pl.PadValue.zero), length, length)
        # P@V 的两个 K 轴共用 token_tile，尾部有效长度始终为 length。
        p_left = pl.move(pl.move(probabilities, target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left)
        v_mat = pl.load(v, [begin, 0], [token_tile, feature_tile],
                        valid_shape=[length, head_dim], target_memory=pl.Mem.Mat, clamp=True)
        v_right = pl.move(v_mat, target_memory=pl.Mem.Right)
        attended = pl.set_validshape(pl.matmul(p_left, v_right), length, head_dim)
        context = pl.store(pl.cast(attended, pl.BF16), [begin, 0], context)
        if remaining:
            context = rest(q, k, v, context)
        return context

    return kernel


@lru_cache(maxsize=64)
def attention_output_projection(rows: int, hidden: int):
    physical_rows = (rows + 15) // 16 * 16
    physical_hidden = (hidden + 15) // 16 * 16

    @pl.jit.incore
    def project(context: pl.Tensor, weight: pl.Tensor, dot: pl.Out[pl.Tensor]):
        a = pl.load(context, [0, 0], [physical_rows, physical_hidden],
                    valid_shape=[rows, hidden], target_memory=pl.Mem.Mat)
        b = pl.load(weight, [0, 0], [physical_hidden, physical_hidden],
                    valid_shape=[hidden, hidden], target_memory=pl.Mem.Mat)
        left = pl.move(a, target_memory=pl.Mem.Left)
        right = pl.move(pl.tile.transpose_view(b), target_memory=pl.Mem.Right)
        result = pl.set_validshape(pl.matmul(left, right), rows, hidden)
        dot = pl.store(result, [0, 0], dot)
        return dot

    return project


@lru_cache(maxsize=64)
def vision_attention(rows: int, hidden: int, heads: int, boundaries: tuple[int, ...]):
    head_dim, half = hidden // heads, hidden // heads // 2
    attend = attention_segments(boundaries, rows, head_dim)
    project_output = attention_output_projection(rows, hidden)

    @pl.jit
    def kernel(x: pl.Tensor, qkv_weight: pl.Tensor, qkv_bias: pl.Tensor,
               proj_weight: pl.Tensor, proj_bias: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor,
               out: pl.Out[pl.Tensor]):
        context = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        qkv_buffer = pl.create_tensor([rows, 3 * hidden], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP):
            # OP-09：BF16 输入投影；各 head 共用同一 QKV 投影结果。
            projected = pl.matmul(x, qkv_weight, out_dtype=pl.FP32, b_trans=True)
            qkv_buffer[:] = pl.cast(pl.col_expand_add(projected, pl.cast(qkv_bias, pl.FP32)), pl.BF16)
        for head in pl.unroll(heads):
            q_buffer = pl.create_tensor([rows, head_dim], dtype=pl.BF16)
            k_buffer = pl.create_tensor([rows, head_dim], dtype=pl.BF16)
            v_buffer = pl.create_tensor([rows, head_dim], dtype=pl.BF16)
            head_context = pl.create_tensor([rows, head_dim], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP):
                begin_q = head * head_dim
                begin_k = hidden + begin_q
                begin_v = 2 * hidden + begin_q
                q32 = pl.cast(qkv_buffer[:, begin_q:begin_q + head_dim], pl.FP32)
                k32 = pl.cast(qkv_buffer[:, begin_k:begin_k + head_dim], pl.FP32)
                # half-split RoPE：[a,b] -> [-b,a]，FP32 旋转后舍入 BF16。
                q_half = pl.concat(pl.mul(q32[:, half:], -1.0), q32[:, :half])
                k_half = pl.concat(pl.mul(k32[:, half:], -1.0), k32[:, :half])
                q_buffer[:] = pl.cast(pl.add(pl.mul(q32, cos), pl.mul(q_half, sin)), pl.BF16)
                k_buffer[:] = pl.cast(pl.add(pl.mul(k32, cos), pl.mul(k_half, sin)), pl.BF16)
                v_buffer[:] = qkv_buffer[:, begin_v:begin_v + head_dim]
            with pl.at(level=pl.Level.CORE_GROUP):
                # 每个片段独立完成 QK、FP32 Softmax 和 PV。
                head_context = attend(q_buffer, k_buffer, v_buffer, head_context)
            with pl.at(level=pl.Level.CORE_GROUP):
                head_tile = pl.load(head_context, [0, 0], [rows, head_dim])
                context = pl.store(head_tile, [0, head * head_dim], context)
        projected_out = pl.create_tensor([rows, hidden], dtype=pl.FP32)
        projected_out = project_output(context, proj_weight, projected_out)
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
