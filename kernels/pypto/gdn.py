"""Complete K-06 in hw-native-sys/PyPTO: FP8 projections and FP32 State.

Recurrent and chunk are two algorithms inside one logical operator. Chunk uses
prefix decays, unit-lower solves and a State scan, not recurrent updates in blocks.
"""

from functools import lru_cache

import pypto.language as pl

from kernels.pypto.fp8 import fp8_linear_bf16


@lru_cache(maxsize=32)
def gdn_recurrent(sequence: int, key_dim: int, value_dim: int, eps: float = 1e-6):
    key_tile = (key_dim + 31) // 32 * 32
    value_tile = (value_dim + 31) // 32 * 32

    @pl.jit.incore
    def core(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, beta: pl.Tensor, g: pl.Tensor,
             initial_state: pl.Tensor, out: pl.Out[pl.Tensor], state_out: pl.Out[pl.Tensor]):
        initial = pl.load(initial_state, [0, 0], [key_tile, value_tile], valid_shape=[key_dim, value_dim])
        for token, (current,) in pl.range(sequence, init_values=(initial,)):
            # 单 Token Q/K 物理补足8行，使 [M,1] 归约结果满足32B对齐；仅首行有效。
            qt = pl.cast(pl.load(query, [token, 0], [8, key_tile], valid_shape=[1, key_dim]), pl.FP32)
            kt = pl.cast(pl.load(key, [token, 0], [8, key_tile], valid_shape=[1, key_dim]), pl.FP32)
            q_tmp = pl.create_tile([8, key_tile], dtype=pl.FP32)
            k_tmp = pl.create_tile([8, key_tile], dtype=pl.FP32)
            qt = pl.row_expand_mul(qt, pl.rsqrt(pl.add(pl.row_sum(pl.mul(qt, qt), q_tmp), eps)))
            kt = pl.row_expand_mul(kt, pl.rsqrt(pl.add(pl.row_sum(pl.mul(kt, kt), k_tmp), eps)))
            qt = pl.mul(qt, key_dim ** -0.5)
            # 单行向量重解释为列向量，不执行转置搬运。
            q_col = pl.reshape(pl.tile.slice(qt, [1, key_tile], [0, 0]), [key_tile, 1])
            k_col = pl.reshape(pl.tile.slice(kt, [1, key_tile], [0, 0]), [key_tile, 1])
            vt = pl.cast(pl.load(value, [token, 0], [1, value_tile], valid_shape=[1, value_dim]), pl.FP32)
            gate = pl.load(g, [token, 0], [1, 8], valid_shape=[1, 1])
            decay = pl.read(pl.exp(gate), [0, 0])
            beta_tile = pl.cast(pl.load(beta, [token, 0], [1, 16], valid_shape=[1, 1]), pl.FP32)
            bt = pl.read(beta_tile, [0, 0])
            # S_decay=exp(g_t)*S；prediction=k_t.T@S_decay。
            decayed = pl.mul(current, decay)
            prediction = pl.col_sum(pl.row_expand_mul(decayed, k_col))
            delta = pl.mul(pl.sub(vt, prediction), bt)
            # State=S_decay+k_t[:,None]*delta[None,:]；out=q_t.T@State。
            outer = pl.row_expand_mul(pl.col_expand(decayed, delta), k_col)
            updated = pl.add(decayed, outer)
            token_out = pl.col_sum(pl.row_expand_mul(updated, q_col))
            out = pl.store(pl.cast(token_out, pl.BF16), [token, 0], out)
            final_state = pl.yield_(updated)
        state_out = pl.store(final_state, [0, 0], state_out)
        return out, state_out

    return core


@lru_cache(maxsize=32)
def gdn_chunk_block(length: int, key_dim: int, value_dim: int, eps: float = 1e-6):
    chunk_tile = (length + 15) // 16 * 16
    key_tile = (key_dim + 31) // 32 * 32
    value_tile = (value_dim + 31) // 32 * 32

    @pl.jit.incore
    def block(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, beta: pl.Tensor, g: pl.Tensor,
              initial_state: pl.Tensor, out: pl.Out[pl.Tensor], state_out: pl.Out[pl.Tensor]):
        q = pl.cast(pl.load(query, [0, 0], [chunk_tile, key_tile], valid_shape=[length, key_dim]), pl.FP32)
        k = pl.cast(pl.load(key, [0, 0], [chunk_tile, key_tile], valid_shape=[length, key_dim]), pl.FP32)
        v = pl.cast(pl.load(value, [0, 0], [chunk_tile, value_tile], valid_shape=[length, value_dim]), pl.FP32)
        b = pl.cast(pl.load(beta, [0, 0], [chunk_tile, 1], valid_shape=[length, 1]), pl.FP32)
        gates = pl.load(g, [0, 0], [chunk_tile, 1], valid_shape=[length, 1])
        q = pl.fillpad(q, pad_value=pl.PadValue.zero)
        k = pl.fillpad(k, pad_value=pl.PadValue.zero)
        v = pl.fillpad(v, pad_value=pl.PadValue.zero)
        b = pl.fillpad(b, pad_value=pl.PadValue.zero)
        current = pl.load(initial_state, [0, 0], [key_tile, value_tile], valid_shape=[key_dim, value_dim])
        q_tmp = pl.create_tile([chunk_tile, key_tile], dtype=pl.FP32)
        k_tmp = pl.create_tile([chunk_tile, key_tile], dtype=pl.FP32)
        q = pl.row_expand_mul(q, pl.rsqrt(pl.add(pl.row_sum(pl.mul(q, q), q_tmp), eps)))
        k = pl.row_expand_mul(k, pl.rsqrt(pl.add(pl.row_sum(pl.mul(k, k), k_tmp), eps)))
        q = pl.mul(q, key_dim ** -0.5)
        kb = pl.row_expand_mul(k, b)
        vb = pl.row_expand_mul(v, b)

        # G_i=sum_{t<=i}g_t，每个 chunk 单独起算；标量循环表达精确前缀依赖。
        prefix_row = pl.tile.full([1, chunk_tile], dtype=pl.FP32, value=0.0)
        prefix = pl.const(0.0, pl.FP32)
        for row in pl.unroll(length):
            prefix = prefix + pl.read(gates, [row, 0])
            pl.tile.write(prefix_row, [0, row], prefix)
        cumulative = pl.reshape(prefix_row, [chunk_tile, 1])
        matrix_shape = pl.tile.full([chunk_tile, chunk_tile], dtype=pl.FP32, value=0.0)
        gi = pl.row_expand(matrix_shape, cumulative)
        gj = pl.col_expand(matrix_shape, pl.reshape(cumulative, [1, chunk_tile]))
        causal = pl.tri(0, [chunk_tile, chunk_tile], dtype=pl.FP32)
        # D_ij=exp(G_i-G_j),j<=i；无效上三角在 exp 之前置0，exp 后再乘 mask。
        decay = pl.mul(pl.exp(pl.mul(pl.sub(gi, gj), causal)), causal)
        system = pl.tile.full([chunk_tile, chunk_tile], dtype=pl.FP32, value=0.0)
        intra = pl.tile.full([chunk_tile, chunk_tile], dtype=pl.FP32, value=0.0)
        for row in pl.unroll(length):
            ki = pl.tile.slice(kb, [1, key_tile], [row, 0])
            qi = pl.tile.slice(q, [1, key_tile], [row, 0])
            scratch = pl.create_tile([chunk_tile, key_tile], dtype=pl.FP32)
            a_column = pl.row_sum(pl.col_expand_mul(k, ki), scratch)
            p_column = pl.row_sum(pl.col_expand_mul(k, qi), scratch)
            d_row = pl.tile.slice(decay, [1, chunk_tile], [row, 0])
            a_row = pl.mul(pl.reshape(a_column, [1, chunk_tile]), d_row)
            p_row = pl.mul(pl.reshape(p_column, [1, chunk_tile]), d_row)
            for column in pl.unroll(length):
                pl.tile.write(system, [row, column], pl.read(a_row, [0, column]))
                pl.tile.write(intra, [row, column], pl.read(p_row, [0, column]))
            pl.tile.write(system, [row, row], pl.const(0.0, pl.FP32))

        # 解 (I+strict_lower(A))*U=beta*V、(I+strict_lower(A))*W=beta*K*exp(G)。
        exp_g = pl.exp(cumulative)
        rhs_w = pl.row_expand_mul(kb, exp_g)
        u = pl.tile.full([chunk_tile, value_tile], dtype=pl.FP32, value=0.0)
        w = pl.tile.full([chunk_tile, key_tile], dtype=pl.FP32, value=0.0)
        for row in pl.unroll(length):
            coefficients = pl.reshape(pl.tile.slice(system, [1, chunk_tile], [row, 0]), [chunk_tile, 1])
            u_row = pl.sub(pl.tile.slice(vb, [1, value_tile], [row, 0]), pl.col_sum(pl.row_expand_mul(u, coefficients)))
            w_row = pl.sub(pl.tile.slice(rhs_w, [1, key_tile], [row, 0]), pl.col_sum(pl.row_expand_mul(w, coefficients)))
            for column in pl.unroll(value_dim):
                pl.tile.write(u, [row, column], pl.read(u_row, [0, column]))
            for column in pl.unroll(key_dim):
                pl.tile.write(w, [row, column], pl.read(w_row, [0, column]))

        q_exp = pl.row_expand_mul(q, exp_g)
        end_g = pl.read(cumulative, [length - 1, 0])
        k_tail = pl.row_expand_mul(k, pl.exp(pl.mul(pl.sub(cumulative, end_g), -1.0)))
        delta = pl.tile.full([chunk_tile, value_tile], dtype=pl.FP32, value=0.0)
        previous = pl.tile.full([chunk_tile, value_tile], dtype=pl.FP32, value=0.0)
        for row in pl.unroll(length):
            wi = pl.reshape(pl.tile.slice(w, [1, key_tile], [row, 0]), [key_tile, 1])
            q_state_column = pl.reshape(pl.tile.slice(q_exp, [1, key_tile], [row, 0]), [key_tile, 1])
            predicted = pl.col_sum(pl.row_expand_mul(current, wi))
            delta_row = pl.sub(pl.tile.slice(u, [1, value_tile], [row, 0], valid_shape=[1, value_dim]), predicted)
            previous_row = pl.col_sum(pl.row_expand_mul(current, q_state_column))
            for column in pl.unroll(value_dim):
                pl.tile.write(delta, [row, column], pl.read(delta_row, [0, column]))
                pl.tile.write(previous, [row, column], pl.read(previous_row, [0, column]))
        for row in pl.unroll(length):
            pi = pl.reshape(pl.tile.slice(intra, [1, chunk_tile], [row, 0]), [chunk_tile, 1])
            within = pl.col_sum(pl.row_expand_mul(delta, pi))
            result = pl.add(pl.tile.slice(previous, [1, value_tile], [row, 0]), within)
            out = pl.store(pl.cast(pl.set_validshape(result, 1, value_dim), pl.BF16), [row, 0], out)

        # K_tail.T@delta 按 Token 外积累加；State 相关计算全部保持 FP32。
        contribution = pl.set_validshape(pl.tile.full([key_tile, value_tile], dtype=pl.FP32, value=0.0), key_dim, value_dim)
        for row in pl.unroll(length):
            k_tail_column = pl.reshape(pl.tile.slice(k_tail, [1, key_tile], [row, 0]), [key_tile, 1])
            di = pl.tile.slice(delta, [1, value_tile], [row, 0])
            outer = pl.row_expand_mul(pl.col_expand(current, di), k_tail_column)
            contribution = pl.add(contribution, outer)
        next_state = pl.add(pl.mul(current, pl.read(exp_g, [length - 1, 0])), contribution)
        state_out = pl.store(next_state, [0, 0], state_out)
        return out, state_out

    return block


@lru_cache(maxsize=32)
def gdn_chunk(sequence: int, key_dim: int, value_dim: int, chunk_size: int = 64, eps: float = 1e-6):
    length = min(sequence, chunk_size)
    remainder = sequence - length
    block_kernel = gdn_chunk_block(length, key_dim, value_dim, eps)
    if remainder:
        rest = gdn_chunk(remainder, key_dim, value_dim, chunk_size, eps)

    @pl.jit.inline(auto_scope=False)
    def core(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, beta: pl.Tensor, g: pl.Tensor,
             initial_state: pl.Tensor, out: pl.Out[pl.Tensor], state_out: pl.Out[pl.Tensor]):
        q = pl.slice(query, [length, key_dim], [0, 0])
        k = pl.slice(key, [length, key_dim], [0, 0])
        v = pl.slice(value, [length, value_dim], [0, 0])
        b = pl.slice(beta, [length, 1], [0, 0])
        gates = pl.slice(g, [length, 1], [0, 0])
        destination = pl.slice(out, [length, value_dim], [0, 0])
        if remainder:
            middle = pl.create_tensor([key_dim, value_dim], dtype=pl.FP32)
            destination, middle = block_kernel(q, k, v, b, gates, initial_state, destination, middle)
            qr = pl.slice(query, [remainder, key_dim], [length, 0])
            kr = pl.slice(key, [remainder, key_dim], [length, 0])
            vr = pl.slice(value, [remainder, value_dim], [length, 0])
            br = pl.slice(beta, [remainder, 1], [length, 0])
            gr = pl.slice(g, [remainder, 1], [length, 0])
            output_rest = pl.slice(out, [remainder, value_dim], [length, 0])
            output_rest, state_out = rest(qr, kr, vr, br, gr, middle, output_rest, state_out)
        else:
            destination, state_out = block_kernel(q, k, v, b, gates, initial_state, destination, state_out)
        return out, state_out

    return core


@lru_cache(maxsize=32)
def gdn(batch: int, sequence: int, hidden: int, *, qk_heads: int, value_heads: int,
        key_dim: int, value_dim: int, conv_kernel: int = 4, eps: float = 1e-6,
        fp8_execution: str = "bf16_carrier", path: str = "recurrent", chunk_size: int = 64):
    if path not in ("recurrent", "chunk"):
        raise ValueError("path must be recurrent or chunk")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if fp8_execution != "bf16_carrier":
        raise ValueError("Full K-06 currently requires the exact BF16 FP8-value carrier on native A2/A3")
    if min(batch, sequence, hidden, qk_heads, value_heads, key_dim, value_dim) < 1 or value_heads % qk_heads:
        raise ValueError("Positive GDN dimensions and a whole Q/K-to-value-head ratio are required")
    rows = batch * sequence
    key_width, value_width = qk_heads * key_dim, value_heads * value_dim
    channels = 2 * key_width + value_width
    ratio = value_heads // qk_heads
    qkv_projection = fp8_linear_bf16(rows, hidden, channels)
    z_projection = fp8_linear_bf16(rows, hidden, value_width)
    out_projection = fp8_linear_bf16(rows, value_width, hidden)
    core_kernel = (gdn_chunk(sequence, key_dim, value_dim, chunk_size, eps) if path == "chunk"
                   else gdn_recurrent(sequence, key_dim, value_dim, eps))

    @pl.jit.incore
    def input_mask(x: pl.Tensor, mask: pl.Tensor, masked: pl.Out[pl.Tensor]):
        for row in pl.range(rows):
            xt = pl.cast(pl.load(x, [row, 0], [1, (hidden + 15) // 16 * 16], valid_shape=[1, hidden]), pl.FP32)
            keep = pl.cast(pl.read(mask, [row, 0]), pl.FP32)
            masked = pl.store(pl.cast(pl.mul(xt, keep), pl.BF16), [row, 0], masked)
        return masked

    @pl.jit.incore
    def ab_projection(flat: pl.Tensor, wa: pl.Tensor, wb: pl.Tensor,
                      a_out: pl.Out[pl.Tensor], b_out: pl.Out[pl.Tensor]):
        a = pl.cast(pl.matmul(flat, wa, out_dtype=pl.FP32, b_trans=True), pl.BF16, mode="rint")
        b = pl.cast(pl.matmul(flat, wb, out_dtype=pl.FP32, b_trans=True), pl.BF16, mode="rint")
        a_out = pl.assemble(a_out, a, [0, 0])
        b_out = pl.assemble(b_out, b, [0, 0])
        return a_out, b_out

    @pl.jit.incore
    def gates(a: pl.Tensor, b: pl.Tensor, alog: pl.Tensor, dt: pl.Tensor,
              beta_out: pl.Out[pl.Tensor], g_out: pl.Out[pl.Tensor]):
        for row in pl.range(rows):
            at = pl.cast(pl.load(a, [row, 0], [8, (value_heads + 15) // 16 * 16],
                                valid_shape=[1, value_heads]), pl.FP32, mode="rint")
            bt = pl.cast(pl.load(b, [row, 0], [8, (value_heads + 15) // 16 * 16],
                                valid_shape=[1, value_heads]), pl.FP32, mode="rint")
            bias = pl.cast(pl.load(dt, [0, 0], [1, (value_heads + 15) // 16 * 16],
                                  valid_shape=[1, value_heads]), pl.FP32, mode="rint")
            log_a = pl.cast(pl.load(alog, [0, 0], [1, (value_heads + 15) // 16 * 16],
                                   valid_shape=[1, value_heads]), pl.FP32, mode="rint")
            af = pl.col_expand_add(at, bias)
            small_branch = pl.log(pl.add(pl.exp(pl.minimum(af, 20.0)), 1.0))
            softplus = pl.tile.select(pl.cmps(af, 20.0, cmp_type=4), af, small_branch)
            g_value = pl.col_expand_mul(softplus, pl.neg(pl.exp(log_a)))
            beta_value = pl.recip(pl.add(pl.exp(pl.neg(bt)), 1.0))
            pl.store(pl.cast(beta_value, pl.BF16, mode="rint"), [row, 0], beta_out)
            pl.store(g_value, [row, 0], g_out)
        return beta_out, g_out

    @pl.jit.incore
    def causal_conv(qkv: pl.Tensor, history: pl.Tensor, weight: pl.Tensor,
                    activated_out: pl.Out[pl.Tensor], history_out: pl.Out[pl.Tensor]):
        for bb in pl.range(batch):
            for token in pl.range(sequence):
                initial = pl.set_validshape(pl.tile.full([1, (channels + 15) // 16 * 16], dtype=pl.FP32, value=0.0), 1, channels)
                for tap, (acc,) in pl.range(conv_kernel, init_values=(initial,)):
                    # 历史与本次 qkv 在时间轴上逻辑拼接，直接读取对应原始投影。
                    position = token + tap
                    if position < conv_kernel - 1:
                        window = pl.load(history, [bb * (conv_kernel - 1) + position, 0], [1, (channels + 15) // 16 * 16], valid_shape=[1, channels])
                    else:
                        window = pl.load(qkv, [bb * sequence + position - conv_kernel + 1, 0], [1, (channels + 15) // 16 * 16], valid_shape=[1, channels])
                    coefficient_column = pl.load(weight, [0, tap], [(channels + 15) // 16 * 16, 1], valid_shape=[channels, 1])
                    coefficient = pl.reshape(coefficient_column, [1, (channels + 15) // 16 * 16])
                    accumulated = pl.add(acc, pl.mul(pl.cast(window, pl.FP32), pl.cast(coefficient, pl.FP32)))
                    conv_result = pl.yield_(accumulated)
                # Conv 输出先舍入 BF16，再计算 SiLU 并舍入 BF16。
                cf = pl.cast(pl.cast(conv_result, pl.BF16), pl.FP32)
                silu = pl.div(cf, pl.add(pl.exp(pl.neg(cf)), 1.0))
                activated_out = pl.store(pl.cast(silu, pl.BF16), [bb * sequence + token, 0], activated_out)
            for tail in pl.range(conv_kernel - 1):
                position_tail = sequence + tail
                if position_tail < conv_kernel - 1:
                    history_row = pl.load(history, [bb * (conv_kernel - 1) + position_tail, 0], [1, (channels + 15) // 16 * 16], valid_shape=[1, channels])
                else:
                    history_row = pl.load(qkv, [bb * sequence + position_tail - conv_kernel + 1, 0], [1, (channels + 15) // 16 * 16], valid_shape=[1, channels])
                history_out = pl.store(history_row, [bb * (conv_kernel - 1) + tail, 0], history_out)
        return activated_out, history_out

    @pl.jit.incore
    def gated_norm(core: pl.Tensor, z: pl.Tensor, norm_weight: pl.Tensor, out: pl.Out[pl.Tensor]):
        for row in pl.range(rows):
            for head in pl.range(value_heads):
                # 每个 head 独立 RMSNorm；8行物理补齐，逻辑仅一行有效。
                u = pl.cast(pl.load(core, [row, head * value_dim], [8, (value_dim + 15) // 16 * 16], valid_shape=[1, value_dim]), pl.FP32)
                tmp = pl.create_tile([8, (value_dim + 15) // 16 * 16], dtype=pl.FP32)
                mean_square = pl.div(pl.row_sum(pl.mul(u, u), tmp), value_dim)
                normalized = pl.cast(pl.row_expand_mul(u, pl.rsqrt(pl.add(mean_square, eps))), pl.BF16)
                weight = pl.cast(pl.load(norm_weight, [0, 0], [1, (value_dim + 15) // 16 * 16], valid_shape=[1, value_dim]), pl.FP32)
                weighted = pl.cast(pl.col_expand_mul(pl.cast(normalized, pl.FP32), weight), pl.BF16)
                zf = pl.cast(pl.load(z, [row, head * value_dim], [8, (value_dim + 15) // 16 * 16], valid_shape=[1, value_dim]), pl.FP32)
                gate = pl.div(zf, pl.add(pl.exp(pl.neg(zf)), 1.0))
                result = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), gate), pl.BF16)
                out = pl.store(result, [row, head * value_dim], out)
        return out

    @pl.jit.incore
    def extract_head(activated: pl.Tensor, beta: pl.Tensor, g: pl.Tensor,
                     query: pl.Out[pl.Tensor], key: pl.Out[pl.Tensor], value: pl.Out[pl.Tensor],
                     head_beta: pl.Out[pl.Tensor], head_g: pl.Out[pl.Tensor],
                     bb: pl.Scalar[pl.INDEX], hv: pl.Scalar[pl.INDEX]):
        # 连续的 head 缓冲显式保留 Token/head 映射，供两种核心共用。
        # 递归 chunk 的子窗口因此具有实际连续步长，不依赖隐式视图传播。
        for token in pl.range(sequence):
            qh = hv // ratio
            qr = pl.load(activated, [bb * sequence + token, qh * key_dim], [1, (key_dim + 15) // 16 * 16], valid_shape=[1, key_dim])
            kr = pl.load(activated, [bb * sequence + token, key_width + qh * key_dim], [1, (key_dim + 15) // 16 * 16], valid_shape=[1, key_dim])
            vr = pl.load(activated, [bb * sequence + token, 2 * key_width + hv * value_dim], [1, (value_dim + 15) // 16 * 16], valid_shape=[1, value_dim])
            br = pl.load(beta, [bb * sequence + token, hv], [1, 16], valid_shape=[1, 1])
            gr = pl.load(g, [bb * sequence + token, hv], [1, 8], valid_shape=[1, 1])
            query = pl.store(qr, [token, 0], query)
            key = pl.store(kr, [token, 0], key)
            value = pl.store(vr, [token, 0], value)
            head_beta = pl.store(br, [token, 0], head_beta)
            head_g = pl.store(gr, [token, 0], head_g)
        return query, key, value, head_beta, head_g

    @pl.jit.incore
    def assemble_head(head_values: pl.Tensor, core: pl.InOut[pl.Tensor], bb: pl.Scalar[pl.INDEX], hv: pl.Scalar[pl.INDEX]):
        for token in pl.range(sequence):
            values = pl.load(head_values, [token, 0], [1, (value_dim + 15) // 16 * 16], valid_shape=[1, value_dim])
            core = pl.store(values, [bb * sequence + token, hv * value_dim], core)
        return core

    @pl.jit
    def kernel(x: pl.Tensor, w_qkv: pl.Tensor, s_qkv: pl.Tensor, w_z: pl.Tensor, s_z: pl.Tensor,
               w_a: pl.Tensor, w_b: pl.Tensor, w_out: pl.Tensor, s_out: pl.Tensor,
               conv_weight: pl.Tensor, A_log: pl.Tensor, dt_bias: pl.Tensor, norm_weight: pl.Tensor,
               initial_state: pl.Tensor, conv_history: pl.Tensor, mask: pl.Tensor,
               out: pl.Out[pl.Tensor], state_out: pl.Out[pl.Tensor], conv_history_out: pl.Out[pl.Tensor]):
        flat = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        flat_x = pl.reshape(x, [rows, hidden])
        flat_mask = pl.reshape(mask, [rows, 1])
        flat = input_mask(flat_x, flat_mask, flat)
        qkv = pl.create_tensor([rows, channels], dtype=pl.BF16)
        z = pl.create_tensor([rows, value_width], dtype=pl.BF16)
        qkv = qkv_projection(flat, w_qkv, s_qkv, qkv)
        z = z_projection(flat, w_z, s_z, z)
        beta = pl.create_tensor([rows, value_heads], dtype=pl.BF16)
        g = pl.create_tensor([rows, value_heads], dtype=pl.FP32)
        a = pl.create_tensor([rows, value_heads], dtype=pl.BF16)
        b = pl.create_tensor([rows, value_heads], dtype=pl.BF16)
        a, b = ab_projection(flat, w_a, w_b, a, b)
        beta, g = gates(a, b, A_log, dt_bias, beta, g)
        activated = pl.create_tensor([rows, channels], dtype=pl.BF16)
        history = pl.reshape(conv_history, [batch * (conv_kernel - 1), channels])
        history_output = pl.reshape(conv_history_out, [batch * (conv_kernel - 1), channels])
        activated, history_output = causal_conv(qkv, history, conv_weight, activated, history_output)
        core = pl.create_tensor([rows, value_width], dtype=pl.BF16)
        initial_flat = pl.reshape(initial_state, [batch * value_heads * key_dim, value_dim])
        final_flat = pl.reshape(state_out, [batch * value_heads * key_dim, value_dim])
        for bb in pl.unroll(batch):
            for hv in pl.unroll(value_heads):
                query = pl.create_tensor([sequence, key_dim], dtype=pl.BF16)
                key = pl.create_tensor([sequence, key_dim], dtype=pl.BF16)
                value = pl.create_tensor([sequence, value_dim], dtype=pl.BF16)
                head_beta = pl.create_tensor([sequence, 1], dtype=pl.BF16)
                head_g = pl.create_tensor([sequence, 1], dtype=pl.FP32)
                query, key, value, head_beta, head_g = extract_head(activated, beta, g, query, key, value, head_beta, head_g, bb, hv)
                state = pl.slice(initial_flat, [key_dim, value_dim], [(bb * value_heads + hv) * key_dim, 0])
                head_output = pl.create_tensor([sequence, value_dim], dtype=pl.BF16)
                state_view = pl.slice(final_flat, [key_dim, value_dim], [(bb * value_heads + hv) * key_dim, 0])
                head_output, state_view = core_kernel(query, key, value, head_beta, head_g, state, head_output, state_view)
                core = assemble_head(head_output, core, bb, hv)
        gated = pl.create_tensor([rows, value_width], dtype=pl.BF16)
        gated = gated_norm(core, z, norm_weight, gated)
        projected = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        projected = out_projection(gated, w_out, s_out, projected)
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.assemble(out, pl.reshape(projected, [batch, sequence, hidden]), [0, 0, 0])
        return out, state_out, conv_history_out

    return kernel
