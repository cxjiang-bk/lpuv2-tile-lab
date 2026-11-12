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
        # S[K,V]跨Token保持FP32；本次调用只更新输出副本。
        initial = pl.load(initial_state, [0, 0], [key_tile, value_tile], valid_shape=[key_dim, value_dim])
        for token, (current,) in pl.range(sequence, init_values=(initial,)):
            # 单 Token Q/K 物理补足8行，使 [M,1] 归约结果满足32B对齐；仅首行有效。
            qt = pl.cast(pl.load(query, [token, 0], [8, key_tile], valid_shape=[1, key_dim]), pl.FP32)
            kt = pl.cast(pl.load(key, [token, 0], [8, key_tile], valid_shape=[1, key_dim]), pl.FP32)
            q_tmp = pl.create_tile([8, key_tile], dtype=pl.FP32)
            k_tmp = pl.create_tile([8, key_tile], dtype=pl.FP32)
            # L2Norm(q)=q/sqrt(sum(q²)+eps)，随后Q只缩放一次1/sqrt(K)。
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
    def block(query: pl.Tensor[[length, key_dim], pl.BF16], key: pl.Tensor[[length, key_dim], pl.BF16],
              value: pl.Tensor[[length, value_dim], pl.BF16], beta: pl.Tensor[[length, 1], pl.BF16], g: pl.Tensor[[length, 1], pl.FP32],
              initial_state: pl.Tensor[[key_dim, value_dim], pl.FP32],
              out: pl.Out[pl.Tensor[[length, value_dim], pl.BF16]], state_out: pl.Out[pl.Tensor[[key_dim, value_dim], pl.FP32]]):
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
        # A_ij=beta_i*<K_i,K_j>*D_ij；P_ij=<Q_i,K_j>*D_ij，仅保留下三角。
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
        # 前代：X_i=RHS_i-sum_{j<i}(A_ij*X_j)；单位对角线无需除法。
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

        # delta=U-W@S；块输出=(Q*exp(G))@S+P@delta，所有乘积保持FP32。
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
    """按chunk扫描FP32 State；块内系数/三角求解由上方gdn_chunk_block定义。"""
    full_chunks, tail = divmod(sequence, chunk_size)
    if full_chunks:
        full_block = gdn_chunk_block(chunk_size, key_dim, value_dim, eps)
    if tail:
        tail_block = gdn_chunk_block(tail, key_dim, value_dim, eps)

    @pl.jit.inline(auto_scope=False)
    def core(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, beta: pl.Tensor, g: pl.Tensor,
             initial_state: pl.Tensor, out: pl.Out[pl.Tensor], state_out: pl.Out[pl.Tensor]):
        state_work = pl.create_tensor([key_dim, value_dim], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP):
            initial_tile = pl.load(initial_state, [0, 0], [key_dim, value_dim])
            state_work = pl.store(initial_tile, [0, 0], state_work)
        if full_chunks:
            # 内部State逐块更新；每个block先完整读取State，最后才写回同一工作缓冲。
            for chunk in pl.range(full_chunks):
                begin = chunk * chunk_size
                q = pl.slice(query, [chunk_size, key_dim], [begin, 0])
                k = pl.slice(key, [chunk_size, key_dim], [begin, 0])
                v = pl.slice(value, [chunk_size, value_dim], [begin, 0])
                b = pl.slice(beta, [chunk_size, 1], [begin, 0])
                gates = pl.slice(g, [chunk_size, 1], [begin, 0])
                destination = pl.slice(out, [chunk_size, value_dim], [begin, 0])
                destination, state_work = full_block(q, k, v, b, gates, state_work, destination, state_work)
        if tail:
            # 尾块只包含剩余Token，并消费最后一个完整块的FP32 State。
            tail_begin = full_chunks * chunk_size
            qt = pl.slice(query, [tail, key_dim], [tail_begin, 0])
            kt = pl.slice(key, [tail, key_dim], [tail_begin, 0])
            vt = pl.slice(value, [tail, value_dim], [tail_begin, 0])
            bt = pl.slice(beta, [tail, 1], [tail_begin, 0])
            gt = pl.slice(g, [tail, 1], [tail_begin, 0])
            tail_destination = pl.slice(out, [tail, value_dim], [tail_begin, 0])
            tail_destination, state_out = tail_block(qt, kt, vt, bt, gt, state_work, tail_destination, state_out)
        else:
            with pl.at(level=pl.Level.CORE_GROUP):
                final_tile = pl.load(state_work, [0, 0], [key_dim, value_dim])
                state_out = pl.store(final_tile, [0, 0], state_out)
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

    @pl.jit
    def kernel(x: pl.Tensor, w_qkv: pl.Tensor, s_qkv: pl.Tensor, w_z: pl.Tensor, s_z: pl.Tensor,
               w_a: pl.Tensor, w_b: pl.Tensor, w_out: pl.Tensor, s_out: pl.Tensor,
               conv_weight: pl.Tensor, A_log: pl.Tensor, dt_bias: pl.Tensor, norm_weight: pl.Tensor,
               initial_state: pl.Tensor, conv_history: pl.Tensor, mask: pl.Tensor,
               out: pl.Out[pl.Tensor], state_out: pl.Out[pl.Tensor], conv_history_out: pl.Out[pl.Tensor]):
        flat = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        flat_x = pl.reshape(x, [rows, hidden])
        flat_mask = pl.reshape(mask, [rows, 1])
        # 1. 输入mask：padding Token置零后保持BF16。
        with pl.at(level=pl.Level.CORE_GROUP):
            for mask_row in pl.range(rows):
                masked_input32 = pl.cast(pl.load(flat_x, [mask_row, 0], [1, (hidden + 15) // 16 * 16], valid_shape=[1, hidden]), pl.FP32)
                keep_token32 = pl.cast(pl.read(flat_mask, [mask_row, 0]), pl.FP32)
                flat = pl.store(pl.cast(pl.mul(masked_input32, keep_token32), pl.BF16), [mask_row, 0], flat)
        qkv = pl.create_tensor([rows, channels], dtype=pl.BF16)
        z = pl.create_tensor([rows, value_width], dtype=pl.BF16)
        # 2. qkv/z投影：BF16输入在共享FP8 Linear内按K128动态量化。
        qkv = qkv_projection(flat, w_qkv, s_qkv, qkv)
        z = z_projection(flat, w_z, s_z, z)
        beta = pl.create_tensor([rows, value_heads], dtype=pl.BF16)
        g = pl.create_tensor([rows, value_heads], dtype=pl.FP32)
        a = pl.create_tensor([rows, value_heads], dtype=pl.BF16)
        b = pl.create_tensor([rows, value_heads], dtype=pl.BF16)
        # 3. a/b投影：BF16矩阵乘，FP32累加，结果回转BF16。
        with pl.at(level=pl.Level.CORE_GROUP):
            a_projection = pl.cast(pl.matmul(flat, w_a, out_dtype=pl.FP32, b_trans=True), pl.BF16, mode="rint")
            b_projection = pl.cast(pl.matmul(flat, w_b, out_dtype=pl.FP32, b_trans=True), pl.BF16, mode="rint")
            a = pl.assemble(a, a_projection, [0, 0])
            b = pl.assemble(b, b_projection, [0, 0])
        # 4. beta=BF16(sigmoid(b))；g=-exp(A_log)*softplus(FP32(a)+dt_bias)。
        with pl.at(level=pl.Level.CORE_GROUP):
            for gate_row in pl.range(rows):
                a32 = pl.cast(pl.load(a, [gate_row, 0], [8, (value_heads + 15) // 16 * 16],
                                    valid_shape=[1, value_heads]), pl.FP32, mode="rint")
                b32 = pl.cast(pl.load(b, [gate_row, 0], [8, (value_heads + 15) // 16 * 16],
                                    valid_shape=[1, value_heads]), pl.FP32, mode="rint")
                gate_bias = pl.cast(pl.load(dt_bias, [0, 0], [1, (value_heads + 15) // 16 * 16],
                                      valid_shape=[1, value_heads]), pl.FP32, mode="rint")
                a_log32 = pl.cast(pl.load(A_log, [0, 0], [1, (value_heads + 15) // 16 * 16],
                                       valid_shape=[1, value_heads]), pl.FP32, mode="rint")
                gate_input = pl.col_expand_add(a32, gate_bias)
                gate_small_branch = pl.log(pl.add(pl.exp(pl.minimum(gate_input, 20.0)), 1.0))
                gate_softplus = pl.tile.select(pl.cmps(gate_input, 20.0, cmp_type=4), gate_input, gate_small_branch)
                log_decay = pl.col_expand_mul(gate_softplus, pl.neg(pl.exp(a_log32)))
                beta_value = pl.recip(pl.add(pl.exp(pl.neg(b32)), 1.0))
                pl.store(pl.cast(beta_value, pl.BF16, mode="rint"), [gate_row, 0], beta)
                pl.store(log_decay, [gate_row, 0], g)
        activated = pl.create_tensor([rows, channels], dtype=pl.BF16)
        history = pl.reshape(conv_history, [batch * (conv_kernel - 1), channels])
        history_output = pl.reshape(conv_history_out, [batch * (conv_kernel - 1), channels])
        # 5. 内容路径：历史qkv与当前投影拼接读取，DWConv -> BF16 -> SiLU。
        with pl.at(level=pl.Level.CORE_GROUP):
            for conv_bb in pl.range(batch):
                for conv_token in pl.range(sequence):
                    conv_initial = pl.set_validshape(pl.tile.full([1, (channels + 15) // 16 * 16], dtype=pl.FP32, value=0.0), 1, channels)
                    for conv_tap, (conv_acc,) in pl.range(conv_kernel, init_values=(conv_initial,)):
                        # 历史与本次 qkv 在时间轴上逻辑拼接，直接读取对应原始投影。
                        conv_position = conv_token + conv_tap
                        if conv_position < conv_kernel - 1:
                            conv_window = pl.load(history, [conv_bb * (conv_kernel - 1) + conv_position, 0],
                                [1, (channels + 15) // 16 * 16], valid_shape=[1, channels])
                        else:
                            conv_window = pl.load(qkv, [conv_bb * sequence + conv_position - conv_kernel + 1, 0],
                                [1, (channels + 15) // 16 * 16], valid_shape=[1, channels])
                        conv_coefficient_column = pl.load(conv_weight, [0, conv_tap], [(channels + 15) // 16 * 16, 1], valid_shape=[channels, 1])
                        conv_coefficient = pl.reshape(conv_coefficient_column, [1, (channels + 15) // 16 * 16])
                        conv_accumulated = pl.add(conv_acc, pl.mul(pl.cast(conv_window, pl.FP32), pl.cast(conv_coefficient, pl.FP32)))
                        conv_sum32 = pl.yield_(conv_accumulated)
                    # Conv 输出先舍入 BF16，再计算 SiLU 并舍入 BF16。
                    conv_value32 = pl.cast(pl.cast(conv_sum32, pl.BF16), pl.FP32)
                    conv_silu = pl.div(conv_value32, pl.add(pl.exp(pl.neg(conv_value32)), 1.0))
                    activated = pl.store(pl.cast(conv_silu, pl.BF16), [conv_bb * sequence + conv_token, 0], activated)
                for conv_tail in pl.range(conv_kernel - 1):
                    conv_position_tail = sequence + conv_tail
                    if conv_position_tail < conv_kernel - 1:
                        conv_history_row = pl.load(history, [conv_bb * (conv_kernel - 1) + conv_position_tail, 0],
                            [1, (channels + 15) // 16 * 16], valid_shape=[1, channels])
                    else:
                        conv_history_row = pl.load(qkv, [conv_bb * sequence + conv_position_tail - conv_kernel + 1, 0],
                            [1, (channels + 15) // 16 * 16], valid_shape=[1, channels])
                    history_output = pl.store(conv_history_row, [conv_bb * (conv_kernel - 1) + conv_tail, 0], history_output)
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
                # 6. 映射共享Q/K head，提取连续Q/K/V与beta/g，供两种核心路径使用。
                with pl.at(level=pl.Level.CORE_GROUP):
                    for head_token in pl.range(sequence):
                        head_qh = hv // ratio
                        head_qr = pl.load(activated, [bb * sequence + head_token, head_qh * key_dim],
                            [1, (key_dim + 15) // 16 * 16], valid_shape=[1, key_dim])
                        head_kr = pl.load(activated, [bb * sequence + head_token, key_width + head_qh * key_dim],
                            [1, (key_dim + 15) // 16 * 16], valid_shape=[1, key_dim])
                        head_vr = pl.load(activated, [bb * sequence + head_token, 2 * key_width + hv * value_dim],
                            [1, (value_dim + 15) // 16 * 16], valid_shape=[1, value_dim])
                        head_br = pl.load(beta, [bb * sequence + head_token, hv], [1, 16], valid_shape=[1, 1])
                        head_gr = pl.load(g, [bb * sequence + head_token, hv], [1, 8], valid_shape=[1, 1])
                        query = pl.store(head_qr, [head_token, 0], query)
                        key = pl.store(head_kr, [head_token, 0], key)
                        value = pl.store(head_vr, [head_token, 0], value)
                        head_beta = pl.store(head_br, [head_token, 0], head_beta)
                        head_g = pl.store(head_gr, [head_token, 0], head_g)
                state = pl.slice(initial_flat, [key_dim, value_dim], [(bb * value_heads + hv) * key_dim, 0])
                head_output = pl.create_tensor([sequence, value_dim], dtype=pl.BF16)
                state_view = pl.slice(final_flat, [key_dim, value_dim], [(bb * value_heads + hv) * key_dim, 0])
                head_output, state_view = core_kernel(query, key, value, head_beta, head_g, state, head_output, state_view)
                # 将核心的逐head BF16结果写回Token的对应特征段。
                with pl.at(level=pl.Level.CORE_GROUP):
                    for assembly_token in pl.range(sequence):
                        assembly_values = pl.load(head_output, [assembly_token, 0], [1, (value_dim + 15) // 16 * 16], valid_shape=[1, value_dim])
                        core = pl.store(assembly_values, [bb * sequence + assembly_token, hv * value_dim], core)
        gated = pl.create_tensor([rows, value_width], dtype=pl.BF16)
        # 7. 每head gated RMSNorm：FP32归一化 -> BF16乘weight -> FP32乘SiLU(z) -> BF16。
        with pl.at(level=pl.Level.CORE_GROUP):
            for norm_row in pl.range(rows):
                for norm_head in pl.range(value_heads):
                    # 每个 head 独立 RMSNorm；8行物理补齐，逻辑仅一行有效。
                    core32 = pl.cast(pl.load(core, [norm_row, norm_head * value_dim], [8, (value_dim + 15) // 16 * 16],
                        valid_shape=[1, value_dim]), pl.FP32)
                    norm_scratch = pl.create_tile([8, (value_dim + 15) // 16 * 16], dtype=pl.FP32)
                    norm_mean_square = pl.div(pl.row_sum(pl.mul(core32, core32), norm_scratch), value_dim)
                    norm_normalized = pl.cast(pl.row_expand_mul(core32, pl.rsqrt(pl.add(norm_mean_square, eps))), pl.BF16)
                    norm_gamma = pl.cast(pl.load(norm_weight, [0, 0], [1, (value_dim + 15) // 16 * 16], valid_shape=[1, value_dim]), pl.FP32)
                    norm_weighted = pl.cast(pl.col_expand_mul(pl.cast(norm_normalized, pl.FP32), norm_gamma), pl.BF16)
                    z32 = pl.cast(pl.load(z, [norm_row, norm_head * value_dim], [8, (value_dim + 15) // 16 * 16],
                        valid_shape=[1, value_dim]), pl.FP32)
                    silu_gate32 = pl.div(z32, pl.add(pl.exp(pl.neg(z32)), 1.0))
                    norm_result = pl.cast(pl.mul(pl.cast(norm_weighted, pl.FP32), silu_gate32), pl.BF16)
                    gated = pl.store(norm_result, [norm_row, norm_head * value_dim], gated)
        projected = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        # 8. 合并head后执行FP8输出投影，输出恢复[B,S,D]的BF16边界。
        projected = out_projection(gated, w_out, s_out, projected)
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.assemble(out, pl.reshape(projected, [batch, sequence, hidden]), [0, 0, 0])
        return out, state_out, conv_history_out

    return kernel
