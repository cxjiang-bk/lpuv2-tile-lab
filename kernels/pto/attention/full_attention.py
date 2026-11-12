"""K-05 Full Attention: FP8 projections, Partial MRoPE, GQA and explicit KV."""

import math

import pypto.language as pl

from kernels.pto.common.activations import sigmoid
from kernels.pto.common.linear import bf16_linear, fp8_linear
from kernels.pto.common.rmsnorm import rmsnorm
from kernels.pto.common.softmax import softmax


_FREQUENCY_STEP = -math.log(10_000_000) / 32
_BF16_MIN = -3.3895313892515355e38


@pl.jit.incore
def _split_query_gate(projected: pl.Tensor, query: pl.Out[pl.Tensor], gate: pl.Out[pl.Tensor]):
    dim = query.shape[1]
    heads = gate.shape[1] // dim
    for token in pl.range(projected.shape[0]):
        for head in pl.range(heads):
            for col in pl.range(0, dim, 128):
                valid = pl.min(128, dim - col)
                q = pl.load(projected, [token, head * 2 * dim + col], [1, 128], valid_shape=[1, valid])
                g = pl.load(projected, [token, head * 2 * dim + dim + col], [1, 128], valid_shape=[1, valid])
                pl.store(q, [token * heads + head, col], query)
                pl.store(g, [token, head * dim + col], gate)
    return query, gate


@pl.jit.incore
def _mrope_tables(positions: pl.Tensor, cosine: pl.Out[pl.Tensor], sine: pl.Out[pl.Tensor]):
    # Fixed Qwen profile: 32 half-frequencies, THW sections (11,11,10).
    indices = pl.tile.arange(0, [1, 32], dtype=pl.INT32)
    inverse = pl.exp(pl.mul(pl.cast(indices, pl.FP32), _FREQUENCY_STEP))
    sequence = positions.shape[2]
    for token in pl.range(cosine.shape[0]):
        angles = pl.tile.full([1, 32], pl.FP32, 0.0)
        for lane in pl.range(32):
            coordinate = pl.cast(pl.read(positions, [lane % 3, token // sequence, token % sequence]), pl.FP32)
            frequency = pl.tile.read(inverse, [0, lane])
            pl.tile.write(angles, [0, lane], coordinate * frequency)
        cos_bf16 = pl.cast(pl.cos(angles), pl.BF16)
        sin_bf16 = pl.cast(pl.sin(angles), pl.BF16)
        pl.store(cos_bf16, [token, 0], cosine)
        pl.store(sin_bf16, [token, 0], sine)
    return cosine, sine


@pl.jit.incore
def _rotate_heads(x: pl.Tensor, cosine: pl.Tensor, sine: pl.Tensor, positions: pl.Tensor,
                  out: pl.Out[pl.Tensor]):
    # Input rows: [B,S,heads,D]. Output rows: [B,heads,S,D].
    dim = x.shape[1]
    sequence = positions.shape[2]
    heads = x.shape[0] // cosine.shape[0]
    for token in pl.range(cosine.shape[0]):
        cos = pl.load(cosine, [token, 0], [1, 32])
        sin = pl.load(sine, [token, 0], [1, 32])
        for head in pl.range(heads):
            src = token * heads + head
            dst = ((token // sequence) * heads + head) * sequence + token % sequence
            first = pl.load(x, [src, 0], [1, 32])
            second = pl.load(x, [src, 32], [1, 32])
            # Both products and their sum keep the model's BF16 boundaries.
            rotated_first = pl.add(pl.mul(first, cos), pl.mul(pl.mul(second, -1.0), sin))
            rotated_second = pl.add(pl.mul(second, cos), pl.mul(first, sin))
            pl.store(rotated_first, [dst, 0], out)
            pl.store(rotated_second, [dst, 32], out)
            for col in pl.range(64, dim, 128):
                valid = pl.min(128, dim - col)
                tail = pl.load(x, [src, col], [1, 128], valid_shape=[1, valid])
                pl.store(tail, [dst, col], out)
    return out


@pl.jit.incore
def _append_kv(k: pl.Tensor, v: pl.Tensor, old_k: pl.Tensor, old_v: pl.Tensor,
               new_k: pl.Out[pl.Tensor], new_v: pl.Out[pl.Tensor], positions: pl.Tensor):
    dim = new_k.shape[1]
    batch = positions.shape[0]
    sequence = positions.shape[1]
    heads = v.shape[1] // dim
    past = old_k.shape[2]
    total = past + sequence
    history_k = pl.reshape(old_k, [batch * heads * past, dim])
    history_v = pl.reshape(old_v, [batch * heads * past, dim])
    for request in pl.range(batch):
        for head in pl.range(heads):
            if past > 0:
                for pos in pl.range(past):
                    for col in pl.range(0, dim, 128):
                        valid = pl.min(128, dim - col)
                        source = (request * heads + head) * past + pos
                        destination = (request * heads + head) * total + pos
                        key = pl.load(history_k, [source, col], [1, 128], valid_shape=[1, valid])
                        value = pl.load(history_v, [source, col], [1, 128], valid_shape=[1, valid])
                        pl.store(key, [destination, col], new_k)
                        pl.store(value, [destination, col], new_v)
            for pos in pl.range(sequence):
                for col in pl.range(0, dim, 128):
                    valid = pl.min(128, dim - col)
                    source = (request * heads + head) * sequence + pos
                    destination = (request * heads + head) * total + past + pos
                    key = pl.load(k, [source, col], [1, 128], valid_shape=[1, valid])
                    value = pl.load(v, [request * sequence + pos, head * dim + col],
                                    [1, 128], valid_shape=[1, valid])
                    pl.store(key, [destination, col], new_k)
                    pl.store(value, [destination, col], new_v)
    return new_k, new_v


@pl.jit.incore
def _append_positions(current: pl.Tensor, old: pl.Tensor, out: pl.Out[pl.Tensor],
                      lengths: pl.Out[pl.Tensor]):
    sequence = current.shape[1]
    past = old.shape[1]
    for request in pl.range(current.shape[0]):
        if past > 0:
            for pos in pl.range(past):
                pl.write(out, [request, pos], pl.read(old, [request, pos]))
        for pos in pl.range(sequence):
            pl.write(out, [request, past + pos], pl.read(current, [request, pos]))
        pl.write(lengths, [request], pl.cast(past + sequence, pl.INT64))
    return out, lengths


@pl.jit.incore
def _scale_mask(scores: pl.Tensor, query_pos: pl.Tensor, key_pos: pl.Tensor,
                out: pl.Out[pl.Tensor], request: pl.INDEX):
    # Scores first exist in BF16; multiplication also rounds to BF16.
    for row in pl.range(scores.shape[0]):
        current = pl.read(query_pos, [request, row])
        for col in pl.range(0, scores.shape[1], 128):
            valid = pl.min(128, scores.shape[1] - col)
            tile = pl.load(scores, [row, col], [1, 128], valid_shape=[1, valid])
            scaled = pl.mul(tile, 0.0625)
            for lane in pl.range(valid):
                key = pl.read(key_pos, [request, col + lane])
                if key > current:
                    pl.tile.write(scaled, [0, lane], pl.const(_BF16_MIN, pl.BF16))
            valid_scores = pl.tile.slice(scaled, [1, 128], [0, 0], valid_shape=[1, valid])
            pl.store(valid_scores, [row, col], out)
    return out


@pl.jit.incore
def _probabilities_bf16(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    for row in pl.range(x.shape[0]):
        for col in pl.range(0, x.shape[1], 128):
            valid = pl.min(128, x.shape[1] - col)
            values = pl.load(x, [row, col], [1, 128], valid_shape=[1, valid])
            result = pl.cast(values, pl.BF16)
            result = pl.tile.slice(result, [1, 128], [0, 0], valid_shape=[1, valid])
            pl.store(result, [row, col], out)
    return out


@pl.jit.incore
def _merge_gated(context: pl.Tensor, gate: pl.Tensor, out: pl.Out[pl.Tensor],
                  request: pl.INDEX, head: pl.INDEX):
    dim = context.shape[1]
    for row in pl.range(context.shape[0]):
        for col in pl.range(0, dim, 128):
            valid = pl.min(128, dim - col)
            c = pl.load(context, [row, col], [1, 128], valid_shape=[1, valid])
            g = pl.load(gate, [request * context.shape[0] + row, head * dim + col],
                        [1, 128], valid_shape=[1, valid])
            result = pl.mul(c, g)
            result = pl.tile.slice(result, [1, 128], [0, 0], valid_shape=[1, valid])
            pl.store(result, [request * context.shape[0] + row, head * dim + col], out)
    return out


@pl.jit.incore
def _value_product(probabilities: pl.Tensor, values: pl.Tensor, out: pl.Out[pl.Tensor]):
    # P[S,T] @ V[T,D]: the cache is read in its existing [T,D] order.
    for row in pl.range(0, probabilities.shape[0], 16):
        for col in pl.range(0, values.shape[1], 16):
            valid_m = pl.min(16, probabilities.shape[0] - row)
            valid_n = pl.min(16, values.shape[1] - col)
            acc = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
            for key in pl.range(0, values.shape[0], 64):
                valid_k = pl.min(64, values.shape[0] - key)
                p = pl.load(probabilities, [row, key], [16, 64], valid_shape=[valid_m, valid_k])
                v = pl.load(values, [key, col], [64, 16], valid_shape=[valid_k, valid_n])
                left = pl.move(pl.tile.fillpad(p), target_memory=pl.MemorySpace.Left)
                right = pl.move(pl.tile.fillpad(v), target_memory=pl.MemorySpace.Right)
                acc = pl.tile.matmul_acc(acc, left, right, init_cond=(key == 0))
            result = pl.cast(pl.move(acc, target_memory=pl.MemorySpace.Vec), pl.BF16)
            result = pl.tile.slice(result, [16, 16], [0, 0], valid_shape=[valid_m, valid_n])
            pl.store(result, [row, col], out)
    return out


@pl.jit
def full_attention(
    x: pl.Tensor, q_gate_weight: pl.Tensor, k_weight: pl.Tensor, v_weight: pl.Tensor, out_weight: pl.Tensor,
    q_norm_weight: pl.Tensor, k_norm_weight: pl.Tensor,
    q_gate_scale: pl.Tensor, k_scale: pl.Tensor, v_scale: pl.Tensor, out_scale: pl.Tensor,
    seq_pos: pl.Tensor, rope_pos: pl.Tensor, old_k: pl.Tensor, old_v: pl.Tensor, old_pos: pl.Tensor,
    out: pl.Out[pl.Tensor], kv_k_out: pl.Out[pl.Tensor], kv_v_out: pl.Out[pl.Tensor],
    kv_seq_pos_out: pl.Out[pl.Tensor], kv_valid_len_out: pl.Out[pl.Tensor],
):
    """Complete fixed-profile K-05; all supplied KV rows are valid and read-only."""
    batch = x.shape[0]
    sequence = x.shape[1]
    hidden = x.shape[2]
    dim = q_norm_weight.shape[0]
    heads = q_gate_weight.shape[0] // (2 * dim)
    kv_heads = k_weight.shape[0] // dim
    past = old_k.shape[2]
    total = past + sequence
    flat_x = pl.reshape(x, [batch * sequence, hidden])
    q_gate = pl.create_tensor([batch * sequence, heads * 2 * dim], dtype=pl.BF16)
    k_projected = pl.create_tensor([batch * sequence, kv_heads * dim], dtype=pl.BF16)
    v_projected = pl.create_tensor([batch * sequence, kv_heads * dim], dtype=pl.BF16)
    fp8_linear(flat_x, q_gate_weight, q_gate_scale, q_gate, scale_bf16=True)
    fp8_linear(flat_x, k_weight, k_scale, k_projected, scale_bf16=True)
    fp8_linear(flat_x, v_weight, v_scale, v_projected, scale_bf16=True)
    query = pl.create_tensor([batch * sequence * heads, dim], dtype=pl.BF16)
    gate = pl.create_tensor([batch * sequence, heads * dim], dtype=pl.BF16)
    _split_query_gate(q_gate, query, gate)
    q_normalized = pl.create_tensor([batch * sequence * heads, dim], dtype=pl.BF16)
    k_normalized = pl.create_tensor([batch * sequence * kv_heads, dim], dtype=pl.BF16)
    rmsnorm(query, q_norm_weight, q_normalized, 1e-6)
    k_head_rows = pl.reshape(k_projected, [batch * sequence * kv_heads, dim])
    rmsnorm(k_head_rows, k_norm_weight, k_normalized, 1e-6)
    cosine = pl.create_tensor([batch * sequence, 32], dtype=pl.BF16)
    sine = pl.create_tensor([batch * sequence, 32], dtype=pl.BF16)
    _mrope_tables(rope_pos, cosine, sine)
    q_rotated = pl.create_tensor([batch * heads * sequence, dim], dtype=pl.BF16)
    k_rotated = pl.create_tensor([batch * kv_heads * sequence, dim], dtype=pl.BF16)
    _rotate_heads(q_normalized, cosine, sine, rope_pos, q_rotated)
    _rotate_heads(k_normalized, cosine, sine, rope_pos, k_rotated)
    k_rows = pl.reshape(kv_k_out, [batch * kv_heads * total, dim])
    v_rows = pl.reshape(kv_v_out, [batch * kv_heads * total, dim])
    _append_kv(k_rotated, v_projected, old_k, old_v, k_rows, v_rows, seq_pos)
    _append_positions(seq_pos, old_pos, kv_seq_pos_out, kv_valid_len_out)
    gate_sigmoid = pl.create_tensor([batch * sequence, heads * dim], dtype=pl.BF16)
    sigmoid(gate, gate_sigmoid, input_bf16=True, output_bf16=True)
    gated = pl.create_tensor([batch * sequence, heads * dim], dtype=pl.BF16)
    for request in pl.range(batch):
        for head in pl.range(heads):
            kv_head = head // (heads // kv_heads)
            q = pl.slice(q_rotated, [sequence, dim], [(request * heads + head) * sequence, 0])
            k = pl.slice(k_rows, [total, dim], [(request * kv_heads + kv_head) * total, 0])
            v = pl.slice(v_rows, [total, dim], [(request * kv_heads + kv_head) * total, 0])
            scores = pl.create_tensor([sequence, total], dtype=pl.BF16)
            bf16_linear(q, k, scores)
            masked = pl.create_tensor([sequence, total], dtype=pl.BF16)
            _scale_mask(scores, seq_pos, kv_seq_pos_out, masked, request)
            probabilities = pl.create_tensor([sequence, total], dtype=pl.FP32)
            softmax(masked, probabilities, input_bf16=True)
            probabilities16 = pl.create_tensor([sequence, total], dtype=pl.BF16)
            _probabilities_bf16(probabilities, probabilities16)
            context = pl.create_tensor([sequence, dim], dtype=pl.BF16)
            _value_product(probabilities16, v, context)
            _merge_gated(context, gate_sigmoid, gated, request, head)
    out_rows = pl.reshape(out, [batch * sequence, hidden])
    fp8_linear(gated, out_weight, out_scale, out_rows, scale_bf16=True)
    return out, kv_k_out, kv_v_out, kv_seq_pos_out, kv_valid_len_out
