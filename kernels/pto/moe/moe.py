"""K-07: Top-8 routed experts and shared FFN, with explicit expert buckets."""

import pypto.language as pl

from kernels.pto.common.activations import sigmoid, silu
from kernels.pto.common.linear import bf16_linear, fp8_linear
from kernels.pto.common.softmax import softmax


@pl.jit.incore
def _select_routes(probabilities: pl.Tensor, expert_id: pl.Out[pl.Tensor], route_weight: pl.Out[pl.Tensor]):
    experts = probabilities.shape[1]
    scratch = pl.create_tile([1, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    for token in pl.range(probabilities.shape[0]):
        remaining = pl.load(probabilities, [token, 0], [1, experts])
        selected = pl.tile.full([1, 8], pl.FP32, 0.0)
        for slot in pl.range(8):
            initial_value = pl.const(-1.0, pl.FP32)
            initial_id = pl.const(0, pl.INDEX)
            for expert, (current_value, current_id) in pl.range(experts, init_values=(initial_value, initial_id)):
                candidate = pl.tile.read(remaining, [0, expert])
                # Ascending scan + strict comparison preserves lower ID on ties.
                if candidate > current_value:
                    chosen_value, chosen_id = pl.yield_(candidate, expert)
                else:
                    chosen_value, chosen_id = pl.yield_(current_value, current_id)
                best_value, best_id = pl.yield_(chosen_value, chosen_id)
            pl.tile.write(selected, [0, slot], best_value)
            pl.write(expert_id, [token, slot], pl.cast(best_id, pl.INT64))
            pl.tile.write(remaining, [0, best_id], pl.const(-1.0, pl.FP32))
        denominator = pl.row_sum(selected, scratch)
        normalized = pl.row_expand_div(selected, denominator)
        pl.store(pl.cast(normalized, pl.BF16), [token, 0], route_weight)
    return expert_id, route_weight


@pl.jit.incore
def _make_buckets(expert_id: pl.Tensor, offsets: pl.Out[pl.Tensor],
                   token_ids: pl.Out[pl.Tensor], slots: pl.Out[pl.Tensor]):
    initial_cursor = pl.const(0, pl.INDEX)
    pl.write(offsets, [0], pl.cast(initial_cursor, pl.INT64))
    for expert, (expert_cursor,) in pl.range(offsets.shape[0] - 1, init_values=(initial_cursor,)):
        for token, (token_cursor,) in pl.range(expert_id.shape[0], init_values=(expert_cursor,)):
            for slot, (cursor,) in pl.range(8, init_values=(token_cursor,)):
                identity = pl.read(expert_id, [token, slot])
                if identity == pl.cast(expert, pl.INT64):
                    pl.write(token_ids, [cursor], pl.cast(token, pl.INT64))
                    pl.write(slots, [cursor], pl.cast(slot, pl.INT64))
                    next_cursor = pl.yield_(cursor + 1)
                else:
                    next_cursor = pl.yield_(cursor)
                token_end = pl.yield_(next_cursor)
            expert_end = pl.yield_(token_end)
        pl.write(offsets, [expert + 1], pl.cast(expert_end, pl.INT64))
        _bucket_end = pl.yield_(expert_end)
    return offsets, token_ids, slots


@pl.jit.incore
def _zero_output(out: pl.Out[pl.Tensor]):
    zeros = pl.tile.full([1, 128], pl.BF16, 0.0)
    for row in pl.range(out.shape[0]):
        for col in pl.range(0, out.shape[1], 128):
            valid = pl.min(128, out.shape[1] - col)
            tile = pl.tile.slice(zeros, [1, 128], [0, 0], valid_shape=[1, valid])
            pl.store(tile, [row, col], out)
    return out


@pl.jit.incore
def _gather_bucket(x: pl.Tensor, token_ids: pl.Tensor, start: pl.INDEX,
                    count: pl.INDEX, gathered: pl.Out[pl.Tensor]):
    for row in pl.range(count):
        token = pl.read(token_ids, [start + row])
        for col in pl.range(0, x.shape[1], 128):
            valid = pl.min(128, x.shape[1] - col)
            tile = pl.load(x, [token, col], [1, 128], valid_shape=[1, valid])
            pl.store(tile, [row, col], gathered)
    return gathered


@pl.jit.incore
def _multiply(lhs: pl.Tensor, rhs: pl.Tensor, out: pl.Out[pl.Tensor]):
    for row in pl.range(lhs.shape[0]):
        for col in pl.range(0, lhs.shape[1], 128):
            valid = pl.min(128, lhs.shape[1] - col)
            a = pl.load(lhs, [row, col], [1, 128], valid_shape=[1, valid])
            b = pl.load(rhs, [row, col], [1, 128], valid_shape=[1, valid])
            product = pl.mul(a, b)
            tile = pl.tile.slice(product, [1, 128], [0, 0], valid_shape=[1, valid])
            pl.store(tile, [row, col], out)
    return out


@pl.jit.inline
def _ffn(x: pl.Tensor, gate_weight: pl.Tensor, up_weight: pl.Tensor, down_weight: pl.Tensor,
          gate_scale: pl.Tensor, up_scale: pl.Tensor, down_scale: pl.Tensor, out: pl.Tensor):
    rows = x.shape[0]
    intermediate = gate_weight.shape[0]
    gate = pl.create_tensor([rows, intermediate], dtype=pl.BF16)
    up = pl.create_tensor([rows, intermediate], dtype=pl.BF16)
    activated = pl.create_tensor([rows, intermediate], dtype=pl.BF16)
    product = pl.create_tensor([rows, intermediate], dtype=pl.BF16)
    fp8_linear(x, gate_weight, gate_scale, gate, scale_bf16=True)
    fp8_linear(x, up_weight, up_scale, up, scale_bf16=True)
    silu(gate, activated, input_bf16=True, output_bf16=True)
    _multiply(activated, up, product)
    fp8_linear(product, down_weight, down_scale, out, scale_bf16=True)
    return out


@pl.jit.incore
def _combine_bucket(contribution: pl.Tensor, route_weight: pl.Tensor, token_ids: pl.Tensor,
                     slots: pl.Tensor, start: pl.INDEX, count: pl.INDEX, merged: pl.InOut[pl.Tensor]):
    for row in pl.range(count):
        token = pl.read(token_ids, [start + row])
        slot = pl.read(slots, [start + row])
        weight = pl.load(route_weight, [token, slot], [1, 1])
        for col in pl.range(0, merged.shape[1], 128):
            valid = pl.min(128, merged.shape[1] - col)
            value = pl.load(contribution, [row, col], [1, 128], valid_shape=[1, valid])
            old = pl.load(merged, [token, col], [1, 128], valid_shape=[1, valid])
            scaled = pl.row_expand_mul(value, weight)
            combined = pl.add(old, scaled)
            tile = pl.tile.slice(combined, [1, 128], [0, 0], valid_shape=[1, valid])
            pl.store(tile, [token, col], merged)
    return merged


@pl.jit.incore
def _add_shared(merged: pl.Tensor, shared: pl.Tensor, gate: pl.Tensor, out: pl.Out[pl.Tensor]):
    for row in pl.range(merged.shape[0]):
        weight = pl.load(gate, [row, 0], [1, 1])
        for col in pl.range(0, merged.shape[1], 128):
            valid = pl.min(128, merged.shape[1] - col)
            routed = pl.load(merged, [row, col], [1, 128], valid_shape=[1, valid])
            value = pl.load(shared, [row, col], [1, 128], valid_shape=[1, valid])
            combined = pl.add(routed, pl.row_expand_mul(value, weight))
            tile = pl.tile.slice(combined, [1, 128], [0, 0], valid_shape=[1, valid])
            pl.store(tile, [row, col], out)
    return out


@pl.jit
def moe(
    x: pl.Tensor, router: pl.Tensor, shared_gate_weight: pl.Tensor,
    expert_gate: pl.Tensor, expert_up: pl.Tensor, expert_down: pl.Tensor,
    shared_ffn_gate: pl.Tensor, shared_ffn_up: pl.Tensor, shared_ffn_down: pl.Tensor,
    expert_gate_scale: pl.Tensor, expert_up_scale: pl.Tensor, expert_down_scale: pl.Tensor,
    shared_gate_scale: pl.Tensor, shared_up_scale: pl.Tensor, shared_down_scale: pl.Tensor,
    out: pl.Out[pl.Tensor], expert_id: pl.Out[pl.Tensor], route_weight: pl.Out[pl.Tensor],
    bucket_offset: pl.Out[pl.Tensor], bucket_token_id: pl.Out[pl.Tensor], bucket_slot: pl.Out[pl.Tensor],
):
    """BF16 x[M,H] -> output, Top-8 routes and expert-major bucket records."""
    tokens = x.shape[0]
    hidden = x.shape[1]
    experts = router.shape[0]
    intermediate = expert_gate.shape[1]
    logits = pl.create_tensor([tokens, experts], dtype=pl.BF16)
    probabilities = pl.create_tensor([tokens, experts], dtype=pl.FP32)
    bf16_linear(x, router, logits)
    softmax(logits, probabilities, input_bf16=True)
    _select_routes(probabilities, expert_id, route_weight)
    _make_buckets(expert_id, bucket_offset, bucket_token_id, bucket_slot)
    merged = pl.create_tensor([tokens, hidden], dtype=pl.BF16)
    _zero_output(merged)
    workspace = pl.create_tensor([tokens, hidden], dtype=pl.BF16)
    expert_output = pl.create_tensor([tokens, hidden], dtype=pl.BF16)
    gates = pl.reshape(expert_gate, [experts * intermediate, hidden])
    ups = pl.reshape(expert_up, [experts * intermediate, hidden])
    downs = pl.reshape(expert_down, [experts * hidden, intermediate])
    gate_scales = pl.reshape(expert_gate_scale,
                            [experts * expert_gate_scale.shape[1], expert_gate_scale.shape[2]])
    up_scales = pl.reshape(expert_up_scale, [experts * expert_up_scale.shape[1], expert_up_scale.shape[2]])
    down_scales = pl.reshape(expert_down_scale,
                            [experts * expert_down_scale.shape[1], expert_down_scale.shape[2]])
    for expert in pl.range(experts):
        start = pl.cast(pl.read(bucket_offset, [expert]), pl.INDEX)
        end = pl.cast(pl.read(bucket_offset, [expert + 1]), pl.INDEX)
        count = end - start
        if count > 0:
            _gather_bucket(x, bucket_token_id, start, count, workspace)
            bucket = pl.slice(workspace, [count, hidden], [0, 0])
            contribution = pl.slice(expert_output, [count, hidden], [0, 0])
            gate_weight = pl.slice(gates, [intermediate, hidden], [expert * intermediate, 0])
            up_weight = pl.slice(ups, [intermediate, hidden], [expert * intermediate, 0])
            down_weight = pl.slice(downs, [hidden, intermediate], [expert * hidden, 0])
            gs = pl.slice(gate_scales, [expert_gate_scale.shape[1], expert_gate_scale.shape[2]],
                          [expert * expert_gate_scale.shape[1], 0])
            us = pl.slice(up_scales, [expert_up_scale.shape[1], expert_up_scale.shape[2]],
                          [expert * expert_up_scale.shape[1], 0])
            ds = pl.slice(down_scales, [expert_down_scale.shape[1], expert_down_scale.shape[2]],
                          [expert * expert_down_scale.shape[1], 0])
            _ffn(bucket, gate_weight, up_weight, down_weight, gs, us, ds, contribution)
            _combine_bucket(contribution, route_weight, bucket_token_id, bucket_slot, start, count, merged)
    shared = pl.create_tensor([tokens, hidden], dtype=pl.BF16)
    shared_logits = pl.create_tensor([tokens, 1], dtype=pl.BF16)
    shared_gate = pl.create_tensor([tokens, 1], dtype=pl.BF16)
    _ffn(x, shared_ffn_gate, shared_ffn_up, shared_ffn_down,
          shared_gate_scale, shared_up_scale, shared_down_scale, shared)
    bf16_linear(x, shared_gate_weight, shared_logits)
    sigmoid(shared_logits, shared_gate, input_bf16=True, output_bf16=True)
    _add_shared(merged, shared, shared_gate, out)
    return out, expert_id, route_weight, bucket_offset, bucket_token_id, bucket_slot
