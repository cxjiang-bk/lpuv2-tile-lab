"""Complete K-06 entry; all computation executes in the PTO C++ simulator."""

from pathlib import Path

import torch

from kernels.pypto import gdn as tile_kernels
from pymodel.isa.hw_pypto import TensorProgram


def pad_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
    """Copy FP8 bits into zero-padded 128x128 block storage, without dequantizing."""
    n, k = weight.shape
    storage = torch.zeros(((n + 127) // 128 * 128, (k + 127) // 128 * 128), dtype=torch.uint8)
    storage[:n, :k] = weight.contiguous().view(torch.uint8)
    return storage.view(torch.float8_e4m3fn)


def gdn(x: torch.Tensor, weights: dict[str, torch.Tensor], *, conv_weight, A_log, dt_bias, norm_weight,
        qk_heads=16, value_heads=32, key_head_dim=128, value_head_dim=128,
        initial_state=None, conv_history=None, attention_mask=None, fp8_execution="bf16_carrier",
        path="recurrent", chunk_size=64, artifact_dir: str | Path | None = None):
    if path not in ("recurrent", "chunk"):
        raise ValueError("path must be recurrent or chunk")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if x.ndim != 3 or x.dtype != torch.bfloat16 or x.device.type != "cpu":
        raise ValueError("GDN native entry expects CPU BF16 [B,S,D]")
    batch, sequence, hidden = x.shape
    if min(batch, sequence, hidden, qk_heads, value_heads, key_head_dim, value_head_dim) < 1 or value_heads % qk_heads:
        raise ValueError("Positive dimensions and value_heads divisible by qk_heads are required")
    key_width, value_width = qk_heads * key_head_dim, value_heads * value_head_dim
    channels = 2 * key_width + value_width
    projection_shapes = {"qkv": (channels, hidden), "z": (value_width, hidden), "out": (hidden, value_width)}
    for name, shape in projection_shapes.items():
        weight, scale = weights[name + "_weight"], weights[name + "_scale_inv"]
        if weight.shape != shape or weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{name} weight must retain FP8 E4M3FN {shape}")
        if scale.shape != ((shape[0] + 127) // 128, (shape[1] + 127) // 128) or scale.dtype != torch.bfloat16:
            raise ValueError(f"{name} scale must retain checkpoint BF16 128x128 blocks")
    if any(weights[name + "_weight"].shape != (value_heads, hidden)
           or weights[name + "_weight"].dtype != torch.bfloat16 for name in ("a", "b")):
        raise ValueError("GDN a/b projection weights must be BF16 [H,D]")
    if conv_weight.ndim == 3 and conv_weight.shape[1] == 1:
        conv_weight = conv_weight[:, 0]
    if conv_weight.ndim != 2 or conv_weight.shape[0] != channels or conv_weight.shape[1] < 2:
        raise ValueError("Expected depthwise weight [C,K>=2]")
    taps = conv_weight.shape[1]
    if A_log.shape != (value_heads,) or dt_bias.shape != A_log.shape or norm_weight.shape != (value_head_dim,):
        raise ValueError("GDN gate and Norm parameters have incompatible shapes")
    if any(t.dtype != torch.bfloat16 for t in (conv_weight, A_log, dt_bias, norm_weight)):
        raise ValueError("GDN gate, Conv and Norm parameter storage must be BF16")
    state_shape = (batch, value_heads, key_head_dim, value_head_dim)
    if initial_state is None:
        initial_state = torch.zeros(state_shape, dtype=torch.float32)
    if initial_state.shape != state_shape or initial_state.dtype != torch.float32:
        raise ValueError("GDN recurrent State must be FP32 [B,H,K,V]")
    if conv_history is None:
        conv_history = torch.zeros((batch, taps - 1, channels), dtype=torch.bfloat16)
    if conv_history.shape != (batch, taps - 1, channels) or conv_history.dtype != torch.bfloat16:
        raise ValueError("GDN Conv history must be raw BF16 [B,K-1,C]")
    if attention_mask is None:
        attention_mask = torch.ones((batch, sequence), dtype=torch.bool)
    if attention_mask.shape != (batch, sequence) or attention_mask.dtype != torch.bool:
        raise ValueError("Padding mask must be boolean [B,S]")
    if any(t.device.type != "cpu" for t in (*weights.values(), conv_weight, A_log, dt_bias,
                                           norm_weight, initial_state, conv_history, attention_mask)):
        raise ValueError("Native C++ simulation tensors must be CPU tensors")

    if fp8_execution != "bf16_carrier":
        raise ValueError("This HW simulation profile uses bf16_carrier for exact FP8 values")
    program = TensorProgram.compile(tile_kernels.gdn(batch, sequence, hidden, qk_heads=qk_heads, value_heads=value_heads,
                                                key_dim=key_head_dim, value_dim=value_head_dim, conv_kernel=taps,
                                                fp8_execution=fp8_execution, path=path, chunk_size=chunk_size))
    out = torch.full(x.shape, float("nan"), dtype=x.dtype)
    state_out = torch.full(state_shape, float("nan"), dtype=torch.float32)
    history_out = torch.full(conv_history.shape, float("nan"), dtype=conv_history.dtype)
    program.details["fp8_execution"] = fp8_execution
    program.details["gdn_path"] = path
    program.details["chunk_size"] = chunk_size if path == "chunk" else None
    program.details["native_fp8_cube_validated"] = False
    program.details["fp8_representation"] = (
        "INT32-widened checkpoint bytes; E4M3FN quantization/decode in PTO C++; BF16 exact value carrier"
        if fp8_execution == "bf16_carrier" else "native FP8 E4M3FN cube")
    padded_weights = {name: pad_fp8_weight(weights[name + "_weight"]) for name in ("qkv", "z", "out")}
    if fp8_execution == "bf16_carrier":
        # Widen raw byte codes only. No host dequantization or activation math.
        padded_weights = {name: value.view(torch.uint8).to(torch.int32) for name, value in padded_weights.items()}
    if artifact_dir is not None:
        program.save(artifact_dir)
    program.execute(x.contiguous(), padded_weights["qkv"], weights["qkv_scale_inv"].contiguous(),
                    padded_weights["z"], weights["z_scale_inv"].contiguous(),
                    weights["a_weight"].contiguous(), weights["b_weight"].contiguous(),
                    padded_weights["out"], weights["out_scale_inv"].contiguous(),
                    conv_weight.contiguous(), A_log.reshape(1, value_heads).contiguous(),
                    dt_bias.reshape(1, value_heads).contiguous(), norm_weight.reshape(1, value_head_dim).contiguous(),
                    initial_state.contiguous(), conv_history.contiguous(), attention_mask.to(torch.int32).contiguous(),
                    out, state_out, history_out)
    if artifact_dir is not None:
        program.save(artifact_dir)
    return {"out": out, "state_out": state_out, "conv_history_out": history_out}
