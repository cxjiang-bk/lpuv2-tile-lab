"""Independent Torch reference for the confirmed per-Token K128 FP8 profile."""

import torch


def activation_quantize(x: torch.Tensor) -> dict[str, torch.Tensor]:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not bool(torch.isfinite(x).all()):
        raise ValueError("FP8 activation reference expects finite BF16 [M,K]")
    rows, width = x.shape
    groups = (width + 127) // 128
    # 每个 Token 沿 K 每 128 个元素一组；尾组补 0，得到 [M,G,128]。
    padded = torch.nn.functional.pad(x.float(), (0, groups * 128 - width)).reshape(rows, groups, 128)
    # amax = max(abs(x_group))，scale 为反量化乘数；全零组取 scale=1。
    amax = padded.abs().amax(-1, keepdim=True)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / 448.0)
    # x8 = E4M3FN_RNE(clamp(x32/scale, -448, 448))；payload 与 scale 按组对应。
    quantized = (padded / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return {"payload": quantized.reshape(rows, groups * 128)[:, :width], "scale": scale.squeeze(-1)}


def linear(x: torch.Tensor, weight: torch.Tensor, weight_scale_inv: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("FP8 Linear requires x[M,K] and weight[N,K]")
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError("Checkpoint weight payload must be E4M3FN")
    rows, width = x.shape
    out_width = weight.shape[0]
    groups = (width + 127) // 128
    if weight_scale_inv.shape != ((out_width + 127) // 128, groups):
        raise ValueError("Checkpoint scale must match 128x128 weight blocks")
    # Linear：x[M,K] @ W[N,K].T -> y[M,N]；输入先按同一 K128 profile 量化。
    quant = activation_quantize(x)
    result = torch.zeros((rows, out_width), dtype=torch.float32, device=x.device)
    for group in range(groups):
        begin, end = group * 128, min((group + 1) * 128, width)
        # 每个 K 组先对 FP8 数值做 FP32 点积，得到该组对所有输出列的贡献。
        dot = quant["payload"][:, begin:end].float() @ weight[:, begin:end].float().T
        # 权重每个 [N128,K128] block 共用 scale；按输出列展开对应的 N-block。
        weight_scale = weight_scale_inv[:, group].float().repeat_interleave(128)[:out_width]
        # acc += (dot * activation_scale) * weight_scale_inv，scale_inv 在这里作为乘数。
        result += (dot * quant["scale"][:, group:group + 1]) * weight_scale
    # 各 K 组均已应用各自 scale；完整 FP32 累加结束后输出 BF16。
    return result.bfloat16()
