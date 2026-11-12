"""BF16 Vision operator references, using the pinned eager attention path.

Source: Transformers f90e93c4f839, modeling_qwen3_5_moe.py and vision_utils.py.
Pixel Conv3D emits merge-ordered tokens; position metadata follows IF-04.
"""

import torch
import torch.nn.functional as F

from pymodel.ref.common import layer_norm


def vision_patch_embedding(pixels: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *,
                           stride: tuple[int, int, int] | None = None,
                           spatial_merge_size: int = 2) -> torch.Tensor:
    """OP-08: pixels [B,C,T,H,W] -> merge-ordered BF16 tokens [N,D].

    Valid Conv3D, padding=0, dilation=1, groups=1. The model uses stride=kernel.
    Pixel preprocessing is defined separately; this entry never assumes packed patches.
    """
    if pixels.ndim != 5 or weight.ndim != 5 or pixels.shape[1] != weight.shape[1]:
        raise ValueError("Expected pixels[B,C,T,H,W] and weight[D,C,Kt,Kh,Kw]")
    if pixels.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Pixels must be BF16 or FP32")
    if weight.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16 or bias.shape != (weight.shape[0],):
        raise ValueError("Conv3D requires BF16 weight and bias[D]")
    stride = tuple(weight.shape[2:]) if stride is None else tuple(stride)
    if len(stride) != 3 or any(type(s) is not int or s < 1 for s in stride):
        raise ValueError("stride must contain three positive integers")
    if (any(d < 1 for d in (*pixels.shape, *weight.shape))
            or any(d < k for d, k in zip(pixels.shape[2:], weight.shape[2:]))):
        raise ValueError("Pixels must contain at least one complete Conv3D window")
    if type(spatial_merge_size) is not int or spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be a positive integer")
    ot, oh, ow = ((d - k) // s + 1 for d, k, s in zip(pixels.shape[2:], weight.shape[2:], stride))
    m = spatial_merge_size
    if oh % m or ow % m:
        raise ValueError("Conv3D output height and width must form complete spatial merge groups")
    # 完整像素张量上的 Conv3D：y[b,d,t,h,w]=sum_{c,kt,kh,kw}(x*W)+bias[d]。
    # 与模型一致，FP32 像素先转为投影权重的 BF16 dtype，再参与卷积。
    out = F.conv3d(pixels.to(weight.dtype), weight, bias, stride=stride)
    # [B,D,Ot,Oh,Ow] -> [B,Ot,block_h,block_w,intra_h,intra_w,D]。
    # 连续 m*m 个 Token 对应同一空间组，供位置嵌入和 Merger 直接消费。
    raster = out.permute(0, 2, 3, 4, 1)
    grouped = raster.reshape(pixels.shape[0], ot, oh // m, m, ow // m, m, weight.shape[0])
    return grouped.permute(0, 1, 2, 4, 3, 5, 6).reshape(-1, weight.shape[0])


def vision_position_interpolation(grid_thw: torch.Tensor, table_side: int = 48,
                                   spatial_merge_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate int64 indices [N,4] and FP32 bilinear weights [N,4].

    grid_thw [M,3] contains temporal/height/width Patch counts per image/video;
    N=sum(t*h*w). Output rows follow the processor's spatial-merge block order,
    repeated for each temporal slice. The model uses align_corners=True and
    clamps neighbours to the learned table's border. Results stay on grid_thw.device.
    """
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError("grid_thw must have shape [M,3] with temporal/height/width Patch counts")
    if grid_thw.dtype not in (torch.int32, torch.int64):
        raise TypeError("grid_thw must contain int32 or int64 Patch counts")
    if type(table_side) is not int or table_side < 1:
        raise ValueError("table_side must be a positive integer")
    if type(spatial_merge_size) is not int or spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be a positive integer")
    if bool((grid_thw <= 0).any()):
        raise ValueError("Temporal/height/width Patch counts must be positive")
    if bool((grid_thw[:, 1:] % spatial_merge_size != 0).any()):
        raise ValueError("Patch height and width must be divisible by spatial_merge_size")
    if grid_thw.shape[0] == 0:
        return (torch.empty((0, 4), dtype=torch.int64, device=grid_thw.device),
                torch.empty((0, 4), dtype=torch.float32, device=grid_thw.device))

    # OP-11：把每个 Patch 行映射到所属图像/视频的 (t,h,w) 网格。
    # 一个视频的各帧复用相同的二维空间位置，因此先取 Token 在帧内的序号。
    grid = grid_thw.to(torch.int64)
    counts = grid[:, 0] * grid[:, 1] * grid[:, 2]
    heights = torch.repeat_interleave(grid[:, 1], counts)
    widths = torch.repeat_interleave(grid[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    within_frame = (torch.arange(counts.sum(), device=grid.device) - starts) % (heights * widths)

    # processor 按连续 m*m 小块排列 Patch；把组内顺序还原为真实空间 (row,col)。
    m = spatial_merge_size
    blocks_w = widths // m
    patch_row = (within_frame // (m * m * blocks_w)) * m + (within_frame // m) % m
    patch_col = ((within_frame // (m * m)) % blocks_w) * m + within_frame % m

    # align_corners=True：row_pos=row*(L-1)/(h-1)，col 同理，L=table_side。
    # 坐标使用 FP32；h/w=1 时分母取 1，此轴的唯一位置仍映射到 0。
    table_row = patch_row.float() * (table_side - 1) / (heights - 1).clamp(min=1)
    table_col = patch_col.float() * (table_side - 1) / (widths - 1).clamp(min=1)
    row_floor, col_floor = torch.floor(table_row), torch.floor(table_col)
    offsets = torch.arange(2, device=grid.device)
    # 每个位置取 floor/+1 两个邻点，边界 clamp；二维组合按 y0x0,y0x1,y1x0,y1x1 排列。
    row_indices = (row_floor.long()[:, None] + offsets).clamp(0, table_side - 1)
    col_indices = (col_floor.long()[:, None] + offsets).clamp(0, table_side - 1)

    # 一维权重为 1-|pos-neighbour|，二维权重为 wy*wx；与四个邻点索引逐一对应。
    row_weights = (1 - (table_row[:, None] - row_floor[:, None] - offsets).abs()).clamp(min=0)
    col_weights = (1 - (table_col[:, None] - col_floor[:, None] - offsets).abs()).clamp(min=0)
    interp_indices = (row_indices[:, :, None] * table_side + col_indices[:, None, :]).reshape(-1, 4)
    interp_weights = (row_weights[:, :, None] * col_weights[:, None, :]).reshape(-1, 4)
    return interp_indices, interp_weights


def vision_position_embedding(x: torch.Tensor, position_table: torch.Tensor,
                              interp_indices: torch.Tensor, interp_weights: torch.Tensor) -> torch.Tensor:
    """OP-11: four bilinear neighbours per row, supplied in processor order.

    vision_position_interpolation generates the IF-04 indices/weights with
    bilinear/align_corners=True and merge-compatible ordering.
    FP32 interpolation weights promote the weighted sum before its BF16 cast.
    """
    if x.ndim != 2 or position_table.ndim != 2 or position_table.shape[1] != x.shape[1]:
        raise ValueError("Position table and hidden features must have the same width")
    if interp_indices.shape != (x.shape[0], 4) or interp_weights.shape != interp_indices.shape:
        raise ValueError("Expected four interpolation indices and weights per Patch")
    if interp_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("Interpolation indices must be int32 or int64")
    if interp_weights.dtype != torch.float32:
        raise TypeError("Interpolation weights must retain the reference's FP32 dtype")
    if interp_indices.numel() and (int(interp_indices.min()) < 0 or int(interp_indices.max()) >= len(position_table)):
        raise ValueError("Interpolation index outside position table")
    # Gather 四邻点：[N,4] 索引 -> [N,4,D] 位置向量。
    neighbours = F.embedding(interp_indices, position_table)
    # pos[n,d]=sum_j(table[index[n,j],d]*weight[n,j])，插值和为 FP32。
    position = (neighbours * interp_weights.unsqueeze(-1)).sum(1)
    # 先将插值结果回转为 hidden dtype，再与原 hidden 相加，保留 BF16 舍入位置。
    return x + position.to(x.dtype)


def vision_axial_cos_sin(position_ids: torch.Tensor, head_dim: int = 72,
                         theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 axial frequencies for IF-04 spatial positions [N,2] = (h,w)."""
    if position_ids.ndim != 2 or position_ids.shape[1] != 2 or head_dim <= 0 or head_dim % 4:
        raise ValueError("Expected [N,2] positions and a head_dim divisible by four")
    # h/w 各占半个 head 的旋转维度；每对维度的频率为 theta^(-2*i/spatial_dim)。
    spatial_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, spatial_dim, 2, device=position_ids.device).float() / spatial_dim))
    angles = position_ids.unsqueeze(-1).float() * inv_freq
    # 先拼接 h、w 的角度，再重复一次，匹配 half-split 旋转的前后两半维度。
    frequencies = torch.cat((angles[:, 0], angles[:, 1]), dim=-1)
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos(), frequencies.sin()


def vision_axial_rope(q: torch.Tensor, k: torch.Tensor,
                      cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate whole heads in FP32, then restore the original Q/K dtypes."""
    if q.shape != k.shape or q.ndim != 3 or q.shape[-1] % 2:
        raise ValueError("Expected same-shaped Q/K [N,H,even head_dim]")
    if cos.shape != (q.shape[0], q.shape[-1]) or sin.shape != cos.shape:
        raise ValueError("cos/sin must have shape [N,head_dim]")
    # 旋转全 head：rotate_half([a,b])=[-b,a]，不是相邻偶/奇维配对。
    q32, k32 = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_half = torch.cat((-q32[..., q.shape[-1] // 2:], q32[..., :q.shape[-1] // 2]), dim=-1)
    k_half = torch.cat((-k32[..., k.shape[-1] // 2:], k32[..., :k.shape[-1] // 2]), dim=-1)
    # q_rot=q*cos+rotate_half(q)*sin；Q/K 都在 FP32 旋转后回到原 dtype。
    return (q32 * cos + q_half * sin).to(q.dtype), (k32 * cos + k_half * sin).to(k.dtype)


def vision_attention(x: torch.Tensor, qkv_weight: torch.Tensor, qkv_bias: torch.Tensor,
                      proj_weight: torch.Tensor, proj_bias: torch.Tensor, *, cos: torch.Tensor,
                      sin: torch.Tensor, cu_seqlens: torch.Tensor, num_heads: int = 16) -> torch.Tensor:
    """OP-09: noncausal MHA, independently within each packed visual segment."""
    if x.ndim != 2 or num_heads <= 0 or x.shape[1] % num_heads:
        raise ValueError("Vision hidden width must be divisible by num_heads")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64) or len(cu_seqlens) < 2:
        raise ValueError("cu_seqlens must be a cumulative integer vector")
    boundaries = cu_seqlens.tolist()
    if boundaries[0] != 0 or boundaries[-1] != x.shape[0] or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Visual segments must be positive and partition all Patch rows")
    n, d = x.shape
    head_dim = d // num_heads
    # OP-09 输入投影：[N,D] -> [N,3D]，拆成 Q/K/V[N,H,head_dim]，随后旋转 Q/K。
    q, k, v = F.linear(x, qkv_weight, qkv_bias).reshape(n, 3, num_heads, head_dim).unbind(1)
    q, k = vision_axial_rope(q, k, cos, sin)
    parts = []
    for begin, end in zip(boundaries, boundaries[1:]):
        # 每段视觉 Token 独立做非因果注意力；不同 cu_seqlens 片段之间互不可见。
        qh, kh, vh = (t[begin:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        # score=Q@K.T/sqrt(head_dim)；在 BF16 score 之后用 FP32 Softmax，再回转 BF16。
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * head_dim**-0.5
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        # context=Softmax(score)@V，合并各段与各 head 后做输出投影。
        parts.append(torch.matmul(probabilities, vh).transpose(1, 2).contiguous())
    context = torch.cat(parts, dim=1).reshape(n, d)
    return F.linear(context, proj_weight, proj_bias)


def dense_ffn(x: torch.Tensor, fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
              fc2_weight: torch.Tensor, fc2_bias: torch.Tensor) -> torch.Tensor:
    """OP-10: Vision block Linear -> GELU(tanh) -> Linear, including biases."""
    # OP-10：hidden=x@W1.T+b1；GELU 使用 tanh 近似，随后 y=GELU(hidden)@W2.T+b2。
    # BF16 模型路径保留第一个 Linear、GELU 和第二个 Linear 各自的输出边界。
    hidden = F.linear(x, fc1_weight, fc1_bias)
    return F.linear(F.gelu(hidden, approximate="tanh"), fc2_weight, fc2_bias)


def spatial_merger(x: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor,
                   fc1_weight: torch.Tensor, fc1_bias: torch.Tensor, fc2_weight: torch.Tensor,
                   fc2_bias: torch.Tensor, *, spatial_merge_size: int = 2, eps: float = 1e-6) -> torch.Tensor:
    """OP-12: Norm before grouping, then Linear -> exact GELU -> Linear.

    Incoming rows already follow the processor's contiguous m*m Patch groups.
    """
    group = spatial_merge_size**2
    if x.ndim != 2 or spatial_merge_size < 1 or x.shape[0] % group:
        raise ValueError("Patch rows must contain complete spatial merge groups")
    # OP-12：先逐 Patch 沿 D 维做 LayerNorm，再把 m*m 个相邻 Patch 拼成一个向量。
    # 输入已按 merge 组连续排列；[N,D] -> [N/(m*m),m*m*D]，reshape 不做空间重排。
    normalized = layer_norm(x, norm_weight, norm_bias, eps)
    grouped = normalized.reshape(-1, group * x.shape[-1])
    # 两层投影之间使用 erf 定义的 GELU(x)=0.5*x*(1+erf(x/sqrt(2)))。
    hidden = F.gelu(F.linear(grouped, fc1_weight, fc1_bias), approximate="none")
    return F.linear(hidden, fc2_weight, fc2_bias)
