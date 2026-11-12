"""GDN Torch math, explicit FP32 recurrent state and logical Conv history.

The full graph receives independent reference projection callables. Their FP8
quantization backend is a separate dependency, never an implicit BF16 fallback.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GDNProjections:
    qkv: Callable[[torch.Tensor], torch.Tensor]
    z: Callable[[torch.Tensor], torch.Tensor]
    a: Callable[[torch.Tensor], torch.Tensor]
    b: Callable[[torch.Tensor], torch.Tensor]
    out: Callable[[torch.Tensor], torch.Tensor]

    def __post_init__(self):
        if not all(callable(p) for p in (self.qkv, self.z, self.a, self.b, self.out)):
            raise TypeError("All five independent reference projection callables are required")


def gdn_conv(qkv: torch.Tensor, weight: torch.Tensor,
              history: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Depthwise causal Conv+SiLU. History [B,K-1,C] stores pre-Conv qkv."""
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight[:, 0]
    if qkv.ndim != 3 or qkv.shape[1] == 0 or weight.ndim != 2 or weight.shape[0] != qkv.shape[2]:
        raise ValueError("Expected qkv [B,S>0,C] and depthwise weight [C,K] or [C,1,K]")
    b, _, c = qkv.shape
    k = weight.shape[-1]
    if k < 1:
        raise ValueError("Conv kernel must be nonempty")
    if history is None:
        history = qkv.new_zeros(b, k - 1, c)
    if history.shape != (b, k - 1, c) or history.dtype != qkv.dtype or history.device != qkv.device:
        raise ValueError("Conv history must match [B,K-1,C], qkv dtype and device")
    # 因果 DWConv：把过去 K-1 个原始 qkv 与当前序列拼为 [B,K-1+S,C]。
    # groups=C 表示每个通道独立卷积；Token t 只读取历史及当前位置。
    joined = torch.cat((history, qkv), dim=1)
    convolved = F.conv1d(joined.transpose(1, 2).to(weight.dtype), weight.unsqueeze(1), groups=c)
    # Conv 输出先落在权重 dtype，再做 SiLU(u)=u*sigmoid(u)，保留 BF16 边界。
    out = F.silu(convolved).transpose(1, 2).to(qkv.dtype)
    # 留给后续 Decode 的历史是最后 K-1 个卷积前 qkv，按时间从旧到新排列。
    next_history = joined[:, -(k - 1):].clone() if k > 1 else joined[:, :0].clone()
    return {"out": out, "conv_history_out": next_history}


def gdn_gates(a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor,
               dt_bias: torch.Tensor) -> dict[str, torch.Tensor]:
    """Keep sigmoid(b) in its projection dtype; compute g explicitly in FP32."""
    if a.shape != b.shape or a.shape[-1:] != A_log.shape or dt_bias.shape != A_log.shape:
        raise ValueError("a/b and per-head A_log/dt_bias shapes must agree")
    # beta = sigmoid(b) 控制本 Token 的写入强度，先保持 b 的投影 dtype（基线为 BF16）。
    # g = -exp(A_log) * softplus(a+dt_bias) 是 FP32 log-decay，exp(g) 用于衰减 State。
    return {"beta": b.sigmoid(), "g": -A_log.float().exp() * F.softplus(a.float() + dt_bias)}


def gdn_recurrent(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor | None = None,
                   *, normalize_qk: bool = True) -> dict[str, torch.Tensor]:
    """Raw Q/K -> optional FP32 L2Norm -> Q scaling once -> ordered delta rule."""
    if query.ndim != 4 or query.shape != key.shape or query.shape[:3] != value.shape[:3]:
        raise ValueError("Expected Q/K [B,S,H,K] and V [B,S,H,V]; repeat Q/K heads before the core")
    if query.shape[1] < 1 or beta.shape != query.shape[:3] or g.shape != beta.shape:
        raise ValueError("Expected nonempty sequence and beta/g [B,S,H]")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Q/K/V must have the same dtype")

    # 核心输入 Q/K[B,S,H,K]、V[B,S,H,V]；State[B,H,K,V] 沿 Token 递推。
    # q/k/v/beta/g 与 State 运算均使用 FP32；开启 normalize_qk 时做 L2Norm。
    # L2Norm(x)=x/sqrt(sum(x^2)+eps)，沿最后的 K 维归约。
    q, k, v, beta32, g32 = (t.float() for t in (query, key, value, beta, g))
    if normalize_qk:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)

    # Initialize the FP32 State without modifying the supplied history.
    state_shape = (query.shape[0], query.shape[2], query.shape[-1], value.shape[-1])
    if initial_state is None:
        state = v.new_zeros(state_shape)
    else:
        if initial_state.shape != state_shape or initial_state.dtype != torch.float32:
            raise ValueError("Initial recurrent State must have shape [B,H,K,V] and dtype FP32")
        state = initial_state.clone()

    # Q 额外乘 1/sqrt(K)，用于从 State 读出；K 不做这一缩放。
    q = q / query.shape[-1]**0.5
    outputs = []
    for t in range(query.shape[1]):
        # 先遗忘旧状态：S_decay = exp(g_t) * State，标量 gate 按 head 广播。
        state = state * g32[:, t].exp()[..., None, None]
        # 预测与误差修正：prediction = k_t^T @ S_decay，delta = beta_t*(v_t-prediction)。
        prediction = (state * k[:, t].unsqueeze(-1)).sum(-2)
        delta = (v[:, t] - prediction) * beta32[:, t].unsqueeze(-1)
        # 外积写入：State = S_decay + k_t[:,None] * delta[None,:]，结果为 [K,V]。
        state = state + k[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
        # 用当前 Q 读取更新后的状态：out_t = q_t^T @ State。
        outputs.append((state * q[:, t].unsqueeze(-1)).sum(-2))
    # 核心输出回转原 query dtype；跨 Token 保存的 State 继续保持 FP32。
    return {"out": torch.stack(outputs, dim=1).to(query.dtype), "state_out": state}


def gdn_chunk(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               beta: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor | None = None,
               *, chunk_size: int = 64, normalize_qk: bool = True) -> dict[str, torch.Tensor]:
    """Batched chunk coefficients/triangular solves, followed by a State scan."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if query.ndim != 4 or query.shape != key.shape or query.shape[:3] != value.shape[:3]:
        raise ValueError("Expected Q/K [B,S,H,K] and V [B,S,H,V]; repeat Q/K heads before the core")
    if query.shape[1] < 1 or beta.shape != query.shape[:3] or g.shape != beta.shape:
        raise ValueError("Expected nonempty sequence and beta/g [B,S,H]")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Q/K/V must have the same dtype")

    # 与 recurrent 相同的可选 FP32 L2Norm；chunk 仅改变递推的计算组织。
    q, k, v, beta32, g32 = (t.float() for t in (query, key, value, beta, g))
    if normalize_qk:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)

    # Initialize the FP32 State before the chunk scan.
    state_shape = (query.shape[0], query.shape[2], query.shape[-1], value.shape[-1])
    if initial_state is None:
        state = v.new_zeros(state_shape)
    else:
        if initial_state.shape != state_shape or initial_state.dtype != torch.float32:
            raise ValueError("Initial recurrent State must have shape [B,H,K,V] and dtype FP32")
        state = initial_state.clone()

    b, s, h, kd = q.shape
    vd = v.shape[-1]
    # 把 head 放到 Token 轴前：[B,S,H,K/V] -> [B,H,S,K/V]，每个 head 独立分块。
    q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    beta32, g32 = (t.transpose(1, 2).contiguous() for t in (beta32, g32))
    q = q * kd**-0.5
    padding = (-s) % chunk_size
    # 尾块补齐至 C=chunk_size；补入 q/k/v/beta/g=0，不产生写入或额外衰减。
    q, k, v = (F.pad(t, (0, 0, 0, padding)) for t in (q, k, v))
    beta32, g32 = (F.pad(t, (0, padding)) for t in (beta32, g32))
    chunks = (s + padding) // chunk_size
    # kb=beta*K，vb=beta*V；随后按 [B,H,num_chunks,C,K/V] 组织块内系数。
    kb, vb = k * beta32.unsqueeze(-1), v * beta32.unsqueeze(-1)
    q, k, kb = (t.reshape(b, h, chunks, chunk_size, kd) for t in (q, k, kb))
    vb = vb.reshape(b, h, chunks, chunk_size, vd)
    # G_i=sum_{t<=i}(g_t) 在每块内重新起算；D_ij=exp(G_i-G_j)，仅保留 j<=i。
    cumulative = g32.reshape(b, h, chunks, chunk_size).cumsum(-1)
    upper = torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool).triu(1)
    decay = (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).masked_fill(upper, -torch.inf).exp()
    # A=(beta*K)@K.T * D，P=Q@K.T * D；矩阵尺寸均为 [C,C]。
    system = (kb @ k.transpose(-1, -2)) * decay
    intra = (q @ k.transpose(-1, -2)) * decay
    # 令 L=I+strict_lower(A)：解 L*U=beta*V，L*W=beta*K*exp(G)。
    # unitriangular=True 将对角线视为 1，忽略 system 存储的对角值与上三角。
    u = torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(system, kb * cumulative.exp().unsqueeze(-1),
                                     upper=False, unitriangular=True)
    # Q_exp=Q*exp(G) 读取入块状态；K_tail=K*exp(G_end-G) 将本块写入传到块末。
    q = q * cumulative.exp().unsqueeze(-1)
    k = k * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    chunk_decay = cumulative[..., -1].exp()[..., None, None]
    outputs = []
    for i in range(chunks):
        # 消除入块 State 对 V 的预测：delta=U-W@State，尺寸 [C,V]。
        delta = u[:, :, i] - w[:, :, i] @ state
        # 块内输出由两部分组成：Q_exp@State（历史）+ P@delta（当前块）。
        outputs.append(q[:, :, i] @ state + intra[:, :, i] @ delta)
        # State_next=exp(G_end)*State + K_tail.T@delta；跨块状态保持 FP32。
        state = state * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ delta
    # 合并 chunk，去掉尾部补齐的 Token，再恢复 [B,S,H,V] 与 query dtype。
    out = torch.stack(outputs, dim=2).reshape(b, h, s + padding, vd)[:, :, :s]
    return {"out": out.transpose(1, 2).contiguous().to(query.dtype), "state_out": state}


def gdn_gated_norm(core_output: torch.Tensor, z: torch.Tensor,
                    weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Direct scale weight; retain the BF16 cast before multiplying it."""
    if core_output.shape != z.shape or weight.shape != (core_output.shape[-1],):
        raise ValueError("Gated Norm inputs and last-axis weight must agree")
    # 每个 value head 沿 V 维做 RMSNorm：n=u*rsqrt(mean(u^2)+eps)，不减均值。
    u = core_output.float()
    normalized = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + eps)
    # 按模型顺序先将 n 回转 BF16，再与直接权重 weight 相乘。
    weighted = weight * normalized.to(core_output.dtype)
    # 门控输出 = BF16(weighted * SiLU(FP32(z)))。
    return (weighted * F.silu(z.float())).to(core_output.dtype)


def gdn_reference(x: torch.Tensor, projections: GDNProjections, *, conv_weight: torch.Tensor,
                    A_log: torch.Tensor, dt_bias: torch.Tensor, norm_weight: torch.Tensor,
                    qk_heads: int = 16, value_heads: int = 32, key_head_dim: int = 128,
                    value_head_dim: int = 128, initial_state: torch.Tensor | None = None,
                    conv_history: torch.Tensor | None = None, path: str = "recurrent",
                    chunk_size: int = 64, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """OP-06 graph with explicit projection dependency and three named outputs.

    qkv/z/out need an FP8-capable independent reference for the target profile;
    supplying BF16 Linear modules validates only that declared functional path.
    """
    if x.ndim != 3 or min(qk_heads, value_heads, key_head_dim, value_head_dim) < 1 or value_heads % qk_heads:
        raise ValueError("Expected x [B,S,D] and value_heads divisible by qk_heads")
    if path not in ("recurrent", "chunk"):
        raise ValueError("path must be recurrent or chunk")
    if attention_mask is not None:
        if attention_mask.shape != x.shape[:2] or attention_mask.dtype != torch.bool:
            raise ValueError("GDN padding mask must be boolean [B,S]")
        # 在输入投影前将 padding Token 置零。
        x = (x * attention_mask.unsqueeze(-1)).to(x.dtype)
    b, s, _ = x.shape
    key_width, value_width = qk_heads * key_head_dim, value_heads * value_head_dim
    # 输入投影：qkv/z 供内容与输出门控，a/b 供每 Token、每 value head 的状态门控。
    # 目标 profile 中 qkv/z/out 使用 FP8 Linear，a/b 使用 BF16 Linear，投影结果均为 BF16。
    qkv, z, a, gate_b = (project(x) for project in (projections.qkv, projections.z, projections.a, projections.b))
    if any(t.dtype != x.dtype for t in (qkv, z, a, gate_b)):
        raise TypeError("Reference projections must return the source activation dtype")
    widths = (2 * key_width + value_width, value_width, value_heads, value_heads)
    if any(t.shape != (b, s, width) for t, width in zip((qkv, z, a, gate_b), widths)):
        raise ValueError("Reference projection shapes do not match the configured GDN heads")
    # 内容路径：qkv -> 因果 DWConv -> SiLU -> 拆分 Q/K/V。
    convolved = gdn_conv(qkv, conv_weight, conv_history)
    q, k, v = convolved["out"].split((key_width, key_width, value_width), dim=-1)
    # 一组 Q/K head 供多个 value head 使用；核心收到对齐后的 [B,S,H_value,K/V]。
    q, k = (t.reshape(b, s, qk_heads, key_head_dim).repeat_interleave(value_heads // qk_heads, dim=2)
            for t in (q, k))
    v = v.reshape(b, s, value_heads, value_head_dim)
    # 生成 beta/g，再选择 recurrent 或 chunk 计算同一个 delta-rule 核心。
    gates = gdn_gates(a, gate_b, A_log, dt_bias)
    if path == "recurrent":
        core = gdn_recurrent(q, k, v, gates["beta"], gates["g"], initial_state)
    else:
        core = gdn_chunk(q, k, v, gates["beta"], gates["g"], initial_state, chunk_size=chunk_size)
    # 输出路径：核心 BF16 输出 -> 每 head gated RMSNorm -> 合并 head -> 输出投影。
    gated = gdn_gated_norm(core["out"], z.reshape(b, s, value_heads, value_head_dim), norm_weight)
    out = projections.out(gated.reshape(b, s, value_width))
    if out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError("Output projection must restore the input hidden shape and activation dtype")
    return {"out": out, "state_out": core["state_out"],
            "conv_history_out": convolved["conv_history_out"]}
