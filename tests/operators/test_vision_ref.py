"""BF16 Vision definitions checked against pinned upstream Torch modules."""

from types import SimpleNamespace

import pytest
import torch

from pymodel.ref.common import layer_norm, residual_add
from pymodel.ref.vision import (
    dense_ffn,
    spatial_merger,
    vision_attention,
    vision_axial_cos_sin,
    vision_axial_rope,
    vision_patch_embedding,
    vision_position_embedding,
)


def frozen_module(cls, config):
    with torch.random.fork_rng():
        torch.manual_seed(23)
        return cls(config).to(torch.bfloat16).eval().requires_grad_(False)


def attention_definitions(upstream):
    return upstream("repeat_kv", "eager_attention_forward", "rotate_half", "apply_rotary_pos_emb_vision",
                    "Qwen3_5MoeVisionAttention", "Qwen3_5MoeVisionMLP", "Qwen3_5MoeVisionBlock")


def run_attention(x, module, cos, sin, cu):
    return vision_attention(x, module.qkv.weight, module.qkv.bias, module.proj.weight, module.proj.bias,
                            cos=cos, sin=sin, cu_seqlens=cu, num_heads=module.num_heads)


@pytest.mark.parametrize("hidden", [8, 1152])
def test_pixel_conv3d_matches_pinned_patch_projection(upstream, hidden):
    source = upstream("Qwen3_5MoeVisionPatchEmbed")
    config = SimpleNamespace(patch_size=16, temporal_patch_size=2, in_channels=3, hidden_size=hidden)
    module = frozen_module(source.Qwen3_5MoeVisionPatchEmbed, config)
    pixels = torch.randn(2, 3, 4, 32, 64, generator=torch.Generator().manual_seed(17))
    actual = vision_patch_embedding(pixels, module.proj.weight, module.proj.bias)
    # 仅测试侧按固定模型的输入排列分 Patch；被测 reference 直接调用完整像素 Conv3D。
    # [B,C,Ot,Kt,block_h,m,Kh,block_w,m,Kw] -> merge 连续的 [N,C*Kt*Kh*Kw]。
    patches = pixels.reshape(2, 3, 2, 2, 1, 2, 16, 2, 2, 16)
    patches = patches.permute(0, 2, 4, 7, 5, 8, 1, 3, 6, 9).reshape(-1, 1536)
    torch.testing.assert_close(actual, module(patches), atol=0, rtol=0)
    assert actual.shape == (32, hidden) and actual.dtype == torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pixel_conv3d_overlapping_windows_and_valid_tail(dtype):
    rng = torch.Generator().manual_seed(31)
    pixels = torch.randn(2, 2, 4, 6, 8, generator=rng).to(dtype)
    weight = torch.randn(3, 2, 2, 3, 2, generator=rng).bfloat16()
    bias = torch.randn(3, generator=rng).bfloat16()
    out = vision_patch_embedding(pixels, weight, bias, stride=(1, 2, 3), spatial_merge_size=1)
    # 用独立的显式窗口求和检查批次/时间/通道和非方形空间步长，尾部使用 valid 规则。
    expected = []
    for b in range(2):
        for t in range(3):
            for h in range(0, 4, 2):
                for w in range(0, 7, 3):
                    window = pixels[b, :, t:t+2, h:h+3, w:w+2].bfloat16().float()
                    expected.append(((weight.float() * window).sum((1, 2, 3, 4)) + bias.float()).bfloat16())
    torch.testing.assert_close(out, torch.stack(expected), atol=.01, rtol=.01)


def test_pixel_conv3d_merge_order_preserves_batch_and_time():
    # 单点卷积使输出直接显示空间位置。每四行构成一个 2x2 Merge 组。
    pixels = torch.arange(2 * 2 * 4 * 6).reshape(2, 1, 2, 4, 6).bfloat16()
    weight = torch.ones(1, 1, 1, 1, 1, dtype=torch.bfloat16)
    out = vision_patch_embedding(pixels, weight, torch.zeros(1, dtype=torch.bfloat16))
    order = [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11, 12, 13, 18, 19,
             14, 15, 20, 21, 16, 17, 22, 23]
    expected = torch.tensor([v + frame * 24 for frame in range(4) for v in order]).bfloat16()
    torch.testing.assert_close(out[:, 0], expected, atol=0, rtol=0)


@pytest.mark.parametrize("problem", ["packed", "integer", "short", "merge", "stride"])
def test_pixel_conv3d_rejects_invalid_input(problem):
    pixels = torch.zeros(1, 3, 2, 4, 4)
    weight = torch.zeros(8, 3, 2, 2, 2, dtype=torch.bfloat16)
    bias = torch.zeros(8, dtype=torch.bfloat16)
    kwargs = {}
    if problem == "packed":
        pixels = pixels.reshape(4, -1)
    elif problem == "integer":
        pixels = pixels.int()
    elif problem == "short":
        pixels = pixels[:, :, :1]
    elif problem == "merge":
        pixels = pixels[:, :, :, :2]
    else:
        kwargs["stride"] = (1, 0, 2)
    with pytest.raises((ValueError, TypeError)):
        vision_patch_embedding(pixels, weight, bias, **kwargs)


def test_position_interpolation_casts_before_the_bf16_add():
    x = torch.ones(1, 1, dtype=torch.bfloat16)
    table = torch.tensor([[0], [1], [0], [1]], dtype=torch.bfloat16)
    indices = torch.tensor([[0, 1, 2, 3]])
    fraction = 0.003915
    weights = torch.tensor([[(1 - fraction) / 2, fraction / 2, (1 - fraction) / 2, fraction / 2]])
    actual = vision_position_embedding(x, table, indices, weights)
    assert actual.item() == 1.0
    assert (x.float() + fraction).bfloat16().item() != actual.item()


def test_position_rows_follow_the_supplied_patch_order():
    table = torch.tensor([[0., 1.], [2., 3.], [4., 5.], [6., 7.]], dtype=torch.bfloat16)
    x = torch.zeros(2, 2, dtype=torch.bfloat16)
    indices = torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]])
    weights = torch.tensor([[1., 0., 0., 0.], [0.25, 0.25, 0.25, 0.25]])
    expected = torch.tensor([[6., 7.], [3., 4.]], dtype=torch.bfloat16)
    torch.testing.assert_close(vision_position_embedding(x, table, indices, weights), expected, atol=0, rtol=0)


def test_axial_positions_and_half_split_rotation_match_upstream(upstream):
    source = upstream("Qwen3_5MoeVisionRotaryEmbedding", "rotate_half", "apply_rotary_pos_emb_vision")
    config = SimpleNamespace(hidden_size=1152, num_attention_heads=16, head_dim=72,
                             rope_parameters={"rope_type": "axial", "rope_theta": 10000.0})
    rotary = source.Qwen3_5MoeVisionRotaryEmbedding(config)
    positions = torch.tensor([[0, 0], [1, 2], [3, 1]])
    q = torch.randn(3, 16, 72, generator=torch.Generator().manual_seed(17), dtype=torch.bfloat16)
    cos, sin = vision_axial_cos_sin(positions)
    expected_cos, expected_sin = rotary(q, positions)
    torch.testing.assert_close(cos, expected_cos, atol=0, rtol=0)
    torch.testing.assert_close(sin, expected_sin, atol=0, rtol=0)
    actual = vision_axial_rope(q, q.flip(0), cos, sin)
    expected = source.apply_rotary_pos_emb_vision(q, q.flip(0), cos, sin)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
        assert a.dtype == torch.bfloat16


@pytest.mark.parametrize("hidden,heads", [(16, 2), (1152, 16)])
def test_attention_matches_packed_noncausal_upstream(upstream, hidden, heads):
    source = attention_definitions(upstream)
    config = SimpleNamespace(hidden_size=hidden, num_heads=heads, _attn_implementation="eager")
    module = frozen_module(source.Qwen3_5MoeVisionAttention, config)
    x = torch.randn(8, hidden, generator=torch.Generator().manual_seed(17), dtype=torch.bfloat16)
    cos, sin = vision_axial_cos_sin(torch.tensor([[i // 4, i % 4] for i in range(8)]), hidden // heads)
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    expected = module(x, cu_seqlens=cu, position_embeddings=(cos, sin))
    actual = run_attention(x, module, cos, sin, cu)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    altered = x.clone()
    altered[3:] += 3
    torch.testing.assert_close(run_attention(altered, module, cos, sin, cu)[:3], actual[:3], atol=0, rtol=0)
    altered = x.clone()
    altered[2] += 3
    assert not torch.equal(run_attention(altered, module, cos, sin, cu)[0], actual[0])


@pytest.mark.parametrize("hidden,intermediate", [(16, 29), (1152, 4304)])
def test_dense_ffn_matches_upstream_gelu_tanh(upstream, hidden, intermediate):
    source = upstream("Qwen3_5MoeVisionMLP")
    config = SimpleNamespace(hidden_size=hidden, intermediate_size=intermediate, hidden_act="gelu_pytorch_tanh")
    module = frozen_module(source.Qwen3_5MoeVisionMLP, config)
    x = torch.randn(3, hidden, generator=torch.Generator().manual_seed(17), dtype=torch.bfloat16)
    actual = dense_ffn(x, module.linear_fc1.weight, module.linear_fc1.bias,
                       module.linear_fc2.weight, module.linear_fc2.bias)
    torch.testing.assert_close(actual, module(x), atol=0, rtol=0)


def test_spatial_merger_matches_upstream_norm_before_grouping_and_exact_gelu(upstream):
    source = upstream("Qwen3_5MoeVisionPatchMerger")
    config = SimpleNamespace(hidden_size=8, spatial_merge_size=2, out_hidden_size=6)
    module = frozen_module(source.Qwen3_5MoeVisionPatchMerger, config)
    x = torch.randn(8, 8, generator=torch.Generator().manual_seed(17), dtype=torch.bfloat16)
    actual = spatial_merger(x, module.norm.weight, module.norm.bias, module.linear_fc1.weight,
                            module.linear_fc1.bias, module.linear_fc2.weight, module.linear_fc2.bias)
    torch.testing.assert_close(actual, module(x), atol=0, rtol=0)
    with pytest.raises(ValueError, match="complete spatial merge groups"):
        spatial_merger(x[:7], module.norm.weight, module.norm.bias, module.linear_fc1.weight,
                       module.linear_fc1.bias, module.linear_fc2.weight, module.linear_fc2.bias)


def test_vision_block_composes_the_independent_references(upstream):
    source = attention_definitions(upstream)
    config = SimpleNamespace(hidden_size=16, num_heads=2, intermediate_size=29,
                             hidden_act="gelu_pytorch_tanh", _attn_implementation="eager")
    module = frozen_module(source.Qwen3_5MoeVisionBlock, config)
    x = torch.randn(8, 16, generator=torch.Generator().manual_seed(17), dtype=torch.bfloat16)
    cos, sin = vision_axial_cos_sin(torch.tensor([[i // 4, i % 4] for i in range(8)]), 8)
    cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    u = layer_norm(x, module.norm1.weight, module.norm1.bias)
    hidden = residual_add(x, run_attention(u, module.attn, cos, sin, cu))
    u = layer_norm(hidden, module.norm2.weight, module.norm2.bias)
    actual = residual_add(hidden, dense_ffn(u, module.mlp.linear_fc1.weight, module.mlp.linear_fc1.bias,
                                           module.mlp.linear_fc2.weight, module.mlp.linear_fc2.bias))
    torch.testing.assert_close(actual, module(x, cu, position_embeddings=(cos, sin)), atol=0, rtol=0)
