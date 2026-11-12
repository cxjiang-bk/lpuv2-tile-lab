"""Complete MoE with different routes, shared contribution and Token recovery."""

import torch
from torch.nn import functional as F

from pymodel.ref.moe import moe


def test_moe_known_experts_route_back_to_their_tokens():
    x = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bfloat16)
    experts = 10
    router = torch.stack((torch.arange(experts), -torch.arange(experts)), dim=1).bfloat16()
    weights = {"router": router, "shared_gate": torch.tensor([[0, 2]], dtype=torch.bfloat16)}
    for name in ("gate", "up"):
        weights[f"expert_{name}"] = torch.ones(experts, 2, 2).to(torch.float8_e4m3fn)
        weights[f"shared_ffn_{name}"] = torch.ones(2, 2).to(torch.float8_e4m3fn)
    weights["expert_down"] = torch.arange(1, experts + 1).float()[:, None, None].expand(-1, 2, 2).to(torch.float8_e4m3fn)
    weights["shared_ffn_down"] = torch.ones(2, 2).to(torch.float8_e4m3fn)
    scales = {name: torch.ones((experts, 1, 1) if name.startswith("expert_") else (1, 1), dtype=torch.bfloat16)
              for name in weights if name.startswith(("expert_", "shared_ffn_"))}

    result = moe(x, weights, scales)
    expected_ids = torch.tensor([[9, 8, 7, 6, 5, 4, 3, 2], [0, 1, 2, 3, 4, 5, 6, 7]])
    probabilities = F.softmax(F.linear(x.reshape(2, 2), router).float(), dim=-1)
    selected = probabilities.gather(-1, expected_ids)
    expected_route_weight = (selected / selected.sum(-1, keepdim=True)).bfloat16()
    assert torch.equal(result["expert_id"], expected_ids)
    torch.testing.assert_close(result["route_weight"], expected_route_weight, atol=0, rtol=0)

    # gate/up output is exactly 1 in this example; each expert's Down uses
    # constant integer weight e+1. This makes the FFN result directly readable.
    activation = F.silu(torch.ones(2, dtype=torch.bfloat16))
    expected = torch.zeros(2, 2, dtype=torch.bfloat16)
    for expert in range(experts):
        token, slot = torch.where(expected_ids == expert)
        expert_value = F.linear(activation, torch.full((2, 2), expert + 1, dtype=torch.bfloat16))
        expected[token] += expert_value * expected_route_weight[token, slot, None]
    shared = F.linear(activation, torch.ones(2, 2, dtype=torch.bfloat16))
    expected += shared * torch.sigmoid(F.linear(x.reshape(2, 2), weights["shared_gate"]))
    torch.testing.assert_close(result["out"], expected.reshape_as(x), atol=1e-2, rtol=1e-2)
    assert result["bucket_offset"][-1].item() == 16
    assert torch.equal(torch.sort(result["bucket_token_id"] * 8 + result["bucket_slot"]).values, torch.arange(16))


def test_moe_public_topk_ties_and_flat_token_input(moe_case):
    weights, scales = moe_case
    weights["router"].zero_()
    result = moe(torch.ones(2, 8, dtype=torch.bfloat16), weights, scales)
    assert result["out"].shape == (2, 8)
    assert result["out"].dtype == torch.bfloat16
    assert torch.equal(result["expert_id"], torch.arange(8).expand(2, 8))
    torch.testing.assert_close(result["route_weight"], torch.full((2, 8), 0.125, dtype=torch.bfloat16), atol=0, rtol=0)
