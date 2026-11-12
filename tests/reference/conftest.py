"""Small synthetic parameters for complete B-route reference calls."""

import pytest
import torch


@pytest.fixture
def attention_case():
    generator = torch.Generator().manual_seed(51)
    hidden, head_dim, q_heads, kv_heads = 8, 256, 2, 1
    shapes = {"q_gate": (2 * q_heads * head_dim, hidden), "k": (kv_heads * head_dim, hidden),
              "v": (kv_heads * head_dim, hidden), "out": (hidden, q_heads * head_dim)}
    weights = {name: torch.randint(-2, 3, shape, generator=generator).to(torch.float8_e4m3fn)
               for name, shape in shapes.items()}
    weights.update(q_norm=torch.zeros(head_dim, dtype=torch.bfloat16),
                   k_norm=torch.zeros(head_dim, dtype=torch.bfloat16))
    scales = {name: torch.full(((n + 127) // 128, (k + 127) // 128), 0.02, dtype=torch.bfloat16)
              for name, (n, k) in shapes.items()}
    x = torch.randn(2, 3, hidden, generator=generator, dtype=torch.bfloat16)
    positions = torch.tensor([[5, 6, 7], [10, 11, 12]], dtype=torch.int64)
    return x, weights, scales, positions


@pytest.fixture
def moe_case():
    generator = torch.Generator().manual_seed(53)
    experts, hidden, intermediate = 10, 8, 4
    weights = {"router": torch.randn(experts, hidden, generator=generator, dtype=torch.bfloat16),
               "shared_gate": torch.randn(1, hidden, generator=generator, dtype=torch.bfloat16)}
    scales = {}
    for name, shape in {"gate": (intermediate, hidden), "up": (intermediate, hidden),
                        "down": (hidden, intermediate)}.items():
        weights[f"expert_{name}"] = torch.randint(-2, 3, (experts, *shape), generator=generator).to(torch.float8_e4m3fn)
        weights[f"shared_ffn_{name}"] = torch.randint(-2, 3, shape, generator=generator).to(torch.float8_e4m3fn)
        scales[f"expert_{name}"] = torch.full((experts, 1, 1), 0.02, dtype=torch.bfloat16)
        scales[f"shared_ffn_{name}"] = torch.full((1, 1), 0.02, dtype=torch.bfloat16)
    return weights, scales
