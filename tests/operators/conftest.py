"""Run selected, hash-verified upstream Torch fallbacks as independent oracles."""

import __future__
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="session", autouse=True)
def reference_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="session")
def upstream():
    root = Path(__file__).resolve().parents[2]
    baseline = json.loads((root / "configs/models/qwen3_6_35b_a3b_fp8/baseline.json").read_text())
    def load(*names, source_file="modeling_qwen3_5_moe.py"):
        source = root / "tests/reference_sources" / (source_file + ".txt")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == (
            baseline["source_files"][source_file]["sha256"]
        ), "Upstream test source differs from the pinned model baseline"
        tree = ast.parse(source.read_text())
        selected = [node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        assert {node.name for node in selected} == set(names), "Unknown upstream definition"
        for node in selected:
            node.decorator_list = []
        module = ast.Module(body=selected, type_ignores=[])
        namespace = {
            "torch": torch, "nn": torch.nn, "F": F,
            "ACT2FN": {"silu": F.silu, "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh")},
            "is_torchdynamo_exporting": lambda: False,
            "is_flash_attention_requested": lambda config: False,
            "ALL_ATTENTION_FUNCTIONS": SimpleNamespace(get_interface=lambda name, fallback: fallback),
            "GradientCheckpointingLayer": torch.nn.Module,
            "auto_docstring": lambda function: function,
            "force_accelerate_hooks": lambda *args: lambda function: function,
        }
        exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag,
                     dont_inherit=True), namespace)
        return SimpleNamespace(**{name: namespace[name] for name in names})

    return load
