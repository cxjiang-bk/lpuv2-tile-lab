"""K-14/15/16: compare full Tile entries with independent output Goldens."""

import json
from pathlib import Path

import pytest
import torch

from kernels.pto.mtp.input import feature_concat, mtp_input_projection
from kernels.pto.output.lm_head import lm_head
from pymodel.common.case import Case, OutputCheck
from pymodel.isa.torch_codegen import torch_codegen
from pymodel.ref.output import feature_concat as concat_ref
from pymodel.ref.output import lm_head as lm_head_ref
from pymodel.ref.output import mtp_input_projection as projection_ref
from pymodel.task_gen.validate import run_case


def _compile_entry(kernel, args, tmp_path):
    program = kernel.specialize(*args)
    ir_path = tmp_path / "program_ir.py"
    code_path = tmp_path / "program_torch.py"
    ir_path.write_text(program.as_python())
    code = torch_codegen(program, check_shapes=True)
    code_path.write_text(code)
    namespace = {}
    exec(compile(code, str(code_path), "exec"), namespace)
    root = Path(__file__).resolve().parents[2]
    upstream = json.loads((root / "data/upstream.lock.json").read_text())
    details = {
        "pypto_commit": upstream["pypto"]["commit"],
        "device": "cpu",
        "adapter": "pymodel.isa.torch_codegen",
        "ir": str(ir_path),
        "generated_code": str(code_path),
    }
    return namespace, details


@pytest.mark.parametrize("hidden,vocab", [(64, 16), (67, 19)], ids=["full_tiles", "tails"])
def test_lm_head_selects_request_local_positions(hidden, vocab, tmp_path):
    case = Case(case_id=f"lm_head_h{hidden}_v{vocab}", seed=61,
                numerical_profile="pto-functional-v0", op_id="OP-14", kernel_id="K-14",
                outputs={"logits": OutputCheck(atol=1e-2, rtol=1e-2)})
    generator = case.generator()
    h = torch.randn(2, 4, hidden, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(vocab, hidden, generator=generator, dtype=torch.bfloat16)
    # Per-request local indices, deliberately reordered and repeated.
    positions = torch.tensor([[3, 0, 3], [0, 2, 1]], dtype=torch.int64)
    shape = (2, 3, vocab)
    namespace, details = _compile_entry(lm_head, (h, weight, positions,
                                                 torch.empty(shape, dtype=torch.bfloat16)), tmp_path)

    def reference(inputs, parameters):
        return {"logits": lm_head_ref(inputs["h"], inputs["weight"], inputs["positions"])}

    def implementation(inputs, parameters):
        out = torch.full(shape, torch.nan, dtype=torch.bfloat16)
        namespace["lm_head"](inputs["h"], inputs["weight"], inputs["positions"], out)
        return {"logits": out}

    result = run_case(case, inputs={"h": h, "weight": weight, "positions": positions},
                      reference=reference, implementation=implementation,
                      backend="pypto_torch_codegen_adapted", backend_details=details,
                      artifact_root=tmp_path / "runs")
    assert result.passed, result.report


@pytest.mark.parametrize("hidden", [128, 131], ids=["full_tile", "tail"])
def test_feature_concat_keeps_embedding_before_hidden(hidden, tmp_path):
    case = Case(case_id=f"concat_h{hidden}", seed=62,
                numerical_profile="pto-functional-v0", op_id="OP-15", kernel_id="K-15",
                outputs={"features": OutputCheck(atol=0, rtol=0)})
    # Different row/column markers make swaps and misplaced writes visible.
    e_norm = (torch.arange(6 * hidden).reshape(2, 3, hidden) % 113).to(torch.bfloat16)
    h_norm = -e_norm - 1
    shape = (2, 3, 2 * hidden)
    namespace, details = _compile_entry(
        feature_concat,
        (e_norm.view(-1, hidden), h_norm.view(-1, hidden),
         torch.empty(6, 2 * hidden, dtype=torch.bfloat16)), tmp_path,
    )

    def reference(inputs, parameters):
        return {"features": concat_ref(inputs["e_norm"], inputs["h_norm"])}

    def implementation(inputs, parameters):
        out = torch.full(shape, torch.nan, dtype=torch.bfloat16)
        namespace["feature_concat"](inputs["e_norm"].view(-1, hidden),
                                     inputs["h_norm"].view(-1, hidden), out.view(-1, 2 * hidden))
        return {"features": out}

    result = run_case(case, inputs={"e_norm": e_norm, "h_norm": h_norm},
                      reference=reference, implementation=implementation,
                      backend="pypto_torch_codegen_adapted", backend_details=details,
                      artifact_root=tmp_path / "runs")
    assert result.passed, result.report


@pytest.mark.parametrize("hidden", [32, 35], ids=["full_tiles", "tails"])
def test_mtp_projection_uses_independent_fc(hidden, tmp_path):
    case = Case(case_id=f"mtp_projection_h{hidden}", seed=63,
                numerical_profile="pto-functional-v0", op_id="OP-16", kernel_id="K-16",
                outputs={"hidden": OutputCheck(atol=1e-2, rtol=1e-2)})
    generator = case.generator()
    features = torch.randn(2, 3, 2 * hidden, generator=generator, dtype=torch.bfloat16)
    fc_weight = torch.randn(hidden, 2 * hidden, generator=generator, dtype=torch.bfloat16)
    namespace, details = _compile_entry(
        mtp_input_projection,
        (features.view(-1, 2 * hidden), fc_weight, torch.empty(6, hidden, dtype=torch.bfloat16)),
        tmp_path,
    )

    def reference(inputs, parameters):
        return {"hidden": projection_ref(inputs["features"], inputs["fc_weight"])}

    def implementation(inputs, parameters):
        out = torch.full((2, 3, hidden), torch.nan, dtype=torch.bfloat16)
        namespace["mtp_input_projection"](inputs["features"].view(-1, 2 * hidden),
                                           inputs["fc_weight"], out.view(-1, hidden))
        return {"hidden": out}

    result = run_case(case, inputs={"features": features, "fc_weight": fc_weight},
                      reference=reference, implementation=implementation,
                      backend="pypto_torch_codegen_adapted", backend_details=details,
                      artifact_root=tmp_path / "runs")
    assert result.passed, result.report
