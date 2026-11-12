"""Run the two-pass RMSNorm Tile kernel against the independent Golden."""

import json
from pathlib import Path

import pytest
import torch
from pypto.debug import torch_codegen

from kernels.pto.common.rmsnorm import rmsnorm
from pymodel.common.case import Case, OutputCheck
from pymodel.ref.normalization import rmsnorm as rmsnorm_ref
from pymodel.task_gen.validate import run_case


@pytest.mark.parametrize(
    "shape,eps",
    [((2, 3, 256), 1e-6), ((2, 2048), 1e-6), ((2, 259), 0.25)],
    ids=["head_dim", "hidden_dim", "column_tail"],
)
def test_rmsnorm_matches_golden(shape, eps, tmp_path):
    case = Case(
        case_id=f"rmsnorm_d{shape[-1]}",
        seed=43,
        numerical_profile="pto-functional-v0",
        outputs={"out": OutputCheck(atol=1e-2, rtol=1e-2)},
        parameters={"eps": eps},
        op_id="OP-02",
        kernel_id="K-02",
    )
    generator = case.generator()
    x = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], generator=generator, dtype=torch.bfloat16) / 4
    # Unequal column blocks expose accidental per-block normalization. Include
    # a zero row so epsilon must keep that row finite.
    x[..., 128:] *= 4
    x.reshape(-1, shape[-1])[0].zero_()
    rows = x.view(-1, shape[-1])
    program = rmsnorm.specialize(rows, weight, torch.empty_like(rows), eps)
    code = torch_codegen(program, check_shapes=True)
    ir_path = tmp_path / "rmsnorm_ir.py"
    code_path = tmp_path / "rmsnorm_torch.py"
    ir_path.write_text(program.as_python())
    code_path.write_text(code)
    namespace = {}
    exec(compile(code, str(code_path), "exec"), namespace)

    def reference(inputs, parameters):
        return {"out": rmsnorm_ref(inputs["x"], inputs["weight"], eps=parameters["eps"])}

    def implementation(inputs, parameters):
        out = torch.full_like(inputs["x"], torch.nan)
        namespace["rmsnorm"](
            inputs["x"].view(-1, shape[-1]), inputs["weight"],
            out.view(-1, shape[-1]), parameters["eps"],
        )
        return {"out": out}

    root = Path(__file__).resolve().parents[2]
    upstream = json.loads((root / "data/upstream.lock.json").read_text())
    result = run_case(
        case,
        inputs={"x": x, "weight": weight},
        reference=reference,
        implementation=implementation,
        backend="pypto_torch_codegen",
        artifact_root=tmp_path / "runs",
        backend_details={
            "pypto_commit": upstream["pypto"]["commit"],
            "device": "cpu",
            "tile_shape": [1, 128],
            "ir": str(ir_path),
            "generated_code": str(code_path),
        },
    )
    assert result.passed, result.report
