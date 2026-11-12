"""Compare the indexed Tile kernel with the independent Embedding Golden."""

import json
from pathlib import Path

import pytest
import torch
from pypto.debug import torch_codegen

from kernels.pto.common.embedding import embedding
from pymodel.common.case import Case, OutputCheck
from pymodel.ref.embedding import embedding as embedding_ref
from pymodel.task_gen.validate import run_case


@pytest.mark.parametrize("hidden", [64, 67], ids=["full_tiles", "column_tail"])
def test_embedding_matches_golden(hidden, tmp_path):
    case = Case(
        case_id=f"embedding_h{hidden}",
        seed=42,
        numerical_profile="pto-functional-v0",
        outputs={"out": OutputCheck(atol=0, rtol=0)},
        op_id="OP-01",
        kernel_id="K-01",
    )
    token_ids = torch.tensor([[0, 10, 3], [3, 8, 0]], dtype=torch.int64)
    weight = torch.randn(11, hidden, generator=case.generator(), dtype=torch.bfloat16)
    sample_out = torch.empty(2, 3, hidden, dtype=torch.bfloat16)
    program = embedding.specialize(token_ids, weight, sample_out)
    code = torch_codegen(program, check_shapes=True)
    ir_path = tmp_path / "embedding_ir.py"
    code_path = tmp_path / "embedding_torch.py"
    ir_path.write_text(program.as_python())
    code_path.write_text(code)
    namespace = {}
    exec(compile(code, str(code_path), "exec"), namespace)

    def reference(inputs, parameters):
        return {"out": embedding_ref(inputs["token_ids"], inputs["embedding_weight"])}

    def implementation(inputs, parameters):
        out = torch.full((2, 3, hidden), torch.nan, dtype=torch.bfloat16)
        namespace["embedding"](inputs["token_ids"], inputs["embedding_weight"], out)
        return {"out": out}

    root = Path(__file__).resolve().parents[2]
    upstream = json.loads((root / "data/upstream.lock.json").read_text())
    result = run_case(
        case,
        inputs={"token_ids": token_ids, "embedding_weight": weight},
        reference=reference,
        implementation=implementation,
        backend="pypto_torch_codegen",
        artifact_root=tmp_path / "runs",
        backend_details={
            "pypto_commit": upstream["pypto"]["commit"],
            "device": "cpu",
            "tile_shape": [1, 32],
            "ir": str(ir_path),
            "generated_code": str(code_path),
        },
    )
    assert result.passed, result.report
