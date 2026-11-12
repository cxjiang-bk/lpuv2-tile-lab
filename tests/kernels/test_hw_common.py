"""Native PTO C++ boundary checks for K-03 and K-04, one process per case."""

import hashlib
import importlib.util
import shutil
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    shutil.which("ptoas") is None or importlib.util.find_spec("simpler") is None,
    reason="Source scripts/hw-env.sh for native PTO C++ simulation",
)
ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "residual_offset_tail": {
        "op": "residual", "shape": (5, 147), "tile_rows": 2, "tile_cols": 64,
        "row_offset": 1, "col_offset": 5, "valid_rows": 3, "valid_cols": 129,
    },
    "residual_flattened_rows": {
        "op": "residual", "shape": (2, 3, 129), "tile_rows": 2, "tile_cols": 128,
        "row_offset": 2, "col_offset": 0, "valid_rows": 3, "valid_cols": 129,
    },
    "residual_narrow_window": {
        "op": "residual", "shape": (5, 17), "tile_rows": 3, "tile_cols": 7,
        "row_offset": 2, "col_offset": 3, "valid_rows": 2, "valid_cols": 7,
    },
    "layer_norm_width7": {
        "op": "layer_norm", "shape": (5, 7), "tile_rows": 2,
        "row_offset": 1, "valid_rows": 3,
    },
    "layer_norm_width129": {
        "op": "layer_norm", "shape": (2, 3, 129), "tile_rows": 2,
        "row_offset": 2, "valid_rows": 3,
    },
    "layer_norm_width1152": {
        "op": "layer_norm", "shape": (4, 1152), "tile_rows": 3,
        "row_offset": 1, "valid_rows": 2,
    },
}


def _run_native_case(case_name, artifact_dir):
    import torch

    from pymodel.ops.common import tile
    from pymodel.ref import common as reference

    torch.set_num_threads(1)
    case = CASES[case_name]
    shape = case["shape"]
    cols = shape[-1]
    rows = 1
    for dimension in shape[:-1]:
        rows *= dimension
    rng = torch.Generator().manual_seed(430)
    x = (torch.randn(shape, generator=rng) * .75 + .1).bfloat16()
    seed = (torch.arange(x.numel()).reshape(shape) % 97).float().mul(.25).bfloat16()
    expected = seed.clone()
    row_slice = slice(case["row_offset"], case["row_offset"] + case["valid_rows"])
    mask = torch.zeros((rows, cols), dtype=torch.bool)
    options = {key: case[key] for key in ("tile_rows", "row_offset", "valid_rows")}
    options.update(output_seed=seed, artifact_dir=artifact_dir / "program")

    if case["op"] == "residual":
        residual = torch.randn(shape, generator=rng).bfloat16()
        inputs = {"x": x, "residual": residual, "output_seed": seed}
        before = {name: value.clone() for name, value in inputs.items()}
        col_slice = slice(case["col_offset"], case["col_offset"] + case["valid_cols"])
        mask[row_slice, col_slice] = True
        golden = reference.residual_add(x, residual).reshape(rows, cols)
        expected.reshape(rows, cols)[mask] = golden[mask]
        actual = tile.residual_add(
            x, residual, tile_cols=case["tile_cols"], col_offset=case["col_offset"],
            valid_cols=case["valid_cols"], **options,
        )["out"]
        assert torch.equal(actual, expected), "Residual output must match BF16 addition exactly"
    else:
        # A constant valid row detects padding included in the mean or variance.
        x.reshape(rows, cols)[case["row_offset"]].fill_(2.5)
        weight = (.5 + torch.rand(cols, generator=rng)).bfloat16()
        bias = (torch.randn(cols, generator=rng) * .2).bfloat16()
        inputs = {"x": x, "weight": weight, "bias": bias, "output_seed": seed}
        before = {name: value.clone() for name, value in inputs.items()}
        mask[row_slice] = True
        golden = reference.layer_norm(x, weight, bias, eps=1e-6).reshape(rows, cols)
        expected.reshape(rows, cols)[mask] = golden[mask]
        actual = tile.layer_norm(x, weight, bias, eps=1e-6, **options)["out"]
        torch.testing.assert_close(actual, expected, atol=.01, rtol=.01)

    assert actual.shape == x.shape and actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert torch.equal(actual.reshape(rows, cols)[~mask], seed.reshape(rows, cols)[~mask])
    for name, value in inputs.items():
        assert torch.equal(value, before[name]), f"Input {name} was modified"
    assert actual.data_ptr() != seed.data_ptr(), "Returning the seed would alias caller storage"
    report = {
        "case": case_name,
        "max_abs_error": float((actual.float() - expected.float()).abs().max()),
        "active_elements": int(mask.sum()),
        "preserved_elements": int((~mask).sum()),
        "inputs_unchanged": True,
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize("case_name", CASES)
def test_hw_common_boundary(case_name, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), case_name, str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-18000:]
    program = tmp_path / "program"
    evidence = json.loads((program / "program.json").read_text())
    assert evidence["backend"] == "hw_pto_cpp_sim"
    assert evidence["platform"] == "a2a3sim"
    assert evidence["native_compilation"] == evidence["native_simulation"] == "completed"
    assert evidence["generated_pto_count"] > 0
    assert list((program / "native-0").rglob("*.pto"))
    assert evidence["binary_sha256"]
    for name, expected_sha in evidence["binary_sha256"].items():
        binary = (program / "native-0" / name).read_bytes()
        assert binary.startswith(b"\x7fELF")
        assert hashlib.sha256(binary).hexdigest() == expected_sha
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["inputs_unchanged"]
    assert metrics["active_elements"] > 0 and metrics["preserved_elements"] > 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in CASES:
        _run_native_case(sys.argv[1], Path(sys.argv[2]))
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
