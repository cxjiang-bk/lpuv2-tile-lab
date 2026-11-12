"""Stage 3 step 4: public Tile numerical functions against independent Goldens."""

import json
from pathlib import Path

import pytest
import torch

from kernels.pto.common.activations import sigmoid, silu
from kernels.pto.common.linear import bf16_linear, fp8_linear
from kernels.pto.common.quantization import dynamic_quantize
from kernels.pto.common.softmax import softmax
from pymodel.common.case import Case, OutputCheck
from pymodel.isa.torch_codegen import torch_codegen
from pymodel.ref.linear import bf16_linear as bf16_linear_ref
from pymodel.ref.linear import fp8_linear as fp8_linear_ref
from pymodel.ref.quantization import dynamic_quantize as dynamic_quantize_ref
from pymodel.ref.normalization import softmax as softmax_ref
from pymodel.task_gen.validate import run_case


def _compile(program, path):
    code = torch_codegen(program, check_shapes=True)
    path.write_text(code)
    path.with_name(path.stem.replace("_torch", "_ir") + ".py").write_text(program.as_python())
    namespace = {}
    exec(compile(code, str(path), "exec"), namespace)
    return namespace, code


def _backend_details(root, ir_path, code_path, tile_shape):
    upstream = json.loads((root / "data/upstream.lock.json").read_text())
    return {
        "pypto_commit": upstream["pypto"]["commit"],
        "device": "cpu",
        "tile_shape": tile_shape,
        "ir": str(ir_path),
        "generated_code": str(code_path),
        "adapter": "pymodel.isa.torch_codegen",
    }


@pytest.mark.parametrize("m,n,k", [(16, 16, 64), (17, 19, 130)], ids=["full_tiles", "tails"])
def test_bf16_linear_matches_golden_with_k_and_output_tails(m, n, k, tmp_path):
    case = Case(
        case_id="bf16_linear_tails",
        seed=51,
        numerical_profile="pto-functional-v0",
        outputs={"out": OutputCheck(atol=1e-2, rtol=1e-2)},
    )
    generator = case.generator()
    x = torch.randn(m, k, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(n, k, generator=generator, dtype=torch.bfloat16)
    sample_out = torch.empty(m, n, dtype=torch.bfloat16)
    program = bf16_linear.specialize(x, weight, sample_out)
    ir_path = tmp_path / "bf16_linear_ir.py"
    code_path = tmp_path / "bf16_linear_torch.py"
    namespace, _ = _compile(program, code_path)
    ir_path.write_text(program.as_python())

    def reference(inputs, parameters):
        return {"out": bf16_linear_ref(inputs["x"], inputs["weight"])}

    def implementation(inputs, parameters):
        out = torch.full((m, n), torch.nan, dtype=torch.bfloat16)
        namespace["bf16_linear"](inputs["x"], inputs["weight"], out)
        return {"out": out}

    root = Path(__file__).resolve().parents[2]
    result = run_case(
        case,
        inputs={"x": x, "weight": weight},
        reference=reference,
        implementation=implementation,
        backend="pypto_torch_codegen_adapted",
        artifact_root=tmp_path / "runs",
        backend_details=_backend_details(root, ir_path, code_path, [16, 16, 64]),
    )
    assert result.passed, result.report


def test_dynamic_quantize_matches_golden_group_tail(tmp_path):
    case = Case(case_id="quantization_group_tail", seed=55, numerical_profile="pto-functional-v0",
                outputs={"payload": OutputCheck(atol=0, rtol=0),
                         "scale": OutputCheck(atol=0, rtol=0)})
    x = torch.randn(2, 257, generator=torch.Generator().manual_seed(55), dtype=torch.bfloat16)
    x[0, :128] *= 0.25
    x[0, 128:256] *= 4
    x[1, :128] = 0
    x[1, 256] = 0
    actual_payload = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    actual_scale = torch.empty(2, 3, dtype=torch.float32)
    program = dynamic_quantize.specialize(x, actual_payload, actual_scale)
    namespace, _ = _compile(program, tmp_path / "quantization_torch.py")

    def reference(inputs, parameters):
        payload, scale = dynamic_quantize_ref(inputs["x"])
        return {"payload": payload, "scale": scale}

    def implementation(inputs, parameters):
        payload = torch.full_like(inputs["x"], torch.nan, dtype=torch.float8_e4m3fn)
        scale = torch.full((2, 3), torch.nan)
        namespace["dynamic_quantize"](inputs["x"], payload, scale)
        return {"payload": payload, "scale": scale}

    root = Path(__file__).resolve().parents[2]
    result = run_case(case, inputs={"x": x}, reference=reference, implementation=implementation,
                      backend="pypto_torch_codegen_adapted", artifact_root=tmp_path / "runs",
                      backend_details=_backend_details(root, tmp_path / "quantization_ir.py",
                                                       tmp_path / "quantization_torch.py", [1, 128]))
    assert result.passed, result.report
    actual = torch.load(result.run_dir / "actual.pt", weights_only=True)
    expected = torch.load(result.run_dir / "reference.pt", weights_only=True)
    assert torch.equal(actual["payload"].view(torch.uint8), expected["payload"].view(torch.uint8))


@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32])
def test_fp8_linear_matches_golden_grouped_accumulation(scale_dtype, tmp_path):
    case = Case(case_id="fp8_linear_groups", seed=54, numerical_profile="pto-functional-v0",
                outputs={"out": OutputCheck(atol=1e-2, rtol=1e-2)})
    generator = torch.Generator().manual_seed(54)
    x = torch.randn(2, 129, generator=generator, dtype=torch.bfloat16)
    weight = torch.randint(-3, 4, (129, 129), generator=generator).to(torch.float8_e4m3fn)
    scale = ((torch.arange(2 * 2, dtype=torch.float32) + 1).reshape(2, 2) / 32).to(scale_dtype)
    actual = torch.empty(2, 129, dtype=torch.bfloat16)
    program = fp8_linear.specialize(x, weight, scale, actual, scale_bf16=(scale_dtype == torch.bfloat16))
    namespace, _ = _compile(program, tmp_path / "fp8_linear_torch.py")

    def reference(inputs, parameters):
        return {"out": fp8_linear_ref(inputs["x"], inputs["weight"], inputs["scale"])}

    def implementation(inputs, parameters):
        out = torch.full((2, 129), torch.nan, dtype=torch.bfloat16)
        namespace["fp8_linear"](inputs["x"], inputs["weight"], inputs["scale"], out)
        return {"out": out}

    root = Path(__file__).resolve().parents[2]
    result = run_case(case, inputs={"x": x, "weight": weight, "scale": scale},
                      reference=reference, implementation=implementation,
                      backend="pypto_torch_codegen_adapted", artifact_root=tmp_path / "runs",
                      backend_details=_backend_details(root, tmp_path / "fp8_linear_ir.py",
                                                       tmp_path / "fp8_linear_torch.py", [1, 16, 128]))
    assert result.passed, result.report


def test_quantization_rne_bytes_and_linear_fp32_cancellation(tmp_path):
    x = torch.tensor([[448, 1.0625, 1.1875, 1.3125, -1.0625, 2**-10, 3 * 2**-10, 0]],
                     dtype=torch.bfloat16)
    payload = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(1, 1)
    namespace, _ = _compile(dynamic_quantize.specialize(x, payload, scale), tmp_path / "rounding_torch.py")
    namespace["dynamic_quantize"](x, payload, scale)
    expected_bytes = torch.tensor([[0x7E, 0x38, 0x3A, 0x3A, 0xB8, 0, 2, 0]], dtype=torch.uint8)
    assert torch.equal(payload.view(torch.uint8), expected_bytes)
    torch.testing.assert_close(scale, torch.ones(1, 1), atol=0, rtol=0)

    x = torch.zeros(1, 129, dtype=torch.bfloat16)
    x[0, 0], x[0, 1], x[0, 64] = 256, 1, -256
    weight = torch.ones(1, 129, dtype=torch.bfloat16)
    out = torch.empty(1, 1, dtype=torch.bfloat16)
    namespace, _ = _compile(bf16_linear.specialize(x, weight, out), tmp_path / "cancellation_torch.py")
    namespace["bf16_linear"](x, weight, out)
    torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_softmax_matches_golden_across_column_tail(dtype, tmp_path):
    case = Case(
        case_id="softmax_column_tail",
        seed=52,
        numerical_profile="pto-functional-v0",
        outputs={"out": OutputCheck(atol=1e-5, rtol=1e-4)},
    )
    x = torch.randn(3, 129, generator=case.generator(), dtype=dtype) * 4 - 1000
    x[0, :128] = -torch.inf
    x[1, 128] = -torch.inf
    sample_out = torch.empty_like(x, dtype=torch.float32)
    program = softmax.specialize(x, sample_out, input_bf16=(dtype == torch.bfloat16))
    ir_path = tmp_path / "softmax_ir.py"
    code_path = tmp_path / "softmax_torch.py"
    namespace, _ = _compile(program, code_path)
    ir_path.write_text(program.as_python())

    def reference(inputs, parameters):
        return {"out": softmax_ref(inputs["x"])}

    def implementation(inputs, parameters):
        out = torch.full_like(inputs["x"], torch.nan, dtype=torch.float32)
        namespace["softmax"](inputs["x"], out)
        return {"out": out}

    root = Path(__file__).resolve().parents[2]
    result = run_case(
        case,
        inputs={"x": x},
        reference=reference,
        implementation=implementation,
        backend="pypto_torch_codegen_adapted",
        artifact_root=tmp_path / "runs",
        backend_details=_backend_details(root, ir_path, code_path, [1, 128]),
    )
    assert result.passed, result.report


@pytest.mark.parametrize("name", ["sigmoid", "silu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_activation_matches_torch(name, dtype, tmp_path):
    case = Case(
        case_id=f"{name}_tile",
        seed=53,
        numerical_profile="pto-functional-v0",
        outputs={"out": OutputCheck(atol=1e-2, rtol=1e-2) if dtype == torch.bfloat16
                 else OutputCheck(atol=1e-5, rtol=1e-4)},
    )
    x = torch.randn(2, 129, generator=case.generator(), dtype=dtype)
    x[0, :3] = torch.tensor([-100, 0, 100], dtype=dtype)
    sample_out = torch.empty_like(x)
    kernel = sigmoid if name == "sigmoid" else silu
    program = kernel.specialize(x, sample_out, input_bf16=(dtype == torch.bfloat16),
                                output_bf16=(dtype == torch.bfloat16))
    ir_path = tmp_path / f"{name}_ir.py"
    code_path = tmp_path / f"{name}_torch.py"
    namespace, _ = _compile(program, code_path)
    ir_path.write_text(program.as_python())

    def reference(inputs, parameters):
        fn = torch.sigmoid if name == "sigmoid" else torch.nn.functional.silu
        return {"out": fn(inputs["x"])}

    def implementation(inputs, parameters):
        out = torch.full_like(inputs["x"], torch.nan)
        namespace[name](inputs["x"], out)
        return {"out": out}

    root = Path(__file__).resolve().parents[2]
    result = run_case(
        case,
        inputs={"x": x},
        reference=reference,
        implementation=implementation,
        backend="pypto_torch_codegen_adapted",
        artifact_root=tmp_path / "runs",
        backend_details=_backend_details(root, ir_path, code_path, [1, 128]),
    )
    assert result.passed, result.report
