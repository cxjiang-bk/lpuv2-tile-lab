"""Check K-06's FP8 semantics on the native PTO C++ simulator, using exact BF16 value carriers."""

import argparse
import json
from pathlib import Path
import shutil
from uuid import uuid4

import torch

from kernels.pypto import fp8
from pymodel.common.case import Case, OutputCheck
from pymodel.isa.hw_pypto import TensorProgram, environment_details
from pymodel.ops.gdn.kernel import pad_fp8_weight
from pymodel.ref import fp8 as reference
from pymodel.task_gen.validate import run_case


def validate(kind, root):
    rng = torch.Generator().manual_seed(43)
    rows, width, output_width = 4, 129, 129
    x = torch.randn((rows, width), generator=rng).bfloat16()
    x[0] = 0
    x[1, :10] = torch.tensor([448., -448., 432., -432., 400., -400., 0., -0., 2. ** -10, -2. ** -10])
    # All positive/negative E4M3FN adjacent midpoints are BF16-representable.
    values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (values[:-1] + values[1:]) * .5
    x[2, :126], x[3, :126] = midpoints.bfloat16(), -midpoints.bfloat16()
    x[2, 126:128], x[3, 126:128] = torch.tensor([448., -0.]), torch.tensor([-448., -0.])
    inputs = {"x": x}
    padded_k = (width + 127) // 128 * 128
    program_dir = Path(root) / "programs" / uuid4().hex

    if kind == "quantize":
        outputs = {name: OutputCheck(atol=0, rtol=0) for name in ("codes", "decoded", "scales")}
        outputs["codes"] = OutputCheck()
        kernel = fp8.activation_quantize_bf16(rows, width)

        def golden(tensors, _):
            q = reference.activation_quantize(tensors["x"])
            codes = torch.zeros((rows, padded_k), dtype=torch.int32)
            codes[:, :width] = q["payload"].view(torch.uint8).int()
            decoded = codes.to(torch.uint8).view(torch.float8_e4m3fn).float().bfloat16()
            return {"codes": codes, "decoded": decoded, "scales": q["scale"]}

        def actual(tensors, _):
            decoded = torch.full((rows, padded_k), float("nan"), dtype=torch.bfloat16)
            codes = torch.full((rows, padded_k), -99, dtype=torch.int32)
            scales = torch.full((rows, padded_k // 128), float("nan"))
            program.execute(tensors["x"], decoded, scales, codes)
            program.save(program_dir)
            return {"codes": codes, "decoded": decoded, "scales": scales}
    else:
        # Different scales in both N/K block directions expose axis mistakes.
        inputs["weight"] = (torch.randn((output_width, width), generator=rng) * 3).to(torch.float8_e4m3fn)
        inputs["scale"] = torch.tensor([[.02, .03], [.04, .05]], dtype=torch.bfloat16)
        outputs = {"out": OutputCheck(atol=.01, rtol=.01)}
        kernel = fp8.fp8_linear_bf16_sim(rows, width, output_width)

        def golden(tensors, _):
            return {"out": reference.linear(tensors["x"], tensors["weight"], tensors["scale"])}

        def actual(tensors, _):
            out = torch.full((rows, output_width), float("nan"), dtype=torch.bfloat16)
            weight_codes = pad_fp8_weight(tensors["weight"]).view(torch.uint8).int()
            program.execute(tensors["x"], weight_codes, tensors["scale"], out)
            program.save(program_dir)
            return {"out": out}

    program = TensorProgram.compile(kernel)
    program.details.update(fp8_execution="bf16_carrier", native_fp8_cube_validated=False,
                           validation_scope="K-06 numerical primitive; not a complete GDN operator")
    case = Case(f"a-native-fp8-{kind}", 43, "pto-functional-v0", outputs,
                {"rows": rows, "width": width, "output_width": output_width,
                 "fp8_execution": "bf16_carrier"}, op_id="OP-06", kernel_id="K-06")
    result = run_case(case, inputs=inputs, reference=golden, implementation=actual,
                      backend="hw_pto_cpp_sim", artifact_root=root, backend_details=environment_details())
    shutil.move(program_dir, result.run_dir / "program")
    path = result.run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["backend_details"].update(program.details)
    manifest["files"].extend(str(p.relative_to(result.run_dir)) for p in (result.run_dir / "program").rglob("*")
                             if p.is_file())
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"FP8 {kind}: {'PASS' if result.passed else 'FAIL'} {result.run_dir}", flush=True)
    print(json.dumps(result.report), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("quantize", "linear"), required=True)
    parser.add_argument("--artifact-root", default="artifacts/a-native-fp8")
    args = parser.parse_args()
    torch.set_num_threads(1)
    return 0 if validate(args.case, args.artifact_root).passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
