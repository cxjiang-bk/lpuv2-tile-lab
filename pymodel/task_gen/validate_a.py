"""Validate complete A operators on the compiled HW PTO C++ simulator.

Run after: source scripts/hw-env.sh
python -m pymodel.task_gen.validate_a --artifact-root artifacts/a-native
Each operator runs in its own process and saves PTO, C++ and native binary evidence.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import torch
import torch.nn.functional as F

from pymodel.common.case import Case, OutputCheck, load_case
from pymodel.isa.hw_pypto import environment_details
from pymodel.ref import common as ref_common
from pymodel.ref import fp8 as ref_fp8
from pymodel.ref import gdn as ref_gdn
from pymodel.ref import multimodal as ref_multimodal
from pymodel.ref import vision as ref_vision
from pymodel.task_gen.validate import run_case

OP_IDS = ("03", "04", "06", "08", "09", "10", "11", "12", "13")


def make_inputs(op, *, real_dimensions=False, seed=43, gdn_sequence=2, gdn_batch=1):
    rng = torch.Generator().manual_seed(seed)

    def bf(*shape, scale=.1):
        return (torch.randn(shape, generator=rng) * scale).bfloat16()

    width = 1152 if real_dimensions else 32
    if op == "03":
        return {"x": bf(3, width), "weight": bf(width), "bias": bf(width)}, {"eps": 1e-6}
    if op == "04":
        return {"x": bf(3, 2048 if real_dimensions else 129),
                "residual": bf(3, 2048 if real_dimensions else 129)}, {}
    if op == "08":
        patch_shape = (3, 2, 16, 16) if real_dimensions else (2, 2, 2, 2)
        image_shape = (1, 3, 2, 32, 32) if real_dimensions else (2, 2, 4, 4, 8)
        # FP32 像素直接传入 native，验证权重 dtype 转换确实发生在 kernel 内。
        pixels = torch.randn(image_shape, generator=rng) * .1
        return {"pixels": pixels, "weight": bf(width, *patch_shape), "bias": bf(width)}, {}
    if op == "10":
        intermediate = 4304 if real_dimensions else 64
        return {"x": bf(3, width), "fc1_weight": bf(intermediate, width), "fc1_bias": bf(intermediate),
                "fc2_weight": bf(width, intermediate), "fc2_bias": bf(width)}, {}
    if op == "12":
        intermediate = 4608 if real_dimensions else 64
        output_width = 2048 if real_dimensions else 32
        return {"x": bf(8, width), "norm_weight": bf(width), "norm_bias": bf(width),
                "fc1_weight": bf(intermediate, 4 * width), "fc1_bias": bf(intermediate),
                "fc2_weight": bf(output_width, intermediate), "fc2_bias": bf(output_width)}, {"spatial_merge_size": 2}
    if op == "09":
        heads = 16 if real_dimensions else 2
        head_dim = width // heads
        angles = torch.randn((5, head_dim), generator=rng)
        return {"x": bf(5, width), "qkv_weight": bf(3 * width, width), "qkv_bias": bf(3 * width),
                "proj_weight": bf(width, width), "proj_bias": bf(width), "cos": angles.cos(), "sin": angles.sin(),
                "cu_seqlens": torch.tensor([0, 2, 5], dtype=torch.int64)}, {"num_heads": heads}
    if op == "11":
        grid = torch.tensor([[1, 4, 4], [2, 2, 2]], dtype=torch.int64)
        return {"grid_thw": grid, "x": bf(24, width), "position_table": bf(2304, width)}, {}
    if op == "13":
        return {"text_embedding": bf(2, 4, 2048 if real_dimensions else 32),
                "visual_features": bf(3, 2048 if real_dimensions else 32),
                "placeholder_rows": torch.tensor([6, 1, 4], dtype=torch.int64)}, {}
    if op == "06":
        batch, sequence, hidden = gdn_batch, gdn_sequence, 128
        qh, vh, kd, vd = 1, 2, 32, 32
        channels, value_width = 2 * qh * kd + vh * vd, vh * vd
        weights = {}
        for name, shape in {"qkv": (channels, hidden), "z": (value_width, hidden),
                            "out": (hidden, value_width)}.items():
            weights[name + "_weight"] = (torch.randn(shape, generator=rng) * 3).to(torch.float8_e4m3fn)
            weights[name + "_scale_inv"] = torch.full(((shape[0] + 127) // 128,
                                                       (shape[1] + 127) // 128), .02, dtype=torch.bfloat16)
        weights["a_weight"], weights["b_weight"] = bf(vh, hidden), bf(vh, hidden)
        inputs = {"x": bf(batch, sequence, hidden), **weights, "conv_weight": bf(channels, 4),
                  "A_log": bf(vh), "dt_bias": bf(vh), "norm_weight": bf(vd),
                  "initial_state": torch.randn((batch, vh, kd, vd), generator=rng) * .01,
                  "conv_history": bf(batch, 3, channels)}
        if batch > 1:
            inputs["attention_mask"] = torch.ones((batch, sequence), dtype=torch.bool)
            inputs["attention_mask"][-1, -1] = False
        return inputs, {"qk_heads": qh, "value_heads": vh, "key_head_dim": kd, "value_head_dim": vd}
    raise ValueError(op)


def reference(op, inputs, parameters):
    if op == "03":
        return {"out": ref_common.layer_norm(**inputs, **parameters)}
    if op == "04":
        return {"out": inputs["x"] + inputs["residual"]}
    if op == "08":
        stride = parameters.get("stride", inputs["weight"].shape[2:])
        grid = [(d - k) // s + 1 for d, k, s in zip(inputs["pixels"].shape[2:],
                                                    inputs["weight"].shape[2:], stride)]
        return {"out": ref_vision.vision_patch_embedding(**inputs, **parameters),
                "grid_thw": torch.tensor([grid] * inputs["pixels"].shape[0], dtype=torch.int64)}
    if op == "09":
        return {"out": ref_vision.vision_attention(**inputs, **parameters)}
    if op == "10":
        return {"out": ref_vision.dense_ffn(**inputs)}
    if op == "12":
        return {"out": ref_vision.spatial_merger(**inputs, **parameters)}
    if op == "13":
        return {"out": ref_multimodal.visual_feature_scatter(**inputs)}
    if op == "11":
        indices, weights = ref_vision.vision_position_interpolation(inputs["grid_thw"])
        return {"out": ref_vision.vision_position_embedding(inputs["x"], inputs["position_table"], indices, weights),
                "interp_indices": indices, "interp_weights": weights}
    if op == "06":
        projections = ref_gdn.GDNProjections(
            qkv=lambda x: ref_fp8.linear(x.reshape(-1, x.shape[-1]), inputs["qkv_weight"],
                                        inputs["qkv_scale_inv"]).reshape(*x.shape[:2], -1),
            z=lambda x: ref_fp8.linear(x.reshape(-1, x.shape[-1]), inputs["z_weight"],
                                      inputs["z_scale_inv"]).reshape(*x.shape[:2], -1),
            a=lambda x: F.linear(x, inputs["a_weight"]),
            b=lambda x: F.linear(x, inputs["b_weight"]),
            out=lambda x: ref_fp8.linear(x.reshape(-1, x.shape[-1]), inputs["out_weight"],
                                        inputs["out_scale_inv"]).reshape(*x.shape[:2], -1))
        args = {k: inputs[k] for k in ("conv_weight", "A_log", "dt_bias", "norm_weight", "initial_state", "conv_history")}
        if "attention_mask" in inputs:
            args["attention_mask"] = inputs["attention_mask"]
        return ref_gdn.gdn_reference(inputs["x"], projections, **args, **parameters)
    raise ValueError(op)


def implementation(op, inputs, parameters, artifact_dir):
    if op in ("03", "04"):
        from pymodel.ops.common import tile as common
    elif op == "06":
        from pymodel.ops.gdn.kernel import gdn
    elif op == "11":
        from pymodel.ops.vision.position import position_embedding
    else:
        from pymodel.ops.vision import kernels as vision

    kwargs = dict(parameters, artifact_dir=artifact_dir)
    if op == "03":
        return common.layer_norm(**inputs, **kwargs)
    if op == "04":
        return common.residual_add(**inputs, **kwargs)
    if op == "08":
        return vision.patch_embedding(**inputs, **kwargs)
    if op == "09":
        return vision.vision_attention(**inputs, **kwargs)
    if op == "10":
        return vision.dense_ffn(**inputs, **kwargs)
    if op == "11":
        return position_embedding(**inputs, **kwargs)
    if op == "12":
        return vision.spatial_merger(**inputs, **kwargs)
    if op == "13":
        return vision.visual_feature_scatter(**inputs, **kwargs)
    if op == "06":
        weights = {k: v for k, v in inputs.items() if k.endswith("_weight") or k.endswith("_scale_inv")}
        weights.pop("conv_weight")
        weights.pop("norm_weight")
        args = {k: inputs[k] for k in ("conv_weight", "A_log", "dt_bias", "norm_weight", "initial_state", "conv_history")}
        if "attention_mask" in inputs:
            args["attention_mask"] = inputs["attention_mask"]
        return gdn(inputs["x"], weights, **args, **kwargs)
    raise ValueError(op)


def validate(op, root, *, real_dimensions=False, fp8_execution="bf16_carrier",
             gdn_path="recurrent", chunk_size=64, gdn_sequence=2, gdn_batch=1):
    if op == "06":
        if real_dimensions:
            raise ValueError("Model-sized GDN is not enabled; use the documented small presets")
    if not real_dimensions:
        case_path = Path(__file__).resolve().parents[2] / "cases/kernels" / f"a-native-op{op}-small.json"
        case = load_case(case_path)
        inputs, _ = make_inputs(op, seed=case.seed, gdn_sequence=gdn_sequence, gdn_batch=gdn_batch)
        actual_shapes = {name: list(t.shape) for name, t in inputs.items()}
        variant = op == "06" and (gdn_path != "recurrent" or gdn_sequence != 2 or gdn_batch != 1)
        if not variant and actual_shapes != case.parameters["input_shapes"]:
            raise ValueError("Case shapes must match the selected input generator preset")
        parameters = {name: value for name, value in case.parameters.items()
                      if name not in ("input_shapes", "validation_shape")}
        if op == "06":
            parameters.update(path=gdn_path, chunk_size=chunk_size)
        if variant:
            case = Case(f"a-native-op06-{gdn_path}-b{gdn_batch}-s{gdn_sequence}-c{chunk_size}",
                        case.seed, case.numerical_profile, case.outputs,
                        dict(parameters, input_shapes=actual_shapes, validation_shape="small"),
                        op_id="OP-06", kernel_id="K-06")
    else:
        inputs, parameters = make_inputs(op, real_dimensions=True)
        outputs = {"out": OutputCheck(atol=.01, rtol=.01)}
        if op == "08":
            outputs["grid_thw"] = OutputCheck(logical_indices=True)
        if op == "11":
            outputs.update(interp_indices=OutputCheck(logical_indices=True),
                           interp_weights=OutputCheck(atol=1e-5, rtol=1e-4))
        case = Case(f"a-native-op{op}-model-dims", 43, "pto-functional-v0", outputs,
                    dict(parameters, validation_shape="model-dims"), op_id=f"OP-{op}", kernel_id=f"K-{op}")
    program_dir = Path(root) / "programs" / f"op{op}_{uuid4().hex[:12]}"

    def golden(tensors, _):
        return reference(op, tensors, parameters)

    def actual(tensors, _):
        kernel_parameters = dict(parameters, fp8_execution=fp8_execution) if op == "06" else parameters
        if op == "06":
            state_before = tensors["initial_state"].clone()
            history_before = tensors["conv_history"].clone()
        outputs = implementation(op, tensors, kernel_parameters, program_dir)
        if op == "06" and (not torch.equal(tensors["initial_state"], state_before)
                           or not torch.equal(tensors["conv_history"], history_before)):
            raise RuntimeError("GDN modified its supplied State or Conv history")
        return outputs

    result = run_case(case, inputs=inputs, reference=golden, implementation=actual, backend="hw_pto_cpp_sim",
                      artifact_root=root, backend_details=environment_details())
    shutil.move(program_dir, result.run_dir / "program")
    manifest_path = result.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    details = json.loads((result.run_dir / "program/program.json").read_text())
    manifest["backend_details"].update(details)
    manifest["files"].extend(str(p.relative_to(result.run_dir)) for p in (result.run_dir / "program").rglob("*")
                             if p.is_file())
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"OP-{op} {'PASS' if result.passed else 'FAIL'} {result.run_dir}", flush=True)
    print(json.dumps(result.report, ensure_ascii=False), flush=True)
    return result


def validate_gdn_decode(root, *, prefill_path="recurrent", chunk_size=64):
    """Continuous Decode, or chunk Prefill -> recurrent Decode, against golden."""
    handoff = prefill_path == "chunk"
    inputs, parameters = make_inputs("06", gdn_sequence=5 if handoff else 2)
    parameters.update(path=prefill_path, chunk_size=chunk_size)
    checks = load_case(Path(__file__).resolve().parents[2] / "cases/kernels/a-native-op06-small.json").outputs
    state_check = "state_after_prefill" if handoff else "state_after_first_token"
    history_check = "history_after_prefill" if handoff else "history_after_first_token"
    checks = dict(checks, **{state_check: checks["state_out"], history_check: checks["conv_history_out"]})
    case_id = "a-native-op06-chunk-prefill-decode" if handoff else "a-native-op06-decode-continuity"
    scenario = "chunk_prefill_recurrent_decode" if handoff else "continuous_decode"
    case = Case(case_id, 43, "pto-functional-v0", checks,
                dict(parameters, scenario=scenario, fp8_execution="bf16_carrier"),
                op_id="OP-06", kernel_id="K-06")
    program_dir = Path(root) / "programs" / f"op06_decode_{uuid4().hex[:12]}"

    def golden(tensors, _):
        expected = reference("06", tensors, parameters)
        prefix = 4 if handoff else 1
        first = reference("06", dict(tensors, x=tensors["x"][:, :prefix].contiguous()), parameters)
        return dict(expected, **{state_check: first["state_out"], history_check: first["conv_history_out"]})

    def actual(tensors, _):
        state, history = tensors["initial_state"], tensors["conv_history"]
        token_outputs, states, histories, invocations = [], [], [], []
        steps = [(0, 4, "chunk"), (4, 5, "recurrent")] if handoff else [(0, 1, "recurrent"), (1, 2, "recurrent")]
        for token, (begin, end, path) in enumerate(steps):
            step_inputs = dict(tensors, x=tensors["x"][:, begin:end].contiguous(),
                               initial_state=state, conv_history=history)
            before_state, before_history = state.clone(), history.clone()
            step_dir = program_dir / f"token-{token:03d}"
            outputs = implementation("06", step_inputs, dict(parameters, path=path, fp8_execution="bf16_carrier"), step_dir)
            if not torch.equal(state, before_state) or not torch.equal(history, before_history):
                raise RuntimeError("Decode modified its supplied State or Conv history")
            state, history = outputs["state_out"], outputs["conv_history_out"]
            token_outputs.append(outputs["out"])
            states.append(state)
            histories.append(history)
            invocations.append(str(step_dir.relative_to(program_dir) / "program.json"))
        details = json.loads((program_dir / invocations[0]).read_text())
        details.update(execution_scenario=scenario, invocations=invocations)
        (program_dir / "program.json").write_text(json.dumps(details, indent=2) + "\n")
        return {"out": torch.cat(token_outputs, dim=1), "state_out": state, "conv_history_out": history,
                state_check: states[0], history_check: histories[0]}

    result = run_case(case, inputs=inputs, reference=golden, implementation=actual, backend="hw_pto_cpp_sim",
                      artifact_root=root, backend_details=environment_details())
    shutil.move(program_dir, result.run_dir / "program")
    path = result.run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["backend_details"].update(json.loads((result.run_dir / "program/program.json").read_text()))
    manifest["files"].extend(str(p.relative_to(result.run_dir)) for p in (result.run_dir / "program").rglob("*")
                             if p.is_file())
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"OP-06 {scenario} {'PASS' if result.passed else 'FAIL'} {result.run_dir}", flush=True)
    print(json.dumps(result.report), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=OP_IDS)
    parser.add_argument("--real-dimensions", action="store_true")
    parser.add_argument("--gdn-decode-continuity", action="store_true")
    parser.add_argument("--gdn-path", choices=("recurrent", "chunk"), default="recurrent")
    parser.add_argument("--gdn-chunk-size", type=int, default=64)
    parser.add_argument("--gdn-sequence", type=int, default=2)
    parser.add_argument("--gdn-batch", type=int, default=1)
    parser.add_argument("--fp8-execution", choices=("bf16_carrier",), default="bf16_carrier",
                        help="BF16 carries exact FP8 values in the compiled PTO C++ simulator")
    parser.add_argument("--artifact-root", default="artifacts/a-native")
    args = parser.parse_args()
    if min(args.gdn_chunk_size, args.gdn_sequence, args.gdn_batch) < 1:
        parser.error("GDN chunk size, sequence and batch must be positive")
    if args.case != "06" and (args.gdn_path != "recurrent" or args.gdn_chunk_size != 64
                             or args.gdn_sequence != 2 or args.gdn_batch != 1):
        parser.error("GDN path/shape options require --case 06")
    if args.gdn_decode_continuity:
        if (args.case != "06" or args.real_dimensions or args.fp8_execution != "bf16_carrier"
                or args.gdn_sequence != 2 or args.gdn_batch != 1):
            parser.error("Decode continuity uses --case 06, fixed small presets and bf16_carrier")
        torch.set_num_threads(1)
        return 0 if validate_gdn_decode(args.artifact_root, prefill_path=args.gdn_path,
                                       chunk_size=args.gdn_chunk_size).passed else 1
    if not args.case:
        root = Path(args.artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        summary = {"backend": "hw_pto_cpp_sim", "operators": {}, "passed": True}
        for op in OP_IDS:
            cmd = [sys.executable, "-m", "pymodel.task_gen.validate_a", "--case", op,
                   "--artifact-root", str(root), "--fp8-execution", args.fp8_execution]
            if args.real_dimensions:
                cmd.append("--real-dimensions")
            log_path = root / f"op{op}.log"
            with log_path.open("w") as log:
                try:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=600)
                    status = {"returncode": result.returncode, "log": str(log_path)}
                except subprocess.TimeoutExpired:
                    status = {"returncode": None, "error": "timeout", "log": str(log_path)}
            summary["operators"]["OP-" + op] = status
            summary["passed"] = summary["passed"] and status["returncode"] == 0
            print(f"OP-{op} {'PASS' if status['returncode'] == 0 else 'FAIL'}: {log_path}", flush=True)
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return 0 if summary["passed"] else 1
    torch.set_num_threads(1)
    return 0 if validate(args.case, args.artifact_root, real_dimensions=args.real_dimensions,
                         fp8_execution=args.fp8_execution, gdn_path=args.gdn_path,
                         chunk_size=args.gdn_chunk_size, gdn_sequence=args.gdn_sequence,
                         gdn_batch=args.gdn_batch).passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
