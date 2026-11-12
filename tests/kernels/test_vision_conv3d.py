"""Raw pixel Conv3D and exact im2col on compiled PTO C++ simulation."""

import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    shutil.which("ptoas") is None or importlib.util.find_spec("simpler") is None,
    reason="Source scripts/hw-env.sh for native PTO C++ simulation",
)


@pytest.mark.parametrize("preset", ["overlap", "model", "tail"])
def test_native_pixel_conv3d(preset, tmp_path):
    script = tmp_path / "conv3d_case.py"
    script.write_text(textwrap.dedent('''\
        import json
        import sys
        from pathlib import Path
        import torch
        from pymodel.common.case import Case, OutputCheck
        from pymodel.isa.hw_pypto import environment_details
        from pymodel.ops.vision.kernels import patch_embedding
        from pymodel.task_gen.validate import run_case

        torch.set_num_threads(1)
        preset, root = sys.argv[1], Path(sys.argv[2])
        if preset == "overlap":
            shape, kernel, stride, hidden, merge = (2,2,4,6,8), (2,3,2), (1,2,3), 32, 1
            dtype = torch.bfloat16
        elif preset == "tail":
            shape, kernel, stride, hidden, merge = (1,2,2,4,6), (2,2,3), (2,2,3), 130, 2
            dtype = torch.bfloat16
        else:
            shape, kernel, stride, hidden, merge = (1,3,2,32,32), (2,16,16), (2,16,16), 1152, 2
            dtype = torch.float32
        rng = torch.Generator().manual_seed(53)
        inputs = {"pixels": (torch.randn(shape, generator=rng)*.1).to(dtype),
                  "weight": (torch.randn((hidden,shape[1],*kernel), generator=rng)*.1).bfloat16(),
                  "bias": (torch.randn(hidden, generator=rng)*.1).bfloat16()}
        originals = {k: v.clone() for k,v in inputs.items()}
        case = Case("a-native-op08-"+preset, 53, "pto-functional-v0",
                    {"out": OutputCheck(atol=.01,rtol=.01), "grid_thw": OutputCheck(logical_indices=True)},
                    {"input_shapes": {k:list(v.shape) for k,v in inputs.items()},
                     "stride":list(stride), "spatial_merge_size":merge}, op_id="OP-08", kernel_id="K-08")
        def golden(a, _):
            # 独立完整 F.conv3d；测试用显式坐标遍历构造期望 Token 顺序。
            raster = torch.nn.functional.conv3d(a["pixels"].bfloat16(), a["weight"], a["bias"], stride=stride)
            b,d,t,h,w = raster.shape
            rows = [raster[ib,:,it,ih+dh,iw+dw] for ib in range(b) for it in range(t)
                    for ih in range(0,h,merge) for iw in range(0,w,merge)
                    for dh in range(merge) for dw in range(merge)]
            return {"out":torch.stack(rows), "grid_thw":torch.tensor([[t,h,w]]*b,dtype=torch.int64)}
        def actual(a, _):
            return patch_embedding(**a, stride=stride, spatial_merge_size=merge, artifact_dir=root/"program")
        result = run_case(case, inputs=inputs, reference=golden, implementation=actual,
                          backend="hw_pto_cpp_sim", artifact_root=root, backend_details=environment_details())
        assert result.passed, result.report
        assert all(torch.equal(v, originals[k]) for k,v in inputs.items())
        print(json.dumps(result.report), flush=True)
    '''))
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, str(script), preset, str(tmp_path)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-16000:]
    evidence = json.loads((tmp_path / "program/program.json").read_text())
    assert evidence["native_simulation"] == "completed"
    assert evidence["platform"] == "a2a3sim"
    assert evidence["window_extraction"] == "im2col3d_composed_gather"
    assert evidence["dedicated_im2col_instruction"] is False
    assert evidence["host_patchify"] is False
    assert list((tmp_path / "program").rglob("*.so"))
    assert any("pto.tmatmul" in p.read_text() for p in (tmp_path / "program").rglob("*.pto"))
    assert list(tmp_path.glob("a-native-op08-*/**/report.json"))


@pytest.mark.parametrize("preset", ["overlap", "model"])
def test_native_im2col_windows(preset, tmp_path):
    script = tmp_path / "im2col_case.py"
    script.write_text(textwrap.dedent('''\
        import sys
        from pathlib import Path
        import torch
        from kernels.pypto.vision_ops import im2col
        from pymodel.common.case import Case, OutputCheck
        from pymodel.isa.hw_pypto import TensorProgram, environment_details
        from pymodel.task_gen.validate import run_case

        torch.set_num_threads(1)
        preset, root = sys.argv[1], Path(sys.argv[2])
        if preset == "overlap":
            shape, window, stride = (2,2,4,6,8), (2,3,2), (1,2,3)
        else:
            shape, window, stride = (1,3,2,32,32), (2,16,16), (2,16,16)
        batch,channels,t,h,w = shape
        kt,kh,kw = window
        st,sh,sw = stride
        rows = batch * ((t-kt)//st+1) * ((h-kh)//sh+1) * ((w-kw)//sw+1)
        width = channels * kt * kh * kw
        kernel = im2col(shape, window, stride, input_dtype="float32")
        pixels = torch.randn(shape, generator=torch.Generator().manual_seed(71))
        before = pixels.clone()
        case = Case("a-native-op08-im2col-"+preset, 71, "pto-functional-v0",
                    {"columns":OutputCheck(atol=0,rtol=0)},
                    {"stage":"im2col", "input_shape":list(shape), "kernel":list(window),
                     "stride":list(stride)}, op_id="OP-08",kernel_id="K-08")
        def golden(a, _):
            # 独立 Torch unfold：时间/空间滑窗后，按 B,Ot,Oh,Ow,C,Kt,Kh,Kw 展平。
            windows = a["pixels"].bfloat16().unfold(2,kt,st).unfold(3,kh,sh).unfold(4,kw,sw)
            return {"columns":windows.permute(0,2,3,4,1,5,6,7).reshape(rows,width)}
        def actual(a, _):
            program = TensorProgram.compile(kernel)
            columns = torch.full((rows,width),float("nan"),dtype=torch.bfloat16)
            program.execute(a["pixels"],columns)
            program.save(root/"program")
            return {"columns":columns}
        result = run_case(case, inputs={"pixels":pixels}, reference=golden, implementation=actual,
                          backend="hw_pto_cpp_sim",artifact_root=root,backend_details=environment_details())
        assert result.passed, result.report
        assert torch.equal(pixels,before)
    '''))
    result = subprocess.run([sys.executable, str(script), preset, str(tmp_path)], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-16000:]
    assert list((tmp_path / "program").rglob("*.so"))
