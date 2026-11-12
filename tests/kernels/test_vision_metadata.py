"""Native checks for runtime media grids and packed Attention boundaries."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    shutil.which("ptoas") is None or importlib.util.find_spec("simpler") is None,
    reason="Source scripts/hw-env.sh for native PTO C++ simulation",
)


@pytest.mark.parametrize("preset", ["mixed", "single_axis", "single_table_entry"])
def test_native_position_runtime_grids(preset, tmp_path):
    script = tmp_path / "position_case.py"
    script.write_text(textwrap.dedent('''\
        import json
        import sys
        from pathlib import Path
        import torch
        from pymodel.ops.vision.position import position_embedding
        from pymodel.ref.vision import vision_position_interpolation, vision_position_embedding

        torch.set_num_threads(1)
        preset, root = sys.argv[1], Path(sys.argv[2])
        if preset == "mixed":
            grid, merge, side = [[1,2,2],[2,4,6],[1,2,4]], 2, 48
        elif preset == "single_axis":
            grid, merge, side = [[2,1,5],[1,3,1]], 1, 48
        else:
            grid, merge, side = [[1,2,2],[2,2,2]], 2, 1
        grid = torch.tensor(grid, dtype=torch.int64)
        rows = int(grid.prod(-1).sum())
        rng = torch.Generator().manual_seed(107)
        x = torch.randn(rows,32,generator=rng).bfloat16()
        table = torch.randn(side*side,32,generator=rng).bfloat16()
        originals = [v.clone() for v in (grid,x,table)]
        indices, weights = vision_position_interpolation(grid,table_side=side,spatial_merge_size=merge)
        expected = vision_position_embedding(x,table,indices,weights)
        actual = position_embedding(grid,x,table,table_side=side,spatial_merge_size=merge,
                                    artifact_dir=root/"program")
        torch.testing.assert_close(actual["out"],expected,atol=.01,rtol=.01)
        assert torch.equal(actual["interp_indices"],indices)
        assert torch.equal(actual["interp_weights"],weights)
        assert all(torch.equal(v,before) for v,before in zip((grid,x,table),originals))
        assert list((root/"program").rglob("*.so"))
        assert list((root/"program").rglob("*.pto"))
        metrics = {"passed":True,"grids":grid.tolist(),"merge_size":merge,"table_side":side,
                   "out_max_abs_error":(actual["out"].float()-expected.float()).abs().max().item(),
                   "indices_exact":True,"weights_exact":True,"inputs_unchanged":True}
        (root/"metadata-report.json").write_text(json.dumps(metrics,indent=2))
    '''))
    result = subprocess.run([sys.executable, str(script), preset, str(tmp_path)], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-14000:]


def test_native_attention_unequal_segments_across_tile_boundary(tmp_path):
    script = tmp_path / "attention_case.py"
    script.write_text(textwrap.dedent('''\
        import json
        import sys
        from pathlib import Path
        import torch
        from pymodel.ops.vision.kernels import vision_attention
        from pymodel.ref.vision import vision_attention as reference

        torch.set_num_threads(1)
        root = Path(sys.argv[1])
        rng = torch.Generator().manual_seed(109)
        def bf(*shape):
            return (torch.randn(shape,generator=rng)*.3).bfloat16()
        rows, hidden, heads = 20,32,2
        angle = torch.randn(rows,hidden//heads,generator=rng)*.2
        inputs = {"x":bf(rows,hidden),"qkv_weight":bf(3*hidden,hidden),"qkv_bias":bf(3*hidden),
                  "proj_weight":bf(hidden,hidden),"proj_bias":bf(hidden),"cos":angle.cos(),
                  "sin":angle.sin(),"cu_seqlens":torch.tensor([0,2,19,20],dtype=torch.int64)}
        originals = {k:v.clone() for k,v in inputs.items()}
        expected = reference(**inputs,num_heads=heads)
        actual = vision_attention(**inputs,num_heads=heads,artifact_dir=root/"program")["out"]
        torch.testing.assert_close(actual,expected,atol=.01,rtol=.01)
        assert all(torch.equal(v,originals[k]) for k,v in inputs.items())
        assert list((root/"program").rglob("*.so"))
        metrics = {"passed":True,"boundaries":[0,2,19,20],
                   "out_max_abs_error":(actual.float()-expected.float()).abs().max().item(),
                   "inputs_unchanged":True}
        (root/"metadata-report.json").write_text(json.dumps(metrics,indent=2))
    '''))
    result = subprocess.run([sys.executable, str(script), str(tmp_path)], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-14000:]
