"""Stage 3 step 1: execute explicit Tile IR through upstream Torch codegen."""

import torch

from kernels.pto.common.tile_add import tile_add
from pypto.debug import torch_codegen


def test_tile_add_executes_generated_torch_on_cpu(tmp_path):
    a = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16) / 4
    b = torch.full((2, 16), -0.5, dtype=torch.bfloat16)
    out = torch.zeros_like(a)
    program = tile_add.specialize(a, b, out)
    code = torch_codegen(program, check_shapes=True)
    namespace = {}
    exec(compile(code, "<pypto_tile_add>", "exec"), namespace)
    namespace["tile_add"](a, b, out)
    torch.testing.assert_close(out, a + b, atol=0, rtol=0)
    assert out.device.type == "cpu"
    (tmp_path / "tile_add_ir.py").write_text(program.as_python())
    (tmp_path / "tile_add_torch.py").write_text(code)
