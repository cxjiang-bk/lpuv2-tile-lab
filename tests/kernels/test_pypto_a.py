"""Compile PTO IR and run actual hw-native-sys PyPTO C++ simulator kernels."""

import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which('ptoas') is None or importlib.util.find_spec('simpler') is None,
    reason='Native kernels require source scripts/hw-env.sh',
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('op', ['03', '04', '06', '08', '09', '10', '11', '12', '13'])
def test_native_operator_and_execution_evidence(op, tmp_path):
    result = subprocess.run(
        [sys.executable, '-m', 'pymodel.task_gen.validate_a', '--case', op,
         '--artifact-root', str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-12000:]
    manifests = list(tmp_path.glob(f'a-native-op{op}-small/*/manifest.json'))
    assert len(manifests) == 1
    run = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    details = manifest['backend_details']
    assert manifest['backend'] == details['backend'] == 'hw_pto_cpp_sim'
    assert details['native_compilation'] == details['native_simulation'] == 'completed'
    assert details['platform'] == 'a2a3sim'
    assert details['hardware_precision_validated'] is False
    assert list((run / 'program').rglob('*.so'))
    assert list((run / 'program').rglob('*.pto'))
    assert all((run / name).is_file() for name in manifest['files'])
    assert not list(run.rglob('python_semantic.py'))
    if op == '06':
        assert set(manifest['case']['outputs']) == {'out', 'state_out', 'conv_history_out'}
        assert details['fp8_execution'] == 'bf16_carrier'
        assert details['native_fp8_cube_validated'] is False
        assert details['gdn_path'] == 'recurrent'
    if op == '11':
        assert set(manifest['case']['outputs']) == {'out', 'interp_indices', 'interp_weights'}
    if op == '08':
        assert set(manifest['case']['outputs']) == {'out', 'grid_thw'}
        assert details['window_extraction'] == 'im2col3d_composed_gather'
        assert details['host_patchify'] is False
        assert details['pixel_input_dtype'] == 'torch.float32'


@pytest.mark.parametrize('kind', ['quantize', 'linear'])
def test_native_fp8_value_semantics(kind, tmp_path):
    result = subprocess.run(
        [sys.executable, '-m', 'pymodel.task_gen.validate_fp8', '--case', kind,
         '--artifact-root', str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-12000:]
    manifest_path, = tmp_path.glob(f'a-native-fp8-{kind}/*/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['backend_details']['fp8_execution'] == 'bf16_carrier'
    assert manifest['backend_details']['native_fp8_cube_validated'] is False
    assert list((manifest_path.parent / 'program').rglob('*.so'))


def test_native_gdn_continuous_decode(tmp_path):
    result = subprocess.run(
        [sys.executable, '-m', 'pymodel.task_gen.validate_a', '--case', '06',
         '--gdn-decode-continuity', '--artifact-root', str(tmp_path)], cwd=ROOT,
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-12000:]
    manifest_path, = tmp_path.glob('a-native-op06-decode-continuity/*/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['backend_details']['invocations']) == 2
    report = json.loads((manifest_path.parent / 'report.json').read_text())
    assert report['passed']
    assert report['outputs']['state_after_first_token']['passed']
    assert report['outputs']['history_after_first_token']['passed']


@pytest.mark.parametrize('sequence,chunk_size', [(1, 64), (5, 3), (6, 3), (5, 1)])
@pytest.mark.parametrize('path', ['recurrent', 'chunk'])
def test_native_gdn_core_paths(sequence, chunk_size, path, tmp_path):
    # Test the actual internal K-06 stage with stronger Q/K/V and nonzero State;
    # full-OP projection fixtures alone produce very small core values.
    script = tmp_path / 'chunk_core.py'
    script.write_text(textwrap.dedent('''\
        import sys
        import json
        import torch
        import pypto.language as pl
        from kernels.pypto.gdn import gdn_chunk, gdn_recurrent as recurrent_kernel
        from pymodel.isa.hw_pypto import TensorProgram
        from pymodel.ref.gdn import gdn_chunk as reference, gdn_recurrent

        torch.set_num_threads(1)
        sequence, chunk_size = (int(v) for v in sys.argv[1:3])
        kd, vd = 32, 16

        stage = (gdn_chunk(sequence, kd, vd, chunk_size, 1e-6) if sys.argv[4] == "chunk"
                 else recurrent_kernel(sequence, kd, vd, 1e-6))
        @pl.jit
        def kernel(q: pl.Tensor, k: pl.Tensor, v: pl.Tensor, beta: pl.Tensor, g: pl.Tensor,
                   initial: pl.Tensor, out: pl.Out[pl.Tensor], state: pl.Out[pl.Tensor]):
            out, state = stage(q, k, v, beta, g, initial, out, state)
            return out, state

        rng = torch.Generator().manual_seed(83)
        q = torch.randn(sequence, kd, generator=rng).bfloat16()
        k = torch.randn(sequence, kd, generator=rng).bfloat16()
        v = torch.randn(sequence, vd, generator=rng).bfloat16()
        beta = torch.rand(sequence, 1, generator=rng).bfloat16()
        g = -torch.rand(sequence, 1, generator=rng) * .2
        initial = torch.randn(kd, vd, generator=rng) * .1
        before = initial.clone()
        out, state = torch.full_like(v, float('nan')), torch.full_like(initial, float('nan'))
        program = TensorProgram.compile(kernel)
        program.execute(q, k, v, beta, g, initial, out, state)
        program.save(sys.argv[3])
        args = (q.reshape(1, sequence, 1, kd), k.reshape(1, sequence, 1, kd),
                v.reshape(1, sequence, 1, vd), beta.reshape(1, sequence, 1),
                g.reshape(1, sequence, 1), initial.reshape(1, 1, kd, vd))
        expected = reference(*args, chunk_size=chunk_size)
        recurrent = gdn_recurrent(*args)
        metrics = {}
        for name, value in (('out', out), ('state_out', state)):
            target = expected[name].reshape_as(value)
            metrics[name] = {'max_abs_error': (value.float() - target.float()).abs().max().item()}
        from pathlib import Path
        (Path(sys.argv[3]).parent / 'core-report.json').write_text(json.dumps(metrics, indent=2))
        for golden in (expected, recurrent):
            torch.testing.assert_close(out, golden['out'].reshape_as(out), atol=.01, rtol=.01)
            torch.testing.assert_close(state, golden['state_out'].reshape_as(state), atol=1e-5, rtol=1e-4)
        assert torch.equal(initial, before)
    '''))
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, str(script), str(sequence), str(chunk_size),
                             str(tmp_path / 'program'), path], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-12000:]
    assert list((tmp_path / 'program').rglob('*.so'))


@pytest.mark.parametrize('sequence,chunk_size,batch', [(5, 3, 2), (4, 64, 1)])
def test_native_gdn_chunk_complete_operator(sequence, chunk_size, batch, tmp_path):
    result = subprocess.run(
        [sys.executable, '-m', 'pymodel.task_gen.validate_a', '--case', '06', '--gdn-path', 'chunk',
         '--gdn-sequence', str(sequence), '--gdn-chunk-size', str(chunk_size), '--gdn-batch', str(batch),
         '--artifact-root', str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-12000:]
    manifest_path, = tmp_path.glob(f'a-native-op06-chunk-b{batch}-s{sequence}-c{chunk_size}/*/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['backend_details']['gdn_path'] == 'chunk'
    assert manifest['backend_details']['chunk_size'] == chunk_size
    assert manifest['backend_details']['native_simulation'] == 'completed'
    assert list((manifest_path.parent / 'program').rglob('*.so'))


def test_native_gdn_chunk_prefill_to_recurrent_decode(tmp_path):
    result = subprocess.run(
        [sys.executable, '-m', 'pymodel.task_gen.validate_a', '--case', '06', '--gdn-path', 'chunk',
         '--gdn-chunk-size', '3', '--gdn-decode-continuity', '--artifact-root', str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-12000:]
    manifest_path, = tmp_path.glob('a-native-op06-chunk-prefill-decode/*/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['backend_details']['execution_scenario'] == 'chunk_prefill_recurrent_decode'
    paths = [json.loads((manifest_path.parent / 'program' / name).read_text())['gdn_path']
             for name in manifest['backend_details']['invocations']]
    assert paths == ['chunk', 'recurrent']
    report = json.loads((manifest_path.parent / 'report.json').read_text())
    assert report['passed']
    assert report['outputs']['state_after_prefill']['passed']
    assert report['outputs']['history_after_prefill']['passed']



def test_native_repeated_attention_and_fp8(tmp_path):
    # Repeated calls expose premature reuse of head/dot/accumulator buffers.
    script = tmp_path / 'repeat_native.py'
    script.write_text(textwrap.dedent('''\
        import sys
        from pathlib import Path
        import torch
        from pymodel.task_gen.validate_a import make_inputs, reference
        from pymodel.ops.vision.kernels import vision_attention
        from pymodel.ops.gdn.kernel import pad_fp8_weight
        from pymodel.ref.fp8 import linear
        from kernels.pypto.fp8 import fp8_linear_bf16_sim
        from pymodel.isa.hw_pypto import TensorProgram

        torch.set_num_threads(1)
        root = Path(sys.argv[1])
        for boundaries in ([0,2,5], [0,1,2,3,4,5]):
            inputs, params = make_inputs('09')
            inputs['cu_seqlens'] = torch.tensor(boundaries, dtype=torch.int64)
            before = {k:v.clone() for k,v in inputs.items()}
            expected = reference('09', inputs, params)['out']
            for repeat in range(3):
                actual = vision_attention(**inputs, **params, artifact_dir=root/f'attention-{len(boundaries)}-{repeat}')['out']
                torch.testing.assert_close(actual, expected, atol=.01, rtol=.01)
                assert all(torch.equal(v, before[k]) for k,v in inputs.items())
        rng = torch.Generator().manual_seed(43)
        x = torch.randn(4,129,generator=rng).bfloat16()
        x[0] = 0
        x[1,:2] = torch.tensor([448.,-448.])
        weight = (torch.randn(129,129,generator=rng)*3).to(torch.float8_e4m3fn)
        scale = torch.tensor([[.02,.03],[.04,.05]],dtype=torch.bfloat16)
        expected = linear(x,weight,scale)
        codes = pad_fp8_weight(weight).view(torch.uint8).int()
        program = TensorProgram.compile(fp8_linear_bf16_sim(4,129,129))
        for repeat in range(8):
            out = torch.full((4,129),float('nan'),dtype=torch.bfloat16)
            program.execute(x,codes,scale,out)
            torch.testing.assert_close(out,expected,atol=.01,rtol=.01)
        program.save(root/'fp8')
    '''))
    result = subprocess.run([sys.executable, str(script), str(tmp_path)], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr)[-14000:]
    assert list((tmp_path / 'fp8').rglob('*.so'))
