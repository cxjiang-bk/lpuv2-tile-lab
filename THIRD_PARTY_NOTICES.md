# Third-party sources

PTO is referenced as an unmodified git submodule at `third_party/pto-isa`.
Its source, documentation and license remain in the upstream repository:

- https://github.com/hw-native-sys/pto-isa
- [PTO LICENSE](third_party/pto-isa/LICENSE)

This repository's Python/kernel directory organization is inspired by pymodel's separation of responsibilities. Internal pymodel source and its LPUv1 task binary ABI are not included.

## PyPTO dependency

The frontend is [hw-native-sys/pypto](https://github.com/hw-native-sys/pypto),
under the [CANN Open Software License Agreement Version 2.0](https://github.com/hw-native-sys/pypto/blob/main/LICENSE).
The compiler uses [PTOAS](https://github.com/hw-native-sys/PTOAS) and the
[simpler runtime](https://github.com/hw-native-sys/simpler). Their source,
licenses and copyright notices remain in the corresponding upstream repositories.
Packages and toolchain binaries are installed separately and are not vendored here.
Compiler, runtime, assembler and ISA library revisions are fixed in
[the lock file](data/upstream.lock.json); execution is described in
[A-group implementation](docs/pypto-a1.md).

## Model configuration and reference metadata

The unmodified `config.json` and `generation_config.json` snapshots under
`configs/models/qwen3_6_35b_a3b_fp8/` come from
[Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/tree/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2),
revision `c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`.
Model weights are not included. Header-derived metadata records storage dtype and shape only.

Numerical descriptions reference
[Hugging Face Transformers](https://github.com/huggingface/transformers/tree/f90e93c4f83974f634860e246cdfd2c68666d605),
commit `f90e93c4f83974f634860e246cdfd2c68666d605`.
Source URLs and SHA-256 values are recorded in `baseline.json`.

Unmodified snapshots of `modeling_qwen3_5_moe.py` and `vision_utils.py` are included as non-package
test fixtures in `tests/reference_sources/`. Operator tests verify their SHA-256 values
and extract selected Torch fallback definitions for independent comparisons.
The original copyright notice is preserved; the Apache 2.0 license is included
as [LICENSE.transformers](tests/reference_sources/LICENSE.transformers).
