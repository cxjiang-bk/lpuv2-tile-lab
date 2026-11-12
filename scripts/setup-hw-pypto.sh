#!/usr/bin/env bash
# Install the locked HW compiler/runtime in private project directories.
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TASK_ROOT"
if [[ "${1:-}" == --check ]]; then
    source scripts/hw-env.sh
    python -c 'import json; from pymodel.isa.hw_pypto import environment_details; print(json.dumps(environment_details(), indent=2))'
    exit 0
fi
command -v conda >/dev/null || { echo 'Install Miniconda/Miniforge, or set up the dependencies in docs/pypto-a1.md.' >&2; exit 1; }
TASK_PYTHON=${LPUV2_PYTHON:-python3}
"$TASK_PYTHON" -c 'import sys; assert sys.version_info >= (3,11), "Python >=3.11 required"'
TASK_TOOLS=${LPUV2_TOOLCHAIN_ROOT:-$TASK_ROOT/.tools/hw}
mkdir -p "$TASK_TOOLS/bin"
TASK_TOOLS=$(cd "$TASK_TOOLS" && pwd)
TASK_SOURCE=$TASK_TOOLS/pypto
TASK_PREFIX=$TASK_TOOLS/compiler-ptoas
TASK_REAL_CMAKE=$(command -v cmake)
[[ -n "$TASK_REAL_CMAKE" ]] || { echo 'cmake >=3.28 is required' >&2; exit 1; }
TASK_COMMIT=$("$TASK_PYTHON" -c 'import json; print(json.load(open("data/upstream.lock.json"))["pypto"]["commit"])')
TASK_RUNTIME=$("$TASK_PYTHON" -c 'import json; print(json.load(open("data/upstream.lock.json"))["pypto"]["runtime"]["commit"])')
TASK_SHA=$("$TASK_PYTHON" -c 'import json; print(json.load(open("data/upstream.lock.json"))["pypto"]["ptoas"]["wheel_sha256_x86_64"])')
if [[ ! -d "$TASK_SOURCE/.git" ]]; then
    git clone https://github.com/hw-native-sys/pypto.git "$TASK_SOURCE"
fi
if [[ -n "$(git -C "$TASK_SOURCE" status --porcelain --untracked-files=no)" ]]; then
    echo "Tool source has local changes: $TASK_SOURCE" >&2; exit 1
fi
git -C "$TASK_SOURCE" fetch origin "$TASK_COMMIT"
git -C "$TASK_SOURCE" checkout --detach "$TASK_COMMIT"
git -C "$TASK_SOURCE" submodule update --init --recursive
[[ "$(git -C "$TASK_SOURCE/runtime" rev-parse HEAD)" == "$TASK_RUNTIME" ]]
if [[ ! -d "$TASK_PREFIX/conda-meta" ]]; then
    conda create -y -p "$TASK_PREFIX" -c conda-forge python=3.10 pip gcc_linux-64=15.3 gxx_linux-64=15.3
fi
TASK_WHEEL=$TASK_TOOLS/ptoas-0.72-cp310-cp310-manylinux_2_34_x86_64.whl
if [[ ! -f "$TASK_WHEEL" ]]; then
    curl -fL --retry 3 -o "$TASK_WHEEL" https://github.com/hw-native-sys/PTOAS/releases/download/v0.72/ptoas-0.72-cp310-cp310-manylinux_2_34_x86_64.whl
fi
printf '%s  %s\n' "$TASK_SHA" "$TASK_WHEEL" | sha256sum -c -
"$TASK_PREFIX/bin/python" -m pip install "$TASK_WHEEL"
ln -sfn "$TASK_PREFIX/bin/x86_64-conda-linux-gnu-gcc" "$TASK_TOOLS/bin/gcc-15"
ln -sfn "$TASK_PREFIX/bin/x86_64-conda-linux-gnu-g++" "$TASK_TOOLS/bin/g++-15"
ln -sfn gcc-15 "$TASK_TOOLS/bin/gcc"
ln -sfn g++-15 "$TASK_TOOLS/bin/g++"
ln -sfn "$TASK_ROOT/scripts/cmake-cap.py" "$TASK_TOOLS/bin/cmake"
if [[ ! -x .venv/bin/python ]]; then "$TASK_PYTHON" -m venv .venv; fi
# %q preserves spaces and shell metacharacters in local paths.
{
    printf 'export PATH=%q:"$PATH"\n' "$TASK_ROOT/.venv/bin:$TASK_TOOLS/bin:$TASK_PREFIX/bin"
    printf 'export LD_LIBRARY_PATH=%q:"${LD_LIBRARY_PATH:-}"\n' "$TASK_PREFIX/lib"
    printf 'export CC=%q CXX=%q PTOAS_ROOT=%q\n' "$TASK_TOOLS/bin/gcc-15" "$TASK_TOOLS/bin/g++-15" "$TASK_PREFIX"
    printf 'export LPUV2_REAL_CMAKE=%q LPUV2_CMAKE_LOCK=%q\n' "$TASK_REAL_CMAKE" "$TASK_TOOLS/cmake-build.lock"
    printf 'export PYPTO_BUILD_JOBS=2 PYPTO_TEST_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 MAX_JOBS=2 MAKEFLAGS=-j2\n'
} > "$TASK_TOOLS/env.sh"
printf 'export LPUV2_HW_ENV_FILE=%q\n' "$TASK_TOOLS/env.sh" > .env.hw
source scripts/hw-env.sh
python -m pip install --upgrade pip
python -m pip install scikit-build-core 'nanobind>=2,<3' ninja cloudpickle 'torch==2.10.0' 'numpy>=2' 'pytest>=8,<9.2'
python -m pip install --no-build-isolation "$TASK_SOURCE"
python -m pip install --no-build-isolation --config-settings=build.targets=build_package_a2a3sim "$TASK_SOURCE/runtime"
python -m pip install --no-deps -e .
conda list -p "$TASK_PREFIX" --explicit > "$TASK_TOOLS/compiler-packages.explicit.txt"
bash scripts/setup-hw-pypto.sh --check
printf '\nReady: source scripts/hw-env.sh\n'
