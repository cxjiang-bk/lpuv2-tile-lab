#!/usr/bin/env bash
# Source this file before compiling or running HW PyPTO kernels.
TASK_HW_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$TASK_HW_ROOT/.env.hw" ]]; then source "$TASK_HW_ROOT/.env.hw"; fi
if [[ -n "${LPUV2_HW_ENV_FILE:-}" ]]; then
    if [[ ! -f "$LPUV2_HW_ENV_FILE" ]]; then
        echo "LPUV2_HW_ENV_FILE does not exist" >&2
        return 1
    fi
    source "$LPUV2_HW_ENV_FILE"
fi
export PATH="$TASK_HW_ROOT/.venv/bin:$PATH"
if ! command -v ptoas >/dev/null || ! command -v g++-15 >/dev/null; then
    echo "Install the locked HW PyPTO toolchain; see docs/pypto-a1.md" >&2
    return 1
fi
unset TASK_HW_ROOT
