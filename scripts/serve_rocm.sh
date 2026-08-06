#!/usr/bin/env bash
# ROCm entry point. It reuses serve.sh while selecting the portable RWKV path.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=rocm_env.sh
source "$ROOT/scripts/rocm_env.sh"
export CGMAXBS="${CGMAXBS:-8}"
export MEMFRAC="${MEMFRAC:-0.85}"
exec bash "$ROOT/scripts/serve.sh" "$@"
