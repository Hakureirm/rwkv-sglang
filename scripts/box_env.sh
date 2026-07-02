# Source before ANY box command that triggers triton JIT (kernel tests, the sglang
# server). The rwkv-sgl venv python is the system 3.10 which has no dev headers, so
# triton's cuda_utils JIT can't find Python.h. Point at mamba's Python 3.10 headers.
# Usage on box:  source ~/rwkv_env.sh && ~/envs/rwkv-sgl/bin/python ...
_H=/home/user/.local/share/mamba/pkgs/python-3.10.20-h3c07f61_0_cpython/include/python3.10
export C_INCLUDE_PATH="${_H}:${C_INCLUDE_PATH:-}"
export CPATH="${_H}:${CPATH:-}"
# ModelScope token: NEVER hardcode/commit it. Set it in an untracked, box-only
# file ~/.rwkv_secrets.sh (export MODELSCOPE_API_TOKEN=...), sourced here if present.
[ -f "$HOME/.rwkv_secrets.sh" ] && . "$HOME/.rwkv_secrets.sh"
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"
# Full CUDA 12.9 toolkit exists at /usr/local/cuda-12.9 (nvcc present, not on PATH).
# Needed to JIT-compile custom CUDA (e.g. vendoring albatross WKV/linear kernels).
# torch is cu128; major matches (12), so nvcc 12.9 compiles fine. Build for sm_86.
if [ -d /usr/local/cuda-12.9 ]; then
  export CUDA_HOME=/usr/local/cuda-12.9
  export PATH="$CUDA_HOME/bin:$PATH"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
fi
