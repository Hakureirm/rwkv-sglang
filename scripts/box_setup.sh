#!/bin/bash
# Robust env setup for the gpu-box box. Run DETACHED (nohup) so ssh drops can't
# kill it. Sequential installs avoid bandwidth contention that stalls big wheels.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_DOWNLOADS=never
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=3
export UV_VENV_CLEAR=1
IDX=(--index-url https://mirrors.ustc.edu.cn/pypi/simple
     --extra-index-url https://pypi.org/simple
     --index-strategy unsafe-best-match --prerelease=allow)
PY=/usr/bin/python3.10

retry() { local n=0; until "$@"; do n=$((n+1)); [ $n -ge 4 ] && return 1; echo "[retry $n] $*"; sleep 5; done; }

echo "### START $(date) ###"

echo "### [1/2] sglang env (rwkv-sgl) ###"
uv venv --python "$PY" "$HOME/envs/rwkv-sgl"
retry uv pip install --python "$HOME/envs/rwkv-sgl/bin/python" "${IDX[@]}" "sglang==0.5.10.post1"
"$HOME/envs/rwkv-sgl/bin/python" -c "import sglang,torch;print('SGL_OK',sglang.__version__,torch.__version__,'cuda',torch.cuda.is_available())" && echo "### SGL DONE ###"

echo "### [2/2] oracle env (rwkv-ref) ###"
uv venv --python "$PY" "$HOME/envs/rwkv-ref"
retry uv pip install --python "$HOME/envs/rwkv-ref/bin/python" "${IDX[@]}" rwkv numpy modelscope tokenizers "torch==2.9.1"
"$HOME/envs/rwkv-ref/bin/python" -c "import rwkv,numpy,modelscope;print('REF_OK')" && echo "### REF DONE ###"

echo "### ALL DONE $(date) ###"
