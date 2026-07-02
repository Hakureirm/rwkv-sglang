#!/bin/bash
# Deploy the RWKV-7 sglang overlay (new + edited files, mirroring the sglang
# package tree) into an installed sglang. No build; rsync only. The overlay
# (sglang_overlay/) + this script ARE the deliverable.
#
# Configure via env (defaults are placeholders — override for your setup):
#   BOX  = ssh host/alias of the target machine (use "" / localhost for a local install)
#   SP   = site-packages dir of the target sglang venv on that machine
# Examples:
#   BOX=my-gpu-host SP=/opt/venv/lib/python3.10/site-packages bash scripts/deploy.sh
#   # local install:
#   BOX=localhost   SP="$(python -c 'import site;print(site.getsitepackages()[0])')" bash scripts/deploy.sh
set -euo pipefail
BOX="${BOX:-gpu-box}"
SP="${SP:-/home/user/envs/rwkv-sgl/lib/python3.10/site-packages}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -d sglang_overlay/sglang ]; then echo "no overlay yet"; exit 1; fi
rsync -az -v --exclude='__pycache__' --exclude='*.pyc' sglang_overlay/sglang/ "$BOX:$SP/sglang/"
echo "deployed sglang_overlay/sglang/ -> $BOX:$SP/sglang/"
# bytecode can go stale vs overlaid .py; clear it
ssh "$BOX" "find $SP/sglang/srt/{models,layers/attention,configs,model_executor,utils} -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null; echo pycache-cleared"
