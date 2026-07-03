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

# State prefix cache (req#3): scheduler.py's is_hybrid_ssm doesn't know rwkv7_config,
# so it would build a plain (RNN-incorrect) RadixCache. Teach it about RWKV-7 so the
# state-aware MambaRadixCache is used instead. Idempotent 1-line patch (kept out of the
# full overlay because scheduler.py is huge + churns upstream; same intent as the
# sglang_main_port upstream_edits). Gate: bench/verify_batch.py --radix-on greedy EXACT.
ssh "$BOX" "python3 - <<'PYEOF'
import sglang.srt.managers.scheduler as sch
f = sch.__file__; s = open(f).read()
a = '            or self.tp_worker.model_runner.mamba2_config is not None\n'
add = '            or self.tp_worker.model_runner.rwkv7_config is not None\n'
if 'rwkv7_config is not None' in s:
    print('scheduler is_hybrid_ssm: already patched')
elif a in s:
    open(f,'w').write(s.replace(a, a+add, 1)); print('scheduler is_hybrid_ssm: patched for RWKV-7')
else:
    print('WARN: is_hybrid_ssm anchor not found; state cache will fall back to radix-off')
PYEOF"
