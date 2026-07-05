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
VENV_PY="${SP%/lib/*}/bin/python3"   # the target venv's python (site-packages -> venv root)
cd "$ROOT"
if [ ! -d sglang_overlay/sglang ]; then echo "no overlay yet"; exit 1; fi
rsync -az -v --exclude='__pycache__' --exclude='*.pyc' sglang_overlay/sglang/ "$BOX:$SP/sglang/"
echo "deployed sglang_overlay/sglang/ -> $BOX:$SP/sglang/"
# bytecode can go stale vs overlaid .py; clear it
ssh "$BOX" "find $SP/sglang/srt/{models,layers/attention,configs,model_executor,utils} -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null; echo pycache-cleared"

# Chain-speculative decoding (req#6 / ADR-0006): register RWKV_CHAIN in the
# SpeculativeAlgorithm enum + worker map. Anchored idempotent patch (spec_info.py
# is upstream-churny; the worker itself ships in the overlay as
# speculative/rwkv_chain_worker.py). Wire-up: --speculative-algorithm RWKV_CHAIN
# --speculative-draft-model-path <small rwkv7 dir> [--speculative-num-draft-tokens K].
ssh "$BOX" "$VENV_PY - <<'PYEOF'
import sglang.srt.speculative.spec_info as si
f = si.__file__; s = open(f).read()
if 'RWKV_CHAIN' in s:
    print('spec_info: already patched')
else:
    a1 = '    STANDALONE = auto()\n'
    a2 = '    def is_ngram(self) -> bool:\n'
    a3 = '        elif self.is_ngram():\n'
    add2 = ('    def is_rwkv_chain(self) -> bool:\n'
            '        return self == SpeculativeAlgorithm.RWKV_CHAIN\n\n')
    add3 = ('        elif self.is_rwkv_chain():\n'
            '            if enable_overlap:\n'
            '                raise ValueError(\n'
            '                    \"RWKV_CHAIN requires the non-overlap scheduler (spec V1).\"\n'
            '                )\n\n'
            '            from sglang.srt.speculative.rwkv_chain_worker import RwkvChainWorker\n\n'
            '            return RwkvChainWorker\n')
    if a1 in s and a2 in s and a3 in s:
        s = s.replace(a1, a1 + '    RWKV_CHAIN = auto()\n', 1)
        s = s.replace(a2, add2 + a2, 1)
        s = s.replace(a3, add3 + '        ' + a3.strip() + '\n', 1)
        open(f, 'w').write(s)
        print('spec_info: RWKV_CHAIN registered')
    else:
        print('WARN: spec_info anchors not found; RWKV_CHAIN not registered')
PYEOF"

# State prefix cache (req#3): scheduler.py's is_hybrid_ssm doesn't know rwkv7_config,
# so it would build a plain (RNN-incorrect) RadixCache. Teach it about RWKV-7 so the
# state-aware MambaRadixCache is used instead. Idempotent 1-line patch (kept out of the
# full overlay because scheduler.py is huge + churns upstream; same intent as the
# sglang_main_port upstream_edits). Gate: bench/verify_batch.py --radix-on greedy EXACT.
ssh "$BOX" "$VENV_PY - <<'PYEOF'
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
