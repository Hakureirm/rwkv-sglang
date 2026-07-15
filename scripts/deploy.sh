#!/bin/bash
# Deploy the RWKV-7 integration into an installed sglang. No build.
#
# Layout (F0059, "Option B" — churn-proof): sglang_overlay/ ships ONLY the
# additive RWKV-7-only files (configs/rwkv7.py, models/rwkv7.py,
# layers/attention/linear/rwkv7_backend.py, layers/attention/rwkv7_kernels/**,
# speculative/rwkv_chain_worker.py). The ~129 lines of genuine edits to *upstream*
# files are NOT shipped as full-file copies (that froze 11k+ upstream lines and
# clobbered newer sglang on every image bump). They are delivered as:
#   (a) sglang_main_port/upstream_edits.patch — the 129-line RWKV-7 delta across
#       10 upstream files (config registry in utils/hf_transformers/common.py, the
#       all-linear cell_size==0 guard in model_executor/pool_configurator.py, the
#       Rwkv7NoOpFullAttnBackend + rwkv7_config in model_runner.py, radix-off in
#       server_args.py, and the F0036 v_first PP + cuda-graph fix), applied with
#       `git apply` — idempotent, `--check`ed so a base drift fails LOUDLY; and
#   (b) two anchored idempotent Python injections (spec_info.py, scheduler.py) for
#       the two huge, churn-prone files kept out of the patch.
#
# Configure via env (defaults are placeholders — override for your setup):
#   BOX     = ssh host/alias of the target ("" or "localhost" = local install)
#   SP      = site-packages dir of the target sglang venv (used to locate its python)
#   VENV_PY = target python (default derived from SP; override for non-venv layouts,
#             e.g. VENV_PY=python3 for the lmsysorg/sglang:dev-cu12 editable install)
# Examples:
#   BOX=my-gpu-host SP=/opt/venv/lib/python3.10/site-packages bash scripts/deploy.sh
#   BOX= VENV_PY=python3 bash scripts/deploy.sh          # local / dev-container install
set -euo pipefail
BOX="${BOX-gpu-box}"                 # single-dash: an explicit BOX= (empty) means local
SP="${SP:-/home/user/envs/rwkv-sgl/lib/python3.10/site-packages}"
VENV_PY="${VENV_PY:-${SP%/lib/*}/bin/python3}"   # target venv python (override if not a venv)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$ROOT/sglang_main_port/upstream_edits.patch"
cd "$ROOT"
if [ ! -d sglang_overlay/sglang ]; then echo "no overlay yet"; exit 1; fi
if [ ! -f "$PATCH" ]; then echo "missing $PATCH"; exit 1; fi

# Run on the target: remote via ssh, or locally when BOX is empty / "localhost".
if [ -z "$BOX" ] || [ "$BOX" = localhost ]; then LOCAL=1; DEST=""; else LOCAL=0; DEST="$BOX:"; fi
sh_target() { if [ "$LOCAL" = 1 ]; then bash -c "$1"; else ssh "$BOX" "$1"; fi; }
# Copy a dir's contents into $PKG on the target. Dependency-light (tar over the
# same local/ssh channel — no rsync, which the minimal dev image lacks).
copy_tree() { # $1 = local source dir
  if [ "$LOCAL" = 1 ]; then
    tar -C "$1" -cf - --exclude='__pycache__' --exclude='*.pyc' . | tar -C "$PKG" -xf -
  else
    tar -C "$1" -cf - --exclude='__pycache__' --exclude='*.pyc' . | ssh "$BOX" "tar -C '$PKG' -xf -"
  fi
}

# Locate the installed sglang package on the target. Robust for both a wheel
# install (-> <site-packages>/sglang) and an editable/source install
# (-> <checkout>/python/sglang, e.g. the dev-cu12 image).
PKG="$(sh_target "$VENV_PY -c 'import sglang,os;print(os.path.dirname(sglang.__file__))'" | tr -d '\r' | tail -1)"
if [ -z "$PKG" ]; then echo "could not locate sglang on target"; exit 1; fi
echo "target sglang package: ${DEST}${PKG}"

# (1) Additive RWKV-7-only files: copy the (now additive-only) overlay tree over
#     the installed package. No upstream files here anymore.
copy_tree sglang_overlay/sglang
echo "deployed additive RWKV-7 files -> ${DEST}${PKG}/"

# (2) The 129-line genuine edit to upstream files: apply the patch idempotently.
#     Marker = the rwkv7_config property that lands in model_runner.py. `--check`
#     first so a base drift fails LOUDLY (regenerate the patch) not half-applied.
if [ "$LOCAL" = 1 ]; then TPATCH="$PATCH"; else
  ssh "$BOX" "cat > /tmp/rwkv7_upstream_edits.patch" < "$PATCH"; TPATCH="/tmp/rwkv7_upstream_edits.patch"
fi
sh_target "RWKV7_PATCH='$TPATCH' $VENV_PY - <<'PYEOF'
import os, sys, subprocess, sglang
patch = os.environ['RWKV7_PATCH']
pkg = os.path.dirname(sglang.__file__)                 # .../sglang
target = os.path.join(pkg, 'srt', 'model_executor', 'model_runner.py')
if 'def rwkv7_config' in open(target).read():
    print('upstream_edits: already applied (rwkv7_config present)'); sys.exit(0)
# The patch is generated with a/python/sglang/... paths (sglang's source layout).
# Editable/source install (the dev image): sglang lives inside a git work tree with
# that exact layout -> apply -p1 from the REPO ROOT. git apply resolves paths
# relative to the repo root and SILENTLY ignores paths seen from a subdirectory, so
# applying from .../python would no-op. Wheel install: no repo -> -p2 from the
# package parent, where git apply resolves relative to cwd.
rr = subprocess.run(['git', '-C', pkg, 'rev-parse', '--show-toplevel'],
                    capture_output=True, text=True)
if rr.returncode == 0:
    cwd, strip = rr.stdout.strip(), '-p1'
else:
    cwd, strip = os.path.dirname(pkg), '-p2'
chk = subprocess.run(['git', 'apply', strip, '--check', patch],
                     cwd=cwd, capture_output=True, text=True)
if chk.returncode != 0:
    sys.stderr.write(chk.stderr)
    print('upstream_edits: FAILED to apply — base drift; regenerate '
          'sglang_main_port/upstream_edits.patch against this image'); sys.exit(1)
subprocess.run(['git', 'apply', strip, patch], cwd=cwd, check=True)
if 'def rwkv7_config' not in open(target).read():   # guard the silent-no-op case
    print('upstream_edits: git apply reported OK but model_runner.py is unchanged '
          '(path/-p mismatch) — deploy would be broken'); sys.exit(1)
print('upstream_edits: applied (129-line RWKV-7 delta across 10 upstream files)')
PYEOF"

# Chain-speculative decoding (req#6 / ADR-0006): register RWKV_CHAIN in the
# SpeculativeAlgorithm enum + worker map. Anchored idempotent patch (spec_info.py
# is upstream-churny; the worker itself ships in the overlay as
# speculative/rwkv_chain_worker.py). Wire-up: --speculative-algorithm RWKV_CHAIN
# --speculative-draft-model-path <small rwkv7 dir> [--speculative-num-draft-tokens K].
sh_target "$VENV_PY - <<'PYEOF'
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

# State prefix cache (req#3): on v0.5.10.post1, scheduler.py's is_hybrid_ssm inlines
# the ssm-config chain and doesn't know rwkv7_config, so it would build a plain
# (RNN-incorrect) RadixCache. Teach it about RWKV-7 so the state-aware MambaRadixCache
# is used instead. Idempotent 1-line patch (kept out of the full overlay because
# scheduler.py is huge + churns upstream; same intent as the sglang_main_port
# upstream_edits). Gate: bench/verify_batch.py --radix-on greedy EXACT.
# NOTE (main / F0059): upstream moved this to `is_hybrid_ssm = result.is_hybrid_ssm`,
# fed by the linear-attn registry — which the model_runner.py patch already teaches
# about rwkv7_config. So on main the anchor below is absent and the WARN is benign
# (RWKV-7 is recognized via the patch; default serving is radix-off regardless).
sh_target "$VENV_PY - <<'PYEOF'
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

# Clear stale bytecode across everything we touched (copy + patch + injections),
# so the next import recompiles the edited .py.
sh_target "find '$PKG'/srt -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null; echo pycache-cleared"
