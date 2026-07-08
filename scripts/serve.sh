#!/bin/bash
# Recommended production launch for the RWKV-7 SGLang overlay.
#
# Turns ON the full hand-written fast-path stack. Verification scope (precise):
# every env below is greedy-token-EXACT vs the numpy fp32 oracle including all-on
# together (bench/verify_batch.py OVERALL PASS), the glue/LoRA kernels are
# byte-gated incl. pad-slot (-1) and duplicate-index cases (bench/test_glue.py,
# bench/test_lora_mn.py), and GEMV autotune is restricted to the logits-invariant
# OutTile axis by default (fast_linear.py NUMERICS DISCIPLINE; the full
# threads-crossing space needs RWKV_GEMV_AUTOTUNE_FULL=1 + a greedy re-gate).
# The kernels self-gate (fp16 / M-gate / decode as applicable) and fall back to
# the stock torch path otherwise.
#
# Also applies the cuda_graph_max_bs fix (F0024): SGLang auto-caps it to 24 for
# this RWKV/MambaPool config regardless of free memory, silently forcing eager
# decode for batch>24 (3–8x throughput loss). We set it explicitly.
#
# Deploy the overlay first: `BOX=... SP=... bash scripts/deploy.sh`.
#
# Usage:
#   MODEL=/path/to/rwkv7-1.5b bash scripts/serve.sh              # throughput mode (default)
#   MODEL=/path/to/rwkv7-1.5b MODE=statecache bash scripts/serve.sh
#   MODEL=/path/to/rwkv7-1.5b PORT=30000 CGMAXBS=512 bash scripts/serve.sh -- <extra sglang flags>
#
# Two VERIFIED modes (we don't ship untested combos):
#   throughput (default): cuda-graph ON (max-bs 512) + radix OFF. Full fast-path
#       stack leads plain fp16 at small bsz and at the peak (parity within the
#       run-to-run band at mid-bsz) on the same wall-clock harness (F0028):
#       1.5B fp16 bsz1 154.4->225.9 (+46%), peak 7334 @ bsz384 (+6.5% vs 6885); the
#       w8a8 model reaches ~9152 @ bsz512 (F0025). Greedy-exact; fused LoRA self-gates
#       by M (wins <=4, cuBLAS fallback above), so all-envs-on is optimal per bsz.
#   statecache: state-aware MambaRadixCache ON (req#3, ~98% high-reuse hit,
#       TTFT 784->200ms; F0022) + cuda-graph OFF (the pairing F0022 verified).
set -euo pipefail

MODEL="${MODEL:?set MODEL=/path/to/rwkv7-model-dir}"
PYTHON="${PYTHON:-python}"         # set to the venv python if not on PATH (e.g. /opt/venv/bin/python)
PORT="${PORT:-30000}"
DTYPE="${DTYPE:-float16}"          # fast paths are fp16; bf16/fp32 fall back cleanly
MEMFRAC="${MEMFRAC:-0.85}"
CGMAXBS="${CGMAXBS:-512}"
MODE="${MODE:-throughput}"

# Verified greedy-exact hand-written kernels (see findings F0015/F0020/F0025/F0026):
export RWKV_FAST_LINEAR=1          # fused fp16 GEMV, bsz1 r/k/v/o + ffn proj
export RWKV_SPARSE_FFN=1           # sparse sqrelu channel-mix value-proj (bsz1)
export RWKV_FUSED_LORA=1           # fused 4-chain LoRA (M==1 lora4_m1 + batched lora4_mn)
export RWKV_FUSED_GLUE=1           # fused paged token-shift + lerp (R2 attn+ffn)
export RWKV_GEMV_AUTOTUNE=1        # arch-aware GEMV launch autotune (warmup only)
# W1 reverse-overtake pair (F0051/F0052), promoted to default here 2026-07-08. GATES and
# SQRELU are NOT two independent simultaneous wins: SQRELU only fires when `not self._sparse`
# (models/rwkv7.py, mutually exclusive by construction — the sparse kernel applies its own
# relu^2 and needs the raw un-fused k). With RWKV_SPARSE_FFN=1 above, SPARSE_FFN wins
# whenever it's eligible (a real bandwidth win, skips ~90% of weight reads on real prompts);
# SQRELU is the insurance policy for sparse's own internal dense-fallback path (line ~926,
# "not buildable -> dense from here on"), not a second lever stacked on top. Both are still
# correct to default ON together (harmless when redundant, real when sparse can't apply).
# Verification: each byte-exact gated individually (op-level + end-to-end verify_batch.py
# greedy-EXACT) and quantized-tier blast-radius-contained; re-verified here 2026-07-08 with
# ALL 7 exports above ON simultaneously on 1.5B fp16 cuda-graph vs the numpy oracle —
# OVERALL: PASS (all batches exact), so the literal combo this script ships is gated, not
# assumed to compose. Known scope gap (disclose, don't hide): both are byte-exact-verified
# on sm_89 (L4) + sm_90 (H100) only, not the full 11-card matrix the other 5 exports
# accumulated — a narrower-published-speed-number issue, not a correctness one (an
# arch-specific divergence would fail the routine oracle re-verify this project runs per box).
export RWKV_FUSED_GATES=1          # fused LoRA-gate activations (sigmoids+neg/mul/sub/add)
export RWKV_FUSED_SQRELU=1         # epilogue-fused ffn relu(k)^2 into the key GEMV's store

COMMON=(--model-path "$MODEL" --dtype "$DTYPE" --trust-remote-code
        --port "$PORT" --mem-fraction-static "$MEMFRAC"
        --page-size 1 --attention-backend triton --disable-piecewise-cuda-graph)

case "$MODE" in
  throughput)
    EXTRA=(--disable-radix-cache --cuda-graph-max-bs "$CGMAXBS"
           --chunked-prefill-size 4096 --max-running-requests 512)
    ;;
  statecache)
    # MambaRadixCache (radix ON, enabled by the deploy.sh scheduler patch) +
    # cuda-graph OFF — the F0022-verified pairing (cuda-graph + mamba radix
    # co-existence is a separate follow-up).
    EXTRA=(--disable-cuda-graph)
    ;;
  *)
    echo "unknown MODE=$MODE (use throughput | statecache)"; exit 1 ;;
esac

# strip a leading `--` separator if the caller passed extra flags after it
[ "${1:-}" = "--" ] && shift
echo "[serve] MODE=$MODE dtype=$DTYPE port=$PORT cuda_graph_max_bs=$CGMAXBS (fast-path stack ON)"
exec "$PYTHON" -m sglang.launch_server "${COMMON[@]}" "${EXTRA[@]}" "$@"
