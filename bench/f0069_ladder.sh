#!/bin/bash
# F0069 #59: reproduce the published bsz1 flagship ladder on today's tree.
#
# The published ladder (BENCHMARKS §7a-flagship) stacks env knobs ON TOP of the
# mega_flag_matrix D leg; D alone is only its bottom rung. Re-measuring D and
# comparing it to the headline is the exact convention error this round exists
# to remove, so the three rungs are run back-to-back in ONE session here.
#
#   L0  D                          MEGA+WKV_CUDA+PDL              published 493.0 / (7.2B) --
#   L1  D + ADDLN_WIDE             F0065                          published 502.3
#   L2  L1 + FUSED_LORA_GATED      F0066c headline                published 514.5 / 142.8
#
# (F0066b's sparse-path finalize is UNCONDITIONAL in this tree -- no env -- so
# every rung here already carries it. That is why L0 is expected to read ABOVE
# the 493.0 that F0063 measured before finalize landed.)
#
# Gates, both hard:
#   * greedy fixture EXACT per leg (same fixture as mega_flag_matrix.sh)
#   * L2 must announce '[rwkv7] F0066c fused LoRA gate epilogue ENABLED'
#     and L0/L1 must NOT -- catches a silently-ignored env var, which would
#     otherwise publish a fake number.
#   * ADDLN_WIDE has no announce line; its liveness check is L1 >> L0. If L1
#     lands within noise of L0 the env did nothing and the rung is void.
set -uo pipefail

MODEL="${1:?model dir}"; PORT="${2:-30070}"; OUT="${3:?out dir}"; TAG="${4:?72b|15b}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"

if [ "$TAG" = 72b ]; then
  FIX_PROMPT='[11, 6699, 304, 25740, 109, 37480, 4600, 52151, 4596, 22590, 30449, 4706]'
  FIX_EXPECT='[37138, 45, 44312, 47, 11, 6699, 304, 25740]'
else
  FIX_PROMPT="$(python3 -c "import json;print(json.load(open('$REPO/bench/fixtures/oracle_rwkv7_15b_eiffel.json'))['prompt_tokens'])")"
  FIX_EXPECT="$(python3 -c "import json;print(json.load(open('$REPO/bench/fixtures/oracle_rwkv7_15b_eiffel.json'))['greedy_tokens'])")"
fi
ANNOUNCE='F0066c fused LoRA gate epilogue ENABLED'

run_leg() { # $1 leg, $2 want_announce(0|1), rest: env K=V
  local LEG="$1" WANT="$2"; shift 2
  local LOG="$OUT/serve_${TAG}_${LEG}.log"
  echo "=== LEG $LEG ($*) ==="
  pkill -f sglang.launch_server 2>/dev/null; sleep 4
  ( cd "$REPO" && env MODEL="$MODEL" PYTHON=python3 PORT="$PORT" MEMFRAC=0.85 \
      CGMAXBS=32 RWKV_STATE_FP16=1 "$@" \
      bash scripts/serve.sh -- --max-running-requests 32 > "$LOG" 2>&1 ) &
  for i in $(seq 1 120); do
    grep -q "fired up and ready to roll" "$LOG" 2>/dev/null && break
    grep -qE "Traceback|ValueError|CUDA out of memory" "$LOG" 2>/dev/null && { echo "LEG $LEG: BOOT FAILED"; tail -5 "$LOG"; return 1; }
    sleep 5
  done
  grep -q "fired up" "$LOG" || { echo "LEG $LEG: boot timeout"; return 1; }

  # announce gate: presence must match the leg's intent exactly, both directions
  local SAW=0; grep -qF "$ANNOUNCE" "$LOG" && SAW=1
  if [ "$SAW" != "$WANT" ]; then
    echo "LEG $LEG: ANNOUNCE GATE FAILED (saw=$SAW want=$WANT) -- env not honoured"; return 1
  fi
  echo "LEG $LEG: announce ok (gated=$SAW)"

  local GOT
  GOT=$(curl -s "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
    -d "{\"input_ids\": $FIX_PROMPT, \"sampling_params\": {\"temperature\": 0.0, \"max_new_tokens\": $(python3 -c "print(len($FIX_EXPECT))"), \"ignore_eos\": true}}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["output_ids"])')
  [ "$GOT" = "$FIX_EXPECT" ] || { echo "LEG $LEG: GREEDY SMOKE FAILED: got $GOT"; return 1; }
  echo "LEG $LEG: smoke EXACT"

  python3 "$REPO/bench/bsz_throughput.py" --port "$PORT" --concurrencies 1 \
    --in-len 64 --out-len 256 --out "$OUT/c1_${TAG}_${LEG}.json" 2>&1 | tail -2
}

run_leg L0 0 RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=1
run_leg L1 0 RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=1 RWKV_ADDLN_WIDE=1
run_leg L2 1 RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=1 RWKV_ADDLN_WIDE=1 RWKV_FUSED_LORA_GATED=1
pkill -f sglang.launch_server 2>/dev/null
echo "LADDER DONE -> $OUT"
