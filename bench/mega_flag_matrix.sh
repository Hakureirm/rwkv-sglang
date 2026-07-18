#!/bin/bash
# #50 sm120 flagship matrix: bsz1 serving A/B across the megakernel-line flags.
#
# For each leg: boot the 7.2B fp16 server (full W1' stack + RWKV_STATE_FP16 =
# the legFinal_B anchor config) with the leg's extra flags, wait ready, greedy
# smoke (fixture 8/8 EXACT — hard gate, abort leg on mismatch), then the
# standard 64-in/256-out c=1 sweep (bench/bsz_throughput.py), kill, next.
# Legs:
#   A  anchor        (MEGA=0 WKV_CUDA=0 PDL=0)   expect ~133-134 clean-card
#   B  +MEGA         (grouped r/k/v + o role)
#   C  +MEGA+WKV     (hand-CUDA WKV decode)
#   D  +MEGA+WKV+PDL (the assembled PDL chain)    <- flagship config
# Usage (inside the serving container):
#   bash bench/mega_flag_matrix.sh /models/rwkv7-7.2b-fla 30070 /data/out_dir 72b
#   bash bench/mega_flag_matrix.sh /models/rwkv7-1.5b-fla 30070 /data/out_dir 15b
set -uo pipefail

MODEL="${1:?model dir}"
PORT="${2:-30070}"
OUT="${3:?output dir}"
TAG="${4:?tag (72b|15b)}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"

# Fixture selection by tag FAMILY (prefix match). The original exact-string
# `[ "$TAG" = 72b ]` silently fed the 1.5B fixture to any decorated tag like
# "72b_f0064" — every leg then "fails" the greedy gate on a wrong-fixture
# comparison (found during the F0064 flagship run). Unknown tags now fail loud.
case "$TAG" in
  72b*)
    FIX_PROMPT='[11, 6699, 304, 25740, 109, 37480, 4600, 52151, 4596, 22590, 30449, 4706]'
    FIX_EXPECT='[37138, 45, 44312, 47, 11, 6699, 304, 25740]'
    ;;
  15b*)
    FIX_PROMPT="$(python3 - <<EOF
import json; fx=json.load(open("$REPO/bench/fixtures/oracle_rwkv7_15b_eiffel.json")); print(fx["prompt_tokens"])
EOF
)"
    FIX_EXPECT="$(python3 - <<EOF
import json; fx=json.load(open("$REPO/bench/fixtures/oracle_rwkv7_15b_eiffel.json")); print(fx["greedy_tokens"])
EOF
)"
    ;;
  *) echo "mega_flag_matrix: unknown tag '$TAG' (want 72b*|15b*)" >&2; exit 1 ;;
esac

run_leg() { # $1 leg name, $2 extra env (space-separated K=V)
  local LEG="$1"; shift
  local LOG="$OUT/serve_${TAG}_${LEG}.log"
  echo "=== LEG $LEG ($*) ==="
  pkill -f sglang.launch_server 2>/dev/null; sleep 4
  ( cd "$REPO" && env MODEL="$MODEL" PYTHON=python3 PORT="$PORT" MEMFRAC=0.85 \
      CGMAXBS=32 RWKV_STATE_FP16=1 "$@" \
      bash scripts/serve.sh -- --max-running-requests 32 > "$LOG" 2>&1 ) &
  for i in $(seq 1 120); do
    grep -q "The server is fired up and ready to roll" "$LOG" 2>/dev/null && break
    if grep -qE "Traceback|ValueError|CUDA out of memory" "$LOG" 2>/dev/null; then
      echo "LEG $LEG: BOOT FAILED"; tail -5 "$LOG"; return 1
    fi
    sleep 5
  done
  grep -q "fired up" "$LOG" || { echo "LEG $LEG: boot timeout"; return 1; }
  # greedy smoke: fixture EXACT, hard gate
  local GOT
  GOT=$(curl -s "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
    -d "{\"input_ids\": $FIX_PROMPT, \"sampling_params\": {\"temperature\": 0.0, \"max_new_tokens\": $(python3 -c "print(len($FIX_EXPECT))"), \"ignore_eos\": true}}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["output_ids"])')
  if [ "$GOT" != "$FIX_EXPECT" ]; then
    echo "LEG $LEG: GREEDY SMOKE FAILED: got $GOT expect $FIX_EXPECT"; return 1
  fi
  echo "LEG $LEG: smoke EXACT"
  # co-residency snapshot before/after the timing
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > "$OUT/gpu_${TAG}_${LEG}_pre.txt"
  python3 "$REPO/bench/bsz_throughput.py" --port "$PORT" --concurrencies 1 \
    --in-len 64 --out-len 256 --out "$OUT/c1_${TAG}_${LEG}.json" 2>&1 | tail -2
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > "$OUT/gpu_${TAG}_${LEG}_post.txt"
}

# headline endpoints first (interruption-resilient: A/D are the flagship pair,
# B/C are attribution), sky-yield checked between legs by the caller.
run_leg A RWKV_MEGA=0 RWKV_WKV_CUDA=0 RWKV_PDL=0
run_leg D RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=1
run_leg B RWKV_MEGA=1 RWKV_WKV_CUDA=0 RWKV_PDL=0
run_leg C RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=0
pkill -f sglang.launch_server 2>/dev/null
echo "MATRIX DONE -> $OUT"
