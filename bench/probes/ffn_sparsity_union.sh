#!/bin/bash
# Is the channel-mix sparsity channel-intrinsic (prunable, batchable) or input-dependent
# (dead on arrival at batch)? The union over rows is the discriminator: with ~90% per-row
# zeros, independence predicts 0.9^64 ~ 0.1% union, while shared dead channels predict the
# union stays near the per-row rate.
#
# Two traps already hit and avoided here: cuda-graph capture emits sparsity samples from
# DUMMY inputs (so the graph is disabled), and bench/bsz_throughput.py sends the SAME token
# ids to every concurrent request (so 64 identical inputs made the union trivially equal to
# the per-row rate). This uses 64 distinct prompts and reads only post-boot samples.
set -uo pipefail
REPO="${REPO_DIR:?set REPO_DIR=/path/to/rwkv-sglang checkout}"
PORT=30084
OUT="${OUT_DIR:-/tmp/ffn_union}"
LOG="$OUT/serve.log"
mkdir -p "$OUT"
rm -f "$LOG"

pkill -f sglang.launch_server 2>/dev/null; sleep 6
( cd "$REPO" && env MODEL="${MODEL_DIR:?set MODEL_DIR=/path/to/rwkv7-1.5b}" PYTHON=python3 PORT="$PORT" \
    MEMFRAC=0.60 RWKV_STATE_FP16=1 RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=1 RWKV_LOG_SPARSITY=1 \
    bash scripts/serve.sh -- --disable-cuda-graph > "$LOG" 2>&1 ) &
for i in $(seq 1 180); do
  grep -q "fired up" "$LOG" 2>/dev/null && break
  sleep 5
done
grep -q "fired up" "$LOG" || { echo "BOOT_FAILED"; tail -8 "$LOG"; exit 1; }
echo "server up"

python3 "$(dirname "$0")/union_client.py" "$PORT"

sleep 3
echo "--- post-boot samples, 64 DISTINCT prompts ---"
awk '/fired up/{seen=1} seen' "$LOG" | grep -oE 'rows=64 zero_frac=[0-9.]+ union_frac=[0-9.]+' | tail -10
echo "--- single-row reference ---"
awk '/fired up/{seen=1} seen' "$LOG" | grep -oE 'rows=1 zero_frac=[0-9.]+' | tail -3
pkill -f sglang.launch_server 2>/dev/null
echo "UNION_DONE"
