#!/usr/bin/env bash
# F0068 add_ln tier x FastTree sweep (task #57). Runs each config in its own
# process (the launcher latches RWKV_ADDLN_* in a process-static getenv) and
# lands one JSON line per config plus the graph-node dispatch floor probe.
#
#   bash addln_sweep.sh <cuda_dir> <out_dir> [reps]
set -euo pipefail
CUDA_DIR="${1:?cuda dir}"
OUT="${2:?out dir}"
REPS="${3:-50}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"
: > "$OUT/addln_sweep.jsonl"

for W in 0 1 2 3; do
  for F in 0 1; do
    for N in 2048 4096; do
      RWKV_ADDLN_WIDE=$W RWKV_ADDLN_FASTTREE=$F \
        python3 "$HERE/bench_addln_configs.py" --hidden "$N" --reps "$REPS" \
          --cuda-dir "$CUDA_DIR" 2>/dev/null | tail -1 >> "$OUT/addln_sweep.jsonl"
    done
  done
done

RWKV_ADDLN_WIDE=1 RWKV_ADDLN_FASTTREE=1 \
  python3 "$HERE/bench_kernel_floor.py" --cuda-dir "$CUDA_DIR" --reps "$REPS" \
    2>/dev/null | tail -1 > "$OUT/kernel_floor.json"

python3 - "$OUT/addln_sweep.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
name = {"0": "parity(32,4)", "1": "WIDE(32,16)", "2": "WIDER(32,32)", "3": "SPLIT(NB blocks)"}
print(f"{'tier':<14}{'fasttree':>9}{'N':>7}{'us/call':>10}")
for r in sorted(rows, key=lambda r: (r["N"], r["wide"], r["fasttree"])):
    print(f"{name[r['wide']]:<14}{r['fasttree']:>9}{r['N']:>7}{r['us_per_call_p50']:>10.3f}")
PY
