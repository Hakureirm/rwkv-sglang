#!/bin/bash
# Cross-implementation gate: the upstream numpy reference against our server.
#
# Every other correctness gate in this repo compares our code to our code -- a
# kernel against the torch chain it replaces, a batched path against the M==1
# path, a new binary against the old one. Those catch regressions and cannot
# catch a shared mistake. This one does not share anything:
#
#   reference  bench/oracle_numpy.py, a faithful port of BlinkDL's own
#              RWKV-v7 numpy forward, invoked through its CLI UNMODIFIED.
#              Do not "fix" it to make this pass -- it stops being a reference
#              the moment it is edited for that reason.
#   prompts    taken from a dataset at a fixed stride, so the cases are not
#              chosen by whoever wrote the change under test.
#   coverage   four concurrencies, so the T==1, 2<=T<=gate and T>gate branches
#              are all exercised against the same single-stream ground truth.
#
# Usage:
#   PTH=/path/rwkv7-g1g-1.5b.pth SERVED=/path/rwkv7-1.5b-fla \
#   PARQUET=/path/lambada_test.parquet SGLANG=/path/sglang-checkout/python \
#   bash bench/xcheck_numpy_vs_server.sh
#
# The reference pass is slow (numpy, fp32, CPU: ~1 min per prompt at 1.5B) and
# is cached in $OUT/reference.json; delete it to regenerate.
set -uo pipefail

PTH="${PTH:?set PTH=/path/to/rwkv7-*.pth}"
SERVED="${SERVED:?set SERVED=/path/to/served-model-dir}"
PARQUET="${PARQUET:?set PARQUET=/path/to/lambada_test.parquet}"
SGLANG="${SGLANG:-}"
OUT="${OUT:-./_xcheck}"
NTOK="${NTOK:-12}"
NPROMPT="${NPROMPT:-24}"
STRIDE="${STRIDE:-20}"
PORT="${PORT:-30089}"
mkdir -p "$OUT"

[ -n "$SGLANG" ] && export PYTHONPATH="$SGLANG"
eval "$(grep -oE '^export RWKV_[A-Z_]+=[0-9]+' scripts/serve.sh | tr '\n' ';')"
echo "server tree : $(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')"
echo "RWKV flags  : $(env | grep -c '^RWKV_')"

# ---- prompts, by stride, not by choice ----
python3 - "$PARQUET" "$OUT/prompts.txt" "$NPROMPT" "$STRIDE" <<'PY'
import json, sys
import pandas as pd
df = pd.read_parquet(sys.argv[1])
col = df.columns[0]
n, stride = int(sys.argv[3]), int(sys.argv[4])
with open(sys.argv[2], "w") as f:
    for i in range(0, n * stride, stride):
        f.write(json.dumps(df.iloc[i][col]) + "\n")
PY
echo "prompts     : $(wc -l < "$OUT/prompts.txt")"

# ---- reference: unmodified CLI, one process per prompt ----
if [ ! -f "$OUT/reference.json" ]; then
  : > "$OUT/reference.jsonl"
  i=0
  while IFS= read -r line; do
    p=$(python3 -c "import json,sys;print(json.loads(sys.stdin.read()),end='')" <<<"$line")
    txt=$(python3 bench/oracle_numpy.py --model "$PTH" --prompt "$p" --n "$NTOK" 2>/dev/null \
          | grep "^greedy text:" | sed "s/^greedy text: //")
    python3 -c "import json,sys;print(json.dumps({'i': $i, 'greedy_repr': sys.argv[1]}))" "$txt" \
        >> "$OUT/reference.jsonl"
    i=$((i+1))
  done < "$OUT/prompts.txt"
  mv "$OUT/reference.jsonl" "$OUT/reference.json"
fi
echo "reference   : $(wc -l < "$OUT/reference.json") rows"

python3 -m sglang.launch_server \
    --model-path "$SERVED" --dtype float16 --trust-remote-code \
    --port "$PORT" --mem-fraction-static 0.50 --page-size 1 \
    --attention-backend triton --disable-radix-cache \
    --cuda-graph-max-bs 512 --chunked-prefill-size 4096 \
    --max-running-requests 512 > "$OUT/server.log" 2>&1 &
pid=$!
ready=0
for _ in $(seq 1 300); do
  sleep 2
  curl -sf "http://127.0.0.1:$PORT/health_generate" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $pid 2>/dev/null || break
done
[ "$ready" = 1 ] || { echo "!!! server never came up"; tail -6 "$OUT/server.log"; exit 1; }

python3 - "$OUT/prompts.txt" "$OUT/reference.json" "$PORT" "$NTOK" <<'PY'
import ast, asyncio, json, sys

import aiohttp

prompts = [json.loads(l) for l in open(sys.argv[1])]
# oracle_numpy prints the completion as a Python repr, and the capture keeps it
# verbatim. Unwrapping it here rather than in the capture keeps the reference
# file a record of exactly what the reference printed.
def unrepr(t):
    try:
        return ast.literal_eval(t)
    except (ValueError, SyntaxError):
        return t

ref = {r["i"]: unrepr(r.get("greedy_repr", r.get("greedy_text", ""))) for r in (json.loads(l) for l in open(sys.argv[2]))}
port, ntok = int(sys.argv[3]), int(sys.argv[4])
URL = f"http://127.0.0.1:{port}/generate"


async def run(conc):
    sem = asyncio.Semaphore(conc)
    async with aiohttp.ClientSession() as sess:
        async def one(text):
            async with sem:
                body = {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": ntok}}
                async with sess.post(URL, json=body) as r:
                    return (await r.json())["text"]
        return await asyncio.gather(*(one(p) for p in prompts))


print(f"{'concurrency':>12} {'match':>8} {'of':>4}   first mismatch")
bad_total = 0
for conc in (1, 4, 8, 16):
    outs = asyncio.run(run(conc))
    bad = [i for i, o in enumerate(outs) if o != ref[i]]
    bad_total += len(bad)
    detail = f"  #{bad[0]}: ref={ref[bad[0]]!r} got={outs[bad[0]]!r}" if bad else ""
    print(f"{conc:>12} {len(prompts)-len(bad):>8} {len(prompts):>4}  {detail}")
print("\nRESULT:", "ALL MATCH" if bad_total == 0 else f"{bad_total} MISMATCHES")
sys.exit(0 if bad_total == 0 else 1)
PY
rc=$?
kill $pid 2>/dev/null; wait $pid 2>/dev/null
exit $rc
