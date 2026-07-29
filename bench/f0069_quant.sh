#!/bin/bash
# F0069 #59 cont: the README's 5090 cell pairs fp16 with int4, so replacing only the
# fp16 half would put two different stacks in one cell. Both quantized rungs of the
# §3 ladder are re-measured here in the same convention and the same session as the
# fp16 legs: W1 reproduces the flag set the published numbers were taken under, L2
# is the current flagship stack.
set -uo pipefail
OUT="${1:?out dir}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${MODELS:-/ws/models}"   # where the run found them; override to reproduce elsewhere
mkdir -p "$OUT"
W1="RWKV_FAST_LINEAR=1 RWKV_SPARSE_FFN=1 RWKV_FUSED_LORA=1 RWKV_FUSED_GLUE=1 RWKV_GEMV_AUTOTUNE=1"
FULL="$W1 RWKV_FUSED_GATES=1 RWKV_FUSED_SQRELU=1 RWKV_FUSED_ADDLN=1 RWKV_FUSED_GNGC=1 RWKV_FUSED_RELUSQ=1 RWKV_FUSED_VRESGATE=1"
MEGA="RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=1 RWKV_ADDLN_WIDE=1 RWKV_FUSED_LORA_GATED=1"

run() { # $1 leg, $2 model, rest env
  local LEG="$1" MODEL="$2"; shift 2
  echo "=== LEG $LEG ($MODEL) ==="
  env "$@" python3 "$REPO/bench/serving_scale.py" --model "$MODEL" --dtype float16 \
    --mode batch --context 1024 --batch-sizes 1 --decode-tokens 64 --mem-fraction 0.85 \
    2>&1 | tee "$OUT/${LEG}.log" | grep -E "SERVING-SCALE|^ *1024"
}
run w4_W1   $MODELS/rwkv7-1.5b-w4     RWKV_W4=1 $W1
run w4_L2   $MODELS/rwkv7-1.5b-w4     RWKV_W4=1 $FULL $MEGA RWKV_STATE_FP16=1
run w8_W1   $MODELS/rwkv7-1.5b-w8g64  RWKV_W8=1 $W1
run w8_L2   $MODELS/rwkv7-1.5b-w8g64  RWKV_W8=1 $FULL $MEGA RWKV_STATE_FP16=1
echo "QUANT LEGS DONE -> $OUT"
