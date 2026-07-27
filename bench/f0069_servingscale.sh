#!/bin/bash
# F0069 #59: the README's 1.5B bsz1 figure lives in the serving_scale convention
# (ctx-1024 steady-state, prefill-subtracted, offline Engine), NOT the 64-in/256-out
# c=1 convention that the §7a flagship ladder uses. Replacing one with the other
# would silently mix conventions, so the flagship is re-measured HERE, in the
# README's own convention, alongside in-session reproductions of the two values it
# would replace.
#
#   W1        FAST_LINEAR+SPARSE_FFN+FUSED_LORA+FUSED_GLUE+GEMV_AUTOTUNE
#             = the exact flag set announced in bench/results/ladder_full_5090.log
#             published 409.8
#   W1_SFP16  W1 + RWKV_STATE_FP16                       published 447.3 (F0056)
#   L2        full stack + megakernel + WIDE + LoRA-gate epilogue + STATE_FP16
#             = the current flagship, never yet measured in this convention
set -uo pipefail
MODEL="${1:?model dir}"; OUT="${2:?out dir}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"

W1="RWKV_FAST_LINEAR=1 RWKV_SPARSE_FFN=1 RWKV_FUSED_LORA=1 RWKV_FUSED_GLUE=1 RWKV_GEMV_AUTOTUNE=1"
FULL="$W1 RWKV_FUSED_GATES=1 RWKV_FUSED_SQRELU=1 RWKV_FUSED_ADDLN=1 RWKV_FUSED_GNGC=1 RWKV_FUSED_RELUSQ=1 RWKV_FUSED_VRESGATE=1"
MEGA="RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=1 RWKV_ADDLN_WIDE=1 RWKV_FUSED_LORA_GATED=1"

run() { # $1 leg, rest: env
  local LEG="$1"; shift
  echo "=== LEG $LEG ==="
  env "$@" python3 "$REPO/bench/serving_scale.py" --model "$MODEL" --dtype float16 \
    --mode batch --context 1024 --batch-sizes 1 --decode-tokens 64 \
    --mem-fraction 0.85 2>&1 | tee "$OUT/${LEG}.log" | grep -E "SERVING-SCALE|^ *1024|\[rwkv7\]"
}

run W1        $W1
run W1_SFP16  $W1 RWKV_STATE_FP16=1
run L2        $FULL $MEGA RWKV_STATE_FP16=1
echo "SERVING-SCALE LEGS DONE -> $OUT"
