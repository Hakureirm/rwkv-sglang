#!/bin/bash
# F0069: serving_scale.py measures throughput and gates NOTHING, so the W4/W8 legs
# arrived ungated. A quantized tier cannot be gated against the fp16 fixture -- int4
# is documented as lossy (-24pt MATH500), so a mismatch there would prove nothing.
# What CAN be asserted is that turning on the megakernel flag set does not change
# what the quantized model emits: same model, same prompt, greedy, W1 vs L2 must be
# token-identical. That is the property the throughput claim actually rests on.
#
# w8g64 additionally carries a "greedy-lossless" claim in §4, so it is also checked
# against fp16 -- a real, falsifiable extra gate. And fp16 L2 is checked against the
# fixture's recorded oracle tokens, which closes the fp16 rung too.
#
# Every comparison first requires BOTH files to be non-empty. The first cut of this
# script did not, every leg died at engine init, and `diff` duly reported three
# passing gates on three empty files.
set -uo pipefail
OUT="${1:?out dir}"; REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${MODELS:-/ws/models}"   # where the run found them; override to reproduce elsewhere; mkdir -p "$OUT"
FIX="$REPO/bench/fixtures/oracle_rwkv7_15b_eiffel.json"
W1="RWKV_FAST_LINEAR=1 RWKV_SPARSE_FFN=1 RWKV_FUSED_LORA=1 RWKV_FUSED_GLUE=1 RWKV_GEMV_AUTOTUNE=1"
FULL="$W1 RWKV_FUSED_GATES=1 RWKV_FUSED_SQRELU=1 RWKV_FUSED_ADDLN=1 RWKV_FUSED_GNGC=1 RWKV_FUSED_RELUSQ=1 RWKV_FUSED_VRESGATE=1"
MEGA="RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=1 RWKV_ADDLN_WIDE=1 RWKV_FUSED_LORA_GATED=1"

gen() { # $1 tag, $2 model, rest env
  local TAG="$1" MODEL="$2"; shift 2
  env "$@" python3 "$REPO/bench/f0069_gate_gen.py" "$MODEL" "$FIX" "$OUT/greedy_$TAG.json" \
    > "$OUT/greedy_$TAG.log" 2>&1
  echo "$TAG -> $(cat "$OUT/greedy_$TAG.json" 2>/dev/null || echo '<<NO OUTPUT>>')"
}

gen fp16_L2 $MODELS/rwkv7-1.5b-fla                $FULL $MEGA RWKV_STATE_FP16=1
gen w8_W1   $MODELS/rwkv7-1.5b-w8g64  RWKV_W8=1   $W1
gen w8_L2   $MODELS/rwkv7-1.5b-w8g64  RWKV_W8=1   $FULL $MEGA RWKV_STATE_FP16=1
gen w4_W1   $MODELS/rwkv7-1.5b-w4     RWKV_W4=1   $W1
gen w4_L2   $MODELS/rwkv7-1.5b-w4     RWKV_W4=1   $FULL $MEGA RWKV_STATE_FP16=1

echo "=== ADJUDICATION ==="
rc=0
cmp_leg() { # $1 a, $2 b, $3 claim
  local A="$OUT/greedy_$1.json" B="$OUT/greedy_$2.json"
  if [ ! -s "$A" ] || [ ! -s "$B" ]; then
    echo "VOID  $3 -- missing output ($1 $([ -s "$A" ] && echo ok || echo EMPTY), $2 $([ -s "$B" ] && echo ok || echo EMPTY))"
    rc=1; return
  fi
  if diff -q "$A" "$B" >/dev/null; then echo "PASS  $3"; else
    echo "FAIL  $3"; echo "  $1: $(cat "$A")"; echo "  $2: $(cat "$B")"; rc=1
  fi
}
cmp_oracle() { # $1 tag
  local A="$OUT/greedy_$1.json"
  [ -s "$A" ] || { echo "VOID  $1 vs oracle fixture -- no output"; rc=1; return; }
  if python3 -c "
import json,sys
got=json.load(open('$A')); want=json.load(open('$FIX'))['greedy_tokens']
sys.exit(0 if got==want else 1)"; then echo "PASS  $1 == oracle fixture (greedy EXACT)"; else
    echo "FAIL  $1 != oracle fixture"; echo "  got:  $(cat "$A")"; rc=1
  fi
}
cmp_oracle fp16_L2
cmp_leg w8_W1 w8_L2   "w8g64: megakernel flags do not change the output"
cmp_leg w4_W1 w4_L2   "w4:    megakernel flags do not change the output"
cmp_leg w8_L2 fp16_L2 "w8g64 greedy-lossless vs fp16 (the §4 claim)"
echo "GATE DONE (rc=$rc) -> $OUT"
