---
doc_kind: finding
finding_id: F0020
title: "Fused LoRA kernel (lora4_m1): all four LoRA chains in 2 launches — fp16 bsz1 decode 203.0 → 226.5 tok/s (+11.6%), greedy 24/24 EXACT; per-component profile shows lm_head is now 58.5% of the graphed step (the fp16 bandwidth wall)"
last_verified_commit: "HEAD"
discovered_by: lead (M11), 2026-07-03
severity: info
status: open
related: [F0017, F0018]
---

# Finding F0020: fused LoRA chains (bsz1 fp16 decode)

## What
`rwkv7_lora.cu` op `rwkv7_lora::lora4_m1`: the per-layer w/a/g/v LoRA chains
(each = down-GEMM → activation → up-GEMM [+bias]; ~12 kernels total) collapse
into **2 launches** (stage1: all chains' down+act, one block per down-row;
stage2: all chains' up+bias, warp per output, rank-innermost packed layout for
coalesced loads). fp32 accumulation, IEEE, deterministic order; torch's fp16
intermediate roundings reproduced at both chain points, so residual vs torch is
~1 fp16 ULP of reduction-order drift (measured ≤3.0e-4 rel, identical to
torch16-vs-fp32 at 3 digits). Packing built lazily from the loaded
ReplicatedLinear weights (mirrors `_mix6`); `xs` is a zero-copy view of
`fused_lerp6`'s output block. Opt-in `RWKV_FUSED_LORA=1`; eligible only on
bsz1 + fp16 + tp1 + unquantized; every other path untouched.

## Measured (1.5B, RTX 3090, cuda-graph ON, radix OFF)
*1.5B · fp16 · bsz1 · tp1 · unquantized · RTX 3090 · cuda-graph ON · radix OFF · greedy 24/24 EXACT · RWKV_FUSED_LORA=1 eligible only on bsz1+fp16+tp1+unquantized*
| config | greedy | decode bsz1 tok/s |
|---|---|---|
| best w/o fused lora (in-place WKV + sparse FFN + fast GEMV) | EXACT 24/24 | 203.0 |
| **+ RWKV_FUSED_LORA=1** | **EXACT 24/24** | **226.5 (+11.6%)** |

Raw transcript (gate + fused run + same-session control):
`bench/results/headline/raw/fused_lora_gate_and_throughput.log`.

Micro: fused 2-launch vs torch chain (eager) 15.5–19.7×; the honest e2e gain is
the +23.5 tok/s above (launch count is what it removes under cuda-graph).

Context (same card, same base checkpoint): the competitor VKWR (Albatross
faster3a kernels) measures 224.6 tok/s bsz1 fp16 server-wall-clock — our fp16
single-stream now clears it in engine steady-state, with quant still on top
(w8 227.4 lossless / w4 259.1).

## Per-component profile (bsz1, graphed μs/layer except head)
lm_head **315.9 (58.5%)** · ffn 61.3 · loras 43.4 (pre-fusion) · rkv 38.6 ·
lerp 14.8 · o_proj 14.1 · kk/l2norm 11.8 · token_shift 10.8 · norms 8.5 ·
gate_corr 7.5 · g_norm 4.4 · wkv 3.9. lm_head = 268 MB fp16 read ≈ 91% of the
3090's bandwidth already — **the remaining fp16 wall is the head, not the
layers**. bsz32: lm_head 335.7 (47%), ffn 118, rkv 67, loras 47, wkv 46.

## Next levers (from the profile)
1. small-M extension of lora4 (bsz≤8: loras are 47 μs/layer at bsz32 too).
2. glue mega-fusion: token_shift+lerp6, gate_corr+g_norm (~20-25 μs/layer).
3. head: int8 lm_head would cut ~150 μs (+7%) but risks argmax drift — only
   behind its own gate, never default.

## Cross-references
[[F0017]] · [[F0018]] · `bench/verify_lora_fused.py` ·
`rwkv7_kernels/lora_fused.py` · `cuda/rwkv7_lora.cu`.
