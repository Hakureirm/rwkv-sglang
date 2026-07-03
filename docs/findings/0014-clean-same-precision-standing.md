---
doc_kind: finding
finding_id: F0014
title: "Clean same-precision standing vs albatross (honest): raw speed loses, accuracy ties, VRAM/int8/serving win"
last_verified_commit: "6a21a17"
discovered_by: M-rigor agent + lead, 2026-07-01
severity: info
status: open
related: [F0007, F0011, F0013]
---

# Finding F0014: Clean same-precision standing vs albatross (the honest bar)

## Method
Clean exclusive RTX 3090 (idle 1 MiB, vs the new co-tenant 1304 MiB). Precision-matched
**ours-fp16 (fp32 state) vs albatross-fp16**, ≥7 medianed, one process at a time, reproducible
(`bench/run_clean_comparison.py`). Accuracy via lm-eval (lambada 5153, MMLU 2000/seed42) ours
vs the `rwkv` pip reference on the same `.pth`; albatross greedy-drift via `bench/albatross_accuracy.py`.

## Result (the honest, defensible standing)
| axis | number | verdict |
|---|---|---|
| **raw speed (fp16=fp16)** | decode ours/alb 0.46-0.85× · prefill 0.16-0.83× (gap shrinks with size) | **albatross wins** (hand-tuned WMMA/cublasLt CUDA at ~92% BW peak) |
| **accuracy** | lambada ours-fp16 0.673 vs ref 0.671; MMLU 0.524 vs 0.511 (PARITY); albatross-fp16 ALSO greedy-EXACT on fixtures | **TIE** (both match rwkv-lm; NO accuracy win either way — earlier "fp16 drifts" hypothesis RETRACTED) |
| **VRAM** | ours flat in batch; albatross static B×T → 24GB near-OOM @7.2B bsz32 | **ours wins** |
| **int8** (albatross lacks; cross-precision bonus) | @7.2B ours-int8 vs alb-fp16: decode 0.90/1.21/0.88×, prefill 1.21/1.70/1.18×, −46% weight bytes (−19% reserved pool); drift −0.9/−2.2pt | **ours wins** (feature + 7.2B speed) |
| **serving** | dynamic batching / chunked prefill / state cache | **ours only** (albatross is a static-batch micro-bench) |

## Conclusion & decision
"同精度吊打速度和精度" is **NOT met on raw speed (we lose) nor accuracy (tie)**. Ours wins
VRAM/int8/serving. albatross's fp16 decode is already ~92% of the 3090's bandwidth peak, so
raw same-precision domination is very hard — a CUDA effort realistically reaches ~parity.
**User decision (2026-07-01): go ALL-IN on the CUDA endgame to chase raw speed** (accept the
big/uncertain effort). Plan:
1. **activation-sparse FFN CUDA** (highest leverage; FFN is 38-51% of decode; `sqrelu=relu²`
   is exact-zero → skipping those value-proj columns is **bit-exact** → speed WITHOUT accuracy
   loss; the way past the dense-BW ceiling). Reference: albatross "no-fc" (Apache-2.0).
2. **fused fp16 CUDA** for r/k/v/o + LoRA + epilogues (peak BW, fewer launches).
3. **chunked/tensor-core WKV CUDA** for prefill (small-model prefill is 0.16-0.25× — the WKV
   sequential scan is the killer). Keep fp32 state throughout (accuracy edge).
Build: `/usr/local/cuda-12.9`, TORCH_CUDA_ARCH_LIST=8.6. Gate: greedy EXACT (or lm-eval-parity
if a kernel must approximate) + speed vs albatross-fp16.

## Cross-references
[[F0007]] albatross baseline · [[F0011]] int8 · [[F0013]] fusion/ceiling · `bench/results/{comparison_clean,lm_eval}.md`.
