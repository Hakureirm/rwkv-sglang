---
doc_kind: finding
finding_id: F0011
title: "M4 — w8a8-int8 quant: decode FASTER than bf16 + weight bytes −41-46%, accuracy preserved at scale"
last_verified_commit: "1e35a91"
discovered_by: M4 agent + lead independent verify, 2026-06-30
severity: info
status: open
related: [F0006, F0008]
---

# Finding F0011: M4 quantization (w8a8-int8)

## Hypothesis
Goal: 8/4-bit quant must cut VRAM without being slower than 16-bit. Since
decode is bandwidth-bound (F-profile), int8 weights (½ the bytes/token) should make
decode FASTER, not just smaller.

## Method
Refactored the model's linears (r/k/v/o_proj, ffn key/value, LoRA down/up) to sglang's
quant-aware `ReplicatedLinear` (tp=1) threaded with `quant_config`. WKV recurrence/state +
small per-channel params (x_*, k_k, k_a, r_k, g_norm, norms, emb, lm_head) NEVER quantized
(FLA-free preserved). 8-bit = `w8a8_int8` (per-channel int8 weight, per-token dynamic int8
act, `int8_scaled_mm` on Ampere INT8 tensor cores); offline converter
`tools/quantize_w8a8_int8.py` (closed-form per-channel scales, **no calibration data**).

## Result (RTX 3090)
- **bf16 regression EXACT** (quant_config=None → F.linear, bit-identical): 0.1B/1.5B 24/24,
  lead-verified.
- **decode int8 vs bf16 — FASTER at every bsz**: 1.5B +15% (bsz1) / +34% (bsz8) / +19% (bsz32);
  **7.2B +53% (bsz1: 43.6→66.9) / +47% (bsz8)**. Win grows with scale.
- **Weight bytes −41-46%** (1.5B −41%; 7.2B **−46%** per the safetensors-derived table in
  `bench/results/comparison_clean.md` — 13731→7387 MiB; the earlier ~48% here was a rough
  pre-safetensors estimate). nvidia-smi reserved-pool peak 7.2B −3.3 GB (−19%, mem_fraction-driven).
- **Accuracy (greedy vs numpy oracle)**: **7.2B int8 8/8 EXACT (lead-verified)**; 1.5B int8
  12/24 free-running (diverges tok 12 — int8 drift, shrinks with scale; acceptable at 7.2B).
- **4-bit BLOCKED (honest)**: bnb-nf4 hits a sglang 0.5.10 loader `qweight`/`weight` naming
  bug (affects any ReplicatedLinear model) + bnb is self-admittedly slower; AWQ/GPTQ need
  offline calibration (no HF on the air-gapped box); fp8 needs Hopper (3090=sm_86).

## Conclusion
**w8a8-int8 meets the goal: VRAM↓ AND decode FASTER than bf16** (not merely
not-slower) — and beats upstream rwkv-pip int8 (which is slower than fp16). It's also a
**speed lever** (bandwidth-bound decode), stacking with the upcoming linear-fusion work.
Best quant on Ampere/3090 = w8a8-int8. (int8 decode isn't a full 2× at bsz1 because the
unquantized WKV/token-shift/GroupNorm + per-token act-quant also cost; net is a clear
scale-growing win.)

## Next
Linear FUSION (batch the 8 pathological LoRA GEMVs + collapse ~40 elementwise kernels) on
top of the quant-aware model → stacks with int8 toward ~90% of albatross decode. (4-bit
needs the sglang loader patch — defer / upstream.)

## Cross-references
[[F0006]] [[F0008]] · profiler (`bench/results/profile.md`) · `tools/quantize_w8a8_int8.py` ·
`bench/results/quant.md`.
