---
doc_kind: finding
finding_id: F0010
title: "M3b — deliverable is 100% FLA-free (own WKV kernel for decode+prefill), zero speed cost"
last_verified_commit: "c976c87"
discovered_by: M3b agents + lead independent verify, 2026-06-30
severity: info
status: closed_by_M3b
related: [F0008, F0009]
---

# Finding F0010: M3b de-FLA complete

## Hypothesis
ADR-0004 requires a FLA-free deliverable. Replacing the vendored FLA triton kernels
(decode + chunked prefill) with our own kernel should keep greedy EXACT and not cost speed.

## Method
- Step 1: self-written triton `wkv_recurrent` kernel (authored from the RWKV-7 math, not
  copied from FLA) → replaced `fused_mul_recurrent_rwkv7` for decode + recurrent-prefill.
- Step 2: routed the extend/prefill branch through `wkv_recurrent` too, dropped the
  `chunk_rwkv7` import, deleted the 10 vendored FLA overlay files. Lead-verified clean deploy.

## Result
- **100% FLA-free** (ADR-0004 gate met): no `fla.rwkv7`/`fla.dplr`/`chunk_rwkv7`/
  `fused_mul_recurrent_rwkv7` imports in our RWKV path — only docstring prose mentions FLA.
  Overlay `fla/` dir deleted. (The 2 remaining `...fla...` imports in `server_args.py`/
  `attention_registry.py` are **upstream sglang's own** gated-delta/mamba code, not ours,
  not on the RWKV path.)
- **Correctness preserved** (lead-verified on clean deploy): verify_m1d EXACT 24/24
  (0.1B+1.5B, bf16 cuda-graph + fp32); verify_batch PASS; kernel gate vs naive fp32 worst
  err 1.3e-6.
- **Zero speed cost**: dropping the FLA chunk kernel did NOT regress prefill — our recurrent
  kernel is faster end-to-end on tested sizes (the chunk path's many per-layer sub-kernel
  launches outweigh its scan advantage; chunk only wins as an isolated kernel at T≥4096,
  and not end-to-end there either). comparison.md prefill numbers stand (they were already
  recurrent-based). A subtle bf16 batched-vs-B1 argmax divergence (M3b-1) was fixed by
  pinning the bit-exact triton config per dtype (decode BV=32/nw=4, varlen BV=16/nw=4).
- albatross LICENSE = Apache-2.0 (redistributable, BlinkDL's own) — available if a future
  tensor-core prefill kernel is wanted for extreme contexts (not needed now).

## Conclusion
The RWKV-7 × sglang deliverable is fully self-contained (own WKV kernel) and FLA-free,
satisfying ADR-0004 (political: Bo/BlinkDL vs FLA) with NO accuracy or speed cost. Decode
and prefill both run our `rwkv7_kernels/wkv_recurrent.py`.

## Cross-references
ADR-0004 · [[F0008]] (cuda-graph) · [[F0009]] (comparison/7.2B) ·
`sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/wkv_recurrent.py`.
