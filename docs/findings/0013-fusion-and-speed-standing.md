---
doc_kind: finding
finding_id: F0013
title: "Elementwise fusion (+5-11% decode, EXACT) + the bit-exact↔speed ceiling; speed standing vs albatross"
last_verified_commit: "27e3fe7"
discovered_by: fusion agent + lead verify, 2026-06-30
severity: info
status: open
related: [F0008, F0011, F0007]
---

# Finding F0013: Fusion + the speed standing / bit-exact ceiling

## Result — elementwise fusion (shipped, EXACT)
3 triton kernels (token-shift 6-lerp; kk+k-update+L2norm; gate-correction+residual+gate-mul)
collapse ~40 glue kernels → **78→54 kernels/layer**. `enable_fp_fusion=False` keeps it bit-exact
(triton FMA-contraction otherwise flips knife-edge argmaxes). Gates EXACT (0.1B/1.5B bf16+fp32) +
verify_batch PASS (lead-verified). **Decode: bf16 1.5B +9-11% / 7.2B +4-5%; int8 +5-16%.**
BW util: 1.5B 43%→48%, 7.2B 66%→69%.

## KEY FINDING — the bit-exact ↔ speed ceiling
Matmuls are **77-91%** of the decode step; the big "42%→90%" win needs fusing THEM (LoRA
batching, WMMA/cublasLt GEMMs). But ANY matmul reorder perturbs values ~1 bf16-ULP, and because
the cuBLAS r/k/v/o/ffn linears are themselves **batch-variant**, that flips a knife-edge argmax →
**breaks the strict bit-exact `verify_batch` gate**. The LoRA-batching kernel was BUILT, verified
to break the gate (+ no bsz1 win), and correctly **dropped** (`RWKV_FUSE_LORA=1` to enable). So:
> **Bit-exact-greedy caps fusion at the elementwise subset (~+5-11%). Full matmul-fusion speed
> parity requires switching the accuracy gate from bit-exact-greedy to lm-eval-metric-parity**
> (which is what "达到 rwkv-lm 精度" actually means and what albatross itself does — fp16 fused
> kernels with small numeric drift, matched at the eval-metric level, not bit-for-bit).

## Speed standing vs albatross (3090, decode bsz1; **NOTE: re-measure clean — see rigor note**)
| size | ours (cuda-graph+int8+fusion) | albatross | ≈ratio |
|---|---|---|---|
| 7.2B | ~70-72 | 77 | **~93%, AT the dense bandwidth physical ceiling (~67 dense / int8 higher)** |
| 1.5B | ~175 (int8) | 309 | ~57% |
| 0.1B | ~ (overhead-bound) | 1172 | lowest |
**At 7.2B (albatross's headline size) we are ~at parity AND bit-exact AND lower-VRAM AND a real
serving system.** Small models are overhead/matmul-bound — closing them needs the lm-eval-gated
fast matmul kernels. To EXCEED albatross at 7.2B needs **activation-sparse FFN** (its trick; only
way past the dense bandwidth ceiling).

## Next (the speed roadmap)
1. **M-rigor**: clean unified re-benchmark (definitive defensible numbers) + **lm-eval (MMLU/
   lambada) vs rwkv-lm** — establishes the lm-eval-parity gate.
2. **lm-eval-gated fast kernels**: LoRA batching + WMMA/cublasLt matmul fusion (small models) +
   **activation-sparse FFN** (exceed albatross at 7.2B) — gated on lm-eval-parity, not bit-exact.

## Cross-references
[[F0008]] cuda-graph · [[F0011]] int8 · [[F0007]] albatross baseline · snapshot rigor note.
