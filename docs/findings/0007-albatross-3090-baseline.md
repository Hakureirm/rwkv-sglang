---
doc_kind: finding
finding_id: F0007
title: "Albatross speed baseline on our 3090; quantified gap; kernel-vendoring path"
last_verified_commit: "3eea259"
discovered_by: M3a agent + lead, 2026-06-30
severity: info
status: open
related: [F0006, F0003]
---

# Finding F0007: Albatross 3090 baseline + the speed gap

## Hypothesis
"Match albatross speed/VRAM" needs albatross re-measured on OUR 3090 (its published
numbers are 5090), apples-to-apples with our sglang impl.

## Method
Built BlinkDL/Albatross `faster3a_2605/rwkv7_fast_v3a.py` (custom fp16 CUDA: WMMA/
cublasLt GEMMs, sparse-FFN, fused WKV; static-batch CUDAGraph) on `gpu-box` and
benchmarked RWKV-7 0.1B + 1.5B. Full recipe + methodology in
`bench/results/albatross_3090.md`. **NEW box fact**: a full CUDA 12.9 toolkit is at
`/usr/local/cuda-12.9` (nvcc present; just not on PATH) — custom CUDA compiles for
sm_86 via `TORCH_CUDA_ARCH_LIST=8.6`.

## Result (3090; albatross fp16 kernel-only vs our sglang bf16, cuda-graph OFF)
| model | bsz | metric | albatross | ours | gap |
|---|---|---|---|---|---|
| 0.1B | 1 | decode tok/s | 1171.6 | 20.6 | ~57× |
| 0.1B | 32 | decode tok/s | 24522 | 665 | ~37× |
| 0.1B | 1 | prefill tok/s | 69714 | 8116 | ~8.6× |
| 1.5B | 1 | decode tok/s | 309.1 | 10.5 | ~29× |
| 1.5B | 1 | prefill tok/s | 14646 | 4149 | ~3.5× |

## Conclusion — the gap decomposes into two parts
1. **Launch-overhead (decode, cuda-graph OFF)**: most of the ~30–57× decode gap is
   our eager mode. **M2b cuda-graph** (in flight) should recover the bulk.
2. **Kernel-quality (residual + prefill 3.5–8.6×)**: albatross's hand-tuned fp16
   CUDA vs our fla-triton. Two routes: (a) optimize/tune the triton kernels, or
   (b) **vendor albatross's WKV/linear CUDA kernels into our sglang backend** —
   PROVEN viable (they compile+run on the 3090) and the same approach the closed
   vLLM PR #46269 (LateranLab) took. (b) is the likely path to true parity.

Caveats for fairness: albatross times the forward only (no sampler/scheduler/
EOS, static shape, embedding on CPU) — an upper-bound kernel number, not a serving
system. Our sglang adds real serving value (dynamic batching, chunked prefill,
state cache) albatross lacks. VRAM is ~constant in context (RNN), matching the
constant-VRAM design goal. fp16 (albatross) vs bf16 (ours) ≈ bandwidth-equal.

## Next
M2b cuda-graph → re-measure decode gap → decide kernel route (likely vendor
albatross fp16 WKV/linear kernels behind sglang). Then 7.2B benchmark + lm-eval.

## Cross-references
[[F0006]] our baseline · [[F0003]] acceptance grid · `bench/results/albatross_3090.md` ·
build env: `scripts/box_env.sh` (CUDA_HOME).
