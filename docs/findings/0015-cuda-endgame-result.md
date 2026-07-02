---
doc_kind: finding
finding_id: F0015
title: "CUDA endgame result: fused fp16 GEMV is greedy-EXACT and +5-9% bsz1 decode at 1.5B/7.2B, but cuda-graph amortizes the launch-overhead win; closing the albatross gap needs mega-kernel fusion (rejected as un-elegant)"
last_verified_commit: "9901a9f"
discovered_by: lead (M6), 2026-07-01
severity: info
status: open
related: [F0007, F0013, F0014]
---

# Finding F0015: the CUDA endgame reached its honest ceiling

## What was built (greedy-EXACT + batch-invariant, FLA-free)
`rwkv7_kernels/cuda/rwkv7_fast.cu` — one hand fp16 decode kernel adapted from
BlinkDL/Albatross (Apache-2.0, re-attributed in cuda/NOTICE): `gemv_m1` (M==1 exact
GEMV for the r/k/v/o + ffn projections). Wired into `models/rwkv7.py::_proj_gemv`
behind `RWKV_FAST_LINEAR=1`, engaged ONLY when M==1 + fp16 activation + fp16
contiguous weight + K%4==0 + N even (else the quant-aware ReplicatedLinear/cuBLAS
path — never crashes on an odd shape). JIT-built via `fast_linear.py` (CUDA 12.9,
sm_86), **without `--use_fast_math`** (IEEE arithmetic, no FTZ / approx transcendentals),
so the gates hold on defensible math. (An earlier draft also vendored fused `lora_down`/
`lora_up` kernels; they were **removed** — dead code, since the LoRA path uses the
batch-invariant triton `grouped_gemm`. Only the measured `gemv_m1` ships.)

## Gate (met)
- Standalone numerics (`bench/verify_fast_linear.py`): `gemv_m1` matches an fp32 torch
  reference to the SAME ULP as torch's own fp16 matmul (rel err ~2-6e-4).
- End-to-end greedy-EXACT with **cuda-graph ON, fp16**: 0.1B 24/24, 1.5B 24/24,
  7.2B 8/8 (`bench/verify_m1d.py --dtype float16 --cuda-graph`, RWKV_FAST_LINEAR=1).
- **Batch-invariant with the flag ON** (`bench/verify_batch.py`, fp16, cuda-graph ON,
  RWKV_FAST_LINEAR=1): 0.1B + 1.5B both PASS (identical / shared-prefix / mixed batches
  all exact vs their B=1 references). So although `gemv_m1` (B==1) and cuBLAS (B>1) are
  different kernels, the greedy output does not diverge on the fixtures — no worse than
  the baseline (cuBLAS is itself only empirically batch-invariant); the fp32-accum GEMV
  is in fact MORE batch-stable than the bf16 `_FUSE_LORA` grouped-gemm was (which failed
  this gate at 0.1B). Strict bit-level cross-load determinism is still not *guaranteed*
  (two kernels), the same caveat as the accepted cuBLAS default.

## The measured result (the honest ceiling) — cuda-graph ON, decode tok/s, RTX 3090
| size | bsz | baseline (cuBLAS) | fast (gemv_m1) | fast/base |
|---|---|---|---|---|
| 0.1B | 1 | 585.6 | 557.3 | **0.95x** (regression) |
| 1.5B | 1 | 159.9 | 167.6 | **1.05x** |
| 7.2B | 1 | 45.9  | 49.9  | **1.09x** |
| (bsz8, any size — fast does not engage at M>1) | | | | ~1.00x |

vs albatross-fp16 (79.6 @7.2B bsz1): the gap narrows from **0.58x -> 0.63x**, still a loss.
(`bench/results/fast_linear/{base,fast}_*.json`.)

## Why the standalone 1.1-1.6x did NOT translate
`verify_fast_linear.py` micro-benched in **eager** mode and showed gemv_m1 1.09-1.61x
faster than cuBLAS at M=1. That advantage is almost entirely **kernel-launch + cuBLAS
heuristic-dispatch overhead**, which **cuda-graph eliminates** (the production config
captures the whole forward and replays only GPU work). Under graph replay the
comparison is pure compute, where cuBLAS's tuned M=1 kernel is competitive; our win
shrinks to +5-9% at 1.5B/7.2B (real bandwidth work) and inverts at 0.1B (so tiny that
the extra fixed cost of many small blocks loses to cuBLAS). **Lesson: never quote an
eager micro-bench as a cuda-graph speedup.**

## Why we stop here (the mega-kernel trade)
Swapping individual GEMVs cannot close the albatross gap because it does not reduce
the number of **global-memory round-trips**: our decode is many separate kernels
(lerp -> proj -> LoRA -> WKV -> norm -> gate -> o_proj), each reading/writing
intermediates to HBM. albatross fuses the entire time-mix into whole-forward
WMMA/cublasLt mega-kernels that keep intermediates on-chip, and it is already at
~92% of the 3090's bandwidth peak. Matching it requires the same **whole-time-mix
mega-kernel fusion**, which:
- destroys the clean, maintainable, first-class sglang integration (the user's
  "优雅" / elegance bar) — it becomes an opaque monolith;
- is high-risk to keep greedy-EXACT across dtypes/GPUs;
- buys raw-kernel latency that only matters for **single-stream, static-batch**
  decode — the one regime where albatross is a no-serving micro-bench and we already
  win on the axes that matter in production.

## Decision
- Keep `gemv_m1` as a **verified, opt-in** path (`RWKV_FAST_LINEAR=1`), documented
  with these exact per-size numbers. Default OFF (matches the fully reproducible cuBLAS
  baseline; avoids the 0.1B regression). The exploratory fused-LoRA kernels were removed
  (dead code — the LoRA path stays on the batch-invariant triton grouped_gemm).
- **Do NOT** claim we match/beat albatross on same-precision raw kernel speed. The
  honest standing (F0014) stands: albatross wins raw fp16 latency; we win int8
  speed, VRAM, and real serving (dynamic batching / chunked prefill / state cache).
- The mega-kernel remains a documented, deliberately-declined option, not a TODO we
  pretend is coming.

## Cross-references
[[F0014]] honest standing · [[F0013]] elementwise fusion ceiling · [[F0007]] albatross
baseline · `bench/verify_fast_linear.py` · `bench/results/fast_linear/`.
