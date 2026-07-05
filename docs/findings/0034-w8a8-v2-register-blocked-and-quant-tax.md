# F0034 — w8a8 V2 register-blocked GEMM, the activation-quant tax, and where int8 is decisive

**Date:** 2026-07-06 · **Status:** V2 SHIPPED (bit-exact gate + e2e sweep); V3 fusion tried and rejected · **Prior:** F0033 (sm120 int8 probe), F0018 (w8 kernels)

## What shipped

**V2 register-blocked w8a8 GEMM** (`rwkv7_w8a8.cu`, commits a515ea7 + the V2 e2e
sweep). V1's 32×128 block issued one A-load + two B-loads per k-step for two MMAs
(arithmetic-per-smem-load 0.67) — smem-load-bound at the decode shapes. V2 uses a
64×128 block where each warp holds a 32×32 register tile (2×2 wmma fragments →
arithmetic-per-load 1.0) and a per-warp 16×16 int32 epilogue stage (8 KB vs V1's
16.9 KB full-tile buffer → 3 blocks/SM). The rescale math is unchanged, so **V2 is
bit-identical to V1** (gate: 17/17 exact per algo + a V2==V1 cross-check). The
launcher auto-dispatches V2 at M≥384 and V1 below (V2's larger tile under-fills the
grid at small M).

**Standalone GEMM (RTX 5090, vs fp16 cuBLAS)** — every real projection shape now
exceeds fp16 at M≥512:

| shape | M512 | M1024 | M4096 |
|---|---|---|---|
| attn 2048² | 1.08× | 1.33× | 1.52× |
| ffn.k 8192×2048 | 1.45× | 1.52× | 1.55× |
| ffn.v 2048×8192 | 1.03× | 1.28× | 1.53× |

**e2e serving (1.5B, 5090, cuda-graph ON, 64-in/256-out):** peak **20,991 tok/s @
c512 = 0.9466× fp16** (22,175), up from V1's 20,518 (0.9253×); +0.5–2.3% at every
concurrency.

## Why e2e is 0.9466× when the GEMM is >1× — the honest gap

The int8 GEMM beats fp16, yet e2e stays just under. Measured breakdown at M=512:

- `per_token_quant_int8` measured **~0.014 ms standalone, launch-latency bound**
  (a [512,8192] tensor quantizes in the same time as [512,2048] — 4× the data, same
  wall time → fixed launch cost dominates, not bandwidth).
- In a tight microbench loop the quant launches pipeline and the cost nearly
  vanishes (q+V2 ≈ 0.99–1.41× fp16 at M512). But **real decode runs ~144
  heterogeneous kernels per step with dependencies between them, so the ~144
  per-projection quant launches do NOT pipeline** — the launch latency is paid in
  full, ~4% of a decode step. That, against our own already-excellent fp16
  full-stack (the 22,175 baseline is *our* tuned kernels, not stock), is the gap.

This is an honest ceiling on 1.5B/5090: int8's compute win is real but the residual
activation-quant launch tax plus a very strong fp16 baseline keep e2e at ~0.95×.

## V3 (fused quant-in-GEMM) — tried, rejected

Hypothesis: fuse the per-token quant into the GEMM prologue (kernel takes fp16 x,
computes per-row absmax, quantizes inline) to remove the separate launch. Built it
(bit-exact bf16 gate; fp16 within int8 quant-noise, rel-L1 ≤1.3e-5). **Result:
2.5× SLOWER than V2** — the internal absmax pass + synchronous fp16 A-staging break
the cp.async pipeline that makes V2 fast. Rejected; not shipped. Lesson recorded:
a working quant fusion must keep A on cp.async (fp16→smem, then smem→smem quantize)
or fuse the quant into the *preceding* norm/residual op — a larger redesign whose
ceiling (≈ removing 4%) only reaches fp16 parity on 1.5B, so it is not the highest
-value next step.

## Where int8 is decisive (the strategic conclusion)

Beating our own tuned fp16 peak with int8 on **1.5B** is marginal: at 1.5B the model
is not VRAM-limited, so int8's weight-halving buys nothing in concurrency, and the
battle is purely compute where fp16 is already excellent. int8's decisive,
fp16-unreachable win is on **7.2B**, where fp16 weights (14.4 GB) leave little of a
32 GB card for the recurrent-state pool — capping max concurrency — while w8a8
weights (7.2 GB) free ~2× the headroom for state, unlocking a higher concurrency
ceiling and therefore a peak throughput fp16 cannot reach on this card. That
comparison is the next measurement (F0035).

## What stands, honestly scoped

On sm120 (Blackwell consumer), where upstream cutlass `int8_scaled_mm` does not
exist at all: rwkv-sglang serves w8a8 end-to-end (V1+V2), the int8 GEMM itself beats
fp16 cuBLAS (1.03–1.55× at M≥512), weight VRAM halves, and greedy numerics are
lambada-certified (F-handoff: 0.6486 vs 0.6509 cutlass). No other RWKV serving stack
has any int8 on this architecture.

## Cross-references

`bench/results/bsz_sweep_w8a8{,v2}_5090main.json` · `bench/verify_w8a8.py` (gate +
microbench) · F0033 (probe) · BENCHMARKS §4.
