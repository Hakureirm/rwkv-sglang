---
doc_kind: finding
finding_id: F0017
title: "Hand-written weight-only int4: faster than (or ties) fp16 at EVERY bsz≤32 (1.03–1.56×; gemv_m1 + gemm_w4_small + tensor-core gemm_w4_tc with in-smem dequant + deterministic split-K) + ~4x weight-VRAM cut; 7.2B: 102.8 tok/s bsz1 (1.29× albatross-fp16 cross-precision), fixture-EXACT 8/8, lambada 0.7161 (−2.64pt RTN), 9.8GB total, live-verified on a 16GB T4. 1.5B accuracy: GPTQ −3.34pt / RTN −4.95pt. bsz64 0.77× (M=64 long-K ffn shapes — tiling work remains)."
last_verified_commit: "HEAD"
discovered_by: lead (M7), 2026-07-02
severity: info
status: open
related: [F0011, F0014, F0015, F0016]
---

# Finding F0017: weight-only int4

Goal: 8-bit AND 4-bit quant where VRAM drops and speed is not worse than
16-bit. The landscape survey ([[project-rwkv-competitive-landscape]]) found **no known RWKV-7
serving implementation has any working 4-bit** (vllm-rwkv/vkwr: none; hf-adapter: bnb loads but
decode is slower than fp16 → fails the speed clause). This is the widest-open gap. We built a
hand-written int4 path — no bitsandbytes, no FLA.

## What was built
- **`rwkv7_w4.cu::gemv_w4_m1`** — weight-only group-wise (GROUP=64) symmetric int4 decode GEMV
  for the r/k/v/o + ffn key/value projections. 8 nibbles/uint32, fp32 accumulate, per-group
  scale folded per word, IEEE (no fast-math), cuda-graph safe. Decode (M==1) is
  weight-bandwidth-bound, so reading int4 (~1/4 the bytes) is **faster than fp16**.
  Standalone (`bench/verify_w4.py`): kernel vs dequant reference rel-err **~2e-4** (same ULP as
  torch fp16 matmul); **1.7–3.4× faster** than a cuBLAS fp16 GEMV at M==1.
- **`rwkv7_w4.cu::dequant_w4`** — memory-bound int4→fp16 unpack for the M>1 path (→ cuBLAS).
- **`bench/quant_w4.py`** — offline quantizer: fla → int4 checkpoint (`.qweight` uint8 [N,K/2] +
  `.scale` fp16 [N,K/64]); keeps LoRA/norms/emb/lm_head at original precision.
- **Model integration** (`models/rwkv7.py`): `W4Linear` (buffers + M==1 GEMV / M>1 dequant),
  `_make_proj` (W4Linear under `RWKV_W4=1` else ReplicatedLinear), w4-aware `_proj_gemv`,
  `load_weights` handles `.qweight`/`.scale` buffers. Opt-in (`RWKV_W4=1`); default path
  unchanged (regression: non-w4 1.5B still **greedy-EXACT 24/24**).

## End-to-end results (1.5B, RTX 3090, cuda-graph ON, fp16)
| bsz | fp16 decode tok/s | w4 decode tok/s | w4/fp16 | w4 path |
|----:|------------------:|----------------:|--------:|---|
|   1 |             166.5 |       **259.1** | **1.56× faster** | `gemv_w4_m1` |
|   2 |             299.5 |       **434.9** | **1.45× faster** | `gemm_w4_small` |
|   4 |             574.1 |       **773.2** | **1.35× faster** | `gemm_w4_small` |
|   8 |            1112.9 |      **1153.0** | **1.04× faster** | `gemm_w4_small` |
|  16 |            2243.3 |      **2619.8** | **1.17× faster** | `gemm_w4_tc` |
|  32 |            3872.6 |      **3978.2** | **1.03× faster** | `gemm_w4_tc` |
|  64 |            6574.4 |          5064.6 | 0.77× | `gemm_w4_tc` (M=64 ffn shapes) |

**`gemm_w4_small` (added 2026-07-02)** closes the small-batch gap: a template kernel for
2≤M≤8 where ONE int4 weight-word read feeds all M rows (weight bandwidth amortized across the
batch). Each row's k-iteration/accumulation order is identical to `gemv_w4_m1`, so every row is
**BIT-identical to the M==1 kernel** (verified `torch.equal` in `bench/verify_w4.py`) →
batch-invariant by construction. Standalone vs fp16 cuBLAS: M=2 2.3×, M=4 1.8–2.0×, M=8
1.07–1.75×; vs the old dequant+cuBLAS fallback: 3–9× (75–149µs → 14–42µs).

**`gemm_w4_tc` (added 2026-07-02)** covers 8<M≤64 with TENSOR CORES: wmma m16n16k16 (fp16 in,
fp32 accum), int4→fp16 dequant **in shared memory** per K-step (K_TILE == GROUP == 64 → exactly
one scale per (n,k-tile)); one block holds all M rows in register fragments so the weight tile
is dequanted ONCE per block (weight HBM traffic stays 1/4 of fp16), and **deterministic split-K**
(f32 partials, fixed-order reduce, no atomics) fills the GPU for small-N shapes. Numerics
rel-err ~2.7e-4 vs the dequant reference; per-row reduction order is fixed (batch-composition
independent). Standalone vs fp16 cuBLAS: M=16 1.16–1.23× (all shapes), M=32/64 1.17× at 2048²
but 0.54–0.86× at the long-K/wide-N ffn shapes — further tiling work (larger N-tiles,
cp.async pipelining on sm80+) is the remaining lever. M>64 (prefill) stays dequant→cuBLAS
(compute-bound; amortized).

## 7.2B results (RTX 3090, RTN g64, fp16, cuda-graph ON) — added 2026-07-02
| metric | fp16 (best, 3 opt-in kernels) | albatross-fp16 | **w4 RTN** |
|---|---|---|---|
| decode bsz1 tok/s | 65.7 | 79.6 | **102.8** (1.56× ours-fp16, **1.29× albatross-fp16**, cross-precision) |
| greedy vs oracle fixture | 24/24-class EXACT | EXACT | **8/8 EXACT** (7.2B quant robustness) |
| lambada (full 5153) | 0.7425 (bf16) | — | **0.7161 (−2.64pt)** — RTN only; cf. 1.5B RTN −4.95pt |
| peak serve VRAM (mem-frac 0.55) | ~17.5 GB weights alone | — | **9.8 GB total** |
| checkpoint | 14.4 GB | — | **4.8 GB** (3.0×) |

**Verified live on a real 16 GB T4** (sm7.5): greedy **8/8 EXACT**, decode **32.9 tok/s** bsz1 /
**65.3** bsz4, peak VRAM **6,735 MiB** of 14,913 — 7.2B serves on a 16 GB Turing card with more
than half the VRAM free (raw: `bench/results/allcards.json` entry `T4-72b-w4`).

7.2B GPTQ is deferred: the `RWKV_CALIB` hook accumulates fp32 Hessians on-GPU and the ffn.value
input dim is 16384 → 1 GB/layer × 32 layers won't fit 24 GB; needs streamed/CPU accumulation.

- **Correctness**: w4 greedy on the oracle fixture = **14/24, first-div @14 — bit-identical to
  the offline fake-quant reference**, confirming the kernel path == dequant (as verify_w4 predicts).
- **VRAM**: peak serve VRAM 8202 vs 9152 MiB at 1.5B bsz1 (−950 MiB); checkpoint 1.2G vs 2.9G
  (2.4× — emb/lm_head stay bf16, so the ratio grows with model size as linear weights dominate).
- **Accuracy** (lambada, full 5153): baseline 0.6724, int8 0.6509 (−2.15), **w4 GPTQ g64 0.6390
  (−3.34)**, w4 RTN g64 0.6229 (−4.95). Calibration-free sweep: g64 > g128; **MSE-clip and
  asymmetric both HURT** end-to-end (weight-MSE-optimal ≠ task-optimal — clipping harms
  functionally-important outlier weights; exactly why AWQ/GPTQ are activation-aware). **GPTQ**
  (activation-aware error feedback; Hessians captured via the `RWKV_CALIB` hook on wikitext,
  `bench/{calib_run,gptq_w4}.py`) recovers **+1.6pt** over RTN → within ~1.2pt of int8, and
  produces the SAME `.qweight`/`.scale` format (kernel/model unchanged). Further push (act-order
  GPTQ) would need per-column `g_idx` in the kernel (breaks contiguous groups) — deferred.

## Honest limitations & the endgame
- **Through bsz 32, w4 is faster than (or ties) fp16 at every batch size** (1.03–1.56×), via the
  three-kernel dispatch (gemv_m1 / gemm_w4_small / gemm_w4_tc). **bsz64 is 0.77×** — the M=64
  long-K/wide-N ffn shapes (4096², 2048×8192) lose to tensor-core cuBLAS (0.54–0.56× standalone);
  remaining levers: larger N-tiles per block, cp.async double-buffering (sm80+; would need an
  arch-guard to keep Turing), better smem swizzle. Prefill (M>64) = dequant→cuBLAS at ~0.95× fp16.
- **Prefill numerics note**: short prefill chunks (8<M≤64) now route through `gemm_w4_tc` — a
  different (still fp32-accumulate, deterministic) summation order than dequant→cuBLAS; lambada
  re-verified after the switch: **0.6227 vs 0.6229 pre-switch** (Δ0.0002, within noise) —
  accuracy unaffected.
- **Accuracy**: GPTQ closed RTN's −4.95pt to **−3.34pt** (within ~1.2pt of int8). Closing the
  last bit toward Q4_K_M would need act-order GPTQ (per-column `g_idx` in the kernel — breaks the
  contiguous-group assumption) or more calibration; deferred as diminishing returns.

## Net
The only working, greedy-consistent 4-bit among RWKV-7 serving implementations: faster-than-fp16 at bsz1,
a real weight-VRAM cut, and GPTQ accuracy within ~1.2pt of int8 — a distinctive 4-bit capability.
Remaining lever: a fused int4 GEMM for M>1 throughput (currently ~0.5× fp16, batch is
compute-bound), documented honestly.
