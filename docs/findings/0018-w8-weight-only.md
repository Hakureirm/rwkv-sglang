---
doc_kind: finding
finding_id: F0018
title: "Hand-written weight-only int8 (w8a16): greedy-EXACT 24/24 (lossless in practice), faster than (or tied with) fp16 at every bsz≤32 (1.02–1.37× e2e), and JIT-runs on every arch — unlike cutlass w8a8 (sm80–90 only)"
last_verified_commit: "HEAD"
discovered_by: lead (M8), 2026-07-02
severity: info
status: open
related: [F0011, F0012, F0017]
---

# Finding F0018: weight-only int8 (w8a16) kernel family

## Why a second int8 path (vs the existing w8a8)
The sgl-kernel cutlass w8a8 int8 GEMM only ships **sm80–90** configs — measured failures on
Turing sm75 (`Error Internal`) and Blackwell sm100/120 (`NotImplementedError`, F0012). And w8a8
quantizes activations too, costing a small accuracy drift (1.5B lambada −2.15pt, MMLU −0.9pt).
A **weight-only** int8 (w8a16) with our proven w4 kernel skeleton fixes both: it JIT-builds
per-arch (runs everywhere the int4 family runs, Turing→Blackwell) and keeps activations fp16.

## What was built (`rwkv7_w8.cu`, mirrors the w4 family)
- `gemv_w8_m1` (M=1) + `gemm_w8_small` (2≤M≤8, one weight-word feeds all M rows, every row
  BIT-identical to the M=1 kernel — torch.equal-verified) + `gemm_w8_tc` (8<M≤64: wmma
  tensor cores, int8→fp16 dequant in shared memory per K-step so weight HBM traffic stays
  1/2 of a cuBLAS fp16 GEMM, fp32 accumulators, deterministic split-K — the w8 sibling of
  `gemm_w4_tc`) + `dequant_w8` (M>64 → cuBLAS).
- Group-wise symmetric int8, GROUP=64 (same structure as w4): 4 int8 per uint32, fp32
  accumulate, IEEE, cuda-graph safe. Quantizer: `bench/quant_w4.py --bits 8`.
- Model: `W8Linear` under `RWKV_W8=1` (same dispatch shape as W4Linear); default path untouched.

## Results (1.5B, RTX 3090, fp16, cuda-graph ON)
**Correctness: greedy 24/24 EXACT vs the numpy oracle** — per-group int8 weight RTN is
lossless in practice (matrix-level quant error 5.9e-3 rel, 18× smaller than int4's 1.05e-1).

Standalone kernels vs fp16 cuBLAS (`bench/verify_w8.py`): **1.13–2.29× at every M∈{1,2,4,8}**
(scalar family, numerics 2.1e-4); `gemm_w8_tc` 1.05–1.47× at M=16, mixed at M=32
(1.22×@2048², 0.79×@4096²), loses at M=64 on long-K shapes (0.51–0.61×) — same crossover
shape-dependence as `gemm_w4_tc`; numerics 2.9e-4.

End-to-end decode:
| bsz | fp16 tok/s | w8 tok/s | w8/fp16 |
|----:|-----------:|---------:|--------:|
| 1 | 166.5 | **227.4** | **1.37×** |
| 2 | 299.5 | **391.7** | **1.31×** |
| 4 | 574.1 | **731.9** | **1.27×** |
| 8 | 1112.9 | **1180.5** | **1.06×** |
| 16 | 2243.3 | **2512.7** | **1.12×** (gemm_w8_tc) |
| 32 | 3872.6 | **3935.9** | **1.02×** (gemm_w8_tc) |
| 64 | 6574.4 | 4895.7 | 0.74× (TC loses on the M=64 long-K ffn shapes — same as int4's 0.77×; honest) |

VRAM: peak serve 8,502 vs 9,152 MiB (bsz1); checkpoint 1.8 GB vs 2.9 GB fp16.

## Positioning (three quant modes, honest)
| mode | accuracy | speed sweet spot | arch coverage |
|---|---|---|---|
| **w8a16 (this)** | **greedy-EXACT** | bsz≤32: 1.02–1.37× fp16 | **all** (JIT per-arch) |
| w8a8 (cutlass) | −2.15pt lambada | large batch (+46–59% vs bf16) | sm80–90 only |
| w4 (ours) | GPTQ −3.34pt | bsz≤32: 1.03–1.56× fp16; max VRAM cut | **all** (JIT per-arch) |

## Cross-references
[[F0011]] (w8a8) · [[F0012]] (arch coverage + cutlass limits) · [[F0017]] (w4 family) ·
`bench/verify_w8.py` · `bench/quant_w4.py --bits 8`.
