# w4 — hand-written weight-only int4 (4-bit quantization)

Group-wise (GROUP=64) symmetric int4 for the big r/k/v/o + ffn key/value projections, via the
hand-written `rwkv7_w4.cu` GEMV. Opt-in (`RWKV_W4=1`), default off. No bitsandbytes, no FLA.
See [`../../../docs/findings/0017-w4-int4-quant.md`](../../../docs/findings/0017-w4-int4-quant.md)
for the full write-up; kernel test `bench/verify_w4.py`, quantizer `bench/quant_w4.py`.

## Kernels (standalone, `bench/verify_w4.py`, RTX 3090)

**`gemv_w4_m1` (M=1):**
| K×N | kernel vs dequant (rel) | int4 GEMV vs fp16 GEMV (M=1) |
|---|---|---|
| 2048×2048 | 2.0e-4 | **2.10×** |
| 4096×4096 | 2.1e-4 | **2.02×** |
| 4096×14336 | 2.1e-4 | **3.41×** |

**`gemm_w4_small` (2≤M≤8, one weight read feeds all M rows; every row BIT-identical to the
M=1 kernel — `torch.equal`-verified):**
| M | K×N | rows == M1 kernel | vs fp16 cuBLAS | vs dequant+cuBLAS |
|---|---|---|---|---|
| 2 | 4096×4096 | BIT-EXACT | **2.27×** | 7.5× |
| 4 | 4096×4096 | BIT-EXACT | **1.79×** | 6.0× |
| 8 | 4096×4096 | BIT-EXACT | **1.07×** | 3.6× |

**`gemm_w4_tc` (8<M≤64, tensor cores):** wmma m16n16k16 with fp32 accumulators; the int4 weight
tile is dequantized to fp16 **in shared memory** each K-step (K_TILE == GROUP == 64 → exactly one
scale per (n, k-tile)), so weight HBM traffic stays 1/4 of a cuBLAS fp16 GEMM; one block covers
all M rows (weight dequanted once per block) and **deterministic split-K** (f32 partials + a
fixed-order reduce, no atomics) restores GPU-filling parallelism. Numerics vs the dequant
reference: rel-err ~2.7e-4 at every shape. Standalone vs fp16 cuBLAS (RTX 3090):

| M | 2048×2048 | 4096×4096 | 2048×8192 |
|---|---|---|---|
| 16 | **1.20×** | **1.16×** | **1.23×** |
| 32 | **1.17×** | 0.81× | 0.86× |
| 64 | **1.17×** | 0.56× | 0.55× |

M>64 (prefill) stays on dequant→cuBLAS (compute-bound; weight read amortized over many tokens).

## End-to-end (1.5B, sglang, cuda-graph ON, fp16)
| bsz | fp16 tok/s | w4 tok/s | w4/fp16 | path |
|----:|-----------:|---------:|--------:|---|
|   1 |      166.5 | **259.1** | **1.56×** | gemv_w4_m1 |
|   2 |      299.5 | **434.9** | **1.45×** | gemm_w4_small |
|   4 |      574.1 | **773.2** | **1.35×** | gemm_w4_small |
|   8 |     1112.9 | **1153.0** | **1.04×** | gemm_w4_small |
|  16 |     2243.3 | **2619.8** | **1.17×** | gemm_w4_tc |
|  32 |     3872.6 | **3978.2** | **1.03×** | gemm_w4_tc |
|  64 |     6574.4 |   5064.6 | 0.77× | gemm_w4_tc (M=64 ffn shapes drag — see kernel table) |

**int4 is faster than (or ties) fp16 at every batch size through 32** (1.03–1.56×); bsz64 is
0.77× (honest — the M=64 long-K ffn shapes lose to tensor-core cuBLAS; further tiling work).
w4 prefill ≈ 0.95× fp16 (13.3–13.8k vs 14.0–14.4k tok/s).

- Checkpoint: **1.2 GB vs 2.9 GB** fp16 (2.4× at 1.5B; grows with model size — emb/lm_head stay bf16).
- Serve VRAM (bsz1): **8202 vs 9152 MiB** (−950 MiB at 1.5B).
- Correctness: w4 greedy on the oracle fixture = 14/24 (first-div @14) — **bit-identical to the
  offline dequant reference**, so the int4 kernel path == the quantizer; unchanged after the
  small-M kernel (bit-identical rows, verified end-to-end).

## 7.2B (RTX 3090, RTN g64, fp16, cuda-graph ON)
| metric | ours fp16 best | albatross-fp16 | **w4 RTN** |
|---|---|---|---|
| decode bsz1 tok/s | 65.7 | 79.6 | **102.8** — 1.56× ours-fp16, **1.29× albatross-fp16** (cross-precision) |
| greedy vs oracle fixture | EXACT | EXACT | **EXACT 8/8** |
| lambada (full 5153) | 0.7425 (bf16) | — | **0.7161 (−2.64pt, RTN)** |
| peak serve VRAM | ~17.5 GB weights | — | **9.8 GB total** (fits a 16 GB card) |
| checkpoint | 14.4 GB | — | **4.8 GB** (3.0×) |

**7.2B on a real 16 GB T4** (`allcards.json` entry `T4-72b-w4`): greedy **8/8 EXACT**, decode
**32.9 tok/s** bsz1 / **65.3** bsz4, prefill ~1,012 tok/s, peak VRAM **6,735 MiB** — the 7.2B
model serves on a 16 GB Turing card with more than half the VRAM to spare.

7.2B GPTQ deferred (ffn.value Hessian = 16384² × fp32 = 1 GB/layer × 32 — needs streamed accumulation).

## Accuracy (lambada_openai, full 5153, lm-eval local-completions)
| model | acc | Δ vs bf16 |
|---|---|---|
| bf16 baseline | 0.6724 | — |
| int8 (w8a8) | 0.6509 | −2.15 |
| **w4 GPTQ g64** (calibrated) | **0.6390** | **−3.34** |
| w4 RTN sym g64 (calibration-free best) | 0.6229 | −4.95 |
| w4 RTN g128 | 0.6158 | −5.66 |
| w4 MSE-clip g64 | 0.6113 | −6.11 |
| w4 MSE-clip g128 | 0.5880 | −8.44 |

**GPTQ** (activation-aware error feedback; Hessians from wikitext calibration via the `RWKV_CALIB`
hook, `bench/{calib_run,gptq_w4}.py`) recovers **+1.6pt** over RTN → within ~1.2pt of int8, same
`.qweight`/`.scale` format (kernel unchanged). Among calibration-free schemes, **RTN sym g64
max-scale is best** — smaller groups, MSE-clip, and asymmetric all *hurt* end-to-end
(weight-MSE-optimal ≠ task-optimal). Reproduce GPTQ:
```bash
RWKV_CALIB=1 RWKV_CALIB_OUT=<dir> RWKV_CALIB_TOKENS=20000 python bench/calib_run.py --model <fla> --corpus <wikitext.txt>
python bench/gptq_w4.py --model <fla> --hessians <dir>/calib_hessians.pt --out <w4gptq> --group 64
```

## Reproduce
```bash
python bench/verify_w4.py                                    # kernel numerics + speed
python bench/quant_w4.py --model <fla> --out <w4> --group 64 # make int4 checkpoint
RWKV_W4=1 python bench/throughput.py --model <w4> --dtype float16 --cuda-graph ...  # e2e speed
# accuracy: sglang server on <w4> (--dtype float16) + lm_eval local-completions lambada
```
