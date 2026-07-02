# w4 — hand-written weight-only int4 (4-bit quantization)

Group-wise (GROUP=64) symmetric int4 for the big r/k/v/o + ffn key/value projections, via the
hand-written `rwkv7_w4.cu` GEMV. Opt-in (`RWKV_W4=1`), default off. No bitsandbytes, no FLA.
See [`../../../docs/findings/0017-w4-int4-quant.md`](../../../docs/findings/0017-w4-int4-quant.md)
for the full write-up; kernel test `bench/verify_w4.py`, quantizer `bench/quant_w4.py`.

## Kernel (standalone, `bench/verify_w4.py`, RTX 3090)
| K×N | kernel vs dequant (rel) | int4 GEMV vs fp16 GEMV (M=1) |
|---|---|---|
| 2048×2048 | 2.0e-4 | **2.10×** |
| 4096×4096 | 2.1e-4 | **2.02×** |
| 4096×14336 | 2.1e-4 | **3.41×** |
Kernel is ULP-accurate vs the dequant reference; 1.7–3.4× faster than cuBLAS fp16 at M==1.

## End-to-end (1.5B, sglang, cuda-graph ON, fp16)
| bsz | fp16 tok/s | w4 tok/s | w4/fp16 |
|----:|-----------:|---------:|--------:|
|   1 |      166.5 | **256.1** | **1.54× faster** |
|   8 |     1112.9 |    541.1 | 0.49× (dequant→cuBLAS fallback) |
|  32 |     3872.6 |   2014.6 | 0.52× |

- Checkpoint: **1.2 GB vs 2.9 GB** fp16 (2.4× at 1.5B; grows with model size — emb/lm_head stay bf16).
- Serve VRAM (bsz1): **8202 vs 9152 MiB** (−950 MiB at 1.5B).
- Correctness: w4 greedy on the oracle fixture = 14/24 (first-div @14) — **bit-identical to the
  offline dequant reference**, so the int4 kernel path == the quantizer.

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
