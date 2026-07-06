---
doc_kind: finding
finding_id: F0039
title: "MLX Apple-Silicon weight quantization (w8g64 / w4g64): the decode-bandwidth lever F0038 pointed at — w8 is greedy-lossless and speeds bsz1 decode +28–49% at −20–33% memory; w4 trades accuracy for the smallest footprint"
last_verified_commit: "HEAD"
discovered_by: Opus 4.8 (agent-assisted), 2026-07-06
severity: info
status: open
related: [F0038, F0040]
---

# Finding F0039: MLX-native weight quantization (Apple Silicon)

## Context
F0038 showed MLX bsz1 decode is **weight-bandwidth-bound** (1.5B reads ~2.88 GB/token; at the M5's
123 GB/s ceiling that caps decode at ~43 tok/s) — so the decode lever is *fewer weight bytes*, not
more fp16 kernel tuning. This finding adds opt-in MLX-native group weight-quant and measures it.

## Mechanism
`load_model(..., quant="w8"|"w4")` (or `RWKV_MLX_QUANT`). Uses `mx.quantize` / `mx.quantized_matmul`
with **group_size=64**, mirroring the CUDA **w8g64 / w4-g64** modes. Quantized: every big `[out,in]`
projection — `r/k/v/o_proj`, ffn `key/value`, and `lm_head` (grouping is over the input dim, i.e.
per-output-channel g64). Left unquantized: **emb** (it is a 1-row gather at decode; MLX has no
quantized gather), and all **LoRA chains / norms / token-shift / the WKV state** (fp32, per the
port's precision policy). Activations stay bf16 — this is weight-only quant.

**fp16 stays the exact default.** Quant is strictly opt-in; with `quant=None` the code path is
byte-for-byte the fp16 one, and `gate_oracle.py` remains **GATE_ALL_PASS** (24/24 & 8/8, both WKV
paths) — verified after the change. Quant is *not* expected bit-exact and is gated by its own ruler
(greedy-vs-oracle here; compression rate in F0040), never by the oracle gate.

## Results (M5, metal, bf16 activations; `bench_mlx.py --quant`)
Decode = bsz1 greedy, 128 steady-state, median/best of 5; prefill = 1024 tokens, median of 3;
peak = single live model. fp16 rows are the F0038 build.

| model | mode | greedy vs oracle | decode tok/s (med / best) | prefill tok/s | peak mem |
|---|---|---|---:|---:|---:|
| 0.1B | fp16 | 24/24 | 325.6 / 331.9 | 11,485.8 | 0.54 GiB |
| 0.1B | **w8** | **24/24** | **417.3 / 475.7** | 8,458.1 | 0.43 GiB |
| 0.1B | w4 | 4/24 | 588.1 / 593.4 | 7,830.7 | 0.36 GiB |
| 1.5B | fp16 | 24/24 | 37.3 / 39.1 | 1,904.5 | 3.38 GiB |
| 1.5B | **w8** | **24/24** | **55.5 / 56.0** | 1,908.0 | 2.28 GiB |
| 1.5B | w4 | 24/24* | 94.0 / 95.8 | 1,974.8 | 1.65 GiB |
| 7.2B | fp16 | 8/8 | 7.5 / 7.9 | 441.2 | 14.64 GiB |
| 7.2B | **w8** | **8/8** | **12.6 / 12.9** | 484.1 | 8.88 GiB |
| 7.2B | w4 | 8/8* | 22.0 / 22.9 | 513.3 | 5.76 GiB |

*w4's greedy match on this 24/8-token fixture is coincidental agreement, NOT losslessness — the
compression ruler (F0040) is where w4's real accuracy cost shows (cf. CUDA int4 GPTQ: +0.0429 bpb,
−24pt MATH500). The 0.1B w4 already diverges greedily (4/24).

## What the numbers say
- **w8 is the decode win — and greedy-lossless — and it grows with model size.** bsz1 decode
  **+28% (0.1B) / +49% (1.5B) / +68% (7.2B)** with greedy output **identical to the fp32 oracle
  (24/24 & 8/8)** — the same "w8g64 is greedy-lossless" result the CUDA side reports, reproduced on
  Apple Silicon. Peak memory drops **−20% / −33% / −39%** (7.2B: 14.64→8.88 GiB). Exactly F0038's
  prediction: halving the weight bytes lifts the bandwidth-bound decode ceiling, and the bigger the
  model the more bandwidth-bound it is.
- **Prefill under quant tracks the compute/bandwidth balance:** 0.1B −26%, 1.5B ~flat, **7.2B
  +10%**. Small-model prefill is compute-bound (M=256 GEMMs) where `quantized_matmul`'s per-tile
  dequant costs more than the byte saving buys; the 7.2B is large enough that even prefill is
  partly bandwidth-bound, so quant helps there too. If TTFT on a *small* model matters more than
  decode, stay fp16.
- **w4 is the footprint + max-decode play, at an accuracy cost.** Even bigger decode —
  **+81% (0.1B) / +152% (1.5B: 37.3→94.0) / +193% (7.2B: 7.5→22.0)** — and the smallest footprint
  (**−33% / −51% / −61%**; 7.2B fits in 5.76 GiB). But int4 is where accuracy is spent: 0.1B already
  diverges greedily (4/24), and the compression ruler (F0040) resolves the real gap. Use w4 when
  memory/latency dominates and the accuracy budget allows; otherwise w8 (lossless) is the default
  quant.

## Accuracy
Greedy-vs-oracle (24 tokens) is only a sanity signal. The real accuracy ruler is the **uncheatable
compression rate** — measured for fp16 AND w8/w4 in F0040. The greedy result already shows the
expected ordering: **w8 lossless, w4 diverges** (int4 carries a real accuracy cost, matching the
CUDA int4 finding where perplexity-style rulers understate the reasoning damage).

## Honesty / host load
Shared M5 (load ~8). The table's absolute rows are separate runs, so cross-run load differs; the
**drift-cancelled** decode number is an interleaved one-process 1.5B A/B (median of 4 rounds,
fp16/w8/w4 measured back-to-back per round): **fp16 31.4 → w8 52.0 (+66%) → w4 85.1 (+171%)**. The
interleaved ratios exceed the table's cross-run ratios because the table's fp16 row happened to be
measured on a quieter box; the effect is far larger than the ±5% jitter either way.

## Cross-references
`mlx_port/rwkv7_mlx.py` (`quant` arg, `qbig`, `_proj`, head matmul) · `mlx_port/bench_mlx.py`
(`--quant`) · F0038 (why decode is bandwidth-bound) · F0040 (compression-rate accuracy for all
precisions).
