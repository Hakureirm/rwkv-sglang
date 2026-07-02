---
doc_kind: finding
finding_id: F0017
title: "Hand-written weight-only int4: bsz1 decode 1.54x FASTER than fp16 + ~4x weight-VRAM cut, greedy-consistent, integrated end-to-end in sglang. Accuracy: GPTQ g64 lambada 0.6390 (−3.34pt, within ~1.2pt of int8) vs RTN g64 0.6229 (−4.95pt). M>1 uses a dequant→cuBLAS fallback (~0.5x fp16); a fused int4 GEMM is the throughput endgame."
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
| bsz | fp16 decode tok/s | w4 decode tok/s | w4/fp16 |
|----:|------------------:|----------------:|--------:|
|   1 |             166.5 |       **256.1** | **1.54× faster** |
|   8 |            1112.9 |           541.1 | 0.49× |
|  32 |            3872.6 |          2014.6 | 0.52× |

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
- **M>1 decode is ~0.5× fp16.** At batch, the GEMM is compute-bound (weight read amortized), so
  int4's bandwidth edge is gone AND the dequant→cuBLAS fallback adds an HBM round-trip. A **fused
  int4 GEMM (marlin-style, dequant in shared memory, no fp16 materialization)** is the throughput
  endgame; until then w4 is best used as a **single-stream latency + VRAM mode** (bsz1: 1.54×
  faster + weight VRAM cut), opt-in and default-off.
- **Accuracy**: GPTQ closed RTN's −4.95pt to **−3.34pt** (within ~1.2pt of int8). Closing the
  last bit toward Q4_K_M would need act-order GPTQ (per-column `g_idx` in the kernel — breaks the
  contiguous-group assumption) or more calibration; deferred as diminishing returns.

## Net
The only working, greedy-consistent 4-bit among RWKV-7 serving implementations: faster-than-fp16 at bsz1,
a real weight-VRAM cut, and GPTQ accuracy within ~1.2pt of int8 — a distinctive 4-bit capability.
Remaining lever: a fused int4 GEMM for M>1 throughput (currently ~0.5× fp16, batch is
compute-bound), documented honestly.
