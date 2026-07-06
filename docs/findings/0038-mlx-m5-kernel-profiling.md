---
doc_kind: finding
finding_id: F0038
title: "MLX Apple-Silicon (M5): single-stream hotspot profiling + a bit-exact WKV kernel win — decode is already at ~79% of the hard memory-bandwidth ceiling, so the decode lever is fewer bytes (quant), not more kernel tuning"
last_verified_commit: "HEAD"
discovered_by: Opus 4.8 (agent-assisted), 2026-07-06
severity: info
status: open
related: [F0037, F0039]
---

# Finding F0038: MLX M5 kernel profiling + decay-precompute WKV kernel

## Context
F0037 made the fused-Metal WKV scan the MLX default. This finding profiles the single-stream
hotspots on Apple **M5** (32 GB unified, MLX 0.31.2), measures the hardware ceilings, ships the one
**bit-exact** kernel win that survived, and records the optimizations that did NOT (so they are not
re-tried). The red line is unchanged: `python mlx_port/gate_oracle.py` stays **GATE_ALL_PASS**
(greedy token-exact vs the numpy fp32 oracle; 24/24 for 0.1B & 1.5B, 8/8 for 7.2B, both WKV paths).

## M5 hardware ceilings (measured)
- **Memory bandwidth: ~123 GB/s** (large elementwise copy; 64 MB→96, 256 MB→117, 1 GB→123 GB/s).
- **Matmul: ~11.4 TFLOP/s @2048², ~13.2 TFLOP/s @4096²** (bf16 ≈ fp16 within noise).

## Decode is bandwidth-bound — and already near the wall
bsz1 decode reads every weight once per token. For 1.5B that is **~2.88 GB/token** (24 layers ×
~108 MB + the 268 MB lm_head; emb is a 1-row gather). At the 123 GB/s ceiling that caps decode at
**~42.7 tok/s**; we measure **33.6** = **79% of the hard ceiling**. The remaining 21% is the many
small per-layer GEMV/elementwise launches (MLX dispatches each op as its own Metal kernel; a lone
[1,2048]@[2048,2048] GEMV runs at ~25 GB/s = launch/occupancy-bound, not bandwidth-bound). An
in-graph ablation confirms the shape: zeroing the big projections takes decode from ~34 to ~400
tok/s — decode *is* the weight read. **Conclusion: the decode lever is fewer weight bytes
(quantization, F0039), not more fp16 kernel tuning.** Micro-opts that don't change bytes land in the
measurement noise on this box (see negatives).

## Prefill: 22–41% WKV scan, the rest GEMM/elementwise
Ablating the WKV scan out of prefill: it is **41% of 0.1B** prefill and **22% of 1.5B** (the scan is
a serial recurrence over T — one thread per V-column walking the sequence). The scan is
serial-latency-bound, not occupancy-bound: packing multiple heads per threadgroup (HPG=2/4/8) did
**not** speed it up (1107–1345 µs vs 1120 µs baseline).

## Shipped (bit-exact): decay-precompute in the WKV Metal kernel
The scan computed `metal::precise::exp(w[k])` **inside** the K-loop, so the D=64 V-column threads of
a head each recomputed the same D decays — **D² exp per timestep**. Precomputing `exp(w)` once per
element during the threadgroup staging (**D exp per timestep**, 64× fewer) and reading it in the loop
is the **same `precise::exp` on the same fp32 input → bit-identical** (oracle gate: 24/24 and 8/8
unchanged). Controlled interleaved A/B (median of 3 rounds, chunk=256, full model prefill):

| model | prefill before | prefill after | Δ |
|---|---:|---:|---:|
| 0.1B | 10,189.8 | 10,269.1 | +0.8% |
| 1.5B | 1,827.9 | 1,861.3 | +1.8% |
| 7.2B | 432.3 | 445.8 | +3.1% |

(The end-to-end gain scales with head count — 7.2B has 64 heads — and is smaller than the scan-only
microbench because projections dominate prefill at chunk=256.) Prefill chunk-size was swept
{128…1024}: **256 is near-optimal** for the models that matter (384 ties on 1.5B; ≥512 degrades as
the per-dispatch serial scan grows; 768 helps only 0.1B, at +65% peak memory) — default kept at 256.

## After (M5, metal, bf16, this build — `bench_mlx.py`, decode median/best of 5, prefill median of 3)

| model | decode tok/s (median / best) | prefill tok/s | peak mem |
|---|---|---:|---:|
| 0.1B | 325.6 / 331.9 | 11,485.8 | 0.54 GiB |
| 1.5B | 37.3 / 39.1 | 1,904.5 | 3.38 GiB |
| 7.2B | 7.5 / 7.9 | 441.2 | 14.64 GiB |

Peak memory is unchanged from F0037 (0.54 / 3.38 / 14.64 GiB) — the decay-precompute adds no memory.
**Read decode honestly:** it is *unchanged* by this build (the kernel change only touches the WKV
scan, a small slice of bandwidth-bound decode). The higher decode here vs F0037's baseline (298 /
33.6 / 7.6) is host-load variance on this shared box, not a speedup — the rigorous, drift-cancelled
result is the +0.8/1.8/3.1% **prefill** A/B above.

## Negative results (recorded so they are not re-tried)
- **Compiling the T>1 prefill forward: +13% (1.5B) but NOT bit-exact — reverted.** Reusing the
  shape-keyed compiled `self._step` for the prefill chunk fuses/reorders fp ops, shifting the state
  by ~2e-7 and last-logit by ~0.25. Harmless on 1.5B/7.2B (still 24/24 & 8/8) but it flipped 0.1B's
  greedy token #4 → **5/24**. Bit-exactness is the red line, so prefill stays eager. (Decode's
  compiled **T==1** step is separately gate-validated as exact — only the T>1 fusion diverges.)
- **T=1 specialized decode step** (drop the token-shift concat): bit-exact but within measurement
  noise (0.1B +1–5%, 1.5B −2–4% on this loaded box) — not shipped.
- **`mx.set_wired_limit` residency**: no reliable decode gain (0.1B ~+1%, 1.5B ~−2%).
- **Multi-head-per-threadgroup WKV scan**: no scan speedup (serial-bound, above).

## Honesty / host load
This M5 is a shared box (load average ~8 during this work: another project's ASR server + other
users), so single-stream decode carries ±3–5% run-to-run jitter. All comparisons above are
**interleaved A/B in one process** (baseline and variant measured back-to-back per round) so
host/thermal drift cancels; absolute numbers are median (with best reported by `bench_mlx.py`).

## Cross-references
`mlx_port/rwkv7_mlx.py` (`_METAL_SRC` decay-precompute; `prefill()` stays eager with the reason
inline) · F0037 (fused-Metal default) · F0039 (MLX weight quantization — the decode lever this
finding points at).
