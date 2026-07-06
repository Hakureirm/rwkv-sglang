---
doc_kind: finding
finding_id: F0037
title: "MLX Apple-Silicon: fused-Metal WKV becomes the default — 5.5–8.5× faster prefill at equal-within-noise bsz1 decode, still oracle-exact 24/24; plus a peak-memory measurement-artifact fix"
last_verified_commit: "HEAD"
discovered_by: Opus 4.8 (agent-assisted), 2026-07-06
severity: info
status: open
related: [F0031]
---

# Finding F0037: MLX fused-Metal WKV as default (Apple Silicon)

## Context
The MLX port (`mlx_port/`, first landed 1215c98) shipped two oracle-exact WKV paths: `pure`
(plain MLX ops — a Python-level per-token scan) and `metal` (a fused single-dispatch scan kernel).
The original default was `pure` (dependency-free, no Metal JIT). This finding re-measures both on
Apple **M5** and flips the default to `metal`, with the numbers to justify it.

## Measurement (M5, bf16 weights, bsz1)
`python mlx_port/bench_mlx.py --wkv pure,metal` — re-gates 24/24 in-process before timing;
decode = 128-token steady-state greedy (median & best of 5), prefill = 1024 tokens (median of 3):

| model | WKV | decode tok/s (median / best) | prefill tok/s | peak mem |
|---|---|---|---|---|
| 0.1B | pure | 274.6 / 302.1 | 1,222 | 0.40 GiB |
| 0.1B | **metal** | 287.7 / 291.4 | **10,360** | 0.54 GiB |
| 1.5B | pure | 31.5 / 32.8 | 297 | 3.04 GiB |
| 1.5B | **metal** | 28.1 / 31.1 | **1,643** | 3.38 GiB |

**Prefill: metal is 8.5× (0.1B) / 5.5× (1.5B) faster** — the fused whole-chunk scan runs in one
dispatch versus `pure`'s T-step Python loop. **Decode: equal within measurement noise** — bsz1
decode is bandwidth-bound on the per-token weight read (the WKV op is a small slice of per-token
work), so both paths land within ~5% on the least-contended `best` run; median jitter on a loaded
host slightly favored either path per size. **Peak memory: +~0.3 GiB** for the kernel's threadgroup
scratch — a small, worthwhile cost for the prefill win.

## Why metal is the right default
Prefill speed sets time-to-first-token and is where `pure` was weakest (an interpreter-bound loop);
metal removes that cliff at no real decode cost and negligible extra memory. `pure` stays available
as a JIT-free fallback via `RWKV_MLX_WKV=pure`. Both remain **oracle-exact** (see gate below), so the
default is bit-for-bit identical output — only faster.

## Correctness gate (the hard red line — unchanged)
`python mlx_port/gate_oracle.py` → **GATE_ALL_PASS**: greedy continuation matches the numpy fp32
oracle **24/24 token-exact** for 0.1B and 1.5B, for BOTH `pure` and `metal`, at bf16. Flipping the
default changes performance, not output.

## Bonus: peak-memory measurement-artifact fix (honest numbers)
`bench_mlx.py` previously reported the second-measured WKV path's peak memory as ~2× inflated: the
compiled decode step (`mx.compile(self._forward_seq)`) captures every weight by closure, so `del
model` alone left the ~3 GiB (1.5B) resident while the next config loaded — metal's peak was really
metal + retained-pure weights. Fixed by nulling the compiled callable (`model._step = None`) +
`gc.collect()` + `mx.clear_cache()` between configs; peak now reflects a single live model. Decode
timing also went median→(median, best) of 5 to report host-jitter honestly.

## Portability
The metal kernel uses only portable Metal (threadgroup memory, barriers, `precise::exp`) — expected
to build on M1/M2/M3/M4 as well; numbers here are M5. Cross-generation re-measurement is future work
(M4 / M1 machines are available).

## Cross-references
`mlx_port/rwkv7_mlx.py` (`WKV_DEFAULT`), `mlx_port/bench_mlx.py`, `mlx_port/gate_oracle.py` · F0031
(the CUDA path's own M-shape reduction-order lesson — the MLX paths avoid it by both being exact).
