---
finding_id: F0089
title: "Single-stream fp16 has 17% of headroom, not a factor: we are at 86% of the card's achievable read bandwidth and Albatross at 92%"
date: 2026-08-06
status: closed
severity: low
---

# F0089 — The single-stream roof

**Date:** 2026-08-06 · **Card:** RTX 5090 · **Model:** rwkv7-1.5b-fla fp16

## TL;DR

A bsz1 decode step reads every weight once and almost nothing else, so its floor
is memory bandwidth. Measured rather than quoted:

| | tok/s | us/token | GB/s effective | % of achievable |
|---|---:|---:|---:|---:|
| ours, fp16 | 514.6 | 1,943 | 1,447 | **85.7%** |
| Albatross, fp16 | 554.1 | 1,805 | 1,558 | **92.3%** |
| the roof | **600.3** | 1,651 | 1,688 | 100% |

**Single-stream fp16 has 17% of headroom in total**, and the 7.7% gap to Albatross
is about half of it. That bounds every further fusion, megakernel and PDL step on
this path: they are competing for 17%, not for a factor. The lever that is not
bounded this way is reading fewer bytes — the int4 path already runs at 742.6
tok/s, which is *above* this fp16 roof because it does not read fp16 weights.

## The denominator, and how the error announced itself

The file is 3.0549 GB, and using that put Albatross at **100.3% of the measured
roof** — which is not a close call, it is an accounting error, because nothing
runs above the bandwidth it has.

`model.embeddings.weight` is 268.4 MB and a decode step reads **one row** of it,
not the table. `lm_head.weight` is the same size and *is* read in full, every
step, which is why the two must be classified separately even though they are
byte-for-byte the same shape.

| | GB |
|---|---:|
| file total | 3.0549 |
| embedding (gathered, one row per token) | −0.2684 |
| weights read per token | 2.7864 |
| + WKV state, fp32, read and written | +0.0252 |
| **per-token traffic** | **2.8116** |

The recurrent state is 0.9% of the total — worth stating because a linear-attention
model is exactly where a reader would expect state traffic to matter, and at 1.5B
it does not. (`RWKV_STATE_FP16` halves it, i.e. buys at most 0.45%.)

## The bandwidth number

Achievable read bandwidth, a large fp16 reduction that touches each byte once:

| size | time | GB/s |
|---|---:|---:|
| 1 GiB | 0.643 ms | 1,670.6 |
| 2 GiB | 1.272 ms | 1,687.9 |
| 4 GiB | 2.545 ms | 1,687.6 |

Three sizes within 1%, so this is the card and not the measurement. Spec sheet
numbers are not used anywhere above: what a kernel gets is the only relevant
figure and it is 1,688 GB/s here.

**A naive GEMV chain gets nowhere near it**: 64 dependent `[1,2048]@[2048,2048]`
matmuls run at 887 GB/s, 53% of the roof, because a dependent chain is
latency-bound rather than bandwidth-bound. Our deployed path reaches 1,447 GB/s
on the same card — 1.63x that naive chain — which is what the fused GEMV and
megakernel line bought and is a fairer statement of their value than any
tok/s delta.

## What this says about the next step

- **fp16 single-stream**: at most +17%, realistically +8% to match Albatross. Worth
  doing only with cheap fusions; not worth a large restructuring.
- **int4 / lower precision**: not bounded by this roof at all, and already ahead.
  This is where single-stream throughput actually lives.
- **Concurrency**: a different regime entirely — at batch 8 the weights are
  amortised across 8 tokens, so the roof is 8x further away and the limit is the
  kernels, which is where F0086-F0088 were working.
