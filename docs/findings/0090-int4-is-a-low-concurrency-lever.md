---
finding_id: F0090
title: "int4 wins below c=32 by up to 1.75x and loses above c=64 by up to 1.87x — its headroom is bandwidth headroom, and bandwidth stops binding as the batch grows"
date: 2026-08-06
status: closed
severity: medium
---

# F0090 — int4 is a low-concurrency lever, not a serving-throughput one

**Date:** 2026-08-06 · **Card:** RTX 5090 · **Model:** rwkv7-1.5b, fp16 vs w4
(int4 g64) · 64-in/256-out sweep, one session, one card

## TL;DR

F0089 measured int4 at 72% of the card's read bandwidth against fp16's 86% and
concluded that int4 is where the headroom is. That is true **and it does not
generalise past low concurrency**, which is the first thing to check rather than
the last:

| concurrency | fp16 | int4 | int4 / fp16 |
|---:|---:|---:|---:|
| 1 | 496 | **716** | **1.44** |
| 4 | 1,194 | **2,089** | **1.75** |
| 8 | 2,232 | **2,897** | 1.30 |
| 16 | 4,111 | **5,620** | 1.37 |
| 32 | 8,498 | **9,057** | 1.07 |
| 64 | **14,300** | 12,895 | 0.90 |
| 128 | **19,960** | 10,686 | **0.54** |
| 256 | **23,054** | 15,251 | 0.66 |
| 384 | **22,582** | 16,894 | 0.75 |
| 512 | **23,617** | 18,708 | 0.79 |

Two arms per tier, agreeing to 0.4% at every point.

**The crossover is between c=32 and c=64.** Below it int4 is worth up to 1.75x;
above it fp16 is worth up to 1.87x.

## Why, and why it follows from F0089 rather than contradicting it

int4's advantage is that it reads 1.64 GB per token where fp16 reads 2.81. That
only buys time while **weight traffic is what the step is waiting on** — which is
the bsz1 regime F0089 measured. As the batch grows, the same weight read serves
every sequence in it, so weight traffic per token falls like 1/batch and the step
becomes compute-bound. Dequantisation is compute that fp16 does not do at all, so
past the crossover int4 is paying a cost with nothing left to buy.

The headroom in F0089 is real. It is **bandwidth** headroom, and it is cashable
only where bandwidth binds.

## The cliff at 128 is not the crossover, it is a separate thing

int4 goes 12,895 at c=64 to **10,686 at c=128** — it gets *slower* with more work.
`W4Linear` dispatches M>64 to dequantise-then-cuBLAS, whose effective weight
traffic (~36 bits/element) is worse than just storing fp16, and the model's own
comment already names this as "the measured M=64 concurrency cliff".

`RWKV_W4_TC_LARGE_M=1` is the intended answer and it barely moves: 11,259 at
c=128 (+5%), 19,297 at c=512 (+3%) — and it changes semantics from w4a16 to w4a8,
so it is opt-in pending accuracy certification. **A 5% recovery of a 46% cliff is
not the fix**, and this is the first measurement that says so at this scale.

## What to deploy

There is no single best configuration, and asking for one is the wrong question:

- **c <= 32** — int4. Biggest at c=4 (1.75x).
- **c >= 64** — fp16. Biggest at c=128 (1.87x).

Both tiers ship; what was missing was the number that says where to switch.

## Two process notes, because both cost time here

**The int4 arm produced no rows on the first attempt and nearly read as a
result.** `scripts/serve.sh` does not export `RWKV_W4` — it is per-model opt-in —
so `load_weights` hit `unexpected checkpoint key: ...attn.k_proj.qweight` and the
server died at startup. The driver log said `DIED` on one line that the summary
grep did not match, so the first read of the output was "the int4 rows are
missing" rather than "the int4 server never ran".

**Then the liveness check fired falsely.** Having added a check that the w4
kernels announce themselves, it reported zero announce lines on a run that was
unambiguously int4 (bs1 at 716 vs fp16's 496, and the checkpoint will not even
load without the flag). The w4 path **had no announce line at all** — every other
fast path in this model has one, and this one was never given one. A liveness
check with nothing to look for is not a liveness check. Added in this change; it
now prints `[rwkv7] W4 int4 weight path ENABLED (fp16 activations, group=64;
M-tiered gemv/gemm)` once, and that line was verified to appear before this
finding was written.

## Quality caveat, stated with its limit

On the one prompt used as a smoke test, fp16 continued *"The capital of France
is"* with `Paris. The capital of France is Paris.` and int4 with
`__________. (A) Paris (B) London (C`. That is one greedy continuation, not an
evaluation, and it is recorded only because it was seen. The measured accuracy
position is F0082's: the published int4 tiers differ from each other by 6.9
MATH500 points, and speed tables should not be read without it.
