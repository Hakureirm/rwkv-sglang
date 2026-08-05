---
finding_id: F0088
title: "lora4_mn stage2 re-read its weights once per token: −33% on the kernel, +1.6% on the step, and the gate still should not move"
date: 2026-08-06
status: closed
severity: low
---

# F0088 — stage2, M-inner

**Date:** 2026-08-06 · **Card:** RTX 5090 · **Model:** rwkv7-1.5b-fla fp16

## TL;DR

`lora4_mn`'s second stage mapped one warp to one output element `y[m,c,n]`, with
the grid flattened over `M*C*H`. So the `u_cat` row segment for `(c,n)` was read
once **per m**, from a different block each time — 8.4 MB of weight traffic at M=8
for 1.05 MB of data, with no L1 reuse because the m's were on different SMs.

Warp → `(c,n)` with m as the inner loop, `h` staged in shared memory for all m:
**stage2 −33% at M=8, −42% at M=16**, byte-identical. End to end that is
**+1.6% at concurrency 8** and nothing anywhere else, which is what the component
share predicts. The batch gate stays at 8.

## The kernel

| M | stage1 | stage2 before | stage2 after | fused before → after |
|---:|---:|---:|---:|---:|
| 4 | 3.3 | 6.35 | 4.70 | 9.66 → 8.02 |
| 8 | 4.6 | 11.35 | **7.55** | 15.91 → **12.13** |
| 16 | 7.4 | 21.32 | **12.37** | 28.66 → **19.73** |
| 32 | 12.4 | 41.17 | 33.12 | 53.49 → 45.56 |

stage2 was 71–77% of the kernel, which is why it was the one worth touching —
measured before writing anything, because stage1 and stage2 have completely
different shapes and the wrong one absorbs a rewrite and returns nothing.

Byte-exactness is structural rather than tested-into-existence: for a fixed
`(m,c,n)` the lane assignment, the `r` indices, the fp32 `fma` order and the
`warp_sum` shuffle tree are unchanged. Only the loop nesting and where the
operands are read from moved. `bench/test_lora_mn.py` (mn ≡ m1 per token, incl.
C=3, odd M, odd rank) is EXACT, and the greedy oracle passes at gate 8 and 16.

**One thing that did not work:** holding each lane's `u` segment in a register
array across the m loop. The array is indexed by a runtime loop counter, so it
lands in local memory, which costs more than the L1 hits it was meant to save.
Re-reading `u` inside the same warp is the version that ships — the reuse that was
missing was never register reuse, it was L1 reuse across m.

## End to end, and why it is small

Sweep, four arms, up then down, out tok/s:

| c | old gate 8 | new gate 8 | new gate 16 | new gate 24 |
|---:|---:|---:|---:|---:|
| 4 | 1137 / 1138 | 1142 / 1145 | 1136 / 1141 | 1142 / 1142 |
| 8 | 2081 / 2080 | **2111 / 2118** | 2100 / 2112 | 2103 / 2111 |
| 16 | 3791 / 3800 | 3795 / 3759 | 3595 / 3630 | 3626 / 3636 |
| 32 | 7401 / 7353 | 7382 / 6934 | 7303 / 7350 | 7384 / 7328 |
| 64 | 10894 / 11528 | 11450 / 10915 | 11499 / 11504 | 11536 / 11544 |

+1.6% at c=8; c=16 and c=32 are flat because the gate keeps the fused kernel out
of those batches entirely; c=64's arms disagree with themselves by 5%, so nothing
is read from that row.

The size is not a disappointment, it is a prediction that held: at c=8 the step is
3,844 us, stage2 across 24 layers was 272 us of it — **7.1%** — so a third off it
predicts +2.4% and the harness read +1.6%. Worth stating because it makes the next
estimate cheaper: anything below ~5% of the step cannot show up end to end above
this harness's noise, whatever the kernel does.

## The gate still should not move

The microbenchmark said `fused/cuBLAS` fell from 0.84 to 0.58 at M=16 and put the
crossover near M=24, so the obvious next step was to widen the gate. **End to end
that is wrong**: at c=16, gate 16 (3595/3630) is *worse* than gate 8 (3795/3759),
and gate 24 does not recover it.

The microbenchmark's cuBLAS reference is what to distrust. Its column is
non-monotonic in M — 20.7 us at M=1, 47.9 at M=2, 43.4 at M=8, 30.3 at M=24 —
which is a launch-bound measurement under a profiler, not the cost of the GEMMs.
A reference that does not behave monotonically in the size of the problem is not
measuring the problem, and any ratio taken against it inherits that. The end-to-end
sweep is the number that decides, and it says 8.
