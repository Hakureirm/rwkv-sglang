---
finding_id: F0086
title: "A measured +10.7% never shipped: the default was raised in the tree that was about to be deleted"
date: 2026-08-06
status: closed
severity: high
---

# F0086 — A measured improvement that never shipped

**Date:** 2026-08-06 · **Card:** RTX 5090 · **Model:** rwkv7-1.5b-fla fp16

## TL;DR

`bench/results/samecard_btl/README.md` says the bs8 decode gap was *"partly closed
since"* by raising the fused-LoRA batch gate from 4 to 8, worth +12.2%. The commit
that raised it, `86bc77e`, edited

```
sglang_overlay/sglang/srt/models/rwkv7.py
```

— the retired v0.5.10 line, which `4761ca0` deleted three commits later. The tree
that ships and runs, `sglang_mainline/srt/models/rwkv7.py:149`, still reads `"4"`.
**The published claim was true of a measurement and false of the code.**

Re-measured on the shipped tree and it is worth having: **+10.7%** at concurrency
8, and 8 also beats 16 and 32 at every concurrency up to 32. The default is now 8.

## The A/B, on the tree that runs

Alternating, two rounds, because the prefill column of this harness once moved 2.1x
between identical runs ordered by run order alone (F0084):

| round | gate 4 | gate 8 |
|---|---:|---:|
| 1 | 1,742.0 tok/s | 1,930.2 tok/s |
| 2 | 1,744.6 tok/s | 1,926.7 tok/s |

Within-arm spread 0.2%; between arms **+10.7%**. Concurrency 8, 128-in/128-out,
`--cuda-graph-max-bs 512`, `--disable-radix-cache`, production env from
`scripts/serve.sh`, `_sglspec` verified byte-identical to `sglang_mainline` first
(29 files, zero differences).

## The profiler said the opposite, and was measuring the wrong function

`bench/profile_components.py`'s `loras(8 matmul)+gate-math` component called the
four LoRA modules and the torch gate chain directly. That is a path
`RWKV_FUSED_LORA_MAX_BS` **cannot reach** — the gate lives in
`Rwkv7Attention.forward`, and `Rwkv7LoRA` is a plain `up(act(down(x)))` module
with no knowledge of it. The component's numbers moved anyway when the gate was
swept, and the first version of this finding read a mechanism out of them
("all of the difference is launch overhead"). **There was no mechanism. The
component was not on the code path being swept.**

The component now takes the branch the model takes and labels which one it took.
Re-measured, graphed (GPU-busy) us per layer:

| M | fused (`lora4_mn`) | torch fallback |
|---:|---:|---:|
| 8 | **30.5** | 45.7 |
| 16 | 46.3 | **29.6** |

That is the crossover, from the model side, and it lands between 8 and 16 — which
is what the server ladder shows and what the original "~M=4 to 8" range meant. At
M=16 the fused kernel is 1.6x the cost of the cuBLAS path, so a gate of 16 or 32
does not merely stop helping, it actively pays.

The `eager` column is not usable for this. Across processes running the *same*
branch it varied by 3-7x (M=8 fused: 227 us at one gate setting, 63 us at another;
M=16 fallback: 139 us and 470 us). Whatever that is, it is not the model. Use
`graphed`, which held to 5% across the same pairs.

Two lessons, and the second is the one that cost time: a component that does not
call the code under test can still produce numbers that correlate with the setting
by accident, and a noisy column will supply the correlation.

## How it happened

## How it happened

The A/B that produced the original +12.2% was run through the env knob
(`RWKV_FUSED_LORA_MAX_BS=8`), which is correct and reaches the running code. Only
the follow-up edit — turning the measured value into the default — went to the
wrong file. Nothing failed: the overlay still parsed, the tests still passed, and
the number in the README stayed true of the experiment that produced it.

This is F0059 one layer up. There, we measured one artifact and published another,
and the fix was to delete the copy that did not run. Here the *fix* was applied to
that same copy, on its way out, six hours before it was deleted. Deleting a tree
does not tell you what was written into it in the meantime.

## Where the gate stops paying, and one cell that does not add up

Concurrency ladder, each gate run twice with the sweep reversed the second time,
128-in/128-out (out tok/s):

| concurrency | gate 8 | gate 16 | gate 32 |
|---:|---:|---:|---:|
| 4  | 1,039.7 / 1,041.2 | 915.2 / 1,041.1 | 1,050.6 / 1,045.1 |
| 8  | 1,934.6 / 1,945.0 | 1,717.0 / 1,688.9 | 1,947.7 / 1,942.9 |
| 16 | 3,708.2 / 3,717.5 | 3,073.2 / 3,007.3 | 3,345.3 / 3,349.0 |
| 32 | **7,208.2 / 7,250.2** | 6,784.0 / 6,794.2 | 5,799.7 / 5,801.4 |

8 wins at every concurrency measured, and the loss above it is large: gate 32 gives
up 20% at c=32, which is the crossover the original range note was describing —
past it the fused kernel is the slower one. Greedy output is oracle-exact at 4, 8,
16 and 32, so nothing here trades correctness for speed.

Most of the shape now has a mechanism: gate 16 and gate 32 both force the fused
kernel at M=16, where the table above shows it costs 1.6x the cuBLAS path, so both
lose to gate 8 at c=16 and c=32.

**What still does not add up:** those two settings take the *same* branch at c=8
and c=16, and they differ by 13% and 10%, reproducibly, in both sweep directions.
Nothing in the model distinguishes them there. Either the running batch is not what
the concurrency setting implies, or this harness carries a per-server-instance
variable we do not control — and the profiler's eager column, which varies 3-7x
across processes on identical code, says something in this environment does vary
that much. Until it is known, **a difference of this size between two server
launches is not by itself evidence of anything**, the same caveat F0084 attached
to the prefill column of the other harness.

That bounds the confidence in the headline: the gate 4 -> 8 result survives it only
because it has a mechanism (M=8 changes which kernel runs), because it replicates
across two independent sessions and six runs, and because the c=16 and c=32 columns
move the same way. It would not survive on a single pair of numbers.

## What changed

- `sglang_mainline/srt/models/rwkv7.py`: default gate 4 -> 8.
- `bench/results/samecard_btl/README.md`: the claim now cites the shipped number
  and this finding rather than the retired measurement.
- `bench/profile_components.py`: the LoRA component takes the model's branch
  instead of a torch copy of it, and says which branch it timed.

Correctness re-gated on the shipped tree at both values, because a gate that only
ever sees the new arm cannot tell a pass from a check that never fired.
