---
finding_id: F0087
title: "W1'': the 2<=T<=gate band was doing in torch what both of its neighbours had a kernel for — +5.2% at bs8, +5.9% at bs4"
date: 2026-08-06
status: closed
severity: medium
---

# F0087 — The band between the two fused paths

**Date:** 2026-08-06 · **Card:** RTX 5090 · **Model:** rwkv7-1.5b-fla fp16

## TL;DR

The LoRA-gate activations (3 sigmoids, the `w_log` scale, the v-residual mix) had a
fused kernel on **both** sides of the batched decode band and none inside it:

| batch | LoRA path | gate activations |
|---|---|---|
| T == 1 | `lora4_m1` | folded into stage2 (F0066c) |
| **2 <= T <= gate** | **`lora4_mn`** | **torch, ~7 launches** |
| T > gate | cuBLAS `ReplicatedLinear` | `ln_fused.vres_gates` |

The middle row now calls the same `vres_gates` the bottom row already used.
**+5.9% at bs4, +5.2% at bs8, bs1 unchanged**, byte-identical output.

## Why the hole existed

`vres_gates` wants contiguous `[T,H]`. `lora4_mn` returns `[T,C,H]`, so `lo_mn[:, c]`
is a strided column and could not be handed over directly — which is what the
comment sitting on those lines said, and where it stopped. The fix is one
transpose: `lo_mn.transpose(0, 1).contiguous()` materialises all C columns as
contiguous rows, and the torch path was already paying most of that anyway — it
materialised `w_log`/`a`/`v` through `sigmoid` and copied `g` outright.

## Measured

Alternating old/new/old/new, two versions of the model file swapped between server
launches rather than an env flag (`RWKV_FUSED_VRESGATE` also controls the `T > gate`
path, so flipping it would move two things at once). Same neutral harness as
`bench/results/samecard_btl/`, decode tok/s, median of 5:

| batch | old | old | new | new | gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 514.2 | 514.4 | 515.3 | 515.4 | +0.2% (noise) |
| 4 | 1,169.0 | 1,169.2 | **1,237.8** | **1,236.5** | **+5.9%** |
| 8 | 2,166.0 | 2,165.9 | **2,279.4** | **2,278.8** | **+5.2%** |

Within-arm spread <= 0.1%. bs1 does not move, which is the control: the band
starts at T=2.

## Gate

- `bench/test_vresgate_mn.py`: **zero differing bytes** against the exact torch
  chain it replaces, at T = 2..8, 12, 16 and both layer roles (layer 0 has no
  v-residual chain). Sabotage-checked: perturbing `_INV_SQRT_E` by 0.1% makes it
  report 258,048 differing bytes, so the comparison discriminates.
- `bench/verify_batch.py` greedy oracle at bsz 8, fp16, cuda-graph: **OVERALL
  PASS**, with the `W1''` announce line present in the same log — a gate that
  cannot see whether the path fired proves nothing about it.

## Also

`bench/profile_components.py`'s LoRA component was updated in the same change, for
the same reason it was fixed earlier today (F0086): it replicates the model's
branch, so when the model's branch gains a kernel the replica has to gain it too
or it drifts back into timing something nobody runs.

## What is left in this band

`lora4_mn` itself. Its own comment says "correctness-first, no smem", and the
measurement agrees: at M=16 it costs 1.6x the cuBLAS path it replaces, which is why
the batch gate sits at 8 rather than higher (F0086). Taking bs16-32 back is a
kernel rewrite, not a threshold.
