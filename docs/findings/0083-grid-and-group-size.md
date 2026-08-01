---
doc_kind: finding
finding_id: F0083
title: "fp4 beats int4 by 4.42 points of MATH500 at the group size we actually ship (separated), which is five times more than its 5% weight-error advantage predicts — so relative weight error ranks configurations within a lattice and mispredicts across them"
last_verified_commit: "HEAD"
discovered_by: Fable 5, 5090, 2026-08-02
severity: info
status: open
related: [F0017, F0081, F0082]
---

# Finding F0083: the lattice is worth more than its error says, and the ruler I reached for was the wrong one

## Why this was asked

[F0081](0081-int4-layer-protection.md) found int4's damage is diffuse: proportional to how
many parameters are 4-bit, indifferent to which ones. That points away from placement and at
the encoding itself. Asked publicly in group 579490404 (2026-08-02) whether the answer is
therefore to swap int4 for fp4.

It is a well-aimed question and the first answer I produced was wrong, which is recorded here
because the mistake is the ordinary one.

## The first answer, and why it was wrong

I compared the two lattices at **our** group size only, got fp4 5% ahead, and was ready to
report that. One point on one axis. The axis I had not varied is the one that turned out to
matter.

## Method

Offline, no kernel and no inference. For each of the 1.5B checkpoint's 144 quantizable
matrices: take the fp16 weights, split each row into groups of `g`, scale each group by its
absmax, snap every value to the nearest lattice point, scale back, and measure
‖Ŵ − W‖ / ‖W‖. Only the lattice and `g` change; everything else is held.

The two lattices carry the same 15 values but space them differently. int4 is uniform on
[−7, 7]. fp4 is E2M1 — one sign bit, two exponent, one mantissa — giving magnitudes
{0, .5, 1, 1.5, 2, 3, 4, 6}, finer near zero and coarser in the tails. Those magnitudes were
derived from the format rather than copied, and `torch.float4_e2m1fn_x2` confirms the format
exists as described.

Scales are charged for honestly: an fp16 scale per group costs 16/g bits per weight, an fp8
scale 8/g. NVFP4 stores block scales in fp8 under a per-tensor fp32 factor, and that
two-level scheme is simulated rather than approximated — quantizing the scale to fp8 *without*
the per-tensor normalisation makes NVFP4 look 1.185× worse than int4 when it is really 0.847×,
an error large enough to invert the conclusion on its own.

## Results (1.5B, all 144 matrices, relative weight error; 1.000 = what we ship today)

| lattice | group | scale | rel. error | vs shipped | bits/weight |
|---|---:|---|---:|---:|---:|
| int4 | 16 | fp8 | 0.088051 | **0.787×** | 4.50 |
| fp4 | 16 | fp8 | 0.094773 | 0.847× | 4.50 |
| **int4** | **32** | **fp8** | **0.100422** | **0.898×** | **4.25** |
| fp4 | 32 | fp8 | 0.101669 | 0.909× | 4.25 |
| int4 | 64 | fp16 | 0.111835 | 1.000× | 4.25 |
| fp4 | 64 | fp16 | 0.106232 | 0.950× | 4.25 |

**At every matched bit budget int4 is ahead of fp4 — except at g64, which is exactly the
configuration we ship.** That is why the one-point comparison inverted the answer: we happened
to sample the only cell where fp4 wins.

The mechanism is unsurprising once the axis is visible. A short group holds values of similar
magnitude, so a uniform lattice spends its 15 levels well. A long group spans a wider dynamic
range, and that is where a float lattice's variable spacing starts to pay. The crossover for
these weights sits between g32 and g64.

**Not a 1.5B accident:** at 7.2B the fp4/int4 ratio at g64 is 0.964× against 1.5B's 0.950×,
and the whole ordering is unchanged.

## Then the accuracy measurement landed, and relative weight error turned out to be a bad ruler

Both lattices at g64, each snapped and dequantised back to fp16 so both serve on the identical
unquantized path. MATH500 avg@32, 500 problems, cluster bootstrap:

| grid (g64) | rel. weight error | MATH500 avg@32 | vs int4 (95% CI) | truncated |
|---|---:|---:|---|---:|
| int4 | 0.111835 | 0.2241 | baseline | 0.382 |
| **fp4 (E2M1)** | 0.106232 | **0.2683** | **+0.0442 [+0.0162, +0.0727] SEPARATED** | 0.351 |

The construction validates: int4 here reads 0.2241 against 0.2198 and 0.2233 from the real int4
kernel at avg@8, so dequantise-and-serve does represent the grid faithfully.

**I predicted about 1 point and it is 4.4.** The prediction came from calibrating against
F0082 — GPTQ carries 38% more weight error and costs 6.9 points, so 5% less error should buy
roughly 0.9. That calibration is wrong, and wrong for a reason worth keeping: **GPTQ's extra
error is deliberately placed where its objective says it does not matter, so it is unusually
cheap per unit. Grid error is not.** Using one to price the other under-counted by five times.

So relative weight error ranks configurations *within* a lattice and fails *across* lattices.
fp4 buys far more accuracy than its 5% L2 advantage can explain, and the likely reason is
distributional rather than magnitude: E2M1 is much finer near zero, which is where nearly all
the weight mass sits, so it protects small weights that a uniform grid rounds away. L2 counts
those errors as tiny; the model apparently does not.

**Which means the g32/fp8 "free 10%" must not be read off this rate.** Extrapolating 4.42pt per
5% would predict ~9 points, and the fp16 endpoint caps the whole gap at 18.2. The honest status
is that a promising configuration has a promising *L2* number and an unmeasured accuracy, and
L2 has just been shown to mispredict across lattices. It gets measured before it gets claimed.

## The question that dissolved it: our int4 can just be non-uniform

Asked whether int4 has to be evenly spaced, and whether fp4 needs new hardware. Both have
the same answer, and it makes the int4-vs-fp4 framing beside the point.

`rwkv7_w4.cu` already converts each nibble to a float before it multiplies:

```
int q0 = (int)((p >> 0) & 0xF);  q0 -= (q0 & 8) << 1;
float part = a0.x * (float)q0 + ...
```

Replacing `(float)q0` with `TABLE[q0]` — sixteen constants — makes the lattice arbitrary. No
fp4 instruction is ever executed, so it runs on every card the existing int4 runs on rather
than only on hardware with native fp4. And because the table is ours to choose, it need not be
E2M1's.

Fitting 15 symmetric levels to RWKV-7's own group-normalised weight distribution by Lloyd-Max:

```
[-0.9467, -0.7312, -0.5667, -0.4308, -0.3096, -0.2012, -0.0993, -0.0027,
  0.0927,  0.1926,  0.3003,  0.4190,  0.5554,  0.7185,  0.9402]
```

| lattice, g64, 4.25 bits/weight | 1.5B | 7.2B |
|---|---:|---:|
| int4 uniform | 1.000× | 1.000× |
| fp4 (E2M1) | 0.950× | 0.964× |
| **fitted table** | **0.854×** | **0.860×** |

**10% better than fp4 with none of fp4's hardware requirement**, and the table fitted on 1.5B
transfers to 7.2B untouched — 0.854× against 0.860× — so it is a property of the architecture's
weight distribution rather than of one checkpoint, which is what allows it to be a compile-time
constant instead of per-model metadata.

**Speed cost: at most 1.8%.** Two kernels moving byte-identical traffic and differing only in
the decode measure 0.1309 ms against 0.1332 ms. That is an upper bound rather than an estimate:
the probe's 8.4 MB working set is L2-resident, which over-weights ALU against the real
weight-streaming path where a lookup has more to hide behind.

**One caught error, because it would have been a good number.** The first fit produced a
**17-level** table, which four bits cannot index — the symmetric construction used `(k+1)//2`
per side instead of `k//2`. It showed 0.786× and was meaningless. Printing the table rather
than only its error is what caught it.

**What is still not known is what any of it is worth in accuracy**, and this finding is the
reason not to guess: relative weight error mispredicted fp4 by five times a few hours ago. A
MATH500 run on the fitted table is in flight.

## What follows

- **At our group size fp4 is the better lattice, worth 4.4 points — against RTN int4, which is
  not what we ship.** The shipped checkpoint is GPTQ at 0.1510 (F0082), so the gap to it is
  larger again, and that is a separate number needing its own paired run. Naming "our int4"
  without saying which one is how these two get conflated.
- **This measured a lattice, not a deployable tier, and the difference matters.** Both arms
  were dequantised to fp16 and served on the unquantized path, so what is established is what
  the grid costs in accuracy. There is no fp4 kernel here: `rwkv7_w4.cu` decodes a uniform
  lattice with an fp16 group scale and nothing else. "Same 4.25 bits, same bandwidth" is a
  property of the format, not of anything that ran — the speed side is entirely unmeasured,
  and an fp4 tier does not exist until that kernel does.
- **But do not generalise it to "fp4 beats int4".** At matched bit budgets int4 has the lower
  weight error at g16 and g32; only at g64 does fp4 lead. What is measured is one cell, and
  that cell is the one we happen to occupy.
- **Relative weight error is retired as a decision metric across lattices.** It mispredicted
  this by five times. It stays useful for ranking within one lattice, which is what the group
  size sweep is, and nothing gets claimed from it without a MATH500 run behind it.
- **Two configurations are now worth measuring, not asserting**: int4 at g32/fp8 (10% lower L2
  at unchanged bits) and fp4 at g32/fp8. Both are cheap to screen with the same
  dequantise-and-serve construction used here.
- **Acting on any of it needs kernel work.** `rwkv7_w4.cu` decodes one fp16 scale per 64
  values on a uniform lattice. fp4 changes the decode table, g32/fp8 changes the stride and
  the scale dtype. Contained, but not free, and it should follow the measurements rather than
  lead them.
