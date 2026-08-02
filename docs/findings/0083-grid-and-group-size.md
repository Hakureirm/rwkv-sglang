---
doc_kind: finding
finding_id: F0083
title: "fp4 beats int4 by 4.42 points of MATH500 at the group size we actually ship (separated), and the mechanism is total weight error rather than the lattice's shape: two lattices held at identical error and 4.6x apart at the origin differ by 1.5 points and do not separate"
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

Everything below comes from `bench/grid_sweep.py`, which both prints this table and writes the
checkpoints that get served, so the configuration whose error is quoted is the one that ran.
It was written against the earlier sweep rather than from it and reproduces all six original
cells to six digits, which is the only reason the g64 arms and the g32 arms can be compared:
the g64 arms were built before the fp8 path existed, and had the fp8 penalty been charged to
one side only the comparison would tilt by more than the effect. Casting a g64 scale to fp16
does not move its error at six digits, so nothing is owed there.

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

## Then the accuracy measurement landed, and relative weight error looked like a bad ruler

> **Retracted below.** This section prices fp4 against GPTQ and concludes weight error is
> five times off. The calibration is the error, not the ruler — see *Refuted* further down,
> where an arm built to test the replacement story kills it and error survives.

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

## The fitted table's extra 10% buys nothing, which mispredicts the other way

Same construction, same avg@32, same 500 problems:

| grid (g64, 4.25 bits) | rel. weight error | MATH500 avg@32 | vs int4 | vs fp4 |
|---|---:|---:|---|---|
| int4 | 1.000× | 0.2241 | baseline | |
| fp4 (E2M1) | 0.950× | 0.2683 | +0.0442 [+0.0161, +0.0727] **SEP** | baseline |
| fitted table | **0.854×** | 0.2704 | +0.0463 [+0.0187, +0.0733] **SEP** | +0.0021 [−0.0286, +0.0326] **not sep** |

The table carries 10% less weight error than fp4 and lands 0.2 points away from it. So L2 has
now mispredicted twice in opposite directions in one afternoon: it undercounted fp4's gain over
int4 by five times, and it overcounted the table's gain over fp4 down to nothing. **A metric
that errs in both directions is not miscalibrated, it is measuring the wrong thing.**

One property separates the two winners from the loser. Spacing at the origin, in units of the
group's absmax:

| lattice | gap either side of zero | MATH500 |
|---|---:|---:|
| fp4 (E2M1) | 0.083 | 0.2683 |
| fitted table | 0.096 | 0.2704 |
| int4 uniform | 0.143 | 0.2241 |

Accuracy tracks that column and not the error column. The reading is that resolution near zero
is what a 4-bit lattice has to buy — nearly all the weight mass sits there — and that past
roughly a tenth of absmax, buying more of it stops paying.

**That story was fitted to three points after seeing them, which is the failure
[F0081](0081-int4-layer-protection.md) is a record of.** So it is written down as a prediction
before the arms that discriminate it exist.

> **This story is dead.** The arm registered below refutes it: at matched error, a 4.6×
> difference at the origin is worth +1.5 points and does not separate. Three points and a
> plausible mechanism, again.

### Registered before the run

int4 at g32 with an fp8 group scale costs the same 4.25 bits and carries **0.898×** the weight
error — *less* than the 0.950× that bought fp4 its 4.42 points — while staying uniform, so its
gap at zero shrinks only with the group absmax, by about 10%. The two accounts disagree:

| | int4 g32/fp8 (0.898×) | table g32/fp8 (0.806×) |
|---|---|---|
| if total error drives it | above fp4's 0.2683, since 0.898 < 0.950 | best of everything, above 0.2704 |
| if the gap at zero drives it | ≈ 0.235, no separation from int4 g64 | ≈ 0.2704, no separation from table g64 |

**Predicting the second.** Concretely: int4 g32/fp8 lands between 0.225 and 0.245, does not
separate from int4 g64, and stays below fp4 g64. If instead it reaches 0.2683 the near-zero
story is dead and total error is back.

Stated limit: at avg@32 a paired difference resolves to about ±0.028, so the two predictions
are separated by roughly one CI width. A result near 0.235 refutes the error account outright
because that account's floor is 0.2683. A result in between refutes neither, and that is the
outcome this screen cannot settle.

### It landed in between, which is the outcome that settles nothing

| grid | rel. err | gap@0 | MATH500 avg@32 | vs int4 g64 |
|---|---:|---:|---:|---|
| int4 g64/fp16 | 1.000× | 0.143 | 0.2241 | baseline |
| **int4 g32/fp8** | **0.898×** | 0.143 | **0.2417** | +0.0176 [−0.0089, +0.0441] not sep |
| fp4 g64/fp16 | 0.950× | 0.083 | 0.2683 | +0.0442 [+0.0161, +0.0727] **SEP** |
| table g64/fp16 | 0.854× | 0.096 | 0.2704 | +0.0463 [+0.0187, +0.0733] **SEP** |
| table g32/fp8 | 0.806× | 0.096 | 0.2782 | +0.0541 [+0.0253, +0.0824] **SEP** |

The registered range was 0.225–0.245 with no separation from int4 g64 and a position below
fp4. All three hold: 0.2417, +0.0176 not separated, below 0.2683.

**The registered refutation does not.** I wrote that a result near 0.235 kills the error
account because its floor is 0.2683 — but the test of "int4 g32 should beat fp4" is the paired
comparison, and that is **−0.0266 with CI [−0.0552, +0.0020]**, zero inside it by 0.002. So
the arm came out exactly where the stated limit says nothing is settled. Naming the floor was
not the same as testing against it, and the point prediction landing is not the claim that
mattered.

What the two g32 arms do establish is the narrower thing: **within a fixed lattice, lower
error does buy accuracy, monotonically.** int4 g64→g32 is +1.76 points for 10.2% less error,
table g64→g32 is +0.78 for 5.6% less. Neither separates alone; both point the same way, and
that is what keeps weight error usable inside one lattice while it fails across lattices.

## The arm where the two accounts predict opposite signs

Every arm so far moved total error and origin spacing **together**, which is why none of them
separated. That is not bad luck, it is structural: the error-optimal placement of levels is
itself "fine where the mass is", so refining the origin always lowers error. Trying to build
a fine-origin lattice at 1.33× error hit this directly — the search family saturated at
1.243× and could go no further, because every remaining move toward a finer origin *reduced*
the error it was supposed to raise.

The lever that does separate them is to hold error fixed and move only the origin. Two
lattices, both snapped into served checkpoints measuring **0.148744** and **0.148742** —
identical error to five figures, 4.6× apart at zero:

| | gap@0 | levels (mirrored about zero, ×absmax) |
|---|---:|---|
| **fine** | 0.0500 | 0.05, 0.10, 0.2046, 0.3092, 0.4138, 0.5184, 1.0 |
| **coarse** | 0.2294 | 0.2294, 0.3794, 0.5094, 0.6294, 0.7494, 0.8747, 1.0 |

`fine` pays for its origin by starving the tail: nothing between 0.518 and 1.0 of a group's
absmax. `coarse` pays the opposite way: nothing below 0.229, so everything under 0.115 snaps
to zero outright. Same total error, opposite priorities.

**Registered before running.** The origin account wins: `fine` lands four or more points above
`coarse` and they separate, with `coarse` the worse arm outright and plausibly catastrophic.
The error account predicts they land on top of each other, within a point or two, not
separated. Both carry a third more error than shipped int4, so both should score below 0.2241
— the comparison is fine-against-coarse, and nothing is claimed from either against int4.

This design cannot distinguish "the origin matters" from "the tail does not". If `coarse`
wins instead, the reading is the reverse of both accounts and the tail is what fp4 was buying.

### Refuted. The origin buys +1.5 points and does not separate

| lattice | rel. err | gap@0 | MATH500 avg@32 | vs int4 g64 |
|---|---:|---:|---:|---|
| **isofine** | 1.330× | 0.0500 | **0.1187** | −0.1054 [−0.1323, −0.0799] SEP |
| **isocoarse** | 1.330× | 0.2294 | **0.1036** | −0.1206 [−0.1487, −0.0922] SEP |

**isofine − isocoarse = +0.0151, 95% CI [−0.0062, +0.0371], not separated.** I registered
four or more points and separation. It is one and a half and it does not separate, which is
the error account's prediction almost exactly — "within a point or two, not separated".

So the story I fitted to three points does not survive the arm built to test it. Resolution at
the origin is not what fp4 and the fitted table were buying. Held against total error, a 4.6×
difference in how finely a lattice represents the region holding nearly all the weight mass is
worth about a point.

**Which forces a correction to this finding's own headline.** "Relative weight error is retired
as a decision metric across lattices" was itself built on a bad calibration: the "five times
off" figure came from pricing fp4 against GPTQ, and this finding already says in the same
breath that GPTQ's error is unusually cheap per unit. Pricing anything with it was the error.
Against the seven arms now measured, error tracks accuracy across the whole 0.806×–1.330×
range:

| rel. err | 0.806 | 0.854 | 0.898 | 0.950 | 1.000 | 1.330 | 1.330 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MATH500 | 0.2782 | 0.2704 | **0.2417** | **0.2683** | 0.2241 | 0.1187 | 0.1036 |

One inversion, int4 g32 against fp4 g64, and it is the pair that does not separate. A straight
line through the endpoints gives about 3.2 points per 0.1 of relative error and leaves
residuals of two to three points — the same size as the intervals. **At this resolution nothing
beyond total weight error is demonstrated, and that is the whole claim.**

**The limit this test has, stated rather than buried.** Both iso arms sit at 1.33× error, where
accuracy has already fallen by half from int4. Differences may compress near that floor, so
what is refuted is "origin resolution is the dominant mechanism", not "origin resolution is
irrelevant at the group sizes we ship". Re-running the pair at 1.05× would test it where the
real lattices live.

### Repeated at 1.05×, and the floor objection does not survive either

Same construction just above the shipped configuration — served checkpoints at **0.117421** and
**0.117427**, both 1.050× int4, 3.3× apart at the origin. Registered beforehand: null again,
the two within two points and not separated, both a little below int4.

| lattice | rel. err | gap@0 | MATH500 avg@32 | vs int4 g64 |
|---|---:|---:|---:|---|
| isofine105 | 1.050× | 0.0500 | 0.2201 | −0.0041 [−0.0320, +0.0232] not sep |
| isocoarse105 | 1.050× | 0.1656 | 0.2169 | −0.0072 [−0.0321, +0.0172] not sep |

**isofine105 − isocoarse105 = +0.0032, CI [−0.0240, +0.0300], not separated.** All three parts
of the registered prediction hold.

And the effect did not grow when the floor was removed — it **shrank**, from +1.5 points at
1.33× to +0.3 at 1.05×, which is the opposite of what a compression artefact does. Where we
actually operate, with accuracy at int4's level and room to move in either direction, a 3.3×
difference in how finely a lattice resolves the region holding nearly all the weight mass is
worth three tenths of a point.

Two independent error levels, both null. The mechanism is total weight error.

## The one arm that contradicted that, and what it turned out to be

int4 g32/fp8 carries **less** total error than fp4 g64/fp16 (0.898× against 0.950×) and
scores below it. At avg@32 that gap did not separate. Doubling the rollouts to avg@64
was registered as the tiebreak, with the prediction that the inversion would not survive
and a stated consequence if it did: lattice shape buys something error does not explain,
and this finding needs an exception.

It survived, and separated:

| pair, avg@64 | difference | 95% CI | |
|---|---:|---|---|
| fp4 g64/fp16 − int4 g32/**fp8** | **+0.0278** | [+0.0002, +0.0557] | **SEPARATED** |

Except the two arms differ in two ways, not one. fp4 carries an fp16 scale per 64
weights; int4 g32 carries an **fp8** scale per 32. Both are 4.25 bits and the error
number folds the scale in, but fp8 rounding through a per-tensor factor puts error
somewhere different from where a shorter group removes it, and no arm had separated
those. So: int4 at g32 with an fp16 scale, changing only the scale dtype. It costs 4.50
bits, so it is a diagnostic, not a tier.

| arm | rel. err | bits | MATH500 avg@64 |
|---|---:|---:|---:|
| int4 g32/fp8 | 0.898× | 4.25 | 0.2400 |
| **int4 g32/fp16** | **0.893×** | 4.50 | **0.2518** |
| fp4 g64/fp16 | 0.950× | 4.25 | 0.2678 |

| pair | difference | 95% CI | |
|---|---:|---|---|
| fp4 − int4 g32/**fp16** | +0.0160 | [−0.0109, +0.0431] | not separated |
| int4 g32/fp16 − int4 g32/fp8 | +0.0118 | [−0.0147, +0.0380] | not separated |

**The exception does not survive the confound.** The only separated inversion in the
whole sweep is the one carrying an fp8 scale; take the fp8 scale away and the same
comparison stops separating. Nothing here establishes that lattice shape buys anything
total error cannot explain.

What it does raise, without establishing it: **an fp8 group scale may cost more than its
contribution to weight error suggests.** Those two int4 arms differ by 0.5% in error and
1.18 points in accuracy — far steeper than anything else in the sweep, where roughly 0.1
of relative error buys 3 points. That is a suspicion with an unseparated difference
behind it, and it is the arm to run next, because the g32/fp8 "free 10%" was the whole
practical proposal and this is the first hint that the fp8 half of it is not free.

**Owed caveat on the separation that started this.** Its lower bound is +0.0002, and by
this point the finding has run well over a dozen paired comparisons. At that many looks
a 95% interval clearing zero by two ten-thousandths is weak on its own, and it is being
read here as "worth chasing", which is what it turned out to deserve.

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
- **Relative weight error is not retired.** An earlier draft of this finding retired it on the
  strength of a GPTQ-based calibration that this same finding calls unreliable two paragraphs
  earlier. Across seven arms spanning 0.806×–1.330× it tracks accuracy with residuals the size
  of the confidence intervals, and the one arm designed to expose a mechanism it misses came
  back null. Use it to rank; do not read points off it without a run behind them, because the
  slope is only known to within a factor of two.
- **The fitted table is not worth its extra complexity over E2M1 on this evidence.** It wins on
  L2 and ties on accuracy. What it does keep is the hardware argument: a lookup runs on every
  card the current int4 runs on, which E2M1 as a native format does not.
- **Acting on any of it needs kernel work.** `rwkv7_w4.cu` decodes one fp16 scale per 64
  values on a uniform lattice. fp4 changes the decode table, g32/fp8 changes the stride and
  the scale dtype. Contained, but not free, and it should follow the measurements rather than
  lead them.
