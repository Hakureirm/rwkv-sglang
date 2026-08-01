---
doc_kind: finding
finding_id: F0081
title: "Mixed-precision by layer for int4: testing the claim that protecting {0, N/4, 3N/4, N-1} rescues the reasoning collapse — and decomposing it to find which of those layers is actually doing the work"
last_verified_commit: "HEAD"
discovered_by: Fable 5, 5090, 2026-08-01
severity: info
status: open
related: [F0017, F0043, F0055]
---

# Finding F0081: which layers, if any, have to survive int4

## Why this is worth a run

Our int4 tier is the fastest single-stream product we ship — 7.2B bsz1 246.2 tok/s against
fp16's 143.2, 1.72× — and it is the one with a publicly documented hole. `docs/BENCHMARKS.md`
§4 states it plainly: 1.5B MATH500 avg@64 **0.1498** symmetric / **0.2199** asymmetric-GPTQ
against fp16's **0.4060**, a 19–26 point collapse that perplexity-style rulers do not see.
F0043 closed 27% of that gap with a better encoding and concluded, on the evidence, that the
residue "is not simply a bit-budget problem" — a bigger scale/zero grid bought back less on
MATH500 than it did on lambada or compression. That is the shape of a problem localized in a
few places rather than spread evenly over the weights.

Two public statements in group 579490404 on 2026-08-01 point at the same place from different
directions. Bo, the model's author: layer 0's `v` is the source of `v_first`, which every later
layer consumes, so an error there cannot be averaged away downstream. Separately, a claim that
protecting four layers at fixed positions `{0, round(N/4), round(3N/4), N-1}` in high precision
recovers what protecting eleven layers does. Neither is a measurement on our stack, and the
second is a *positional formula* — the kind of claim that is easy to adopt whole and hard to
believe without decomposing.

So the run is not "reproduce the recipe". It is: **is there an effect, and if so, is the
positional formula the reason, or is the whole thing one layer wearing a formula's clothes?**

## Design

Four arms. One variable: which layers keep checkpoint precision. Same symmetric g64 quantizer
(`bench/quant_w4.py --keep-layers`), same server flags, same MATH500 harness, same card, one
session. 1.5B is 24 layers, so the positional formula is `{0, 6, 18, 23}`.

| arm | protected | what it isolates |
|---|---|---|
| NONE | — | our shipping int4; also the pipeline's sanity check against the documented 0.1498 |
| L0 | {0} | the `v_first` source alone |
| MID | {6, 18} | the two "dilution" points **without** L0 |
| ALL4 | {0, 6, 18, 23} | the full positional formula |

**MID is the discriminator.** The arms were chosen so that the interesting outcomes are
distinguishable before any of them is measured:

- If `L0 ≈ ALL4` and `MID ≈ NONE` → the effect is entirely `v_first`. The positional formula is
  decoration, and the correct product is "protect layer 0", which costs a quarter of the VRAM
  the formula does. This would be new information, not a reproduction.
- If `ALL4 > L0` and `MID > NONE` → error accumulation is real and distributed, the midpoints
  genuinely reset it, and the formula earns its cost.
- If `ALL4 ≈ NONE` → the collapse is not localized in whole layers at all, and the whole
  direction is closed. F0043's residue would then point at per-*tensor* sensitivity (the
  hypothesis that some projections, not some layers, cannot take 4 bits).

Screened at avg@8 (500 problems × 8 rollouts) rather than avg@64: the effect under test spans
0.15 → 0.40, roughly thirty times the resolution of 4,000 rollouts, and avg@64 costs ~2h per
arm against ~15 min. The winner earns an avg@64 before any number leaves this repo.

`RWKV_SPARSE_FFN` is forced OFF in every arm. It is ineligible on quantized projections but
*would* become eligible on protected ones — precisely the confound that would let the arms
differ for a reason other than precision.

**Two things this design is careful about, because both have bitten this repo before:**

- **The flag is gated for liveness before it is trusted.** An ignored env knob still serves
  plausible numbers. `RWKV_W4_KEEP_LAYERS` was proved load-bearing by a mismatch that *must*
  fail: the ALL4 checkpoint leaves layers 0/6/18/23 as `.weight`, so booting it without the
  flag builds `W4Linear` there and the `.qweight` keys it demands do not exist. Three gates,
  all passed — matched pair boots, ALL4-without-flag dies, NONE-with-flag dies.
- **Arm NONE is a pipeline check, but the published numbers turned out not to be a valid
  reference for it.** The published 1.5B MATH500 figures — 0.1498 symmetric, 0.2199
  asymmetric — are both **GPTQ** checkpoints, calibrated on Hessians this box no longer has.
  All four arms here are plain **RTN** (`bench/quant_w4.py`, round-to-nearest, no
  calibration), because RTN is the one quantizer reproducible from the fp16 checkpoint alone
  and therefore the only way to hold everything but the protected set constant tonight.

  I first wrote that NONE should therefore land at or *below* 0.1498, citing BENCHMARKS §4 for
  RTN being the weaker encoder. That citation does not support the claim: the line compares
  **7.2B**-RTN (+0.0202) against **1.5B**-GPTQ (+0.0429), two different model sizes, and says
  nothing about RTN versus GPTQ at fixed size. We have no same-size RTN-vs-GPTQ MATH500
  comparison at all. The prediction was unfounded and is withdrawn rather than rescued.

  It also failed: NONE measured **0.2198**, above the published symmetric 0.1498 and within
  noise of the published *asymmetric* 0.2199. Since the two runs differ in card (3090 vs
  5090), fused stack, encoder, and rollout count, the honest move is to stop comparing across
  them and measure the reference here instead — see the GPTQ and fp16 arms added below.

**What this design therefore cannot answer.** Whether layer protection *stacks* with GPTQ. The
shipping product would be asymmetric-GPTQ plus whatever protection survives, and the two could
overlap: GPTQ's error feedback may already be absorbing some of the same damage that protecting
layer 0 absorbs, in which case the combined gain is less than the sum. Answering that needs a
fresh calibration pass to regenerate Hessians, which is a separate run and only worth its cost
if the protection effect is real. Stated here so the result is not over-read.

## How much this screen can actually see

Worth stating before any arm is read, because it decides which results are conclusions and
which are silence. `bench/math500_compare.py` (new, this finding) resamples **problems** with
replacement and carries every rollout of a resampled problem with it — a cluster bootstrap.
The naive binomial bar on 500×8 treats 4,000 generations as independent trials, which they are
not: a problem the model can do it does most of the time, one it cannot it never does.

Measured on this run: the baseline's own 95% CI is about **±3pp**, and a *paired* difference
against it about **±2pp**. So this screen resolves effects of roughly **4pp and up**, and is
blind below that. The claim under test — protection recovering what eleven protected layers
recover, against a collapse of 19–26pp — sits far above that floor. An effect small enough for
this screen to miss would not justify the VRAM it costs anyway.

The comparator was checked by breaking it, not just by running it: on synthetic arms drawn from
one distribution it reports "not separated" despite a +1.07pp point estimate; on a planted
+10pp effect it separates; on an empty file it refuses rather than reporting 0.0000.

**A bootstrap does not measure everything that varies, and there is a free control for the
rest.** Resampling tells you how much the estimate would move on a different draw *of
problems*, holding the generations fixed. It cannot tell you how much it would move if you
re-ran the whole thing — and sglang's sampler is not seed-controlled per request
(`bench/math500_avg64.py` L66-67), so a second run of an *identical* configuration draws
entirely new rollouts. Arm **NONE2** is that run: same checkpoint, same flags, nothing changed
but the dice. Whatever NONE and NONE2 differ by is the pipeline's own reproducibility floor,
and no arm separated by less than it has been shown to do anything. It was added as a check
that swapping in the tensor-axis binary was a no-op; it turned out to be the better control.

## Two traps this run walked into, recorded because they generalise

- **A late import costs a whole arm.** The first launch booted a server, measured bsz1,
  generated for fifteen minutes and only then died on `import math_verify` — the grader is
  imported inside the verify worker, which runs after every rollout exists. `math500_avg64.py`
  now checks the grader at startup, and the runners preflight it before spending GPU time.
- **"Wait for the other job to disappear" is not a lock.** The reference and tensor sweeps were
  chained by two host-side scripts, each polling `docker ps` for the other's container to
  vanish. That condition is also true *before* the other container starts, so in a ten-second
  window both launched, and since each sweep begins its arms with `pkill -f
  sglang.launch_server`, they began killing each other's servers. The give-away was
  `VRAM NONE2 11816 MiB` for a model that takes 5908 — almost exactly two of them. The
  contaminated arms were discarded and both chains replaced with one container running arms in
  sequence, plus a per-arm VRAM guard that aborts above 16 GB. A resource this cheap to check
  should be checked, not assumed: nothing else in the run would have looked wrong.

## Results

All arms: 1.5B, RTN symmetric g64, 5090, sglang main, `RWKV_SPARSE_FFN=0`,
`RWKV_STATE_FP16=1 RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=1`, MATH500 avg@8 (500×8), one
session. The state-fp16 flag is a numerics choice that could itself depress the absolute
numbers; it is held constant across every arm including the fp16 ceiling, so the comparison is
unaffected.

| arm | protected | MATH500 avg@8 | vs NONE (95% CI) | bsz1 tok/s | VRAM |
|---|---|---:|---|---:|---:|
| arm | protected | MATH500 avg@8 | vs NONE (95% CI) | bsz1 tok/s | VRAM |
|---|---|---:|---|---:|---:|
| _fp16 ceiling_ | _(not quantized)_ | _0.4020_ | _+0.1823 SEPARATED_ | _433.1_ | _7750 MiB_ |
| NONE | — | 0.2198 | baseline, CI [0.1895, 0.2507] | 710.7 | 5908 MiB |
| NONE2 | — (control) | 0.2233 | +0.0035 [−0.0075, +0.0145] **not separated** | 711.6 | 5908 MiB |
| L0 | {0} | 0.2055 | −0.0143 [−0.0345, +0.0057] **not separated** | 692.5 (−2.6%) | 5970 MiB |
| MID | {6, 18} | 0.2160 | −0.0038 [−0.0245, +0.0165] **not separated** | 675.1 (−5.0%) | 6050 MiB |
| ALL4 | {0, 6, 18, 23} | 0.2387 | +0.0190 [−0.0068, +0.0448] **not separated** | 642.9 (−9.5%) | 6198 MiB |

Protecting layer 0 — the `v_first` source, the single strongest a-priori candidate and the one
Bo named — moved nothing this screen can see, and cost 2.6% of single-stream throughput. Its
point estimate is *negative*. The two dilution midpoints moved nothing either.

**The control did its job twice over.** NONE2 reproduces NONE to **+0.35pp** (711.6 vs 710.7
tok/s, identical 5908 MiB), which settles the thing it was added for — swapping in the
tensor-axis binary is a no-op, so the tensor arms are comparable to these — and also puts a
number on something the bootstrap cannot reach: re-rolling every generation on an unchanged
configuration moves the score by about a third of a point.

**ALL4 is the one arm that is not flat, and it deserves to be stated carefully rather than
either claimed or dismissed.** At +0.0190 it is the only positive point estimate, and its
interval sits ~74% above zero. Against the reproducibility floor it looks more interesting than
that: +1.90pp is roughly five times the +0.35pp two identical runs drifted. But one control run
is one draw of that drift, not its distribution, and the paired bootstrap interval — the
conservative instrument, since it also carries the weight change ALL4 actually makes — still
includes zero. Two honest readings survive: noise (both components measured null, and four arms
give noise four chances to look interesting), or a real effect around 2pp that avg@8 cannot
resolve. Only more rollouts separate those, and that is queued behind the arms that could
change the picture more.

What does **not** depend on resolving it: **the decision.** Taken at face value, +1.9pp is two
orders of magnitude short of the 19–26pp the intervention was being considered for, and it
costs 9.5% of single-stream throughput and 290 MiB. That is a bad trade at its own best-case
point estimate, so the product answer is settled even though the scientific one is not.

**The VRAM ladder is an independent check that the protection was real**, and a quantitative
one rather than a "the flag seemed to do something" one. Per protected layer the model carries
4 attention projections at 2048² plus ffn key and value at 2048×8192 = 50.33M parameters:
int4 g64 stores that as 25.2 MB of nibbles plus 1.6 MB of scales, fp16 as 100.7 MB, so each
protected layer should cost **~70.5 MiB**. Measured, against 0 protected layers: +62, +142,
+290 MiB for 1, 2 and 4 layers — ~72 MiB each, linear in the count. The arms differ in exactly
the way and by exactly the amount the intervention predicts; they simply do not differ in
accuracy.

**And the price is linear too.** Single-stream throughput falls 2.6%, 5.0%, 9.5% for 1, 2 and
4 protected layers — a flat ~2.4% per layer, which is what bandwidth-bound decode should cost
when a layer's weights go from 4 bits to 16. So the trade this intervention offers on our int4
is fully characterized on both sides: **~2.4% of single-stream throughput and ~72 MiB per
protected layer, bought with no measurable accuracy.**

_(ALL4 and the same-stack GPTQ and fp16 references pending.)_

## The flagship: same answer, and the same non-answer

7.2B is 32 layers, so the positional formula moves to `{0, 8, 24, 31}` — 805.3M of the model's
6,443M quantizable parameters, a 12.5% dose. Same box, same session, avg@8:

| arm | MATH500 | vs RTN (95% CI) | bsz1 |
|---|---:|---|---:|
| fp16 | 0.6370 | +0.0165 [−0.0095, +0.0435] not sep. | 109.1 † |
| RTN (no protection) | 0.6205 | baseline | 246.1 |
| RTN + `{0,8,24,31}` | 0.6348 | +0.0142 [−0.0140, +0.0420] **not separated** | 212.7 (−13.6%) |
| GPTQ | 0.6112 | −0.0093 [−0.0405, +0.0220] not sep. | 246.1 |

† handicapped: every arm runs `RWKV_SPARSE_FFN=0` so the arms stay comparable, which costs only
the unquantized one ~24%. Our published fp16 bsz1 is 143.2. Do not quote 109.1.

The positional formula gives **+1.42pt, not separated, for 13.6% of single-stream throughput** —
the same shape as 1.5B's +1.90pt for 9.5%. The product conclusion carries to the flagship
unchanged.

**But be careful reading the 7.2B column: nothing in it separates from anything else, including
fp16 from int4.** Scores here sit near 0.62 with more heterogeneous problems, so the baseline
interval is ±3.8pp and paired intervals run to ±3.1pp — a resolution floor around 5–6pp, above
every effect present. BENCHMARKS §4's avg@64 measurement does resolve GPTQ at −3.1pt from fp16,
and that remains the better estimate. So 7.2B here says "no intervention is worth its cost at
any magnitude this screen can see", which is the product answer, and it does **not** say the
effects are zero.

## What follows

**Do not adopt mixed precision on our int4 — on either axis, at either size.** Seven arms at
1.5B across two axes and four doses, plus the formula at 7.2B, and every one lands where its
parameter count predicts rather than where its mechanism predicts. There is no sensitive layer
and no sensitive projection kind to protect; int4's damage here is diffuse. The exchange rate
is ~1.0pp of MATH500 per 100M parameters restored across the entire affordable half of the
model, against a strictly linear cost, and the half of the weights actually worth paying for is
the half that erases the memory saving you quantized for.

This is not a power problem. The screen resolves 4pp, reproduces both published reference
points, and did separate the two largest-dose arms — it simply found nothing where the
hypotheses said to look.

`RWKV_W4_KEEP_LAYERS` and `RWKV_W4_KEEP_TENSORS` stay in the tree because they are the
instruments that produced this result and the ones that would retest it on another model,
another quantizer, or another bit-width. Both stay **default-empty**, which is a no-op.

**What this does and does not say about the claim it came from.** It says the intervention does
not transfer to weight-only int4 g64 RTN at 1.5B on our stack. It does not say the original
comparison was wrong, and two differences matter enough that I would expect a null here even if
that comparison were exactly right:

- **The claim is a comparison inside another scheme, not a standalone rescue.** What was stated
  publicly is that 4 bf16 layers reach the accuracy of 11 bf16 layers — a comparison of two
  protection budgets *within* an NVFP4+FP8+AWQ mixed-precision design whose primary mechanism
  is assigning different bit-widths per projection kind. I tested something else: whether
  adding 4 protected layers to *our* uniform int4 helps. A null on my question leaves that
  comparison untouched.
- **The baselines are not in the same regime.** That design's unprotected baseline (pure NVFP4)
  was reported at MATH500 8.6% — a model that has essentially stopped reasoning. Ours sits at
  0.2198. An intervention whose job is to reset accumulated error has far more to reset when
  there is far more error, so "recovers a catastrophic baseline" and "improves a degraded one"
  are different claims and only the second was testable here.

## The tensor axis, and an accidental dose-response test

Each RWKV-7 layer carries 50.33M quantizable parameters (r/k/v/o at 2048² = 16.78M, `ffn.key`
and `ffn.value` at 2048×8192 = 16.78M each). Ordering the arms by how many of those parameters
are lifted back to fp16:

| arm | protected | params kept fp16 | MATH500 | vs NONE (95% CI) | bsz1 | VRAM |
|---|---|---:|---:|---|---:|---:|
| NONE | — | 0 | 0.2198 | baseline | 710.7 | 5908 MiB |
| L0 | layer 0 | 50.3M | 0.2055 | −0.0143 not sep. | 692.5 | 5970 MiB |
| MID | layers 6,18 | 100.7M | 0.2160 | −0.0038 not sep. | 675.1 | 6050 MiB |
| ALL4 | layers 0,6,18,23 | **201.3M** | 0.2387 | **+0.0190** not sep. | 642.9 | 6198 MiB |
| KV | `k_proj`,`v_proj` × all 24 | **201.3M** | 0.2402 | **+0.0205** not sep. | 633.2 | 6282 MiB |

**ALL4 and KV protect exactly the same number of parameters — 201.3M — in completely disjoint
sets of weights, and they land in the same place.** The equality is an architectural
coincidence (4 layers × 50.33M = 24 layers × 2 × 4.19M), and it turns two arms that were
designed to test different hypotheses into a controlled dose-versus-location experiment nobody
planned. One protects everything in four layers; the other protects two small attention
projections in every layer. Same dose, no overlap, +1.90 versus +2.05 points.

Read the whole column and the same thing shows: −1.43, −0.38, +1.90, +2.05 as 50M, 101M, 201M,
201M come out of int4 — **monotone in dose, indifferent to placement.** Which is a duller
hypothesis than either of the ones under test, and more useful: there may be no sensitive
location to find. If int4's damage is diffuse, then "protect the right layers" and "protect the
right projections" are both the wrong question, and mixed precision is just a smooth
accuracy-for-VRAM dial with no clever placement available.

**FFNV doubles the dose and doubles the effect — and is the first arm to separate.**

| arm | params kept fp16 | MATH500 | vs NONE (95% CI) | per 100M params | bsz1 |
|---|---:|---:|---|---:|---:|
| ALL4 | 201.3M | 0.2387 | +0.0190 not sep. | +0.94pp | 642.9 |
| KV | 201.3M | 0.2402 | +0.0205 not sep. | +1.02pp | 633.2 |
| **FFNV** | **402.7M** | **0.2622** | **+0.0425 [+0.0192, +0.0658] SEPARATED** | **+1.06pp** | 568.5 |

Three arms, three disjoint weight sets, two different doses — and a constant exchange rate of
about **one point of MATH500 per 100M parameters lifted out of int4**. The two low-dose arms
(L0 at −1.43, MID at −0.38) sit off this line, but their true effects under the same rate would
be +0.5 and +1.0 against a ±2pp noise floor, so they are where noise puts them, not evidence
against it.

**The prediction, written before the last arm reports.** KVFFNV protects k/v *and* `ffn.value`
in every layer: 604.0M parameters, 3× ALL4's dose. The two hypotheses now give different
numbers:

- **dose** (nothing is special, the rate is ~1.02pp/100M) → **+6.2pp**, landing near 0.282
- **placement** (`ffn.value` is special because it feeds the state accumulation, as claimed
  publicly) → adding k/v to FFNV buys little, so **~+4.5pp**, landing near 0.265

These are far enough apart for this screen to tell them apart. Recorded here so the result
cannot be read as whichever one it lands on.

**Measured: +6.00pp, landing at 0.2797, separated — CI [+0.0352, +0.0855].** The prediction was
+6.2pp near 0.282. Dose wins out of sample; placement is out by a mile. `ffn.value` is not
special, and neither is layer 0 — the state-accumulation argument, which is the strongest
mechanistic story anyone offered, does not survive its own prediction.

(One correction to my own arithmetic: I derived +6.2 from a locally constant rate, then fitted a
power law that implied ~+7.3. Both were "dose", so the discriminator held, but the two should
have been reconciled before being written down as one prediction.)

## The complete dose curve, and why it kills mixed precision here

| params kept fp16 | % of quantizable | MATH500 | vs NONE | rate over previous | bsz1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0% | 0.2198 | — | — | 710.7 |
| 50.3M | 4% | 0.2055 | −1.43pp | (noise) | 692.5 |
| 100.7M | 8% | 0.2160 | −0.38pp | (noise) | 675.1 |
| 201.3M | 17% | 0.2387 / 0.2402 | +1.90 / +2.05pp | ~1.0pp/100M | 642.9 / 633.2 |
| 402.7M | 33% | 0.2622 | +4.25pp | ~1.1pp/100M | 568.5 |
| 604.0M | 50% | 0.2797 | +6.00pp | ~0.9pp/100M | 522.4 |
| 1,208M (= fp16) | 100% | 0.4020 | +18.2pp | **~2.0pp/100M** | 433.1 |

The curve is not a power law — fits through different points disagree (exponent 1.32 via the
402.7M point, 1.60 via the 604M point), so it is better described as **two regimes**: a flat
~1.0pp per 100M across the whole first half of the model, then roughly **double that rate**
across the second half. The last half of the weights is worth twice as much per parameter as
the first half.

Cost, meanwhile, is strictly linear: every protected parameter costs the same 4× bytes to store
and stream, and bsz1 falls almost exactly in proportion (−9.5%, −10.9%, −20.0%, −26.5%).

**So partial protection is always the worst part of its own curve.** You buy the cheap half of
the accuracy at full price, and the half that is actually worth paying for is the half you
cannot afford — by then you have given back the memory saving that was the point. Every
affordable configuration is dominated by either shipping plain int4 or shipping fp16.

And all of it is dominated by the free option. The best mixed-precision arm tested — half the
model at fp16 — buys **+6.00pp for 26.5% of single-stream throughput and 888 MiB**.
[F0082](0082-gptq-loses-to-rtn-on-math500.md)'s encoder swap buys **+6.9pp for nothing**: same
format, same size, same speed to within 0.2%. A one-line change to which checkpoint we publish
beats the entire mixed-precision design space explored here.

**The mechanism actually cited was tested too, and it also fails.** The public argument for
protection is not positional — it is that `att.key` sets the attention weight distribution and
`att.value` enters the state accumulation, so 4 bits *there* is what breaks; Bo's `v_first`
point has the same shape. Those are claims about **projection kinds**, so they were tested
directly via `bench/quant_w4.py --keep-tensors` / `RWKV_W4_KEEP_TENSORS` (suffixes verified
against real checkpoint keys — `value` matches `ffn.value` and does not catch `v_proj`). Every
one of those arms lands exactly where its *parameter count* predicts, and nowhere near where
its *mechanism* predicts. Whatever int4 damages here, it is spread across the weights rather
than concentrated in a nameable place.

**There is a real gap, and protection is not what closes it.** The fp16 ceiling measured on this
same stack is **0.4020** — reproducing the published avg@64 0.4060, with matching truncation
(0.146 vs 14.2%) and generation length (584 vs 581 tokens). So int4's collapse is confirmed
here and not an artifact of the rebuilt stack: RTN sits **18.2 points** below fp16. Against that
denominator, the entire positional formula's best-case +1.9pp closes about **10% of the gap**,
and does not separate from zero.

**A second thread opened by accident, and it matters more than the first.** Chasing why arm NONE
disagreed with the published 0.1498 — rather than writing it off as card-and-stack drift — is
what produced [F0082](0082-gptq-loses-to-rtn-on-math500.md): the published GPTQ checkpoint
reproduces its own number here (0.1510), which means plain RTN beats our calibrated, shipped
int4 by **6.9 points, separated**, at identical speed and size. That is three and a half times
what the intervention this finding set out to test could offer, it costs nothing instead of
9.5% of throughput, and it was sitting in a comparison nobody had run. The discrepancy was the
finding.
