---
doc_kind: finding
finding_id: F0082
title: "At 1.5B, our calibrated int4 (GPTQ) is 6.9 points WORSE on MATH500 than plain round-to-nearest — separated, same base model, same session. GPTQ concentrates its weight error on ffn.value at both sizes, which F0080's sparsity result explains; but 7.2B's total quantization damage is only −3.1pt, so the flagship effect is bounded well below the 1.5B one"
last_verified_commit: "HEAD"
discovered_by: Fable 5, 5090, 2026-08-02
severity: medium
status: open
related: [F0017, F0043, F0081]
---

# Finding F0082: the calibration we ship is the worse checkpoint on the metric we say decides

## How this surfaced

Not by looking for it. [F0081](0081-int4-layer-protection.md) was testing whether protecting a
few layers rescues int4's reasoning collapse, and its unprotected RTN baseline came back at
**0.2198** where `docs/BENCHMARKS.md` §4 publishes **0.1498** for symmetric int4 at this size.
The tempting move was to write that off as card-and-stack drift — 3090 vs 5090, a rebuilt fused
stack, avg@64 vs avg@8 — and carry on. Instead the published checkpoint was put through the
same harness on the same box in the same session.

It reproduced. And that turned a bookkeeping discrepancy into a result about what we ship.

## The measurement

1.5B, int4 g64 **symmetric in both arms**, same kernel, same server flags
(`RWKV_SPARSE_FFN=0 RWKV_STATE_FP16=1 RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=1`), same 5090,
same session, MATH500 avg@8 over all 500 problems, cluster-bootstrapped over problems
(`bench/math500_compare.py`, 20,000 resamples).

| checkpoint | rounding | MATH500 | vs RTN (95% CI) | ended-eod | truncated | mean tokens | bsz1 |
|---|---|---:|---|---:|---:|---:|---:|
| `rwkv7-1.5b-w4pNONE` | **RTN** (`bench/quant_w4.py`) | 0.2198 | baseline | 0.621 | 0.379 | 757 | 710.7 |
| same, re-run | **RTN** | 0.2233 | +0.0035 [−0.0075, +0.0145] | 0.617 | 0.384 | 767 | 711.6 |
| `rwkv7-1.5b-w4gptq` | **GPTQ**, Hessian-calibrated | **0.1510** | **−0.0688 [−0.0953, −0.0430] SEPARATED** | 0.379 | 0.583 | 1032 | 712.4 |

**Both published reference points reproduce on this box**, which is what makes the rest of the
table believable. The screen was validated against the two numbers it could be checked against,
and it hit both:

| | published (avg@64, 3090) | measured here (avg@8, 5090) |
|---|---:|---:|
| fp16 | 0.4060, trunc 14.2%, 581 tok | **0.4020**, trunc 0.146, 584 tok |
| int4 GPTQ symmetric | 0.1498, trunc 57.7%, 1022.9 tok | **0.1510**, trunc 0.583, 1032 tok |

Accuracy, truncation rate and generation length all land, at both ends of the range, across a
card change and an 8× change in rollout count. The harness, the card and the rollout count are
not what moved.

**With the ceiling measured on the same stack, the gap decomposes cleanly** (all differences
against fp16 0.4020, all SEPARATED):

| checkpoint | MATH500 | gap to fp16 | note |
|---|---:|---:|---|
| fp16 | 0.4020 | — | 433.1 tok/s at bsz1 |
| int4 **RTN** | 0.2198 / 0.2233 | **−18.2pp** | 711 tok/s, 1.64× fp16 |
| int4 **GPTQ** (what we publish) | 0.1510 | **−25.1pp** | 712 tok/s, same speed, same size |

So the collapse our public docs describe at −24 to −26pt is real and it is **specifically the
GPTQ checkpoint's**. Round-to-nearest gives back 7 of those points for nothing: same format,
same kernel, same file size, same throughput to within 0.2%.

**The two checkpoints differ only in the rounding.** All 651 unquantized tensors — embeddings,
lm_head, LoRA ranks, norms, WKV parameters — are **bit-identical** between the RTN checkpoint,
the GPTQ checkpoint, and the `rwkv7-1.5b-fla` source they were both built from. Same base
model, same group size, same symmetric scheme, same 144 packed matrices. The only thing that
changed is how each weight was rounded onto the grid, and it is worth 6.9 points.

Throughput is unaffected (710.7 / 711.6 / 712.4 tok/s): the two checkpoints are the same
format, so this is a pure accuracy question with no speed trade to weigh against it.

## Why this was never caught

Because the comparison was never run. F0017 established int4-GPTQ against fp16 on **lambada**;
F0043 compared **symmetric against asymmetric** GPTQ on lambada, compression and MATH500. Neither
compared **GPTQ against RTN on MATH500 at 1.5B**. BENCHMARKS §4 does put an RTN number next to a
GPTQ number — "+0.0202 (7.2B, plain RTN) vs +0.0429 (1.5B, the stronger GPTQ)" — but those are
different model sizes and different metric (compression), so the phrase "the stronger GPTQ" was
carrying an assumption, not a measurement. This finding is that assumption failing.

## The reinterpretation this forces on F0043, stated as a hypothesis and not yet a result

F0043's headline was that asymmetric encoding lifts 1.5B MATH500 from 0.1498 to **0.2199**,
"closing 27.4% of the gap" to fp16, and it reasoned from the residue that int4's damage "is not
simply a bit-budget problem".

Plain RTN, measured here, is **0.2198**.

That is close enough to be worth saying out loud and careful enough to need saying properly:
the two numbers come from different runs (avg@64 on a 3090 versus avg@8 on a 5090), so this is
a coincidence across conditions, not a paired comparison. But symmetric GPTQ *did* reproduce
across exactly those conditions, which is the evidence that they are comparable. If it holds,
F0043's gain was not an improvement over the state of the art — it was **GPTQ repairing damage
GPTQ caused**, arriving back where doing nothing clever already was.

Settling it needs asymmetric GPTQ measured on this box, which needs a calibration pass to
regenerate Hessians this box no longer has. Until then this stays a hypothesis with one
striking coincidence behind it.

## The mechanism, with direct evidence — and it is not what I first guessed

The obvious worry about a result like this is that the checkpoint is simply *botched* — a bad
calibration run rather than a real property of the algorithm. That is checkable offline, with no
GPU and no calibration data: unpack both checkpoints through the kernel's own nibble convention
and compare each dequantized matrix against the fp16 weight it came from.

| | relative weight reconstruction error ‖Ŵ−W‖/‖W‖ |
|---|---:|
| RTN | **0.1118** |
| GPTQ | **0.1542** |

**GPTQ's weights are 38% further from the originals than RTN's, on 144 of 144 matrices — every
single one.**

My first reading of this was "the calibration is broken". It is not, and the distinction
matters. GPTQ does not minimize ‖Ŵ−W‖; it minimizes the *activation-weighted* error ‖(Ŵ−W)X‖
on calibration text, and it buys that by deliberately accepting **larger** weight error in
directions the calibration data says are unimportant. Higher plain weight MSE is GPTQ working
as designed, not failing. The published evidence agrees it works: F0017 and BENCHMARKS §4 have
GPTQ beating RTN on lambada by a clear margin at 7.2B (−1.28pt versus RTN's −2.64pt).

So the two facts fit together into one statement, and it is sharper than the hypothesis I
started with:

> GPTQ knowingly makes the weights less accurate in order to make the calibration-text
> activations more accurate. That trade wins on lambada and compression. It loses 6.9 points of
> MATH500.

BENCHMARKS §4 already warns that perplexity-style rulers badly understate int4's reasoning
damage. This is the same warning one level deeper: an *algorithm* tuned against a
perplexity-like objective does not merely fail to see reasoning damage, it will actively spend
weight fidelity to buy improvements the reasoning metric does not want. The behavioural evidence
is consistent — the GPTQ arm's failure is the documented one, losing the thread and never
stopping (truncation 0.583 versus RTN's 0.379, mean tokens 1032 versus 757), a
generation-control failure rather than confident wrongness.

**What is still not ruled out:** that *this particular* calibration was also poor, on top of
being a bad trade. Separating "the objective is wrong for MATH500" from "these Hessians were
bad" needs the activation-weighted error measured, which needs a calibration pass this box no
longer has data for. The 38% figure establishes that GPTQ paid a real price in weight fidelity;
it does not by itself establish that it got fair value on its own terms.

## Where GPTQ spent it — and a matched control that falls out for free

Breaking the error down by projection kind (`bench/quant_weight_error.py`):

| projection | params | RTN | GPTQ |
|---|---:|---:|---:|
| `ffn.value` | 402.7M | 0.1126 | **0.1867** |
| `attn.v_proj` | 100.7M | 0.1154 | 0.1457 |
| `attn.o_proj` | 100.7M | 0.1129 | 0.1434 |
| `attn.r_proj` | 100.7M | 0.1125 | 0.1392 |
| `attn.k_proj` | 100.7M | 0.1135 | 0.1387 |
| `ffn.key` | 402.7M | 0.1095 | **0.1340** |

**RTN's error is flat to within 0.006 across every kind** — round-to-nearest cannot concentrate
error anywhere, it only sees the weight distribution. **GPTQ's is not**: it ranges over 0.053,
and its worst is `ffn.value`, the projection the state accumulation runs through — the exact
tensor named in the public claim that 4 bits there "pollutes the state accumulation path".
F0081's dose experiment could not see that claim's content precisely because it was run on RTN,
which has no concentration to find.

And the table hands over an ideal control. `ffn.key` and `ffn.value` are the same shape, the
same 402.7M parameters, and adjacent in the same block — yet GPTQ damages `value` most (0.1867)
and `key` least (0.1340). So take the shipped GPTQ checkpoint and hand back one of them at a
time (`bench/quant_restore.py` — surgery, not re-quantization, since re-running GPTQ needs
Hessians this box no longer has):

- **dose-only** (F0081's rate, position irrelevant): both restores buy the same ~+4.25pp,
  landing both near **0.194**.
- **concentration** (GPTQ's loss is *where* it put the error): `value`-restore climbs far
  higher than `key`-restore, plausibly recovering most of the 6.9-point gap to RTN.

Identical dose, identical shapes, opposite ends of GPTQ's own damage profile.

**Result: mostly dose, with a real but marginal concentration component on top.**

| checkpoint | restored | params at fp16 | MATH500 | vs GPTQ (95% CI) |
|---|---|---:|---:|---|
| GPTQ | — | 0 | 0.1510 | baseline |
| GPTQ + `ffn.key` back | 24 matrices | 402.7M | 0.2072 | +0.0562 [+0.0318, +0.0807] SEPARATED |
| **GPTQ + `ffn.value` back** | 24 matrices | 402.7M | **0.2338** | **+0.0828 [+0.0570, +0.1090] SEPARATED** |
| _(RTN, for reference)_ | — | 0 | 0.2198 | +0.0688 |

**`value` versus `key`, paired directly: +0.0265, 95% CI [+0.0008, +0.0518] — separated, but
only just.** The lower bound is eight ten-thousandths above zero. That is the weakest verdict
this comparator can return, and it should be read as "detectable" rather than "established".

Both predictions on the record were wrong in the same direction, and the honest reading splits
the difference between them:

- **Dose-only predicted +4.25pt for both.** Both arms beat it — +5.62 and +8.28. So restoring
  402.7M parameters from *GPTQ* is worth substantially more than restoring the same count from
  RTN, wherever you take them from. That is the uniform part of GPTQ's excess error (it is
  worse on 144/144 matrices) and it is the larger share of the effect.
- **Concentration predicted `value` far above `key`.** It is above, by 2.65pt, and the gap
  clears zero by a hair rather than by a margin. Concentration is real and it is the *smaller*
  term, not the dominant one I had claimed before this control existed.

One quantitative consistency worth noting without leaning on it: GPTQ's error ratio between the
two families is 0.1867 / 0.1340 = **1.39**, and the ratio of what restoring them buys is
8.28 / 5.62 = **1.47**. Close enough to be suggestive that recovered accuracy tracks removed
error roughly proportionally, on two points, which is not enough points to claim a law.

A like-for-like check that the story stays coherent: with `ffn.value` at fp16 in both, RTN's
remaining matrices still beat GPTQ's remaining matrices (0.2622 versus 0.2338, F0081's FFNV arm
against this one). GPTQ is worse everywhere — somewhat more so on `ffn.value`.

## The same pathology at 7.2B, measured before its MATH500 lands

| projection | params | RTN | GPTQ |
|---|---:|---:|---:|
| `ffn.value` | 2,147.5M | 0.1106 | **0.1954** |
| `attn.v_proj` | 536.9M | 0.1116 | 0.1380 |
| `attn.o_proj` | 536.9M | 0.1110 | 0.1370 |
| `attn.r_proj` | 536.9M | 0.1097 | 0.1367 |
| `attn.k_proj` | 536.9M | 0.1106 | 0.1348 |
| `ffn.key` | 2,147.5M | 0.1087 | **0.1328** |
| **overall** | 6,443M | **0.1100** | **0.1561** (+42%) |

Identical shape to 1.5B and slightly worse: RTN flat within 0.003, GPTQ spread over 0.063, the
excess concentrated on `ffn.value` — which at 7.2B is 44% of all quantized weight. This is a
property of the algorithm on this architecture, not a one-off bad run on one model.

**Prediction, recorded before the 7.2B MATH500 arms report:** the GPTQ-versus-RTN inversion
reproduces at 7.2B. The mechanism that would produce it is present and marginally stronger.

**The prediction failed. At 7.2B there is no inversion worth the name.**

| 7.2B, avg@8, same box, same session | MATH500 | vs RTN (95% CI) | truncated | bsz1 |
|---|---:|---|---:|---:|
| RTN | 0.6205 | baseline | 0.101 | 246.1 |
| GPTQ | 0.6112 | **−0.0093 [−0.0405, +0.0220] not separated** | 0.136 | 246.1 |

RTN is ahead by 0.9 of a point and the interval comfortably spans zero. The direction matches
1.5B and the truncation gap matches too (0.101 versus 0.136), but at flagship scale the effect
is inside the noise. **The strong prediction — "the inversion reproduces at 7.2B" — is refuted,
and the correction below turned out to be the right read of it.**

**"Not separated" at 7.2B means something weaker than it did at 1.5B, and the difference has to
be stated.** The fp16 ceiling here reads **0.6370** (truncation 0.063 against the published
6.3%), and neither int4 arm separates from it either: RTN −1.65pt [−4.35, +0.95], GPTQ −2.58pt
[−5.92, +0.75]. But BENCHMARKS §4's avg@64 measurement puts symmetric GPTQ at −3.1pt from fp16,
and that is a *real* effect this screen simply cannot see. At 1.5B, where scores sit near 0.22,
the paired intervals were ±2pp and the screen resolved 4pp. At 7.2B, where scores sit near 0.62
and problems are more heterogeneous, the baseline interval is ±3.8pp and paired intervals run
to ±3.1pp — so the resolution floor is roughly **5–6pp**, above the size of every effect in
play.

So the correct 7.2B statements are bounded, not null:

- int4 is **not** shown lossless at 7.2B here; the published −3.1pt stands as the better
  estimate, from 8× the rollouts.
- the GPTQ-versus-RTN gap at 7.2B is **bounded above by roughly 3 points**, not shown to be
  zero. It is certainly not 6.9.

That is enough to decide the product question and not enough to close the scientific one, and
those should not be conflated.

**Four published reference points reproduced**, which is what licenses every comparison in this
finding:

| reference | published (avg@64) | measured here (avg@8) |
|---|---:|---:|
| 1.5B fp16 | 0.4060, trunc 14.2% | 0.4020, trunc 0.146 |
| 1.5B int4 GPTQ | 0.1498, trunc 57.7% | 0.1510, trunc 0.583 |
| 7.2B int4 GPTQ | 0.6108, trunc 14.0% | 0.6112, trunc 0.136 |
| 7.2B fp16 | 0.6418, trunc 6.3% | 0.6370, trunc 0.063 |

Four for four on accuracy and on truncation, across two model sizes, two cards, and an 8×
change in rollout count. The screen is sound; what it lacks at 7.2B is resolution, not
calibration.

**One number in this run is not our fp16 product number.** The fp16 arm reads 109.1 tok/s at
bsz1 against the 143.2 we publish, because every arm here runs `RWKV_SPARSE_FFN=0` — the sparse
channel-mix is ineligible on quantized projections, so it was disabled everywhere to keep the
arms comparable. That costs the *unquantized* arm about 24% and nothing else. Do not quote
109.1 anywhere; it is a deliberately handicapped configuration that exists to make the accuracy
comparison clean.

**And a correction to that prediction, entered before the arms landed rather than after.**
Re-reading BENCHMARKS §4's own 7.2B table undercuts the strong form of it. There, symmetric
GPTQ costs **−3.1pt** against fp16 (61.08% versus 64.18% at avg@64) where at 1.5B it cost
−25.6pt. If GPTQ's *total* damage at 7.2B is three points, then RTN cannot beat it by more than
about three, and the 6.9-point inversion is a 1.5B-scale result that cannot survive at flagship
scale in anything like that magnitude. The first 7.2B arm is consistent with the small version:
RTN reads **0.6205** at avg@8 against that published 61.08% — about a point apart, across
different rollout counts and boxes, which is exactly the cross-run comparison this finding
exists to stop making. The same-box GPTQ arm settles it.

**This also caps how much the weight-error story can claim.** `ffn.value`'s GPTQ error is
*worse* at 7.2B (0.1954) than at 1.5B (0.1867), while the MATH500 damage is eight times
*smaller*. So concentration explains **where** GPTQ puts its error, not **how much that
costs** — the cost is dominated by model scale, and a bigger model absorbs the same relative
weight error far better. Any statement of the form "GPTQ's ffn.value damage predicts MATH500
loss" is wrong as written; the honest version is that it predicts the loss *within* a size, not
across sizes.

## Supporting context we already published without reading it this way

BENCHMARKS §4's own 7.2B table lists three GPTQ variants: symmetric **61.08%**, hybrid
(`ffn.value`+`ffn.key` forced symmetric, rest asymmetric) **56.03%**, all-asymmetric
**47.78%** — a 13-point spread produced purely by encoding choices *within* GPTQ. And the
ordering **inverts with model size**: at 1.5B asymmetric beat symmetric (0.2199 versus 0.1498),
at 7.2B it loses to it by 13 points.

A method whose own variants reorder across model sizes is not being reliably steered by its
objective on this architecture. That was visible in our published numbers before this finding;
what was missing was the comparison against doing nothing clever at all, which is the one
configuration that has no knobs to get wrong.

## Why `ffn.value`, and it is our own earlier measurement that explains it

`ffn.value`'s input is `relu(key(x))²` — and [F0080](0080-ffn-sparsity-does-not-batch.md)
measured exactly what that input looks like: **77–90% of its channels are zero on any given
token, and which ones are zero is a property of the input, not the channel.** On 64 pieces of
genuinely different real text the union of live channels was only 6.5–15.8%.

That is the worst possible input distribution for a calibration-based method. GPTQ weights its
reconstruction by `XᵀX`, so channels the calibration text never activates carry no weight in
the objective, and GPTQ is free to — and does — let those rows drift in order to buy accuracy
on the channels the calibration text does light up. On a dense-input projection there is
nothing to trade, which is precisely why the attention projections sit at 0.135–0.138 while
`ffn.value` sits at 0.195.

Then a math problem arrives and lights up a different subset. F0080 established that the live
set barely overlaps across inputs; this finding is the bill for optimizing against one sample
of it. Two findings that were about unrelated things — a sparse kernel that would not batch,
and a quantizer that scores badly — turn out to be the same fact seen twice.

This is a hypothesis with the pieces in place, not a proven chain — and the surgery that tested
it came back **supporting but modest**. Handing back `ffn.value` does beat handing back
`ffn.key` at identical dose, by 2.65pt, but the interval clears zero by a hair and the larger
share of both arms' gain is the uniform excess error GPTQ carries on all 144 matrices. So the
sparse-input story explains a real component of the damage, not the bulk of it.

The honest ordering of causes for GPTQ's 6.9-point loss at 1.5B, from this evidence:

1. **GPTQ is uniformly worse** — 38% higher weight error on every matrix. Largest term.
2. **GPTQ is additionally worse on `ffn.value`**, the sparse-input projection, which costs a
   further ~2.6pt. Real, marginal, and the part the sparse-activation mechanism explains.

I had written "concentration, decisively" after the first arm and before its control. That was
the first arm agreeing with a hypothesis I liked, and one arm cannot separate dose from
placement — which is precisely why the control was built. It is in the record because the
sequence is the lesson, not the conclusion.

## What follows

- **Do not change the 7.2B checkpoint.** Measured: RTN 0.6205 versus GPTQ 0.6112, not
  separated. Whatever GPTQ costs at flagship scale, it is inside the noise of the ruler this
  repo uses to decide quantization quality, and GPTQ additionally has the better published
  lambada number. `Hakureirm/rwkv7-g1-7.2b-w4gptq` stays as it is.
- **At 1.5B, prefer RTN if int4 is used at all — while noting it probably should not be.** The
  6.9-point gap is real and separated, and it costs nothing to take. But int4 at 1.5B is a bad
  product either way: −18.2pt for RTN against −25.1pt for GPTQ, both far outside what anyone
  would ship for reasoning. So the honest recommendation is "if you insist on 1.5B int4, use
  RTN", not "1.5B int4 is now fine".
- **And the ordering is not monotone in size, which narrows the claim further.** Collecting
  every same-size GPTQ-versus-RTN comparison this repo now has:

  | size | metric | GPTQ | RTN | winner |
  |---|---|---|---|---|
  | 0.1B | coherence (BENCHMARKS §7) | coherent text | **repetition collapse, broken** | GPTQ, decisively |
  | 1.5B | MATH500 | 0.1510 | **0.2198** | RTN, +6.9pt separated |
  | 7.2B | MATH500 | 0.6112 | 0.6205 | tied, not separated |
  | 7.2B | lambada (BENCHMARKS §4) | **−1.28pt** | −2.64pt | GPTQ, +1.36pt |

  GPTQ wins at 0.1B, loses at 1.5B, and ties-or-wins at 7.2B. A finding that reverses twice
  across the size range is not a recommendation to change quantizers — it is evidence that
  calibration helps when the model has little redundancy to spare, hurts in the middle where
  the sparse-input pathology dominates, and washes out once the model is large enough to absorb
  either. Only the 1.5B point is new here; the other three were already measured and were never
  put in one table.
- **State the awkward shape rather than smoothing it.** The inversion is largest exactly where
  the tier should not be used, and vanishes exactly where the tier is good. A reader who takes
  the 6.9 figure and applies it to the flagship gets the wrong answer by a factor of seven.
  That is why the title carries the size.
- **Do not overcorrect either.** GPTQ was adopted on lambada and compression evidence, and that
  evidence is not withdrawn by this. The plausible outcome is a genuine metric-dependent split —
  GPTQ better on perplexity-style rulers, worse on reasoning — in which case the honest product
  answer is to publish both with the split documented, not to silently swap one for the other.
- **BENCHMARKS §4 needs one row, not a rewrite.** Its int4 numbers describe the GPTQ
  checkpoints and remain correct for what we ship. What is missing is the comparison that was
  never run: plain RTN, at both sizes, on MATH500 — 0.2198 versus GPTQ's 0.1510 at 1.5B,
  0.6205 versus 0.6112 at 7.2B. The phrase "the stronger GPTQ" should go, since at neither size
  is GPTQ stronger on this ruler and at one it is decisively weaker.
- **The screen earned its keep and should be reused.** MATH500 avg@8 with a cluster bootstrap
  reproduced three published avg@64 reference points across two model sizes and two cards, at
  an eighth of the cost. Quantization work that currently waits hours for avg@64 can screen at
  avg@8 first and spend avg@64 only on what survives.
