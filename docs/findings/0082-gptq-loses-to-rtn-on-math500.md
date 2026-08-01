---
doc_kind: finding
finding_id: F0082
title: "On MATH500 our calibrated int4 (GPTQ) is 6.9 points WORSE than plain round-to-nearest — separated, same base model, same session — and the w4gptq checkpoints are what we publish"
last_verified_commit: "HEAD"
discovered_by: Fable 5, 5090, 2026-08-02
severity: high
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

## A mechanism that would explain it, offered as motivation for the next test rather than as fact

GPTQ minimizes layer-wise **output reconstruction error** on calibration text — an objective
closely related to perplexity. BENCHMARKS §4 already warns, in its own words, that
perplexity-style rulers badly understate int4's reasoning damage. Put those together and the
sharper statement is available: an algorithm that optimizes a perplexity-like proxy can trade
away exactly what that proxy fails to see. The behavioural evidence is consistent — the GPTQ
arm's failure mode is the documented one, losing the thread and never stopping (truncation
0.583 versus RTN's 0.379, mean tokens 1032 versus 757), which is a generation-control failure
rather than confident wrongness.

That is a story, not a finding. What is established is the 6.9 points.

## What follows

- **Do not act on this at 7.2B yet.** 1.5B is not the flagship, and BENCHMARKS §4 already
  documents quantization costing less at 7.2B than at 1.5B, so the inversion may be
  size-dependent. The 7.2B arms (RTN / GPTQ / fp16 / positional-protection) are queued.
- **Then reconsider what we publish.** ModelScope carries
  `Hakureirm/rwkv7-g1-{1.5b,7.2b}-w4gptq`. If the inversion holds at 7.2B, we are shipping the
  worse checkpoint on the metric this repo itself calls "the ruler that decides quantization
  quality here", at identical speed and identical size.
- **Do not overcorrect either.** GPTQ was adopted on lambada and compression evidence, and that
  evidence is not withdrawn by this. The plausible outcome is a genuine metric-dependent split —
  GPTQ better on perplexity-style rulers, worse on reasoning — in which case the honest product
  answer is to publish both with the split documented, not to silently swap one for the other.
- **Re-measure the tier's public accuracy line.** §4's int4 rows describe the GPTQ checkpoints.
  If RTN is what we recommend, those rows describe something we no longer ship.
