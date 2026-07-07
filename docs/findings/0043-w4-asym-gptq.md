---
doc_kind: finding
finding_id: F0043
title: "Asymmetric (scale+zero) GPTQ for w4, zero kernel changes: closes 27-35% of the fp16 gap across lambada/compression/MATH500 avg@64 (1.5B); MATH500 — the hardest, most decision-relevant metric — closes the least (27%), evidence the reasoning collapse is not simply a bit-budget problem; Stage 2 (K-quant mixed precision) NOT recommended yet, 7.2B avg@64 should be measured first"
last_verified_commit: "HEAD"
discovered_by: Sonnet 5 (agent-assisted, 3090 box), 2026-07-06/07
severity: info
status: open
related: [F0017]
---

# Finding F0043: asymmetric GPTQ closes a real fraction of the w4 gap, but not the part that matters most

## Goal

F0017 established weight-only int4 (group-64, **symmetric** scale, GPTQ Hessian calibration)
at lambada −3.34pt vs fp16, and a separate measurement (not yet an F-numbered doc) found 1.5B
int4-GPTQ's greedy MATH500 collapses catastrophically (0.1560 vs fp16 0.3940) — far more damage
than lambada suggested, because perplexity-style metrics don't see multi-step reasoning failure.
The user asked to push int4 quality toward llama.cpp's Q4_K_M/Q5_K_M band. Q4_K_M's real edge
over a naive symmetric scheme is **asymmetric (scale+zero) encoding plus mixed precision**
(6-bit blocks for sensitive tensors). This finding covers Stage 1: asymmetric alone, GPTQ-error-fed,
same group-64 granularity, no kernel rewrite. Stage 2 (real K-quant mixed precision, new kernel)
was scoped but is a separate, larger undertaking — this finding's job is to decide whether the
data justifies it.

## What was built

`bench/gptq_w4.py`'s existing group-64, Hessian-error-feedback GPTQ loop gained an `asym=` mode:
per-group `scale = (max-min)/15`, `zero = round(-min/scale)` clamped to `[0,15]`, quantized value
`q_unsigned = clip(round(w/scale + zero), 0, 15)`, GPTQ error feedback computed against the true
asymmetric dequant `(q_unsigned - zero) * scale` (not a symmetric approximation) — this is standard
GPTQ practice, not a new algorithm; the only real engineering question was the kernel.

**The kernel needed zero changes.** `rwkv7_w4.cu`'s int4 unpack already sign-extends via two's
complement (`q -= (q & 8) << 1`, decoding to `[-8, 7]`); the existing symmetric scheme only ever
used `[-7, 7]`, leaving one code point unused. Storing `Q_int = q_unsigned - 8` makes the kernel's
existing per-nibble decode exactly equal `q_unsigned - 8` (verified value-by-value). The true
asymmetric dequant differs from the kernel's (symmetric-style) output by a constant that depends
only on the group, not on the activation: `bias[n,g] = (8 - zero[n,g]) * scale[n,g]`. So:
`true_output = kernel_output + Σ_g bias[n,g] * groupsum_x[g]` — an O(N·K/64) correction, a single
extra `groupsum_x @ bias.T` matmul in pure PyTorch (worst case ~1.6% extra FLOPs on the widest FFN
projection), applied once after any of the four existing dispatch paths
(gemv_m1/gemm_small/gemm_tc/dequant+F.linear). This is an **exact** correction, not an
approximation — verified in `bench/verify_w4.py`'s new asymmetric test section: 12 configs
(M∈{1,4,32,96} × 3 shapes), rel-err 7.6e-5–2.2e-4, the same noise floor as the existing symmetric
kernel's own numerics. `models/rwkv7.py` gained an opt-in `RWKV_W4_ASYM=1` flag (default off,
composes with `RWKV_W4=1`) and an optional `zero` buffer on `W4Linear`, registered only when
asymmetric — the existing symmetric path is provably unaffected (buffer absence is enforced by
`load_weights`'s missing/unexpected-key check, not a silent default).

## Results (1.5B, RTX 3090/5090, same calibration Hessians as the symmetric baseline for a fair A/B)

| metric | fp16 | symmetric GPTQ (F0017) | asymmetric GPTQ (this finding) | gap closed |
|---|---:|---:|---:|---:|
| lambada (full, 5153) | 0.6724 | 0.6390 (−3.34pt) | **0.6507 (−2.17pt)** | 35.0% |
| compression (uncheatable, N=300, pooled bpb) | 0.5893 | 0.6330 | **0.6186** | 32.9% |
| MATH500 avg@64 (500×64=32000 rollouts) | 0.4060 | 0.1498 | **0.2199** | 27.4% |

Two behavioral indicators alongside the MATH500 number (symmetric → asymmetric): truncated_rate
57.7% → 30.5%, mean_generated_tokens 1022.9 → 722.6. The "wanders and never stops" pathology is
real and asymmetric quantization measurably reduces it — this is a genuine improvement, not just a
noisier accuracy number moving up.

Asymmetric's lambada number (0.6507) is close to int8's 0.6509 (F0018) — statistically
indistinguishable on that one metric. **This is exactly the kind of result the F0029/int4-MATH500
lesson warned about**: a perplexity-adjacent metric (lambada) says "as good as int8," while the
task-relevant metric (MATH500 avg@64) says the gap to fp16 is still enormous (asymmetric still
loses **0.186** absolute accuracy vs fp16, and int8 loses essentially nothing on the equivalent
MATH500 avg@64 measurement — see `docs/BENCHMARKS.md` §2). Trusting lambada alone here would be a
real mistake.

## Decision: Stage 2 (K-quant mixed precision) — NOT recommended yet

The gap-closure percentage is fairly consistent across lambada (35.0%) and compression (32.9%),
but **meaningfully lower on MATH500 avg@64 (27.4%)** — the one metric this project's own rulers
(compression rate + MATH500 avg@64, per `feedback-benchmark-rigor`) treat as decisive. That
ordering (the harder, more decision-relevant metric improves the least) is evidence, not proof,
that the MATH500 collapse is not purely a "more bits would fix it" problem — it looks more like
compounding error over long autoregressive reasoning chains, which a bigger quantization budget
(Q4_K_M's ~4.5-5 bits/weight average, from mixed 4/6-bit blocks) might reduce further but is not
guaranteed to fix. Naively extrapolating the same ~27% relative gap-closure rate to a bigger bit
budget would land somewhere around 0.24-0.28 rollout_accuracy — still a large, likely
task-disqualifying gap from fp16's 0.4060.

Given that, and given Stage 2 is a real, substantial investment (new superblock/subblock two-level
kernel, new quantizer, full re-verification — not a zero-cost win like Stage 1 was), the more
informative and much cheaper next step is: **measure 7.2B's MATH500 avg@64 for both symmetric and
asymmetric GPTQ** (currently only lambada has been measured at 7.2B — 0.7297 GPTQ vs 0.7425 bf16,
−1.28pt, a much smaller relative hit than 1.5B's −3.34pt, consistent with this project's general
"bigger models quantize better" pattern seen elsewhere, e.g. RTN int4 lambada: 1.5B −4.95pt vs
7.2B −2.64pt). If 7.2B's MATH500 collapse (if any) turns out to be much milder than 1.5B's — which
the lambada pattern suggests is plausible — then int4's honest scope can be "great at 7.2B+, use
int8 instead of int4 for reasoning-heavy workloads at 1.5B," a coherent, defensible position that
doesn't need Stage 2 at all. If 7.2B ALSO collapses badly on MATH500, that raises Stage 2's
priority substantially (a model-size-independent reasoning failure would be a much bigger problem
worth the bigger investment). **Recommendation: measure 7.2B avg@64 next, revisit Stage 2 after.**

## Honest positioning (unchanged from F0017, sharpened by this data)

int4 (symmetric or asymmetric) at 1.5B is a VRAM-savings tool for non-reasoning-heavy workloads,
not a general-purpose lossless tier — that was already F0017's honest framing after the original
MATH500 collapse was found, and this finding doesn't change it, just quantifies how much
asymmetric quantization helps (real, but partial) versus how much of the problem remains
(most of it, on the metric that matters most). w8g64 (int8, weight-only) remains the lossless
quantized tier (oracle 24/24 exact); int4's honest pitch is aggressive VRAM reduction at an
accepted, now-somewhat-smaller-but-still-large accuracy cost for tasks sensitive to multi-step
reasoning.
