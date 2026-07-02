---
doc_kind: adr
adr_id: 0004
title: "No FLA dependency in the deliverable — kernel endgame = albatross (BlinkDL's own) or self-written"
status: accepted
date: 2026-06-30
last_verified_commit: "2aaf68b"
supersedes: []
superseded_by: []
---

# ADR-0004: No FLA dependency in the deliverable

## Context
**Ecosystem constraint (relaying the RWKV community):** RWKV's author **BlinkDL** and the
**flash-linear-attention (FLA)** maintainers are not on good terms. An implementation that
appears "built on FLA" is poorly received in the RWKV community — and, more concretely, FLA's
RWKV-7 is documented as not aligned with the reference, so it is the wrong basis regardless.

Current technical reality (to be clear about what we actually depend on):
- **Accuracy** is anchored to **BlinkDL** only: the `rwkv` pip + `rwkv_v7_numpy.py` reference
  is the oracle; we explicitly **rejected FLA as an accuracy oracle** from the start
  ([[F0003]] [[F0004]]: FLA's RWKV-7 is documented as NOT aligned with BlinkDL's reference).
  Weights come from BlinkDL `.pth` (our own converter [[F0005]]). Speed is benchmarked vs
  **albatross** (BlinkDL's own engine, [[F0007]]).
- The ONLY FLA touch-point is the **vendored FLA triton kernels** (chunk/fused_recurrent/
  dplr) used as a *temporary correctness scaffold* for the WKV recurrence (M1a). Model,
  weights, oracle, and baseline are all non-FLA.

## Options considered (final kernel source)
1. **Keep FLA triton kernels** — politically unacceptable for the deliverable; also the
   slower path (kernel-quality gap [[F0008]]). REJECTED for the final deliverable.
2. **Adapt/vendor albatross's CUDA kernels** (BlinkDL's own) — fast (the parity target) and
   ecosystem-safe (it's BlinkDL's own code). Proven to compile+run
   on the 3090 ([[F0007]]). PRIMARY plan for M3b.
3. **Write our own RWKV-7 WKV (+linear) kernel** from the published math (the numpy reference
   recurrence is short) — fully independent, cleanest IP/politics, no third-party kernel at
   all. FALLBACK / longer-term ideal; harder to reach albatross speed from scratch.

## Decision
**Hard constraint: the FINAL deliverable contains ZERO FLA dependency.** FLA triton kernels
are an early correctness scaffold ONLY and MUST be removed before release.

- **M3b explicit goal**: replace the vendored FLA kernels with **albatross's CUDA kernels**
  (BlinkDL's own — Option 2) as the primary path to both speed parity AND de-FLA. Verify license
  compatibility for redistribution; attribute BlinkDL/albatross.
- **Fallback (Option 3)**: if albatross-kernel integration is blocked (license/layout), write
  our own triton/CUDA WKV kernel from the RWKV-7 math (numpy-reference-equivalent) — also a
  valid, fully-independent de-FLA endgame.
- **Narrative (effective immediately, zero-cost)**: outward-facing framing is *"aligns
  BlinkDL rwkv-lm accuracy + matches albatross speed, on sglang."* Do NOT lead with FLA;
  mention it at most as a footnote ("early triton scaffold, replaced"). README + snapshot
  de-FLA'd now.

## Consequences
### Positive
- Aligned with the RWKV community + albatross; the implementation reads as "rwkv-lm accuracy +
  albatross speed", not "FLA wrapper". De-FLA is also the speed-parity path (M3b), so one
  effort serves both (ecosystem fit + perf).
### Negative / Risk
- Removing FLA is real work (M3b), gated on keeping greedy EXACT vs the oracle.
- albatross-kernel route adds an albatross dependency (BlinkDL's own → safe) + a license check;
  the fully-clean route (own kernel) is more effort.
### Verification
- Done means: `grep -ri "fla" sglang_overlay/` returns nothing in the final deliverable
  (no `fla` imports / vendored fla dirs), greedy still EXACT, and speed ≥ M3b target.

## Outcome (2026-06-30, M3b complete — [[F0010]])
**SATISFIED.** Self-written `rwkv7_kernels/wkv_recurrent.py` replaced the FLA kernels for
BOTH decode and prefill; the 10 vendored FLA overlay files were deleted. Deliverable is
**100% FLA-free** (only docstring prose mentions FLA; the 2 residual `...fla...` imports are
upstream sglang's own gated-delta code, not ours). Greedy still EXACT (0.1B/1.5B/7.2B);
**zero speed cost** (our recurrent kernel is faster end-to-end than the dropped FLA chunk).
We chose **self-written** over vendoring albatross — even cleaner (no third-party kernel);
albatross (Apache-2.0) remains available if a tensor-core prefill kernel is wanted later.

## Cross-references
[[F0003]] [[F0004]] (FLA rejected as oracle) · [[F0007]] (albatross kernels build on 3090) ·
[[F0008]] (kernel-quality gap) · [[F0010]] (de-FLA complete) · ADR-0002 (integration).
