---
doc_kind: finding
finding_id: F0029
title: "Speculative-decoding viability (req#6): 0.1B RWKV-7 draft vs 1.5B target per-token greedy acceptance α = 0.738 → ~2.98 target-tokens/forward at block K=4, net ~2.35× (1.5B target) to ~2.82× (7.2B target) bsz1 decode speedup — the viability gate PASSES, justifying the full spec-decode build (ADR-0006)"
last_verified_commit: "HEAD"
discovered_by: lead (M13), 2026-07-04
severity: info
status: open
related: []
---

# Finding F0029: RWKV-7 speculative-decoding viability (req#6, step 1)

## Measurement
Before building the full spec-decode machinery (ADR-0006: recurrent-verify + O(1)-state
rollback — HIGH effort), measure the number that gates it: the **per-token greedy acceptance rate
α** = how often the 0.1B draft's argmax matches the target's greedy token given the same prefix.
`bench/spec_accept.py` (two-phase, since two sglang servers don't cleanly share one GPU): target
greedy → sequences T; then feed the draft `prompt+T` with `return_logprob, top_logprobs_num=1` and
compare the draft's per-position argmax to T. 8 prompts, 603 tokens, 1.5B target / 0.1B draft, fp16.

## Result
**α = 0.738** (draft argmax == target token 73.8% of positions, n=603). Expected target-tokens per
target-forward (i.i.d. approx, 1 + α + … + α^K):

| block K | target-tokens / target-forward |
|---|---|
| 2 | 2.28 |
| 4 | **2.98** |
| 8 | 3.57 |

**Net bsz1 speedup** (accounting for draft cost, draft(0.1B) ≈ 1/15 the FLOPs of 1.5B, ≈ 1/70 of
7.2B; net ≈ tokens-per-forward / (1 + K·draft_frac)):
- vs **1.5B** target, K=4: ≈ 2.98 / (1 + 4/15) ≈ **2.35×**
- vs **7.2B** target, K=4: ≈ 2.98 / (1 + 4/70) ≈ **2.82×**

## Verdict — viability gate PASSES
α = 0.738 is high (typical draft/target acceptance is 0.6–0.8; the shared RWKV7-G1 family helps), so
spec-decode yields a **~2.3–2.8× single-stream decode speedup**, best on the **7.2B target** (draft
cost negligible there). This decisively justifies the full build (ADR-0006): draft engine + recurrent
verify + conv/temporal state rollback, gated spec-on == spec-off token-identical. Spec-decode is a
capability no other RWKV serving stack has (they have no scheduler), and it is exact-by-construction
(accepts only tokens the target would emit) so it trades nothing for the speedup.

## Method caveats (honest)
- α measured via the `input_top_logprobs` top-1 (draft's argmax per position). A first run mis-aligned
  the position (`logprob_start_len=len(prompt)-1`, off by one → spurious α=0.012); fixed to
  `=len(prompt)` (input_top_logprobs[i] is the prediction FOR position i given prefix[0..i-1]) → 0.738.
- 8 short prompts / 603 tokens is a viability estimate, not a production α — real α varies by domain
  (reasoning/factual tokens accept less than boilerplate). The full build will re-measure end-to-end.
- Speedup figures are FLOP-ratio estimates; the real number depends on the draft/verify/rollback
  overhead the build introduces (measured then).

## Cross-references
ADR-0006 (spec-decode design) · `bench/spec_accept.py` · draft model `rwkv7-0.1b-fla`.
