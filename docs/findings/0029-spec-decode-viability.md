---
doc_kind: finding
finding_id: F0029
title: "Speculative-decoding viability (req#6): 0.1B-class RWKV-7 draft vs 1.5B target per-token greedy acceptance α = 0.738 → ~2.98 target-tokens/forward at block K=4, net ~2.0× (1.5B target) to ~2.7× (7.2B target, same-α assumption) bsz1 decode speedup — the viability gate PASSES, justifying the full spec-decode build (ADR-0006)"
last_verified_commit: "bd08540"
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

**Net bsz1 speedup** (accounting for draft cost; net ≈ tokens-per-forward / (1 + K·draft_frac)).
TRUE parameter counts (safetensors bytes/2, fp16): draft "0.1B" = **0.191B**, target 1.5B = 1.527B,
7.2B = 7.199B — so draft_frac = 0.191/1.527 = **1/8.0** (not the nominal 1/15) and 0.191/7.199 =
**1/37.7** (not 1/70):
- vs **1.5B** target, K=4: ≈ 2.98 / (1 + 4/8.0) ≈ **1.99×**
- vs **7.2B** target, K=4: ≈ 2.98 / (1 + 4/37.7) ≈ **2.69×** (K=8: ≈ 2.95×)

## Update (2026-07-06, Opus) — 7.2B α MEASURED on the 5090 (main); no longer an assumption

Re-ran `bench/spec_accept.py` on the RTX 5090 (sglang main container, fp16, same 8 prompts, gen-len
128), measuring the 0.1B draft against BOTH targets. The 7.2B row above was previously a same-α
*assumption*; it is now measured:

| draft → target | measured α | K=4 tok/forward | net bsz1 (K=4, true draft_frac) |
|---|---|---|---|
| 0.1B → **7.2B** | **0.7485** (n=811) | 3.04 | 3.04 / (1 + 4/37.7) ≈ **2.75×** |
| 0.1B → 1.5B | 0.7330 (n=603) | 2.95 | 2.95 / (1 + 4/8.0) ≈ **1.97×** |

Two facts confirmed: (1) the 1.5B α reproduces on main (0.733 vs the original 0.738 — within
greedy-composition noise); (2) **α is HIGHER against the 7.2B target (0.7485 > 0.733)**, so the
"best on the big target" hypothesis holds on BOTH axes — larger target ⇒ higher acceptance AND
smaller relative draft cost. The 7.2B net-speedup projection tightens to **~2.75×** (was ~2.69×
extrapolated). Raw: `bench/results/spec_acc_{72b,15b}.json`, `bench/logs/spec_alpha_run.log`.

## Verdict — viability gate PASSES
α = 0.738 is high (typical draft/target acceptance is 0.6–0.8; the shared RWKV7-G1 family helps), so
spec-decode yields a **~2.0–2.7× single-stream decode speedup estimate**, best on the **7.2B target**
(draft cost smallest there). This decisively justifies the full build (ADR-0006): draft engine + recurrent
verify + conv/temporal state rollback, gated spec-on == spec-off token-identical. Spec-decode is a
capability no other RWKV serving stack has (they have no scheduler), and it is exact-by-construction
(accepts only tokens the target would emit) so it trades nothing for the speedup.

## Method caveats (honest)
- α measured via the `input_top_logprobs` top-1 (draft's argmax per position). A first run mis-aligned
  the position (`logprob_start_len=len(prompt)-1`, off by one → spurious α=0.012); fixed to
  `=len(prompt)` (input_top_logprobs[i] is the prediction FOR position i given prefix[0..i-1]) → 0.738.
- 8 short prompts / 603 tokens is a viability estimate, not a production α — real α varies by domain
  (reasoning/factual tokens accept less than boilerplate). The full build will re-measure end-to-end.
- n = 603 of the 611 generated target tokens: sglang's logprob format prepends None at
  logprob_start_len, so the FIRST target token of each prompt (8 total) is structurally unscored —
  a negligible per-block-boundary bias, noted for exactness.
- ~~α was measured against the **1.5B** target only; the 7.2B row ASSUMES the same α.~~ **RESOLVED
  (2026-07-06, see Update above):** measured α(0.1B→7.2B) = **0.7485** (n=811), slightly HIGHER than
  1.5B — the same-family draft agrees *more* with the larger target here, not less.
- Speedup figures are parameter-ratio (≈ weight-bandwidth) estimates; at bsz1 the 12-layer draft
  also has a fixed per-token launch/overhead floor, so its real cost exceeds its parameter fraction —
  these estimates are OPTIMISTIC upper-ish bounds. The build measures the real number.

## Cross-references
ADR-0006 (spec-decode design) · `bench/spec_accept.py` · draft model `rwkv7-0.1b-fla`.
