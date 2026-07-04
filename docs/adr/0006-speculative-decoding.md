---
doc_kind: adr
adr_id: 0006
title: "RWKV-7 speculative decoding (req#6): recurrent-verify + O(1)-state rollback — a bespoke draft/target loop, NOT EAGLE"
status: proposed
date: 2026-07-04
last_verified_commit: "924d0f8"
supersedes: []
superseded_by: []
---

# ADR-0006: RWKV-7 speculative decoding

## Context
req#6 asks for speculative decoding (0.1B RWKV draft → larger RWKV target). The reverse-overtake
(ADR-0005) is done; spec-decode is a *new* capability that extends the lead — but RWKV-7 is a
**recurrent** model, so the standard sglang path does not transfer. This ADR is the design (build
is a separate, gated effort); it is grounded in the actual state layout read from the backend.

## Why sglang's EAGLE / tree-spec does NOT apply
sglang's speculative infra (`speculative_algorithm=EAGLE/EAGLE3`, tree attention, `eagle_topk`) is
**transformer-oriented**: the target verifies a *tree* of draft tokens in ONE forward by attending
over the KV cache at all candidate positions in parallel, and rejected branches simply are not
committed to the KV cache. RWKV-7 has **no KV cache and no attention** — its state is an O(1)
recurrence advanced token-by-token. You cannot verify K positions "in parallel via attention"; you
must *run the recurrence* over the K draft tokens. So EAGLE's tree + parallel-verify + KV-drop
mechanism has no RWKV analogue. RWKV needs a **bespoke chain-verify + state-rollback** loop. (ngram
spec is closer — a cheap draft + a linear verify — but still assumes KV-drop rollback we must
replace.)

## State to roll back (measured, `rwkv7_backend.py`)
Per layer, in the MambaPool, indexed by `mamba_cache_indices`:
- `conv[0]`, `conv[1]`: token-shift states, `[size+1, H, 1]` fp32 (attn + ffn prev-token).
- `temporal`: the WKV recurrent state S, `[size+1, H, K, V]` fp32.
Both are **O(1) per token** (do not grow with context) and small (1.5B: ~12.6 MB/slot temporal +
2·H conv). This is the key enabler — checkpoint/rollback is cheap, unlike a transformer KV cache.

## Design
1. **Draft**: a smaller RWKV-7 (0.1B) — same tokenizer/vocab/family (high token agreement with the
   target ⇒ high acceptance) — greedily proposes K tokens (K≈4–8), advancing its own O(1) state.
2. **Verify (target)**: run the target recurrence over the K draft tokens via the **extend path**
   (`recurrence(...)` varlen branch snapshots `init_state = temporal[cache_indices]` and computes on
   that copy — but NOTE it writes `final_state` back to the pool at the end, so the verify call must
   either pass a no-commit flag or snapshot-and-restore `temporal` + `conv` around it; that restore
   is exactly the O(1) rollback below, so it is not extra machinery). One target forward yields K
   logits.
3. **Accept**: longest matching prefix — accept draft token j while `draft[j] == argmax(target_logits[j])`
   (greedy target; the standard spec-decode acceptance for temperature 0). Let J = #accepted (0..K);
   the target's own token at position J (its argmax) is always appended, so each round commits J+1
   tokens for one target forward.
4. **Roll state to J+1**: commit `temporal[cache_indices] = S_{J+1}` and `conv[·] = shift_{J+1}`.
   Two clean options (pick by measurement):
   (a) **checkpoint-per-token**: the verify kernel emits the K intermediate states (O(1)·K memory,
       cheap for small K); restore the (J+1)-th. Fastest, one forward.
   (b) **re-run J+1**: keep only `init_state = S_0`; after choosing J, re-run the recurrence over the
       first J+1 tokens to produce the committed state. Two forwards (verify K + commit J+1), no
       kernel change. Start here (simpler), move to (a) if the re-run cost matters.

## Where it wins — VIABILITY MEASURED (F0029): α = 0.738 → ~2.0–2.7× bsz1 (estimate)
Payoff scales with **target size × acceptance rate**. Measured (F0029, `bench/spec_accept.py`): the
0.191B draft's greedy argmax matches the target token **α = 0.738** of positions (n=603, 1.5B target,
shared RWKV7-G1 family). At block K=4 that is ~2.98 target-tokens/forward → net bsz1 speedup estimate
**≈1.99× vs 1.5B target, ≈2.69× vs 7.2B target** (true param ratios 1/8.0 and 1/37.7; the 7.2B row
assumes the 1.5B-measured α — unmeasured there). The gate PASSES — best on the **7.2B target**.
Metric for the build: measured accepted-tokens/forward × target tok/s.

## Integration + risk
The sglang spec scheduler is EAGLE/tree-shaped; RWKV's chain-verify + state-rollback does not fit it,
so this is a **bespoke path in the RWKV backend + a thin scheduler hook** (draft engine, verify call,
accept/rollback), not a reuse of `speculative_algorithm=EAGLE`. Risk: HIGH (new control flow +
state-rollback correctness). **Gate (non-negotiable)**: spec-decode output must be **token-identical
to plain greedy decode** (spec-decode is exact-by-construction: it only accepts tokens the target
would have produced) — verify with the existing greedy oracle (`bench/verify_*.py`) that spec-on ==
spec-off token-for-token. Build incrementally: (i) draft engine + target verify (re-run rollback,
option b) at bsz1, greedy-exact vs plain; (ii) checkpoint-per-token rollback (option a); (iii)
batched spec; (iv) 7.2B target measurement.

## Consequences
Adds a capability albatross and the other RWKV serving stacks lack entirely (they have no scheduler,
let alone spec-decode). Exact-by-construction, so no accuracy trade-off. Sequenced AFTER the
reverse-overtake (done) as the next major thrust; this ADR unblocks the build.

## Cross-references
[[F0022]] (state cache / MambaPool state layout) · ADR-0005 (reverse-overtake, done) ·
`rwkv7_backend.py` (recurrence extend path = the verify/rollback substrate).
