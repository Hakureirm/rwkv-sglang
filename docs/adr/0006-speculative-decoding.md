---
doc_kind: adr
adr_id: 0006
title: "RWKV-7 speculative decoding (req#6): recurrent-verify + O(1)-state rollback — a bespoke draft/target loop, NOT EAGLE"
status: "built — correctness gate passes (10/10); net speedup not yet achieved (see 2026-07-07 BUILD RESULT)"
date: 2026-07-04
last_verified_commit: "0cc881280 (sglang-upstream/rwkv7-spec-decode)"
supersedes: []
superseded_by: []
---

# ADR-0006: RWKV-7 speculative decoding

## 2026-07-07 BUILD RESULT (Sonnet 5) — Strategy B built for real, 10/10 correctness gate, cuda-graph speedup real but net still negative

Full detail in `docs/findings/0046-spec-decode-strategy-b-build.md`. Summary for anyone scanning
this ADR:

- **Strategy B (the 2026-07-06 pivot below) is now actually implemented**, not just decided — a
  prior handoff had conflated "decided in a memory note" with "built in the worker file," which
  this build caught and corrected by reading the live code rather than trusting the note.
- Two real logic bugs found via instrumentation (not guessed): `intermediate_ssm`/
  `intermediate_conv_window` are indexed by request-ordinal-in-batch, not pool slot; and RWKV-7's
  *second* token-shift (`conv[1]`, the FFN one) is never corrected by the generic upstream commit
  hook, which only handles `conv[0]` — because RWKV-7's model code bypasses the generic
  `AttentionBackend.forward_extend`/`forward_decode` dispatch that GDN/KDA/Lightning's verify
  capture lives on. `rwkv7_backend.py` needed a small `is_target_verify` branch added to close
  this gap (not budgeted in the original pivot plan).
- **Gate: `bench/spec_gate.py` (written this session, didn't exist before) is 10/10** token-identical,
  spec-on vs spec-off, greedy — held at both 128 and 256 generation length. The pre-existing
  non-spec-decode regression suite (`test/registered/models/test_rwkv7.py`) stayed 3/3, zero
  regression, after the shared `models/rwkv7.py` projection layer picked up a `gemv_mb`-routed fix
  for a separate M>1 cuBLAS-reduction-order flip during verify.
- **Speed: a hand-rolled `torch.cuda.CUDAGraph` capture for the draft's own K-1 step eager loop**
  (not sglang's shared `DecodeCudaGraphRunner`, which hardcodes an EAGLE-family
  `capture_forward_mode=TARGET_VERIFY` assumption incompatible with a plain recurrent draft) gives
  a real, clean, all-7-test-prompts-positive **1.5–1.6× speedup on the draft decode step itself**.
  But **net spec-on is still 2.6×–4.5× slower than spec-off** (down from 7× before the graph) —
  correctness is done, speed is a real, gated, honest partial win, not yet the ADR's stated goal.
- Three next-lever hypotheses recorded (per-layer state-clone cost, target-verify overhead,
  per-round Python orchestration) — **none profiled yet**, flagged as inference not measurement.

## 2026-07-06 PIVOT (Opus) — main gained recurrent spec-V2; reuse upstream verify, build only the draft (Strategy B)

The bespoke chain-verify worker below (increment (i), v0.5.10, F0031) was written when sglang
had no recurrent spec support. **Current main (`b28bc10`) has since rewritten the whole spec
subsystem to "spec-V2" and it already implements recurrent/mamba speculative verify+commit**,
which changes the build from "port 595-line worker" to "reuse upstream verify, add only the draft":

- **The RWKV-7 backend already verifies.** `layers/attention/hybrid_linear_attn_backend.py`
  handles `ForwardMode.TARGET_VERIFY`, captures per-draft-token state in
  `MambaPool.SpeculativeState.intermediate_ssm` / `intermediate_conv_window`, and exposes
  `update_mamba_state_after_mtp_verify`.
- **The commit is upstream + target-agnostic.** `speculative/spec_utils.commit_mamba_states_after_verify`
  commits the accepted step's recurrent state (`accept_lens-1` for topk==1 = chain-accept), no-ops
  for non-mamba models. **This is exactly "option (a)" (per-token checkpoint) below — upstream now
  provides it for free**, superseding the "option (b) re-run" plan and the hand-rolled
  snapshot/restore.
- **Registration is a plugin.** `@SpeculativeAlgorithm.register(...)` (`speculative/spec_registry.py`)
  — no enum / scheduler edits. Workers subclass `BaseSpecWorker` and implement
  `forward_batch_generation(batch, on_publish=None) -> GenerationBatchResult`; the V1 separate-worker
  path is gone, so run non-overlap (`--disable-overlap-schedule`).
- **`NGRAMWorker` is the template** (a non-EAGLE draft source that builds a `VerifyInput` and reuses
  the verify path). Our worker = NGRAM's shape, but the draft source is a 0.1B RWKV-7 (own
  `TpModelWorker` with `req_to_token_pool=None` → own MambaPool) run K greedy decode steps → a
  chain (topk=1) verify input (`retrieve_index` linear, lower-triangular mask) → reuse
  `TARGET_VERIFY` + `eagle_sample` + `commit_mamba_states_after_verify`.

**The one genuinely new piece** is the *draft's own* recurrent-state rollback (upstream's commit
handles the target only): draft also captures `intermediate_ssm` and commits at accepted length, or
re-runs J steps, or snapshots its slot. Everything below (draft construction, K-step decode, greedy
chain acceptance) still applies; the target verify/rollback machinery is now upstream's.
**Benefits:** far less code, upstream-maintained, option-(a) speed for free, and **upstreamable**
(RWKV-7 spec support contributed back). Build + gate (spec-on == spec-off token-identical) on the
tower main container under `--disable-overlap-schedule`. The `gemv_mb` primitive (3cafe02) is
demoted to an ε-flip backup unless the upstream verify path itself trips the near-tie flip.

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
