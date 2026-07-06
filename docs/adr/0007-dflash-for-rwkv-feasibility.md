---
doc_kind: adr
adr_id: 0007
title: "DFlash speculative decoding for RWKV-7: feasibility spike — verdict: don't pursue"
status: accepted
date: 2026-07-06
last_verified_commit: "e125583"
supersedes: []
superseded_by: []
---

# ADR-0007: DFlash speculative decoding for RWKV-7 — feasibility

## Context
ADR-0006 scoped our primary spec-decode strategy: a bespoke chain/greedy draft-and-verify
loop (0.1B RWKV-7 draft → larger RWKV-7 target), measured viable at **α ≈ 0.738–0.7485**
([[F0029]]). That build is separate, in-flight work (not touched here). This ADR is a
research-only spike on a **second, independent** sglang-upstream algorithm — **DFlash**
(`--speculative-algorithm DFLASH`) — asking whether it is *also* worth adapting to RWKV-7.

Source: read-only over `/Users/hakureirm/codespace/Study/sglang-upstream` (local fork,
HEAD `4405d9fbf`, 2026-07-06). Primary files read in full: `python/sglang/srt/speculative/
dflash_worker_v2.py` (1706 lines), `dflash_info.py`, `dflash_info_v2.py`, `dflash_utils.py`
(794 lines), `python/sglang/srt/models/dflash.py` (579 lines), `python/sglang/srt/speculative/
triton_ops/dflash.py`, plus supporting reads of `hybrid_linear_attn_backend.py`,
`python/sglang/srt/models/{llama,rwkv7}.py`, `model_executor/model_runner.py`, and
`docs_new/docs/advanced_features/speculative_decoding.mdx`. No sglang-upstream file was
modified. All line numbers below are relative to `python/sglang/srt/` unless stated
otherwise.

**Correction to the initial brief**: `get_dflash_context_layer_ids` does not exist anywhere
in this checkout (repo-wide grep, zero hits). The actual function is
`build_target_layer_ids()` / `DFlashDraftConfig.resolve_target_layer_ids()`
(`speculative/dflash_utils.py:258-295, 407-436`).

## What DFlash actually is

### 1. The draft is a real transformer; it borrows the target's embedding and lm_head
`models/dflash.py:1-4` states this directly: "This model intentionally does not include
token embeddings or an LM head; DFlash uses the target model's embedding/lm_head."
`DFlashAttention` (`models/dflash.py:71-196`) is a standard GQA block: `QKVParallelLinear`,
per-head RMSNorm, RoPE (`self.rotary_emb(positions, q, k)`, line 191), and `RadixAttention`
(line 149-157) with the target's own paged KV pool machinery. There is no ambiguity here —
the draft is attention-native, not recurrent-flavored in any way.

### 2. Target context reaches the draft via direct KV-cache injection, not re-computation
The draft never re-runs its own layers over the full prior context. Instead:
`DFlashDraftModel.project_target_hidden()` (`models/dflash.py:382-394`) projects a
**concatenation of K target-layer hidden states** (`num_context_features * hidden_size` →
`hidden_size`, via `self.fc` + `hidden_norm`) into one draft-hidden-size vector per
committed token. `_append_target_hidden_to_draft_kv_by_loc()`
(`speculative/dflash_worker_v2.py:859-1014`) then, per draft layer, projects that vector to
K/V only (`attn.kv_proj_only`, `models/dflash.py:202-225`), applies k-norm/RoPE, and writes
it straight into the draft's own paged KV cache at the token's slot
(`set_kv_buffer`/`set_kv_buffer_prefix_valid`). This is called once per committed token,
both at prefill (`dflash_worker_v2.py:1268-1290`) and after every verify step
(`dflash_worker_v2.py:1679-1685`).

Which K target layers: `build_target_layer_ids()` (`dflash_utils.py:258-295`) picks layers
spread across target depth (e.g., the single middle layer for a 1-layer draft; evenly spaced
between layer 1 and `num_target_layers-3` otherwise) — a multi-layer feature fusion in the
same spirit as EAGLE3's low/mid/high-layer fusion. Enabling this on a target model is not
free: `model_executor/model_runner.py:1070-1075` hard-requires
`hasattr(self.model, "set_dflash_layers_to_capture")` and raises if absent. Every currently
supported target implements this per-model (e.g. `models/llama.py:825-833`, mirroring
`set_eagle3_layers_to_capture` at lines 811-823, which stashes `layer_ids` into
`self.model.layers_to_capture` and the layer loop appends `hidden_states + residual` into an
`aux_hidden_states` list, `models/llama.py:405-408`). **`models/rwkv7.py` has none of this
today** (grep for `capture`/`layers_to_capture`/`aux_hidden` over the whole file: zero hits).

By default this KV-injected context is **unbounded** — it grows with the whole conversation,
exactly like a normal transformer KV cache, unless `--speculative-dflash-draft-window-size`
is set to bound it to a fixed recent window (`dflash_worker_v2.py:122-125, 547-563`, already
upstream-supported as an explicit opt-in).

### 3. Block drafting is genuinely parallel / masked, not a dressed-up loop — confirmed
This is the crux question and it resolves cleanly. Per decode round
(`dflash_worker_v2.py:1339-1420`): `block_ids` is filled entirely with
`self._mask_token_id`, then **only position 0** is overwritten with the real
previous-round bonus token (`block_ids.fill_(int(self._mask_token_id))` /
`block_ids[:, 0].copy_(draft_input.bonus_tokens)`, lines 1379-1380 and mirrored in the eager
fallback at 1398-1399, and again at the Triton-kernel level,
`speculative/triton_ops/dflash.py:177-178`). That whole block
`[bonus, MASK, MASK, ..., MASK]` is embedded in **one call**
(`noise_embedding = embed_module(block_ids)`, line 1417 — the variable name alone is a
strong signal of the masked/non-autoregressive lineage), then run through **one single**
`self.draft_model_runner.forward(forward_batch)` call (lines 1493-1494) covering all
`block_size` positions at once. The block_size-1 predictions are read off via one batched
matmul over all positions simultaneously (`_DflashDraftSampler.__call__`,
`dflash_worker_v2.py:76-88`, or `_greedy_sample_from_vocab_parallel_head`, lines 1506-1511) —
never a Python-level loop over positions.

Critically, `DFlashVerifyInput.topk` is hardcoded to `1` with the comment "DFLASH verify is
linear (non-tree)" (`speculative/dflash_info.py:33-35`), which the shared spec_v2 attention
plumbing turns into **causal** masking. Because every non-bonus block position holds the
*identical* MASK embedding, causal self-attention within the block conveys no information
about what any other position resolved to — each of the `block_size-1` predictions is made
independently from (a) the frozen injected target context, (b) the real bonus token, and (c)
that position's own RoPE angle. This is a genuine one-shot, non-autoregressive
masked-block prediction (MaskGIT-flavored), **not** sequential drafting wearing a batched
API — confirmed at the tensor-flow level, not just from the docstring.

*(Aside, minor: the docs table, `docs_new/docs/advanced_features/speculative_decoding.mdx`
line 94, claims DFLASH "disables overlap scheduler," but `DFlashDraftInputV2` in
`dflash_info_v2.py` implements a full overlap-scheduling `prepare_for_decode` with
plan-stream KV over-allocation — the doc appears stale relative to the spec-v2 rewrite. Not
load-bearing for this ADR, noted for hygiene.)*

### 4. Verification: exact for greedy; a different, secondary mechanism for sampling
Greedy path — `compute_dflash_correct_drafts_and_bonus()` (`dflash_utils.py:547-585`):
`matches = candidates[:, 1:] == target_predict[:, :-1]`, longest contiguous prefix accepted,
first-mismatch (or end-of-block) target token appended as bonus. This is the textbook
greedy speculative-decoding rule and is **provably token-identical to plain greedy
autoregressive decoding by construction** — the output at every position is always the
target's own argmax, either because the draft happened to guess it or because it is
substituted at the first disagreement. Structurally identical to our own chain-verify rule
in ADR-0006. This is the path that matters for our project's "spec-on == spec-off
token-identical" invariant, and it holds.

Sampling path — `compute_dflash_sampling_correct_drafts_and_bonus()`
(`dflash_utils.py:588-773`) reuses `tree_speculative_sampling_target_only`
(`sgl-kernel/csrc/speculative/speculative_sampling.cu`), the same kernel EAGLE's tree verify
uses, but with a forced chain topology (`retrieve_next_sibling` all `-1`) and
**`draft_probs = torch.zeros_like(target_probs)`** (line 734) — i.e., it is the
"target-only" variant (no real draft distribution `q(x)` exists because the draft chose its
token deterministically), gated by `threshold_single`/`threshold_acc`
(`server_args.py` defaults both `1.0`). This is *not* the classic Leviathan/Chen `p/q`
rejection-sampling algorithm; it is a threshold-style acceptance rule, and it is visibly the
less-mature path: gated behind `is_dflash_sampling_verify_available()` with a silent
fallback to greedy verification when the kernel is unavailable
(`dflash_worker_v2.py:1184-1198`, function named `_validate_phase1_sampling_support`), and
**every single example command in the docs page — EAGLE, DFLASH, STANDALONE, NGRAM alike —
uses `temperature=0`**. For our purposes this is moot either way: our binding correctness
requirement is the greedy invariant, and that path is exact.

### 5. Target-side verify already generalizes beyond pure attention — and RWKV-7 already plugs into it
This was the most consequential and least expected finding. `_update_target_mamba_state_after_verify()`
(`dflash_worker_v2.py:1089-1133`) checks
`hasattr(attn_backend, "update_mamba_state_after_mtp_verify")` and, if present, commits
"Mamba intermediate states for accepted verify steps" using `commit_lens - 1` as the
per-request accepted index. The real implementation
(`layers/attention/hybrid_linear_attn_backend.py:1006-1060`, on `HybridLinearAttnBackend`)
operates generically over `MambaPool`'s `intermediate_ssm` / `intermediate_conv_window`
buffers via a fused gather-scatter kernel — it has no attention-specific logic at all.

Tracing whether RWKV-7 actually reaches this: `model_executor/model_runner.py`'s
`_get_attention_backend_from_str` special-cases RWKV-7 explicitly — *"RWKV-7 is all-linear
(zero full-attention layers). Do not construct a real full-attn backend
... HybridLinearAttnBackend only needs a no-op stub here"* — builds a
`Rwkv7NoOpFullAttnBackend` stub as the "full" half and calls `attn_backend_wrapper`, which
(`layers/attention/attention_registry.py:352-354`) sets
`linear_attn_backend = Rwkv7AttnBackend(runner)` and composes both into a
`HybridLinearAttnBackend`. So **RWKV-7's live `attn_backend` object at runtime is a
`HybridLinearAttnBackend`** — the exact class DFlash's hook already targets — with
`Rwkv7AttnBackend(MambaAttnBackendBase)` (`layers/attention/linear/rwkv7_backend.py:87`) as
its linear half. Additionally, `MambaAttnBackendBase._forward_metadata`
(`hybrid_linear_attn_backend.py:29-160`) already branches on
`forward_batch.forward_mode.is_target_verify()` generically (building `query_start_loc`,
tree-retrieval buffers, etc.) — this is shared, pre-existing base-class machinery, not
something built bespoke for our own chain-verify effort. RWKV-7 inherited working
MTP/spec-decode `TARGET_VERIFY` scaffolding automatically by virtue of being implemented as
a `MambaAttnBackendBase` subclass. **Net: `hasattr(attn_backend,
"update_mamba_state_after_mtp_verify")` is already `True` for an RWKV-7 target today, with
zero new code.** This corroborates (with a precise mechanism, not just a restated claim)
ADR-0006's note that "the RWKV-7 backend already verifies."

## Fit assessment A: target = RWKV-7, draft = transformer
Architecturally coherent, but not free. Concretely missing today:
1. **`set_dflash_layers_to_capture` + aux-hidden accumulation on `Rwkv7Model`/`Rwkv7ForCausalLM`.**
   Does not exist (confirmed above). Mechanical, same shape as `models/llama.py:382-430,
   811-833` — the RWKV-7 layer loop (`models/rwkv7.py:537-570`) already produces `x` per
   layer index; capturing it at configured indices is a small, low-risk addition.
2. **A one-shot, block-wide `TARGET_VERIFY` forward that is correct from a checkpointed
   state.** Partially shared with, and partially still-pending in, our own chain-verify
   design: ADR-0006 frames exactly this ("option (a): checkpoint-per-token") as the
   *not-yet-built* fast path, currently starting from the simpler "option (b) re-run."
   Building it once benefits both strategies — but it is not "already done" the way §5 above
   is.
3. **Mask-token resolution against the RWKV tokenizer** (`_resolve_mask_token_id`,
   `dflash_worker_v2.py:565-645`) — mechanical, assuming the RWKV tokenizer can add a special
   token cleanly.

## Fit assessment B: draft = RWKV-7 (the other direction)
This is **not a faithful DFlash port** — it would be a different algorithm wearing DFlash's
name. Two independent, structural blockers:

1. **KV injection has no recurrent analogue.** DFlash's context-sharing trick fundamentally
   depends on attention's per-position addressability — a K/V slot *is* a specific past
   position that a query can look up. RWKV's state is the opposite by design: a fixed-size,
   already-collapsed recurrence (`conv[0]/conv[1]` token-shift + `temporal` WKV state,
   `[size+1, H, K, V]`, per ADR-0006's own measured layout). There is no "extra slot" to drop
   a projected target feature into; you would have to invent a wholly new
   state-conditioning mechanism (e.g., projecting into the full `[H,K,V]` state tensor
   directly), which is a new research contribution, not an adaptation of the existing `fc:
   (K·hidden)→hidden` per-token projection DFlash actually ships.
2. **Mask-and-parallel-predict needs an explicit per-slot positional signal that RWKV does
   not have.** DFlash tells block position 5 apart from block position 6 — both holding the
   *identical* MASK embedding — purely via RoPE (`models/dflash.py:191, 232-236`, explicit
   `self.rotary_emb(positions, q, k)` calls). RWKV-7 has no analogous mechanism: `positions`
   is accepted as a forward argument for API uniformity
   (`models/rwkv7.py:540, 655, 663`) but is **never referenced** inside
   `Rwkv7Model.forward`'s body (`models/rwkv7.py:537-570`, confirmed by direct read) or
   anywhere else in the file (grep for `position` in `models/rwkv7.py` returns only the three
   pass-through signature/call sites). RWKV encodes order purely through sequential state
   evolution — which requires genuinely different, ordered inputs to produce different
   outputs. Feeding it `block_size-1` copies of the same mask token in one shot, the way
   DFlash's draft does, has no way to convey "this is offset 5 vs. offset 6."

Conclusion: draft-as-RWKV is out of scope as "DFlash for RWKV" — pursuing it would mean
designing a new algorithm, which this ADR does not attempt to scope.

## Would it beat the chain-verify ceiling? (reasoned, not measured)
[[F0029]] measured chain-verify at **α = 0.738–0.7485** (0.191B draft vs. 1.5B/7.2B RWKV-7
targets, same family), giving an **estimated** ~2.0–2.75× bsz1 speedup at K=4 — a number that
degrades geometrically in K because each successive chain step is autoregressive on the
*draft's own* (possibly already-wrong) prior guess.

DFlash's block-parallel scheme does not sidestep decay with block position — it decays for a
**different structural reason**: each block position is predicted blind to every other
still-masked position in the same block (no intra-block self-conditioning at all, per §3
above). This is the same fundamental difficulty multi-token-prediction/Medusa-style heads
face — predicting `i` tokens ahead with zero knowledge of the intervening tokens is
intrinsically harder than predicting one token ahead from full context, and accuracy is
well-documented (in that adjacent literature) to fall off quickly with offset. DFlash's own
multi-layer target-feature fusion (§2) exists specifically to fight this decay, and by report
it works — but only because the draft's `fc`/`hidden_norm` projection and full trunk are
**trained end-to-end against that specific target's hidden-state manifold**. There is no
existing "DFlash draft for RWKV-7" checkpoint (unlike EAGLE3, which sglang's own docs point
to a training framework — SpecForge — DFlash has no training recipe or repo referenced
anywhere in `docs_new/`; the only example checkpoint,
`z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`, is a transformer-target-specific artifact we
cannot reuse). Our chain-verify draft, by contrast, is an **off-the-shelf pretrained 0.1B
RWKV-7 from the same model family** — zero new training required to hit the measured α above.

**This comparison is a structural projection, not a benchmark** — flagged explicitly as such
per this project's own rule that speed/quality claims need numbers or an honest "unmeasured."
We have no a priori reason to expect DFlash's offset-conditioned block accuracy beats a
geometric chain at matched K for RWKV, and every reason to expect that finding out requires
an expensive from-scratch distillation-training project, not a rerun of an existing
benchmark script.

## What it would cost, concretely
- **Target side (§A)**: small-to-medium. One new method + a layer-loop change in
  `models/rwkv7.py`; a state-checkpoint-per-token verify path that mostly overlaps with
  ADR-0006's own still-pending "option (a)."
- **Draft side (§B applied to a transformer draft, since RWKV can't be the draft)**: large and
  open-ended. A brand-new paired (RWKV-7 target, transformer draft) training pipeline with no
  existing tooling, checkpoint, or reference recipe in this codebase or upstream docs —
  qualitatively larger and less certain than anything else in this project's roadmap.
- **Ongoing**: a second, structurally unrelated spec-decode code path (RadixAttention,
  paged KV, RoPE, mask-token bookkeeping) permanently living inside an otherwise
  attention-free, hand-rolled RWKV stack — in tension with this project's own
  no-FLA-dependency / elegance posture ([[F(no-fla)]], ADR-0004): we would be adding a real
  attention/KV-cache dependency purely to serve a speculative side-channel.

## Verdict: **Don't pursue**
Grounded in four concrete findings, not a general "seems hard":
1. **Draft-as-RWKV is not a port at all.** DFlash's two defining mechanisms — KV-cache
   context injection and RoPE-distinguished masked-block prediction — both depend on
   attention-specific properties (addressable per-position K/V slots; an explicit positional
   signal orthogonal to token identity) that RWKV-7 structurally lacks and does not use
   anywhere in its current sglang implementation (`models/rwkv7.py`, verified by direct read).
2. **Target-as-RWKV is feasible but the expensive half is untouched by that feasibility.**
   The target-side machinery is a genuinely pleasant surprise — RWKV-7 already satisfies
   DFlash's Mamba-generalized verify/commit hook today, for free, as an inherited consequence
   of implementing `Rwkv7AttnBackend` as a `MambaAttnBackendBase` subclass wrapped in
   `HybridLinearAttnBackend`. But the dominant cost is the *draft* side (§B is a hard no, so
   the draft must stay a transformer), and that requires an unbounded, from-scratch training
   investment with no existing checkpoint, recipe, or tooling anywhere in this project or
   upstream — a fundamentally different order of effort than reusing an off-the-shelf 0.1B
   RWKV-7 draft.
3. **No structural reason to expect it beats what we already have.** DFlash's block-parallel
   acceptance plausibly decays with block position for reasons analogous to
   multi-token-prediction head decay, which is not obviously better — and could easily be
   worse at the K that matters — than chain-verify's already-measured, already-passing
   α ≈ 0.738–0.7485 ([[F0029]]). We have no evidence either way; the point is that acquiring
   evidence costs a training project, not a rerun.
4. **It reintroduces exactly the dependency this project has spent effort removing.** A
   transformer draft means RadixAttention, paged KV, and RoPE become permanent fixtures of
   the serving stack, in direct tension with the no-FLA / attention-free / hand-rolled-kernel
   posture already committed to elsewhere in this project.

If priorities change later (e.g., a matched DFlash-style draft checkpoint for an RWKV-7
target becomes available from a third party, removing cost item 2), the target-side finding
in §5 means re-opening this ADR would start from a meaningfully better position than a naive
reading of "DFlash needs attention" would suggest — that finding is worth remembering even
though it does not flip today's verdict.

**Adjacent idea, explicitly out of scope**: DFlash's multi-layer target-hidden-state fusion
(§2) is a reusable *idea* independent of its KV-injection *mechanism* — conceivably, a future
increment could feed a projected summary of the target's mid-layer hidden state into the
existing chain-verify RWKV-7 draft's embedding as extra conditioning (closer in spirit to
EAGLE/EAGLE3's feature-passing than to DFlash specifically). This is not scoped, not
evaluated for feasibility here, and should not be read as part of this ADR's verdict — it is
a footnote for a possible future spike, not a recommendation.

## Consequences
- No new build this cycle from this spike; ADR-0006's chain-verify remains the sole
  spec-decode strategy in flight.
- The §5 finding (RWKV-7 already satisfies DFlash's generalized Mamba verify/commit hook) is
  worth surfacing if we ever talk to upstream about RWKV-7 + spec-decode more broadly — it is
  evidence our backend integration chose a good shape (subclassing `MambaAttnBackendBase`
  rather than a bespoke backend), independent of DFlash.
- No sglang-upstream files were modified; no production code was written for this spike.

## Cross-references
[[F0029]] (chain-verify viability, α measurement) · ADR-0006 (chain-verify design, the
active strategy) · ADR-0004 (no-FLA-dependency posture) ·
`speculative/dflash_worker_v2.py`, `speculative/dflash_utils.py`, `models/dflash.py`,
`layers/attention/hybrid_linear_attn_backend.py`, `layers/attention/linear/rwkv7_backend.py`,
`models/rwkv7.py` (sglang-upstream, HEAD `4405d9fbf`).
