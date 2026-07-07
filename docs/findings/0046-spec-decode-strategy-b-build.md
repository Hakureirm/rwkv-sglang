---
doc_kind: finding
finding_id: F0046
title: "RWKV_SPEC on sglang main, Strategy B built for real: 10/10 correctness gate (spec-on == spec-off token-identical, greedy, at gen-len 128 and 256) with zero regression on the existing non-spec suite; hand-rolled draft-decode CUDA graph gives a real 1.5-1.6x speedup on the draft step, but net spec-on remains 2.6x-4.5x slower than spec-off — a genuine partial win, not yet ADR-0006's stated goal"
last_verified_commit: "sglang-upstream/rwkv7-spec-decode @ 0cc881280; delivery repo main @ 29240be"
discovered_by: Sonnet 5 (agent-assisted, tower), 2026-07-07
severity: info
status: open
related: [F0029, F0031, ADR-0006]
---

# Finding F0046: RWKV_SPEC Strategy B — correctness done, speed a real partial win

## Context

ADR-0006 designed RWKV-7 speculative decoding as a bespoke chain-verify + O(1) state rollback,
because RWKV's recurrence has no KV cache for EAGLE's tree-attention verify to exploit. A
2026-07-06 pivot (recorded in ADR-0006 itself) found that sglang main had, in the meantime,
rewritten its whole speculative subsystem to "spec-V2" with a plugin registry and — critically —
already-built recurrent/mamba verify+commit machinery (`commit_mamba_states_after_verify`,
target-agnostic). That changed the plan from "port a 595-line bespoke v0.5.10 worker" to
"reuse upstream verify+commit, build only a thin RWKV draft-glue worker" (Strategy B, modeled on
`NGRAMWorker` — the template for "draft tokens from a non-transformer source").

**This finding is the build.** A prior handoff described Strategy B as already implemented in
`python/sglang/srt/speculative/rwkv_spec_worker.py`. Reading the live file found it was not —
still the older, fully self-contained "Strategy A" (manual snapshot/restore, re-run-on-partial-
acceptance), with the pivot only decided in a memory note, never actually built. That gap —
"decided" vs "implemented" — is itself worth naming: briefs written from memory notes must be
checked against the live file before being trusted, not assumed current.

## What was built

### The Strategy A → B conversion

`RwkvSpecWorker` (subclass of `BaseSpecWorker`, plugin-registered as `RWKV_SPEC` via
`@SpeculativeAlgorithm.register`) now: runs the draft (an independent 0.1B RWKV via its own
`TpModelWorker`, `req_to_token_pool=None` so it owns its own `MambaPool` slot) for K-1 greedy
eager decode steps; builds a chain (topk=1) `NgramVerifyInput` from the draft tokens
(`retrieve_index=arange(bs*K).view(bs,K)`, `retrieve_next_token`/`retrieve_next_sibling` as real
trivial tensors — **not** `None`, despite the dataclass default; `NgramVerifyInput.__init__`'s
`=None` is just a Python default, `sgl_kernel.verify_tree_greedy`'s bound C++ signature has no
None-handling, confirmed by reading `verify_tree_greedy_func`, `eagle_utils.py:353`); calls
`target_worker.forward_batch_generation(batch, is_verify=True)`, reusing upstream's
`eagle_sample` + `commit_mamba_states_after_verify` for the target's own state commit — the
"free" part of the pivot's promise.

### The gap the pivot didn't budget: RWKV-7's backend needed its own verify-capture

`commit_mamba_states_after_verify`'s commit/scatter side is genuinely backend-agnostic (confirmed
in `hybrid_linear_attn_backend.py`). Its *capture* side is not: `gdn_backend.py` / `kda_backend.py`
/ `lightning_backend.py` all have an explicit `is_target_verify` branch in `forward_extend` that
writes into `MambaPool.SpeculativeState.intermediate_ssm`/`intermediate_conv_window` — but
`rwkv7_backend.py` had none, because `models/rwkv7.py` calls `token_shift`/`recurrence` directly,
bypassing the generic `AttentionBackend.forward_extend`/`forward_decode` dispatch those other
backends' capture logic lives on (RWKV-7's own `forward_extend`/`forward_decode` on the backend
class raise `NotImplementedError` — never called). Fix: a K-step Python loop inside
`rwkv7_backend.py`'s decode path, reusing the existing single-step decode kernel (same M=1 batched
call as plain decode — a deliberate choice: since this is always topk=1 chain with small K, a
Python loop was judged not worth a new fused CUDA kernel), that in addition to its normal
in-place `cache.conv`/`cache.temporal` update, writes a second copy of each step's state into
`cache.intermediate_ssm[cache_indices, step]` / `cache.intermediate_conv_window[...]` (shapes
confirmed from `memory_pool.py`: `[pool_size+1, speculative_num_draft_tokens, H, K, V]` and
`[pool_size+1, speculative_num_draft_tokens, dim, K-1]`). After all K steps, the persistent state
holds "as if fully accepted"; the generic `update_mamba_state_after_mtp_verify` scatter then
corrects it back to the true accepted length — same pattern GDN uses with its own fused kernel,
here done in plain Python since K is small and always chain-shaped.

### Two real logic bugs found via instrumentation, not guessed

1. `intermediate_ssm`/`intermediate_conv_window` are indexed by **request-ordinal-in-batch**, not
   by pool slot — confirmed by reading the kernel source rather than assumed from the tensor
   shape alone. Getting this backwards silently corrupts a different request's state at
   batch size > 1.
2. RWKV-7 has **two** token-shift states per layer (`conv[0]` for time-mixing/attention,
   `conv[1]` for the FFN/channel-mix) — the generic upstream commit hook only ever corrected
   `conv[0]`. This is not an oversight in the generic code; GDN/KDA/Lightning only have one
   token-shift-equivalent state, so the hook was never written with a second one in mind.
   `rwkv7_backend.py` needed its own explicit handling for `conv[1]`.

### The remaining verify-path numerical flip, and its fix

Beyond the backend-capture gap, one more source of spec-on/spec-off divergence: the target
verify computes the round's K positions' `r/k/v/w/a/kk` projections as a single batched (M=K)
matmul in `models/rwkv7.py`, while the plain decode baseline computes each token as an M=1 GEMV —
different cuBLAS reduction order, same class of near-tie argmax flip as F0031's original finding.
Fix: route the verify-path (M>1 AND `is_target_verify`) through `gemv_mb` — a primitive already
built and gated bit-identical to `gemv_m1` row-for-row (`bench/verify_gemv_mb.py`) — while the
plain M=1 decode/prefill path is untouched (structurally unreachable to trigger outside
`RWKV_SPEC`'s own verify calls, by construction of the guard condition). `models/rwkv7.py` is the
shared model file every serving path in this project uses, so this change was gated on BOTH
`bench/spec_gate.py` (10/10) AND the pre-existing non-spec-decode regression suite
(`test/registered/models/test_rwkv7.py`, 3/3 with `RWKV_FAST_LINEAR` both off and on) — zero
regression outside spec-decode.

**A genuinely unrelated discovery along the way**: the canonical `models/rwkv7.py` this project
uses in production isn't in the `sglang-upstream` fork at all — it lives in the separate public
delivery repo (`github.com/Hakureirm/rwkv-sglang`, `sglang_overlay/`). That tree had an unrelated
in-flight w8a8 kernel patch sitting uncommitted on the exact file needed here; handled by
inspecting the diff, isolating with `git stash`, committing the gemv_mb fix on a clean base
separately, then `git stash pop` (clean auto-merge, nothing clobbered) — not something to
routinely expect, but a reminder to always check `git status` before editing a shared file whose
working tree you don't already know is clean.

### Gate result

`bench/spec_gate.py` (written this session — it didn't exist yet, another gap between what a
prior handoff assumed and what was actually on disk) is **10/10** token-identical (spec-on vs
spec-off, greedy), verified at both 128 and 256 generation length (~200+ replay rounds per
prompt at the longer setting) and with the draft-decode CUDA graph (below) both on (default) and
off (`RWKV_SPEC_DRAFT_GRAPH=0`, confirming the fallback path itself is also correct, not just the
happy path).

## Speed: a hand-rolled draft-decode CUDA graph — real, positive, not yet sufficient

The draft's K-1 step eager decode loop is the dominant cost: launch overhead for a 0.1B model's
many small per-step kernels, not compute. The natural fix — enable sglang's existing
`DecodeCudaGraphRunner` for the draft, since its per-step `ForwardBatch` is always bs=1/DECODE —
hits a real wall: `DecodeCudaGraphRunner` hardcodes `capture_forward_mode=TARGET_VERIFY` for ANY
draft worker under ANY speculative algorithm, an EAGLE-family assumption (EAGLE-style drafts
capture verify-tree-shaped batches) fundamentally incompatible with a plain recurrent model doing
ordinary eager decode steps — crashes with `RuntimeError("This should not happen")`. Patching that
shared runner (used by every speculative algorithm in sglang) was ruled out as out of this
project's blast radius.

**Built instead**: `_ensure_draft_decode_graph` / `_build_draft_decode_graph` /
`_draft_decode_one_logits_graphed` in `rwkv_spec_worker.py` — a narrow, self-contained
`torch.cuda.CUDAGraph` capture living entirely in the RWKV-specific worker, touching no shared
sglang infrastructure. Static bs=1/n=1 leaf tensors, captured after a 5-iteration side-stream
warmup (needed to trigger the draft model's lazy one-time setup — e.g. a sparse-cmix tiled-weight
build that only happens on first call), using a dedicated *permanent scratch* mamba-pool slot so
warmup/capture practice runs never touch real request state. Falls back permanently to the
pre-existing eager path on any capture failure, or via `RWKV_SPEC_DRAFT_GRAPH=0`.

**Two real capture-safety bugs found and fixed, both in the category of "cuda graphs are stricter
than eager about what can happen inside the captured region," neither a numerics bug**:
- `out_cache_loc`'s allocation isn't replayable (it's a real allocator call) — moved outside the
  graph, copied into a static buffer before each replay.
- `seq_lens_cpu` can't be derived via a `.cpu()` copy inside the capture region
  (`RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless
  pinned`) — replaced with a fixed dummy CPU tensor, safe because that field is confirmed unread
  on this decode path unless a separate, unused flag (`--enable-linear-replayssm`) is on.

**Measured (clean isolated A/B, same server/prompts, only `RWKV_SPEC_DRAFT_GRAPH` toggled,
7 test prompts spanning accept-length 1.2–4.0)**: draft-step decode goes from eager's baseline to
**median 33.2→53.7 tok/s (1.62×), mean 42.3→63.6 tok/s (1.50×), positive on every single prompt
(1.33×–1.74× range)** — a real, consistent, not-noise improvement to the draft's own decode speed.

**Honest net verdict against spec-off (median 240.7 / mean 222.3 tok/s)**: spec-on-graphed lands
at **0.22×–0.29× of spec-off (3.5×–4.5× slower)** — improved from spec-on-eager's 0.14×–0.19×
(7×–7.3× slower), roughly halving the gap, but **not closing it**. ADR-0006's stated goal
(net speedup, not just a faster draft) is not yet met. This is a real, gated, positive step, not
a finished win — say so plainly rather than rounding up.

## What's next (recorded as hypotheses, not measured — do not treat as findings)

Three candidate remaining bottlenecks, none profiled this session:
1. Per-layer state snapshot/restore `.clone()` cost (paid every round regardless of accept length).
2. Target-verify overhead itself (the extend-path forward over K positions).
3. General per-round Python orchestration (scheduler-level bookkeeping between rounds).

Whoever picks this up next should profile before picking one to attack — don't assume from this
list which one dominates.

## Also not done yet

- 7.2B target end-to-end measurement (ADR-0006's build path step (iv)) — all correctness and
  speed numbers here are 1.5B only.
- A pre-existing, unrelated flaky test (`test_batch_state_isolation` in
  `test/registered/models/test_rwkv7.py`) was observed flaking intermittently on both the old and
  new code (verified via A/B against a byte-identical pre-change file) — not caused by this work,
  not fixed by this work, out of scope here.

## Cross-references

`docs/adr/0006-speculative-decoding.md` (design + this build's summary) · F0029 (α=0.738–0.7485
viability) · F0031 (the original increment-(i) prototype and the first sighting of the M-shape
reduction-order flip) · `bench/spec_gate.py`, `bench/verify_gemv_mb.py` (the two gates this build
depends on) · `memory/project-spec-decode.md` (the full session-by-session build log, more
granular than this summary).
