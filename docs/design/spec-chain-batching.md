# Batching the speculative draft chain

**Status:** design note, nothing implemented. Written after F0078 established that
`RWKV_SPEC` is single-request by construction and now says so instead of corrupting memory.

## The one thing to settle before writing any code

At bs=1 the target decodes one token per forward and speculation buys back that latency.
At bs=8 the target is already decoding eight tokens per forward, and its cost per token has
fallen accordingly, while the draft chain's cost has not. Tonight's measurement is the
warning: repairing plain decode from 122.9 to 142.2 tok/s moved the speculative median from
1.29x to 1.23x and pushed a second prompt below 1.0x, with **identical accept lengths**.
Batch size is a much larger lever on the baseline than that repair was.

So the first task is not implementation, it is an economics check:

> Measure plain decode against a *hypothetical* perfectly-batched speculative path at
> bs = 2, 4, 8 before building it. The accept lengths are already known and do not depend
> on batch size (they are a property of the draft/target pair — measured 2.51 / 2.72 / 3.51
> / 4.40 / 4.41 across every configuration tried, sparse on or off, fp16 state or fp32).
> Combine them with measured per-forward costs at each bs to predict the ratio. If the
> predicted ratio is below ~1.1x at bs=4, the work below is not worth doing and the honest
> product statement is "speculation is a single-stream latency feature".

### The measurement, and how far it gets

Cost-vs-batch, 7.2B target and 0.1B draft, sparse off, same card and session. A decode
forward at concurrency C processes C rows, so this *is* the cost-vs-row-count curve, and a
verify forward processes `bs*K` rows — which is why the target was swept past 32.

| rows | target fwd | target agg | draft fwd |
|---:|---:|---:|---:|
| 1 | 9.24 ms | 108.2 tok/s | 0.50 ms |
| 2 | 10.24 | 195.3 | 0.92 |
| 4 | 10.71 | 373.5 | 0.91 |
| 8 | 10.94 | 731.1 | 1.08 |
| 16 | 11.54 | 1387.0 | — |
| 24 | 11.95 | 2009.1 | — |
| 32 | 12.37 | 2586.4 | — |
| 48 | 13.49 | 3558.0 | — |

The shape that matters: **the target's forward is nearly flat in row count** — 46% more time
for 48× the rows. It is bandwidth-bound on weights, so a verify that checks `bs*K` rows costs
barely more than one checking `K`. That is the structural reason speculation suits this model
family, and it does not weaken with batch.

Modelling a round as `(K-1)·draft_fwd(bs) + target_fwd(bs·K) + overhead`, and solving for
`overhead` from the measured bs=1 point (175.1 tok/s at accept 3.51, K=6) gives ~6.7 ms:

| bs | plain (measured) | spec (predicted) | ratio |
|---:|---:|---:|---:|
| 1 | 108.2 | 175.1 | 1.62 (calibration point) |
| 2 | 195.3 | ~311 | ~1.59 |
| 4 | 373.5 | ~605 | ~1.62 |
| 8 | 731.1 | ~1097 | ~1.50 |

**The whole table hangs on one unknown: whether that 6.7 ms/round is fixed or per-request.**
Fixed, the ratio holds near 1.5x. Linear in bs, it becomes 53.6 ms at bs=8, the round costs
72.5 ms, and speculation lands at ~387 against plain's 731 — a 0.53x, i.e. a large loss. The
two answers differ by a factor of three, so the number is not usable until this is settled.

Structurally most of the round is the two forwards, both of which scale well, which argues
for the optimistic branch — but that is an argument, not a measurement. A `RWKV_SPEC_TIMING=1`
run was attempted and is **not** usable: it reported 95 ms/round with the draft phase at 70%,
which cannot be right when five 0.1B forwards measure 2.5 ms, because the instrumentation
syncs per phase and the 50-round window includes Triton first-compiles and graph warmup. The
redo needs a steady-state window (discard the first hundred rounds) and ideally a
sync-free accounting.

Everything after this section assumes that check came back positive.

## What is already batched, and must not be rebuilt

The expensive half of the problem — committing the right recurrent state after a partial
accept, per request — is upstream machinery that already handles bs>1:

- `spec_utils.commit_mamba_states_after_verify` computes `last_correct_step_indices` for the
  whole batch (`accept_index[req_idx, accept_lens - 1] - accept_indices_offset`) and calls
  the backend hook with batched indices. Our worker already calls it.
- `update_mamba_state_after_mtp_verify` comes from the `hybrid_linear_attn_backend` base
  class, so the scatter of the chosen per-step state back into the persistent pool is
  batched for free.
- The overlay's own TARGET_VERIFY path is already bs-general: it derives `bs` from the batch
  and `K = x.shape[0] // bs`, reshapes to `x.view(bs, K, -1)`, and writes
  `interm_conv[req_pos, :K]`. This is the piece F0077 built, and it needs no change.
- `NgramVerifyInput`, `eagle_sample` and the accept bookkeeping are batch-shaped upstream.

## What is bs=1, and what each becomes

1. **Draft state snapshot / restore.** `_snapshot`/`_restore` operate on one pool slot, and
   the fast path's stacked buffers are `(K, L) + state_shape` indexed by chain step only,
   against a single `d_mslot`. These become `(bs, K, L) + state_shape` with a slot *vector*,
   and the restore becomes a gather by `(request, accepted_step)` — which is the same shape
   of operation `commit_mamba_states_after_verify` already does target-side, so the index
   math can be borrowed rather than invented.

2. **The draft's hand-rolled DECODE CUDA graph.** Captured for n=1 today. Upstream's own
   decode graph runner solves this with bucket capture — this server's config already lists
   decode buckets `[1,2,3,4,5,6,7,8,10,12,...,32]` — so the pattern to copy is: capture a
   small bucket set, pad the batch up to the next bucket, and fall back to eager above the
   largest. Do not capture per exact bs.

3. **The verify graph.** Keyed by `K` today (`self._verify_graphs[K]`). It becomes keyed by
   `(bs_bucket, K)`, and the product of the two bucket sets is the capture cost — another
   reason to keep the bucket set small and to reuse a shared graph memory pool, as the
   current code already does.

4. **Per-round bookkeeping in `_verify_round`.** The adaptive-K accept-EMA is already keyed
   per request id; the chain arrays (`d_toks`, candidates, accept lengths) become `[bs, K]`.
   The guard added in F0078 is the marker for where this work lands: when it can handle a
   multi-request batch, that `NotImplementedError` is what gets deleted.

## Verification the change must pass

- The 10-prompt identity gate **at bs>1**. The harness drives sequentially today, so it needs
  a concurrent mode first; a batched chain that is token-identical at bs=1 proves nothing
  about bs=4.
- The 240-request 8-way soak that currently trips the guard, ending clean.
- Accept-length parity against bs=1 on the same prompts. A batched chain that quietly
  accepts less is a regression the speed number alone would hide.
- `RWKV_SPARSE_FFN=0` throughout, per F0078 — sparse is bsz1-decode-only, so it is doubly
  irrelevant here, and leaving it on breaks the identity comparison.
