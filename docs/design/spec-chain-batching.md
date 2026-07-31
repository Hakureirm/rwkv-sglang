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

### Settled: the phase that scales is a Python loop over the lm_head

`RWKV_SPEC_TIMING=1` reports a **cumulative** average, so a long generation gives two windows
whose difference is the steady state. Over 50 rounds it read 95.74 ms/round (Triton
first-compiles and graph warmup dominating); over 100, 57.75. Differencing —
`2·x(100) − x(50)` — gives rounds 51-100:

| phase | steady ms/round | share | how it scales with bs |
|---|---:|---:|---|
| target_fwd | 13.91 | 70% | ~flat (48× the rows costs 46% more) |
| draft | 3.11 | 16% | sublinear (0.50 → 1.08 ms for bs 1 → 8) |
| **head_recompute** | **1.90** | **10%** | **linear in bs·K** |
| sample_commit | 0.26 | 1% | vectorized upstream, ~flat |
| rollback_glue / gap / prep | 0.56 | 3% | ~fixed |
| **total** | **19.74** | | |

That total is worth pausing on: an independent estimate from the measured throughput
(175.1 tok/s ÷ accept 3.51) gives 20.05 ms/round. Two unrelated routes agreeing to 1.5% is
what makes the split trustworthy — and it also retires the "95 ms/round, draft at 70%" reading,
which was warmup, not signal. The draft's steady 3.11 ms across five 0.1B steps matches its
independently measured 0.50 ms forward, which is the second consistency check.

**`head_recompute` is the answer to the open question.** It is this, in `_verify_round`:

```python
logits_output.next_token_logits = torch.cat(
    [torch.matmul(h[i:i+1], w.t()).float()[:, :vocab] for i in range(h.shape[0])], dim=0)
```

A Python loop over every one of the `bs·K` verify rows, each doing a separate `[1,H]@[H,V]`
against the full lm_head — 4096×65536 fp16, 536 MB, **re-read once per row**. Six rows is
3.2 GB of weight traffic, which at this card's bandwidth is ~1.9 ms: exactly what the timer
measures. It exists for a correctness reason (a batched `[M,H]@[H,V]` has a different cuBLAS
reduction order than the M=1 projection the baseline decode uses, and that flips near-ties —
the F0031 class), and it is the one cost in the round that grows linearly with batch.

Carrying each phase to bs=8, K=6 with its own scaling:

| | bs=1 | bs=8, loop kept | bs=8, loop replaced |
|---|---:|---:|---:|
| target_fwd | 13.91 | ~18 | ~18 |
| draft | 3.11 | ~6.7 | ~6.7 |
| head_recompute | 1.90 | **~15.2** | ~0.5 |
| rest | 0.82 | ~1.1 | ~1.1 |
| round | 19.74 | ~41 | ~26.3 |
| spec tok/s | 175 | ~685 | ~1068 |
| plain tok/s (measured) | 108.2 | 731.1 | 731.1 |
| **ratio** | **1.62x** | **0.94x** | **1.46x** |

So the batching decision is not really about batching. **With that loop in place, speculation
at bs=8 is a net loss; without it, it is 1.46x.** Everything in the section below is
conditional on replacing it first.

### Replacing it is not a one-liner, and here is the trap

`gemv_mb` is the obvious candidate — it is exactly a shared-weight M-row GEMV that keeps one
weight pass and is row-for-row bit-identical (bench/verify_gemv_mb.py). But its bit-identity
contract is against **our** `gemv_m1`, while this loop's contract is against **cuBLAS at
M=1**, which is what the plain decode's lm_head actually runs. Swapping it in would make
verify agree with a baseline we do not have, and the identity gate would fail — the same trap
in the opposite direction from the sparse finding in F0078.

The honest options, in order of preference:

1. Route the lm_head through the same batch-invariant kernel on **both** paths — plain decode
   and verify. Then one weight pass serves all rows and identity is structural rather than
   maintained by brute force. This changes plain decode's numerics (cuBLAS → our GEMV), so it
   needs the numeric-oracle gate, not just the spec gate. It is also worth ~9% at bs=1 on its
   own, independent of batching.
2. Chunk the loop to `min(8, rows)` per `gemv_mb` call *if* option 1 lands, since the
   shared-weight variant is capped at M≤8 (above it the kernel still burns M weight passes,
   just in one launch).
3. Leave it and cap speculation at bs=1, which is the status quo the F0078 guard enforces.

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
