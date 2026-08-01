# F0080 — the channel-mix sparsity is mostly input-dependent, so it does not batch

**Status:** CLOSED for the question it asked (can the sparse FFN kernel serve M>1) ·
**OPEN** for the one it uncovered (a ~16% always-dead core worth a pruning look)
**Date:** 2026-08-01 · 5090, RWKV-7 1.5B fp16, sglang main

## The question

[F0079](0079-large-batch-is-gemm-bound.md) ended by retracting its own headroom claim and
inverting its conclusion: at large batch the step is already running at hardware speed, so
effort belongs in *removing work* rather than making kernels faster. The sparse channel-mix
is this repo's biggest example of removing work — it skips ~85-90% of the value projection's
weight rows — but it only fires at M==1, because a row can be skipped only if its activation
is zero. At M>1 that means zero for **every** request in the batch.

Whether that is possible depends on something nobody here had measured: are the zeros a
property of the input, or of the channel?

- **input-dependent** → with ~85% zeros per row, the chance a channel is dead across 64
  requests is 0.85⁶⁴ ≈ 3·10⁻⁵. A batched sparse kernel would skip nothing and should not be
  built.
- **channel-intrinsic** → the union stays near the per-row rate at any batch size, the kernel
  batches, and better still those channels could simply be pruned.

## The measurement

Instrumented the model's existing `RWKV_LOG_SPARSITY` logger to report, alongside the per-row
zero fraction, the fraction of channels zero in **every** row of the batch. 64 concurrent
requests, distinct prompts, cuda-graph off, samples taken only after the server is live.

| | per-row zero | union over 64 rows |
|---|---:|---:|
| measured (10 consecutive layer samples) | 0.77 – 0.90 | **0.10 – 0.24** |
| if zeros were independent | — | ≈ 0.00 |
| if zeros were channel-intrinsic | — | ≈ per-row |

**Both hypotheses are wrong, and the answer is nearer the pessimistic one.** A ~16% union is
far above what independence predicts, so a shared always-dead core exists; but it is far
below the 85% each row enjoys alone, so the great majority of the sparsity is genuinely a
property of the input.

## What follows

**Batching the sparse kernel is not worth building.** At bs=64 it could skip ~16% of the
weight rows against the ~85% it skips at bs=1, and the union can only shrink further with
batch. That is a decisive negative result, and it is cheaper to have it as a measurement than
as a half-finished kernel.

**A ~16% always-dead core is worth a look, separately.** If those channels are dead across
inputs generally — not just across these 64 — they are static pruning candidates, which beats
a sparse kernel outright because it costs nothing at runtime. This measurement does not
establish that: the prompts were random token ids rather than natural text, one model, one
batch, short generations. The follow-up is a wider sample over real corpora before anyone
touches a weight.

## Two traps this probe walked into first, recorded because they generalise

- **CUDA-graph capture emits activations from dummy inputs.** The first run reported a 65-79%
  union — a spectacular false positive — because the server OOMed during capture and every
  sample came from synthetic capture batches rather than real tokens. Disabling the graph for
  the probe removed the contamination.
- **`bench/bsz_throughput.py` sends the same token ids to every concurrent request.** The
  second run reported union ≈ per-row to four decimals, which reads as a perfect
  channel-intrinsic result and is really 64 copies of one input. Any batch-diversity question
  measured with that harness will produce this artifact.
