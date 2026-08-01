# F0080 — the channel-mix sparsity is mostly input-dependent, so it does not batch

**Status:** CLOSED, both questions — the kernel cannot batch, and the always-dead core it
seemed to uncover shrinks under a better sample rather than holding up
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

**The ~16% always-dead core does not survive a better sample.** It looked like a static
pruning candidate — dead channels cost nothing to remove and beat any sparse kernel — so the
caveat attached to it (random token ids drive the model off-distribution) was worth spending
a run on. Repeated with 64 pieces of genuinely different natural text (English and Chinese
prose, code, SQL, dialogue, clinical notes, German, a legal clause):

| prompts | per-row zero | union over 64 |
|---|---:|---:|
| random token ids | 0.77 – 0.90 | 0.10 – 0.24 |
| **real text** | 0.78 – 0.89 | **0.065 – 0.158** |

The per-row rate is unchanged; the union *falls*. A genuine always-dead set would not care
what the inputs are — it would hold or grow as the sample gets more representative. Shrinking
under a better sample is the signature of the other thing: no dead set, just a tail of rarely
active channels, and the tail thins as you look at more inputs. With 64 prompts it reads 12%,
and there is no reason to expect anything left at 640.

So the pruning lead closes too. This is worth one more sentence than the result deserves,
because the shape of the mistake recurs: the encouraging number came from the *less*
representative sample, and the instinct on seeing it was to plan the follow-up work rather
than to attack the sample first.

## Two traps this probe walked into first, recorded because they generalise

- **CUDA-graph capture emits activations from dummy inputs.** The first run reported a 65-79%
  union — a spectacular false positive — because the server OOMed during capture and every
  sample came from synthetic capture batches rather than real tokens. Disabling the graph for
  the probe removed the contamination.
- **`bench/bsz_throughput.py` sends the same token ids to every concurrent request.** The
  second run reported union ≈ per-row to four decimals, which reads as a perfect
  channel-intrinsic result and is really 64 copies of one input. Any batch-diversity question
  measured with that harness will produce this artifact.
