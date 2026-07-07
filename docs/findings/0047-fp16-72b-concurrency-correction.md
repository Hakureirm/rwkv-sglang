# F0047 — RWKV-7 7.2B fp16 full-stack peak was undertested: true peak is 6,709 tok/s @ c320, not 5,983 @ c192

**Date:** 2026-07-07 · **Status:** MEASURED (RTX 5090, re-swept twice at independent
`--cuda-graph-max-bs` settings, agreeing within 0.5%) · **Supersedes:** the fp16 side of
[F0035](0035-7b-int8-concurrency-headroom.md) · **Trigger:** a bf16 stock-path sweep run in a
separate benchmark project (`memory/project-qwen35-benchmark.md`, round 4) found RWKV-7 7.2B
bf16 peaking at 6,171 tok/s @ c256 — higher than the published fp16-hand-kernel headline of
5,983 @ c192, which should not happen (fp16 + hand kernels should not lose to bf16 stock).

## The question

Was the published fp16 full-stack peak (5,983 tok/s @ c192, max concurrency "221, OOMs
above") ever actually the true peak, or did the original sweep grid (1, 32, 64, 128, 192,
221 — see `bench/results/72b/sweep_72b_fp16.json`) simply stop before finding it? And
separately: does fp16's hand-written kernel stack use meaningfully more memory than bf16's
stock path, in a way that would genuinely cap fp16 lower than bf16 (as opposed to the
221-cap being a plain methodology gap)?

## Method

Deployed RWKV-7 7.2B fp16 with the full hand-kernel stack (`RWKV_FAST_LINEAR=1
RWKV_SPARSE_FFN=1 RWKV_FUSED_LORA=1 RWKV_FUSED_GLUE=1 RWKV_GEMV_AUTOTUNE=1`, matching
`scripts/serve.sh`'s throughput mode) in a fresh isolated container on the RTX 5090 tower,
`--mem-fraction-static 0.85` (the `scripts/serve.sh` default — not F0035's original 0.93; see
"why 0.85 not 0.93" below). Confirmed all five hand kernels actually armed in the boot log
(M6 fused GEMV, R2 fused glue ×2, M9 fused LoRA, M6 sparse channel-mix) — full-stack was
genuinely active, not silently gated off.

Bisected `--cuda-graph-max-bs` upward from the previously-published-safe range, checking real
GPU memory at each step (not just trusting throughput numbers), and treating a clean boot as
necessary but not sufficient — every candidate ceiling was verified under an actual
concurrency sweep, not just a boot log, per the lesson from a sibling investigation
(Qwen3.5-9B bf16 in the same project: a `--mem-fraction-static 0.92` config booted clean but
OOM'd on the first real burst of concurrent prefills; a separate config produced a
KV-starved false plateau at 3,186.6 tok/s that looked like a peak but was actually
request-queuing). Both failure modes were checked for and ruled out here:

| `--cuda-graph-max-bs` | boots? | behavior under full sweep | verdict |
|---|---|---|---|
| 384 | boots | OOM during decode-graph capture itself (short by 128 MiB of 119 MiB free — i.e. essentially a coin-flip away from fitting) | reject: doesn't even reach serving |
| 368 | boots, 390 MB free after capture | **OOMs at c=32** (`wkv_recurrent`, eager prefill path — allocator fragmentation, "17.12 MiB free" for a 32 MiB alloc) | reject: clean boot, unstable under real load |
| 344 | boots, 1.17 GB free after capture | full sweep to c=344 completes, no crash, 100% GPU util throughout | **accept — safe ceiling** |
| 320 | boots, 1.99 GB free after capture | full sweep to c=320 completes cleanly, largest headroom of the safe points | **accept — used as primary sweep** |

Two independent full sweeps were run (`--cuda-graph-max-bs` 320 and 344) over the union grid
{1, 8, 32, 64, 128, 192, 221, 256, 320, 336, 344}; every point present in both runs agrees to
within 0.5%, so run-to-run noise is not driving the result.

## Result

| c | 1 | 8 | 32 | 64 | 128 | 192 | 221 | 256 | **320** | 336 | 344 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| out tok/s (run @ cgmax=320) | 123.8 | 667.9 | 2361.4 | 4039.7 | 5694.8 | 5999.0 | 5769.4 | 6186.2 | **6714.2** | — | — |
| out tok/s (run @ cgmax=344) | 123.7 | 666.4 | 2363.2 | 4033.9 | 5688.1 | 6034.1 | 5771.6 | 6205.3 | **6709.0** | 6039.2 | 6111.2 |

**True peak: 6,709–6,714 tok/s @ c=320**, confirmed bracketed by decline at c=336 (6,039.2)
and c=344 (6,111.2) — both below the c=320 value, with the server staying healthy (no crash,
100% GPU utilization, HTTP 200s throughout) all the way to c=344. This is **+12.1% above**
the previously published 5,983.3 @ c192, and the concurrency ceiling is at least **344**, not
221 — a 55.7% upward correction to the "max concurrency" figure.

The original grid's own data already contained the tell: c=192 (5,983) → c=221 (5,747) is a
*decline*, and the original sweep stopped there, reading the decline as "hit the wall."
Re-tested here at 221 (5,769–5,772, matching within noise), the curve in fact *recovers* by
c=256 (6,186–6,205) and keeps climbing to its real peak at c=320 — the 221 dip was a local
wiggle, not the ceiling. This is the same shape bf16 showed in the sibling investigation
(dip near 221, recovers and peaks at 256) — fp16 simply peaks a bit further out (320 vs 256)
and a bit higher (6,709 vs 6,171).

## Why 0.85 mem-fraction, not F0035's original 0.93

F0035 used `--mem-fraction-static 0.93` specifically to maximize fp16's static (weight +
state-pool) budget, and still reported capping at 221. This re-test used 0.85 (the
`scripts/serve.sh` default, and the setting used for every bf16/Qwen3.5 comparison sweep in
the sibling project, for direct comparability) and reaches a *higher* concurrency (≥344) and
peak than the 0.93 run ever reported. This is not a contradiction: 0.93 pushes more memory
into the static pool (larger state-pool budget) but leaves less "dynamic" headroom for the
eager (non-cuda-graph) prefill path's transient activations — exactly the kind of margin that
368's crash-under-load (at 0.85, even) shows is already thin. It is plausible 0.93 would have
hit the *same* real ceiling somewhere past 221 had it been swept further; this was not
re-tested at 0.93 since 0.85 alone is sufficient to falsify "221 is a hard cap," and 0.85 is
the more standard, already-cross-comparable setting. Flagged as an open, low-priority
follow-up, not required to resolve the headline question.

## Does fp16's hand-kernel stack use more memory than bf16 stock? Yes — checked, not assumed

The original F0035 write-up asserted per-request state is "identical for both [fp16 and
w8a8] — it is fp32 model state, independent of weight quantization," implying fp16 and bf16
should have identical memory footprints at matched settings (same weights bytes, same
state-pool math, hand kernels aside). Checked directly: at the identical launch
(`--cuda-graph-max-bs 320`, `--mem-fraction-static 0.85`, 7.2B, same GPU), **fp16 full-stack
leaves ~2.0 GB free after decode-graph capture; bf16 stock (hand kernels self-gate off for
non-fp16 dtypes, confirmed in the boot log) leaves ~6.4 GB** at the same nominal
configuration (`rwkv7_7.2b_bf16_sweep_5090_v2.json`, prior session). That is a real, ~4.4 GB
difference attributable to the hand-kernel stack's own workspace (GEMV autotune launch
buffers, fused-LoRA and fused-glue scratch, the sparse channel-mix tiled-weight cache) — the
hand kernels are not memory-free. This is why fp16's *safe concurrency ceiling* (≥344) is
lower than what bf16 demonstrated it could sustain (bf16 ran a clean full sweep through
c=320 with margin to spare, per the sibling project). But it does not mean fp16 loses
overall: fp16's corrected throughput peak (6,709 @ c320) is still **8.7% above** bf16's own
measured peak (6,171 @ c256) — the hand kernels buy more raw compute per step than the extra
memory pressure costs in reachable concurrency. The original "suspicious" finding (bf16
beating the published fp16 number) is fully resolved: bf16 was never actually ahead of fp16
fullstack, fp16 fullstack's own published number was just stale.

## Corrected headline numbers (supersedes F0035's fp16 column; w8a8 column unchanged)

| 7.2B on one 5090 | max concurrency | peak output throughput |
|---|---|---|
| fp16 (corrected) | **≥344** (was 221) | **6,709 tok/s @ c320** (was 5,983 @ c192) |
| w8a8 (unchanged) | 640 | 7,587 tok/s @ c640 |
| **ratio (corrected)** | **1.86×** (was 2.90×) | **+13.1%** (was +26.8%) |

int8 is still a genuine, real win on this axis — it is just roughly **half** as dramatic a
win as previously published. The w8a8 measurements themselves were not re-run and are not in
question; only the fp16 comparison point was wrong.

## Downstream documents corrected

`docs/BENCHMARKS.md` §4, `docs/BENCHMARKS.zh-CN.md` §4, `README.md`, `README.zh-CN.md`,
`CONTRIBUTIONS.md` §1 req#5 all cited the 5,983/221/2.90×/26.8% figures and have been updated
to the corrected numbers with a pointer to this finding. `docs/findings/0035` is kept
verbatim as the process record with a superseded-banner at the top pointing here.

## Cross-references

`bench/results/72b/sweep_72b_fp16_v2_5090.json` (cgmax=320 run, primary),
`bench/results/72b/sweep_72b_fp16_v3_5090.json` (cgmax=344 run, cross-check + extended range)
· original `bench/results/72b/sweep_72b_fp16.json` (undertested grid, kept for lineage) ·
[F0035](0035-7b-int8-concurrency-headroom.md) (w8a8 side, unaffected) · bf16 sweep data from
the sibling `memory/project-qwen35-benchmark.md` round 4 (not in this repo — cited by number
only: 6,171.3 @ c256, 5,625.7 @ c320 confirmed declining).
