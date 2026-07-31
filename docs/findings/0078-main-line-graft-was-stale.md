# F0078 — the main-line graft was stale, and the profile said so

**Status:** OPEN (numbers below are measured; the last leg is re-measuring)
**Date:** 2026-07-31 · 5090 (sm120), 7.2B fp16, `lmsysorg/sglang:dev-cu13` + fork worktree
**Supersedes:** the postscript in [F0077](0077-spec-decode-draft-desync-fix.md), whose
attribution of the main-vs-0.5.10 gap to "main-branch per-step runtime overhead" was wrong.

## What this is

The consolidation onto sglang main was reported done. It was not: the grafted model file
was an older snapshot of the delivery overlay, and the backend shipped two fallback stubs
where the fused glue should be. Both were invisible to every gate we had — the greedy
fixture stayed 8/8 EXACT throughout, because nothing here changes numerics. Only a
flag-ladder A/B and then a kernel trace could see it.

## The tell: a ladder that returned nothing

`bench/mega_flag_matrix.sh` runs four legs (A anchor → D flagship) at c=1, 64-in/256-out.
On 0.5.10 that ladder is the F0066c flagship: **133.4 → 142.8 tok/s (+7%)**. On main, at
identical flags, model dir and harness:

| leg | flags | main, before | +model/config graft | +fused glue | 0.5.10 ref |
|---|---|---:|---:|---:|---:|
| A | anchor (MEGA/WKV_CUDA/PDL off) | 119.2 | 122.6 | **139.8** | ~133.4 |
| B | +MEGA | 119.2 | — | — | — |
| C | +MEGA+WKV_CUDA | 118.9 | — | — | — |
| D | +MEGA+WKV_CUDA+PDL | 118.6 | 122.9 | **141.7** | 142.8 |

Four legs within 0.5% of each other is not a result, it is a dead code path. A ladder that
returns nothing is evidence about the ladder, not about the hardware. With both repairs in,
main lands at 141.7 against the 0.5.10 flagship's 142.8 — 0.8% apart, inside the band this
project's cross-session anchors normally sit in — and main's *anchor* leg now beats 0.5.10's
anchor (139.8 vs ~133.4), so nothing about the newer runtime was ever the problem.

## Two things had not crossed over

1. **The model file was a stale snapshot** — 1385 lines against the delivery overlay's
   1620, with zero occurrences of `_MEGA`. The whole #50 megakernel line was absent, and so
   were `_FUSED_GATES`, `_FUSED_SQRELU`, `_FUSED_ADDLN_SHIFT` and `_FUSED_LORA_GATED`.
2. **The `RWKV_STATE_FP16` knob lives in `configs/rwkv7.py`**, and the fork carried the
   lean upstream-port version of that file, which has no such knob. The flag was set on
   every leg and silently did nothing — the `[rwkv7_wkv]` banner said `state=float32` while
   the environment said `RWKV_STATE_FP16=1`.

The two files had diverged in BOTH directions, which is why a wholesale copy either way
would have lost work: the fork's backend carries the spec-decode work (F0077), and the
fork's model file carried the `is_target_verify` batch-invariance plumbing that the
delivery overlay never got. Fixing this meant merging, not copying — the overlay now has
the verify plumbing (threaded through `_proj_gemv` and `_proj_gemv_sqrelu`, six call
sites), and the fork got the overlay's model file plus a surgical config knob.

After the merge every stage announces itself: `#50 Stage-A grouped r/k/v GEMV ENABLED`,
`[rwkv7_pdl] PDL chain ARMED`, `[rwkv7_wkv] hand-CUDA WKV decode ACTIVE (state=float16)`.
The realistic-prompt baseline went **115.1 → 122.9 tok/s**.

## The ladder still returned nothing — so profile it

Every stage armed, and D still equalled A (+0.2%). The hypothesis was that main's scheduler
had moved the bottleneck to the host, where GPU-side wins cannot show. **The trace refuted
it:**

| | 0.5.10 flagship (F0066c) | main, after graft |
|---|---:|---:|
| SPAN/step | 7052.1 us → 141.80 tok/s | 8118.4 us → 123.18 tok/s |
| BUSY/step | 7167.7 us | 8100.8 us |
| gap/step | −115.7 us | +17.7 us |
| kernels/step | 469.1 | 968.6 |
| overlapped transitions | 96.5% | 33.0% |

The GPU is busy 8100 of 8118 us. There is no host-side hole to find; serving (122.9) and
the kernel-loop framing (123.18) agree to 0.2%. The step simply issues **twice the
kernels**, and the per-kernel table says exactly which ones:

| kernel | 0.5.10 /step | main /step | main us/step |
|---|---:|---:|---:|
| `shift_lerp6_kernel<256>` | 32.26 | **0** | — |
| `shift_lerp1_kernel<256>` | 32.26 | **0** | — |
| `index_elementwise` (gather) | 1.02 | **64.36** | 347.2 |
| `index_elementwise` (index_put) | 2.05 | **64.37** | 280.6 |
| `direct_copy` (two variants) | ~1 | **191.3** | 253.0 |

~880 us/step of torch gather/scatter, against a 933 us/step busy-time deficit. The fused
paged token-shift was not running: `try_fused_shift_lerp6/1` in the main-line backend were
unconditional `return None` stubs, added earlier to stop an `AttributeError` at a call site
whose contract treats `None` as "fall back". The fallback is correct, which is why no gate
complained, and slow, which nothing measured until now. The collapse of overlapped
transitions (96.5% → 33.0%) follows: the PDL chain is built around those glue kernels, and
torch fallbacks interleaved between the chain's links break it into fragments.

## Also found, because speculative decoding boots a second model

`wkv_recurrent`'s eligibility block mirrored the kernel's requirements for T/K/V/dtype but
not for the head count, while the kernel `TORCH_CHECK`s H to be a power of two. Every
single-model shape we had ever benchmarked is 32 or 64 heads. The 0.1B draft is 12
(768/64), it inherits the target's armed stack, and the server died on the first draft step.
Every other fast path in this overlay self-gates and falls back; this one raised. Fixed by
mirroring the check at the dispatch site.

## Speculative decoding, re-measured at the corrected baseline

**Superseded — read the caveat.** The table below was taken against the 122.9 baseline,
i.e. after the model/config graft but BEFORE the fused glue was restored. The plain leg has
since moved to ~141, so every ratio here is optimistic by roughly that factor and is being
re-measured at the final configuration. It is kept because the accept lengths and the
shape of the workload dependence are unaffected by the baseline.

K=6, greedy fixture 8/8 EXACT, long-form prompts, against the 122.9 baseline:

| prompt | tok/s | accept | vs plain |
|---|---:|---:|---:|
| story | 114.5 | 2.51 | **0.93x** |
| explain | 125.3 | 2.75 | 1.02x |
| history | 158.9 | 3.51 | 1.29x |
| math | 189.6 | 4.40 | 1.54x |
| code | 199.2 | 4.41 | 1.62x |

Median **1.29x**. The story prompt is a net loss and is reported as one. The same server on
`bench/bsz_throughput.py`'s synthetic shape reads 225.6 vs 122.8 (1.84x) — do not quote
that as a workload number: random `input_ids` with `ignore_eos` decode into degenerate
repetition, which a draft model predicts almost perfectly.

## Corrected elsewhere

BENCHMARKS called the 142.8 flagship an `sgl.Engine` measurement in four places. The
artifact (`bench/results/f0066c/c1_72b_F1.json`) records host 127.0.0.1 port 30070: it is
the HTTP server via `bench/bsz_throughput.py` at c=1. The substance — real request path,
scheduler included — was right; the API name was wrong. Relabelled in both languages.

## Why the ladder was flat, restated

Once the glue is back the ladder works again (A 139.8 → D 141.7). It was never that the
megakernel stages did nothing: they were shaving GPU time off a step whose cost was
dominated by ~880 us of torch gather/scatter that no flag in the ladder touches, and whose
interleaving also broke the PDL chain the D leg is supposed to close. Fix the glue and both
recover together. The lesson for reading an A/B: a lever that measures zero has two
explanations — the lever is worthless, or something upstream of it is spending the budget —
and only a profile distinguishes them.

## What this cost us to learn

Three gates existed and all three passed while the line ran 14% slow: greedy fixture
8/8 EXACT, the numeric oracle, and the spec token-identity gate. None of them is a
performance gate. The flag-ladder A/B is the performance gate, and it was not re-run after
the graft — the graft was verified by "does it boot and answer correctly", which it did.
The banners are an activity gate for flags, but a stub that returns `None` announces
nothing, so the banner set looked complete while two of its members were missing.
