# F0078 — the main-line graft was stale, and the profile said so

**Status:** CLOSED — repaired and re-measured end to end (7.2B bsz1 118.6 → 141.7)
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

### First, the identity gate fails — and the bisect says which flag

The per-leg smoke gate is 8 tokens. The real gate is `bench/spec_gate.py`: 10 prompts,
128 tokens, spec output must be byte-identical to the plain server's. At the repaired
configuration it came back **9/10** — one prompt (`def fibonacci(n):`) diverging at a
single position, everything after it a different-but-plausible continuation, accept length
a healthy 3.38. That is the F0031 near-tie signature, not a desync (a desync shows up as
accept ≈ 1.2, not 3.4).

Two hypotheses died to their own measurement before the third stuck:

| run | config | verdict |
|---|---|---|
| fp16 state | flagship, `RWKV_STATE_FP16=1` | FAIL at pos 126 |
| fp32 state | same, state fp32 | FAIL at pos 90 — **fp16 state exonerated** |
| anchor | `MEGA=WKV_CUDA=PDL=0`, fp32 | FAIL at pos **90, identical** — the whole megakernel line exonerated, and positive evidence its three kernels really are bit-identical |
| no sparse | `RWKV_SPARSE_FFN=0` | **PASS 10/10**, accept 3.39 (vs 3.38) |

The sparse channel-mix SpMV skips zero rows, so its fp32 summation *order* differs from the
dense projection — mathematically equal, not bitwise. A plain server runs it on every decode
step; a speculating server's target never decodes at all, it only verifies, and verify is
M>1, which the sparse kernel does not serve. So the two servers disagree by ~1 ULP and a
near-tie argmax eventually flips.

### The decision this forces

| config | plain | spec (K=6) | ratio | gate |
|---|---:|---:|---:|---|
| sparse ON | **142.2** | 178.2 | 1.25x | **FAIL 9/10** |
| sparse OFF | 109.1 | 175.1 | 1.61x | **PASS 10/10** |

Read across, not down. Sparse is worth +30% on plain decode and **nothing** on the
speculative path (178.2 vs 175.1) — under speculation the only sparse work left is the
0.1B draft's own decode. So the choice a user actually faces is *best correct spec* against
*best plain*: **175.1 vs 142.2 = 1.23x**, with the identity gate passing. On the gate's
short prompts the same comparison is 130.9 vs 107.0 = 1.22x; with sparse on there it is
131.5 vs 139.0, i.e. **speculation is a net loss AND fails the gate** — that combination is
strictly the worst of the three and is the one our own `serve.sh` defaults would have picked.

We warn rather than auto-disable. Turning sparse off inside the speculating process would
not restore the guarantee — the mismatch is against a *separate* plain server we do not
control — and it would cost ~1.8% by slowing the draft. The warning fires once, on the
target's first FFN forward, and is gated: verified present on a spec server and absent on a
plain one with identical flags. The guarantee, stated precisely, is against a plain server
in the **same kernel configuration**; to verify it, run both with `RWKV_SPARSE_FFN=0`.

### Speed, at the gate-passing configuration

K=6, greedy fixture 8/8 EXACT, long-form prompts, plain median **142.2** (sparse on; the
spec column below is sparse-off, i.e. the configuration that passes the identity gate):

| prompt | plain (sparse on) | spec (sparse off) | accept | ratio |
|---|---:|---:|---:|---:|
| story | 142.2 | 126.4 | 2.51 | **0.89x** |
| explain | 142.2 | 138.4 | 2.75 | **0.97x** |
| history | 142.5 | 175.1 | 3.51 | 1.23x |
| math | 138.6 | 208.4 | 4.40 | 1.50x |
| code | 142.3 | 219.0 | 4.41 | 1.54x |

Median **1.23x**, and two of five prompts are net losses. Measured against the 122.9
baseline earlier in the same session the ratios read 1.29x median with one loss — **the
accept lengths are identical to two decimals across all of it** (2.51/2.72/3.51/4.40/4.41,
in every configuration tried tonight, sparse on or off, fp16 state or fp32). Nothing about
the draft ever changed; the target got cheaper per token while the draft chain cost did not,
so the same acceptance buys less. Speculation is a ratio against a baseline, and improving
the baseline is a way to lose it: tonight's decode-side repair moved the median from 1.29x
to 1.23x, and the low-acceptance prompts fall through 1.0x first.

The same server on `bench/bsz_throughput.py`'s synthetic shape reads 249.0 vs 141.9 (1.75x).
Do not quote that as a workload number: random `input_ids` with `ignore_eos` decode into
degenerate repetition, which a draft model predicts almost perfectly.

## Size dependence: speculation is a large-target feature, and 1.5B is past the edge

Same protocol, same stack, same 0.1B draft, K=6, sparse off, measured 2026-08-01:

| target | plain median | spec median | ratio |
|---|---:|---:|---:|
| 7.2B | 142.2 (sparse on, its best) | 193.5 | **1.36x** |
| 1.5B | 435.0 | 285.7 | **0.66x** |

Every 1.5B prompt is a loss, and not because the draft agrees less — accept lengths come
back 2.81-3.61, the same band as the 7.2B's. The target is simply too cheap: a 1.5B plain
step is 2.30 ms, so a round that spends the draft chain plus a verify to buy ~3 tokens cannot
win. The ratio the draft has to overcome scales with how much bigger the target is, and
0.1B against 1.5B is 6.7% where 0.1B against 7.2B is 1.4%.

F0077 recorded this as 0.87x. It is worse now (0.66x) for the reason that keeps recurring in
this finding: the plain baseline got faster while the draft chain did not. Against the 1.5B's
*best* configuration (sparse on, 516 tok/s) it would read 0.55x.

So the rule to ship with the feature is a size rule, not a tuning knob: speculation is for
the large targets. Somewhere between 1.5B and 7.2B it crosses 1.0x, and nothing measured here
locates that crossing.

## Speculation has never run under concurrency, and does not

Every speculative number in this project — tonight's and F0077's — is bsz1 and sequential.
Pointing 8 concurrent clients at the same server kills it on the **first prefill**, before
any model code runs:

```
RuntimeError: The size of tensor a (19) must match the size of tensor b (109)
  eager_runner._execute_extend -> load_batch
  -> cuda_graph_buffer_registry.fill_from -> _grouped_foreach_copy_
```

`fill_from` slices each registered destination to the batch (`slot.buffer[:raw_n]`, raw_n =
19 tokens here) and copies the ForwardBatch field into it; the source is 109 rows, which is
the request-pool size at this memory fraction, not anything batch-shaped. So a slot whose
source is pool-sized is being filled as if it were per-batch.

Three runs pin it down. Adaptive K is **not** the cause — fixed K=8 dies identically (a=19
vs the adaptive run's a=16, same b=109) — and neither is anything repaired tonight: restoring
the pre-repair model file, config and backend from backup reproduces the identical error.
It was invisible because no benchmark or gate in this project has ever sent overlapping
requests to a speculative server.

**Cause, at source level.** Patching the registry to print the slot name instead of two bare
numbers took one run: `slot=positions dst=(6,) src=(105,) raw_bs=1 raw_tokens=6`, and the
source's first six entries were `[0,1,2,3,4,5]` with an uninitialized tail. `_build_forward`
derives the draft's batch with `dataclasses.replace` and overrides `extend_lens` and
`prefix_lens` but not `extend_num_tokens`; `ForwardBatch.init_new` passes that to
`compute_position`, and `compute_position_triton` allocates `torch.empty(extend_seq_lens_sum)`
while the kernel fills only what the per-sequence lengths cover. The draft therefore
inherited the *target's* token count. At bsz1 the two counts coincide and nothing is wrong;
at 8-way the target prefills ~105 tokens while the draft chunk is 6. One-line fix, and it is
a contract worth stating generally: **a derived batch must override every field carrying a
length, because `replace` silently keeps the parent's.**

**Behind it, a limit rather than a bug.** With that fixed the batch reaches verify and dies
on an asynchronous illegal memory access. This worker drives one request's chain at a time —
snapshot buffers, hand-rolled draft/verify graphs and rollback bookkeeping are all bs=1 and
worker-shared — so a batch of 2+ walks off the end of the snapshot stack. That is also, in
hindsight, the failure F0077 recorded against adaptive K and could not explain: a **fixed**
chain length reproduces it identically, so adaptive K was never the variable. It was
concurrency all along, and adaptive K happened to be what was running when requests first
overlapped.

`_verify_round` now refuses a multi-request batch with a message that names the limitation,
which turns memory corruption into an instruction. Verified both ways: the 8-way soak logs
zero illegal-access lines and the refusal instead, and single-stream is unchanged — identity
gate still 10/10, accept 3.39, 107.7 → 131.1 tok/s. The check is per-batch, not on
`max_running_requests`: that is a capacity, every working single-stream deployment here
passes 32, and rejecting on capacity would refuse configurations that are fine.

Consequence for the numbers above: they are correct for what they measure and what they
measure is one stream. Speculative decoding is single-request until the chain is batched,
which is design work, not a patch.

Adaptive K comes out of this cleared of the charge that made it EXPERIMENTAL, since the IMA
was never its doing: 40/40 sequential requests clean on the repaired stack, K distribution
{4: 589, 6: 2005, 8: 206}, genuinely switching. The default stays off anyway — the soak that
would promote it is the concurrent one, and concurrency is now explicitly out of scope.

## Corrected elsewhere

BENCHMARKS called the 142.8 flagship an `sgl.Engine` measurement in four places. The
artifact (`bench/results/f0066c/c1_72b_F1.json`) records host 127.0.0.1 port 30070: it is
the HTTP server via `bench/bsz_throughput.py` at c=1. The substance — real request path,
scheduler included — was right; the API name was wrong. Relabelled in both languages.

## The repair, checked at the kernel level

The same trace, re-taken after both repairs, against the two earlier columns:

| | 0.5.10 flagship | main, broken | main, repaired |
|---|---:|---:|---:|
| kernels/step | 469.1 | 968.6 | **464.4** |
| SPAN/step | 7052.1 us | 8118.4 us | **6983.1 us** |
| → kernel-loop framing | 141.80 tok/s | 123.18 | **143.20** |
| gap/step | −115.7 us | +17.7 | **−30.3** |
| overlapped transitions | 96.5% | 33.0% | **95.6%** |

The kernel count lands within 1% of the line that had the glue all along, the PDL chain is
back (negative gaps mean same-stream consecutive kernels overlapping, which is the signature
programmatic launch leaves and nothing else does), and main's kernel loop now reads slightly
*faster* than the 0.5.10 flagship's. This is the check that the e2e number alone cannot give:
throughput could have been recovered by luck or by a different mechanism, and the kernel
histogram is what says the specific thing that broke is the specific thing that got fixed.

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

And the correctness gate that did exist was run at the wrong size: the per-leg smoke is 8
tokens, the divergence sat at token 90. Every performance number in this session was taken
on a configuration whose 128-token identity gate had not been re-run, and when it finally
was, it failed. The rule that comes out of this is narrow and cheap: **any configuration
whose numbers get quoted has to have run the full-length gate in that exact configuration**,
not a shorter relative of it.

One more, on the fix rather than the bug: the first version of the sparse/spec repair was a
guard that disabled sparse whenever speculation was on. It would have passed review — it is
two lines, it reads as obviously right, and it is useless: under speculation the target
never decodes, so the only sparse work it removes is the draft's, which is not where the
divergence comes from. It would have cost 1.8% and fixed nothing. The question that killed
it was asking what the code actually does, rather than what it is named after.
