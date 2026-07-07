---
doc_kind: finding
finding_id: F0045
title: "Qwen3.5-2B vs RWKV-7 1.5B, matched MLX benchmark (same M5, same bench_mlx.py protocol, multi-run): RWKV-7 wins bsz1 decode (+19-36% bf16, a near-tie at int4), Qwen3.5 wins prefill (+41-66% both tiers) — a genuine split, not a sweep either way"
last_verified_commit: "HEAD"
discovered_by: Sonnet 5 (agent-assisted), 2026-07-07
severity: info
status: open
related: [F0037, F0038, F0039, F0044]
---

# Finding F0045: Qwen3.5-2B MLX matched benchmark vs RWKV-7 1.5B

## Context / question being asked

F0044 established that Qwen3.5-2B runs on MLX via `mlx-lm` 0.31.3 out of the box (real Metal
kernels, not a slow fallback) but explicitly flagged its numbers as **single-run smoke-test**
throughput, not a `bench_mlx.py`-style gated multi-run result, and listed "a `bench_mlx.py`-equivalent
multi-run median bsz1/prefill sweep for Qwen3.5 on MLX" as the first of three follow-ups before
anything could be cited as a benchmark number. This finding does that follow-up: a real multi-run
benchmark for Qwen3.5-2B using the *exact same protocol* `mlx_port/bench_mlx.py` uses for RWKV-7, and
the resulting matched, same-machine, same-size-tier comparison table.

## Method

New script: [`mlx_port/bench_mlx_qwen35.py`](../../mlx_port/bench_mlx_qwen35.py). It reimplements
`bench_mlx.py`'s `bench_decode`/`bench_prefill` line-for-line against Qwen3.5 via `mlx_lm`'s Python
API (not the `mlx_lm.generate` CLI F0044 used for its smoke test):

- **decode**: prefill a short seed prompt ("The capital of France is" — the same prompt F0044's smoke
  test used), run **16 untimed warmup** greedy steps, then time **128 further greedy steps**. Every
  step is async-pipelined (`mx.async_eval` on the argmax token, fed straight back as next input, no
  per-step host sync mid-loop — the same pattern `rwkv7_mlx.py`'s `greedy_loop` uses, which its own
  docstring calls "the standard mlx-lm generation pattern"). **Median of 5 runs**, reporting median +
  best, exactly like `bench_mlx.py`.
- **prefill**: the seed-prompt tokens tiled to exactly **1024 tokens** (same tiling trick
  `bench_mlx.py` uses), fresh KV/recurrent-state cache, timed end-to-end including forcing logits +
  full cache state to materialize. **Median of 3 runs** after 1 discarded warmup run.
- Peak memory reset before each config, released (`del` + `gc.collect()` + `mx.clear_cache()`) before
  the next config loads — the same discipline `bench_mlx.py` documents to avoid one config's retained
  weights polluting the next config's peak-memory reading.
- **No oracle gate** (none exists for Qwen3.5 in this repo — out of scope, per F0044). A coherence
  sample (24-token greedy continuation, eyeballed for on-topic/non-garbled prose) substitutes, the
  same bar F0044's own probe used.

Checkpoints: the same two local checkpoints F0044 already verified byte-for-byte —
`/private/tmp/qwen35_mlx_test/Qwen3.5-2B` (bf16, native, zero conversion) and
`/private/tmp/qwen35_mlx_test/Qwen3.5-2B-mlx-4bit` (int4 group-64 affine, via `mlx_lm.convert -q
--q-bits 4 --q-group-size 64`, confirmed via `config.json`'s `quantization: {group_size: 64, bits: 4,
mode: affine}`). No re-download, no re-conversion — both were already present and intact. Same
machine/software stack as F0044 and `mlx_port/`: Apple M5, 32 GiB unified, macOS 27.0, MLX core
0.31.2, `mlx-lm` 0.31.3, Python 3.13.13.

**Disclosed, not hidden, methodology difference**: this benchmarks Qwen3.5 through `mlx_lm` — the
opponent's own native, actively-maintained implementation (real hand-written Metal delta-rule kernel
for its Gated-DeltaNet layers, confirmed in F0044) — not a from-scratch port. `mlx_port/`'s "zero
fla/torch/transformers" policy governs what this project ships as its **own** RWKV-7 implementation;
it was never a requirement to also hand-port the competitor's architecture just to benchmark it
(F0044's Decision section; mirrors how the GPU/cloud tier benchmarks Qwen3.5 through sglang's own
native support, not a hand-rolled mirror port).

## Result 1 — Qwen3.5-2B MLX, multi-run (new this pass)

| precision | decode tok/s (median / best) | prefill tok/s (1024 tok, median of 3) | peak mem |
|---|---:|---:|---:|
| bf16 (native checkpoint) | 27.5 / 27.7 | 2,800.5 | 4.65 GiB |
| int4 (mlx_lm.convert, g64 affine) | 89.3 / 89.9 | 2,691.3 | 2.28 GiB |

Coherence samples (greedy, 24 tokens after the seed prompt — no oracle exists for Qwen3.5, so this is
an eyeball check, not a pass/fail gate, same bar as F0044):

- **bf16**: `" Paris.\nA. True\nB. False\n\n<think>\nThinking Process:\n\n1.  **Analyze the"` — correct
  answer, coherent reasoning-model preamble (Qwen3.5 emits a `<think>` block by default).
- **int4**: `" Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe
  capital of France is"` — correct answer, but **degenerates into repetition** rather than reasoning.
  Disclosed honestly: this is a real, visible quality signal from int4 quantization, not swept under
  the rug. It is a coherence observation, not a quantitative accuracy claim — no compression-rate or
  logprob ruler was run for Qwen3.5 (out of scope this pass; see Honest limits below).

Raw JSON: [`mlx_port/results/bench_qwen35_2b_bf16.json`](../../mlx_port/results/bench_qwen35_2b_bf16.json),
[`mlx_port/results/bench_qwen35_2b_int4.json`](../../mlx_port/results/bench_qwen35_2b_int4.json).

int4 vs bf16 for Qwen3.5 itself: decode **+224.7%** (27.5→89.3), prefill **−3.9%** (2,800.5→2,691.3,
a real if small regression — plausibly a dequant tax similar in kind, if not degree, to what
`bench_mlx.py`'s own quant notes record for RWKV at small sizes).

## Result 2 — RWKV-7 1.5B MLX: canonical citation *and* a fresh same-session cross-check

The canonical, previously-published numbers (`docs/BENCHMARKS.md` §12.3, gate-verified 24/24 at the
time, superseding `mlx_port/README.md`'s own slightly older same-config table — see footnote¹):

| precision | decode tok/s (median / best) | prefill tok/s (1024 tok) | peak mem |
|---|---:|---:|---:|
| fp16-labeled² (bf16 weights) | 37.3 / 39.1 | 1,905 | 3.38 GiB |
| w4 (int4 group-64) | 94.0 / 95.8 | 1,975 | 1.65 GiB |

`docs/BENCHMARKS.md` itself warns: *"this is a shared box (load ~8), so single-stream decode has
±3–5% run-to-run jitter."* Rather than take that on faith, this pass re-ran `bench_mlx.py` fresh,
twice, back-to-back, today, on the same machine, minutes apart from the Qwen3.5 runs above (gate
re-verified 24/24 both times before any number was accepted):

| run | fp16-labeled decode (med/best) | fp16 prefill | w4 decode (med/best) | w4 prefill |
|---|---:|---:|---:|---:|
| fresh A | 32.8 / 33.6 | 1,691.6 | 89.5 / 91.8 | 1,913.2 |
| fresh B | 34.4 / 34.5 | 1,776.6 | 90.9 / 92.9 | 1,935.2 |

Peak memory was identical across both fresh runs and the canonical citation (3.38 / 1.65 GiB) —
memory doesn't jitter with load the way timing does. The two fresh runs agree with **each other**
within the documented ±3–5% band (decode 4.9%, prefill 5.0%, w4 decode 1.6%, w4 prefill 1.2%). But
**both fresh runs sit 8–12% below the canonical published fp16-labeled decode/prefill figures**
(32.8–34.4 vs 37.3; 1,691.6–1,776.6 vs 1,905) — a gap bigger than the documented same-session jitter
band, most likely day-to-day system-load variance rather than a measurement error on either side (the
w4 numbers, by contrast, land much closer to canonical: 89.5–90.9 vs 94.0). **This is disclosed, not
smoothed over**, because it directly affects how tight the closest race below (int4 decode) actually
is — see Result 3.

¹ `mlx_port/README.md`'s own measured table (commit `e78ae0e`, 2026-07-06 20:06) shows 36.4 tok/s
decode / 1947.5 prefill / **6.68 GiB** peak for the identical 1.5B-metal-fp16 config — the peak-memory
figure is a stale pre-fix artifact (superseded by `docs/BENCHMARKS.md` at commit `1aa5ab7`,
2026-07-06 22:12, which is later and whose fp16→w8→w4 peak-memory progression, 3.38→2.28→1.65 GiB, is
internally monotonic and consistent with the memory-release fix `bench_mlx.py` documents in its own
code comments). `docs/BENCHMARKS.md` §12.3 is treated as canonical here for that reason.

² `docs/BENCHMARKS.md` §12.1's own header reads "fp16 default", but `mlx_port/README.md`'s precision
policy and `bench_mlx.py --dtype`'s actual default are both **bfloat16** ("bf16 weights for the big
projections... fp32 for everything else... macOS 27.0... bf16 weights + fp32 state"). "fp16" here is
a **label carried over from this project's CUDA-side naming convention for its non-quantized
baseline**, not a claim of literal float16 weights. Qwen3.5's checkpoint is *also* natively bf16
(confirmed in F0044: `mlx_lm.convert`'s own log prints `Using dtype: bfloat16`, and no dtype
conversion is applied for the direct-load bf16 path benchmarked here). **Net effect: there is no
actual precision mismatch between the two "bf16" rows below — both run bfloat16 weights** — but the
RWKV-7 MLX docs' "fp16" label should not be read as literal float16 when comparing the two projects'
numbers side by side. This comparison uses "bf16" throughout to avoid perpetuating the ambiguity.

## Result 3 — head-to-head, matched precision tiers

**Primary comparison** (fresh run A for both sides — measured today, same machine, same working
session, minutes apart, eliminating the cross-session load confound noted above):

| tier | metric | RWKV-7 1.5B | Qwen3.5-2B | winner |
|---|---|---:|---:|---|
| bf16 | decode tok/s (median) | **32.8** | 27.5 | RWKV-7 **+19.3%** |
| bf16 | decode tok/s (best) | **33.6** | 27.7 | RWKV-7 **+21.3%** |
| bf16 | prefill tok/s (1024 tok) | 1,691.6 | **2,800.5** | Qwen3.5 **+65.6%** |
| bf16 | peak mem | **3.38 GiB** | 4.65 GiB | RWKV-7 **−27.3%** |
| int4 | decode tok/s (median) | 89.5 | 89.3 | **statistical tie** (+0.2%) |
| int4 | decode tok/s (best) | **91.8** | 89.9 | RWKV-7 **+2.1%** (near-tie) |
| int4 | prefill tok/s (1024 tok) | 1,913.2 | **2,691.3** | Qwen3.5 **+40.7%** |
| int4 | peak mem | **1.65 GiB** | 2.28 GiB | RWKV-7 **−27.6%** |

**Alternate framing** (canonical `docs/BENCHMARKS.md` RWKV figures vs the same fresh Qwen3.5 numbers
— wider RWKV margins at bf16/int4 decode, everything else directionally identical):

| tier | metric | RWKV-7 1.5B (canonical) | Qwen3.5-2B (fresh) | winner |
|---|---|---:|---:|---|
| bf16 | decode tok/s (median) | **37.3** | 27.5 | RWKV-7 **+35.6%** |
| int4 | decode tok/s (median) | **94.0** | 89.3 | RWKV-7 **+5.3%** |

**Reading it, plainly, in both directions:**

- **RWKV-7 wins bsz1 decode at every framing**, by a comfortable margin at bf16 (+19–36% depending on
  which RWKV run is cited) and by a much smaller margin at int4 — **close enough (0.2–5.3%) that
  which RWKV run you cite decides whether it's "RWKV wins narrowly" or "statistical tie."** This
  report picks the same-session fresh numbers as primary specifically because they remove the
  cross-session confound, and under that framing int4 decode is a tie, full stop — that is stated
  plainly, not softened toward either side.
- **Qwen3.5 wins prefill at both tiers, by a wide and robust margin** (+41–66%, and this doesn't
  depend on which RWKV citation is used since the fresh/canonical RWKV prefill numbers are close to
  each other, 1,691–1,905 fp16 / 1,913–1,975 int4). Qwen3.5's interleaved full-attention layers (6 of
  24) do dense batched matmul over the whole 1024-token window — exactly the shape GPU matmul units
  are best at — while RWKV-7's WKV recurrence, even chunked (256-token chunks), carries a genuinely
  sequential per-token state update that chunking amortizes but cannot fully parallelize away (the
  same structural point `docs/BENCHMARKS.md` §12.2/§12.6 already make about RWKV-7's decode being
  bandwidth/launch-bound rather than compute-bound — here it shows up as a prefill cost instead).
- **RWKV-7 uses ~27% less peak memory at both tiers** — expected and disclosed as such, not framed as
  an architecture-efficiency win: it is substantially explained by RWKV-7 "1.5B" being a nominally
  smaller model than Qwen3.5 "2B" (both are each vendor's own nominal size label; this project did not
  independently recompute exact active/text-only parameter counts for either checkpoint this pass —
  see Honest limits). The bf16 and int4 memory ratios are nearly identical (4.65/3.38 = 1.376×,
  2.28/1.65 = 1.382×), consistent with a roughly proportional, parameter-count-driven effect rather
  than a quant-scheme-specific one.
- **Both sides benefit hugely from int4**, but Qwen3.5 relatively more so on decode. Using each
  side's own canonical/primary bf16→int4 pair: RWKV-7 +152.0% (37.3→94.0, matching
  `docs/BENCHMARKS.md`'s own stated "+152%" for 1.5B exactly) vs Qwen3.5 +224.7% (27.5→89.3). This is
  an observation, not a causally-explained result — no profiling was done this pass to attribute it to
  bit-width efficiency (4.503 measured bits/weight for Qwen3.5), checkpoint byte count, or
  architecture mix; flagged for a future pass rather than asserted.

**This is a genuine split decision — not a sweep for either side, and this report does not round it
into one.** Decode and memory favor RWKV-7; prefill favors Qwen3.5 clearly; int4 decode is close to a
coin flip depending on which RWKV-7 run is cited.

## Repeatability check (why the primary table uses fresh run A, not an average)

Both sides were benchmarked **twice**, independently, this pass:

| | RWKV-7 fp16 decode | RWKV-7 w4 decode | Qwen3.5 bf16 decode | Qwen3.5 int4 decode |
|---|---:|---:|---:|---:|
| run 1 (primary) | 32.8 | 89.5 | 27.5 | 89.3 |
| run 2 (repeat) | 34.4 | 90.9 | 27.4 | 86.1 |
| spread | 4.9% | 1.6% | 0.4% | 3.6% |

All four fall inside or right at the documented ±3–5% shared-box jitter band. Run 1 (the first,
un-cherry-picked run for both models) is used as the primary table above rather than an average of
the two, since averaging two 5-run medians invents a statistic this project's methodology doesn't
otherwise use elsewhere; the repeat runs exist to show the primary numbers are not outliers, and to
make explicit exactly how close the int4-decode race is to measurement noise.

## Honest limits of this comparison

- **No numerical correctness oracle exists for Qwen3.5 in this repo.** The coherence samples above
  are the same "on-topic, non-garbled" bar F0044 used, not RWKV-7's 24/24 exact-match oracle bar. The
  int4 sample's repetition is disclosed as a real, visible signal precisely because there is no
  stronger ruler (compression rate, MATH500) run for Qwen3.5 here to quantify it properly — that
  remains a follow-up (F0044's item 2, still open).
- **bsz1, single-stream only** — this mirrors `mlx_port/README.md`'s own explicit scope ("Single-
  stream inference port, deliberately... NOT the sglang serving stack"). This says nothing about
  concurrent/batched throughput, which is where this project's broader competitive story is centered
  on the CUDA/sglang tier (`docs/BENCHMARKS.md` §5–§8); the MLX tier as a whole has no serving stack to
  batch on, for either model.
- **"1.5B" and "2B" are each vendor's own nominal size label**, not independently-verified matched
  active-parameter counts. Qwen3.5's on-disk bf16 checkpoint still carries stripped-at-load vision-
  tower and MTP weights (F0044), so computing an exact active/text-only parameter count from file size
  would overstate it; this wasn't attempted. The size-pairing (1.5B↔2B) follows this project's existing
  convention for this comparison tier, not a claim of parameter-for-parameter parity.
- **Only the 2B tier is covered.** F0044's third follow-up (whether Qwen3.5's 9B checkpoint is dense
  or MoE on this stack, to scope a 7.2B↔9B second tier) remains open and untouched by this pass.
  int4 group-size and quantization recipe were matched exactly (g64 affine, both via native `mx.
  quantize`), one of the cleaner apples-to-apples aspects of this comparison — but this pass did not
  re-verify Qwen3.5's true bits/weight against RWKV's (RWKV's own bits/weight for w4g64 isn't broken
  out separately in `docs/BENCHMARKS.md`; Qwen3.5's was measured at 4.503 in F0044).
- **No compression-rate or MATH500 ruler was run for Qwen3.5** — this project's own decreed accuracy
  rulers (compression rate + position curve, MATH500 avg@64) were not exercised here; this is a speed
  and coherence comparison only. Treat the int4 "degenerates into repetition" observation as a
  qualitative flag, not a quantified accuracy delta.

## Addendum (2026-07-07, follow-up investigation): the 8–12% gap is confirmed external jitter, not a regression

A separate follow-up pass investigated whether the "8–12% below canonical" gap flagged in Result 2
above is a genuine performance regression (code, library, or OS change) or ordinary machine variance.
**Verdict: variance, not a regression** — the codebase's own history and a live repro both settle it.

- **No functional change to the decode path.** `git log` on `mlx_port/rwkv7_mlx.py` since the
  canonical §12.3 numbers were measured (commit `b8075a8`, 2026-07-06 18:40, the same commit that
  shipped the decay-precompute WKV kernel whose "after" table *is* the canonical 37.3/39.1 reading)
  shows exactly two later commits touching that file: `a87e563` (F0039, opt-in quant) and `42f28b2`
  (F0040, compression-rate scoring). Neither touches the fp16 decode path in a way that could cost
  8–12%: `a87e563` adds `qbig()`/`isinstance(W, tuple)` dispatch that is a pure pass-through to the
  original `big(n)` call when `quant=None` (the default) — one extra `isinstance` check per matmul,
  nanoseconds against a ~28 ms/token bandwidth-bound budget; `42f28b2` only *adds* new methods
  (`_hidden_all`, `_head_logits`, `score_tokens`) never called by `generate`/`greedy_loop`/
  `bench_mlx.py`. `bench_mlx.py`'s own `bench_decode`/`bench_prefill` timing functions are byte-for-byte
  unchanged since before the canonical run. The two commits after that (`e78ae0e`, `0139586`) don't
  touch `rwkv7_mlx.py` at all.
- **No library change.** Installed versions today: `mlx==0.31.2`, `mlx-lm==0.31.3` — identical to what
  every relevant doc (this finding, F0038, BENCHMARKS.md §12) already records as the measurement
  environment. No upgrade occurred between the canonical run and now.
- **The canonical 37.3 reading was already flagged, in its own source finding, as noise-flavored.**
  F0038's own text (the finding that shipped the decay-precompute kernel and produced the 37.3/39.1
  "after" table) says outright: *"the higher decode here vs F0037's baseline (298 / 33.6 / 7.6) is
  host-load variance on this shared box, not a speedup."* i.e. 33.6 tok/s (not 37.3) was F0038's own
  assessed baseline for this exact config on the very same day the "canonical" 37.3 was recorded —
  BENCHMARKS.md §12.2 independently repeats the 33.6 figure. So the "canonical" citation was already,
  by the original author's own account, sitting on the high side of that day's noise, not a clean
  steady-state number.
- **Live repro, 10 fresh consecutive runs** (2026-07-07, `bench_mlx.py --models-root /private/tmp/mlx_models
  --wkv metal`, 1.5B fp16, this Mac, load average 4.6–4.9 per `uptime` — lower than F0038's documented
  ~8 — no thermal or performance warnings per `pmset -g therm`, on AC power, no other Python/MLX/CoreML
  process consuming CPU per `ps aux`):

  | run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | decode tok/s (median of 5) | 36.3 | 36.2 | 36.5 | 36.5 | 36.3 | 36.2 | 36.4 | 36.4 | 36.4 | 36.4 |

  Extremely tight (36.2–36.5, 0.8% spread) — closer to the canonical 37.3 than to this finding's own
  32.8/34.4 readings from earlier the same day, and comfortably above them. This proves neither the
  machine nor the code has settled into a permanently slower state: the same box, same code, same
  library versions reproduce near-canonical numbers minutes after this check began.
- **Revised jitter picture.** Across five independent same-code, same-library 1.5B-fp16-decode
  session medians now on record (33.6 on 2026-07-06 ~17:37, 37.3 on 2026-07-06 ~18:40, 32.8 and 34.4
  earlier on 2026-07-07, 36.2–36.5 in this check later on 2026-07-07), the spread is **32.8–37.3
  tok/s, a ~12% peak-to-trough spread** — roughly 2–3x the ±3–5% figure `docs/BENCHMARKS.md` previously
  quoted as *the* jitter band. That ±3–5% figure was not wrong, it was scoped too narrowly: it
  correctly describes *within-session* spread (this finding's own run1-vs-run2 repeatability check
  landed at 0.4–4.9%, and this check's 10-run cluster at 0.8%), but it does not cover *cross-session*
  spread, which is the comparison Result 2 above was actually making. `docs/BENCHMARKS.md` /
  `.zh-CN.md` §12's intro has been corrected to state both bands separately.
- **No root cause was chased beyond this** (out of scope for this pass's time-box). Plausible
  contributors to cross-session (as opposed to cross-run) drift include background load at the moment
  of the run (this Mac is shared with other work, e.g. an ASR server process — idle at 0.0% CPU during
  this check, but not necessarily idle during earlier sessions), memory-allocator/Metal-driver state
  left over from whatever large models were loaded earlier in a session (F0045's original low readings
  were captured in a session that had just finished loading/unloading multiple multi-GB Qwen3.5
  checkpoints back-to-back; this check's tight cluster was a clean RWKV-only session with nothing else
  loaded first), or ordinary macOS scheduler/QoS variation — none of these were isolated or ruled out
  individually. The actionable conclusion: **absolute single-session decode tok/s on this box is a
  point-in-time sample with a real ~±6% (up to ~12% peak-to-trough) cross-session band, not a fixed
  constant**, while interleaved-A/B *deltas* (the quant comparisons, the head-to-head vs Qwen3.5)
  remain reliable because both arms of an A/B share whatever session-level drift is present.

## Cross-references

`mlx_port/bench_mlx_qwen35.py` (this pass's new script) · `mlx_port/bench_mlx.py` (the RWKV-7
protocol this mirrors) · [F0044](0044-qwen35-mlx-feasibility.md) (the feasibility probe this extends —
its follow-up item 1 is now done; items 2–3 remain open) · [F0037](0037-mlx-fused-metal-default.md) /
[F0038](0038-mlx-m5-kernel-profiling.md) (RWKV-7 MLX baseline numbers and the bandwidth-bound-decode
framing used to read the prefill-vs-decode split above) · [F0039](0039-mlx-weight-quantization.md)
(RWKV-7's own w8g64/w4g64 quant methodology, the scheme this pass's int4-vs-int4 comparison is matched
against) · `docs/BENCHMARKS.md` §12.3 (canonical RWKV-7 1.5B citation) · `mlx_port/results/
bench_qwen35_2b_bf16.json`, `bench_qwen35_2b_int4.json` (raw output of this pass).

## Addendum (2026-07-07): compression-rate accuracy comparison, and why MATH500 stops here

A follow-up session added the **compression rate** ruler (this project's other Bo-mandated accuracy
metric alongside MATH500 avg@64) on MLX, matched-N, both models measured the same way:

| model | precision | N (docs/corpus × 15 corpora) | pooled bpb |
|---|---|---:|---:|
| RWKV-7 1.5B | fp16 | 40 × 15 = 600 | **0.5926** |
| Qwen3.5-2B | bf16 | 40 × 15 = 600 | 0.6719 |

**RWKV-7 1.5B compresses better** (lower bpb) than Qwen3.5-2B on MLX, the same direction already
found on the cloud tier (sglang: RWKV 0.6085 fp16 vs Qwen3.5-2B 0.6729 bf16, §2) — two independent
platforms, two independent implementations, same conclusion, which is exactly the kind of
cross-validation that makes a claim trustworthy rather than a one-off measurement artifact. N=40/corpus
(600 docs total) is a reduced sample vs this project's usual 500/corpus convention, matched on both
sides so the comparison itself is fair even though the absolute numbers carry more sampling noise
than the flagship cloud-tier figures — cite the cloud-tier numbers as primary, this MLX pair as a
corroborating cross-check.

**MATH500 avg@64 was not run on this platform.** The session attempting it ran into real memory
pressure on this Mac (running Qwen3.5's larger vocabulary/generation-heavy workload alongside
everything else resident) and was stopped on the user's direct instruction before producing a result.
This is an honest, disclosed gap, not a silent omission: the Apple-Silicon tier of this comparison has
a **speed** story (this finding, F0044) and now a **partial accuracy** story (compression rate, above),
but no MATH500 avg@64 reading. The cloud tier (sglang, §2/§7-series findings) carries the full
MATH500 picture for this comparison; nothing here should be read as implying an MLX MATH500 result
exists or was attempted further.
