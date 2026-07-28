---
doc_kind: finding
finding_id: F0069
title: "#59 public-number re-measure. Three results. (1) The published flagship numbers are NOT stale — 7.2B 142.8 reproduces exactly and 1.5B 514.5 to −0.23%, same card, same session, announce + greedy-EXACT gates green. (2) What IS wrong is a convention split that nearly became a fabricated regression: `mega_flag_matrix.sh`'s D leg is the BOTTOM rung of the §7a ladder, and the README's 1.5B figure lives in the `serving_scale` ctx-1024 convention, not the 64-in/256-out c=1 one the ladder is quoted in — measured in its OWN convention the current stack reads 535.2 against a published 409.8, so the README undersells the shipped stack by 31%. (3) `bench/serving_scale.py` was silently dropping `disable_piecewise_cuda_graph` after sglang renamed it to `disable_prefill_cuda_graph`, and RWKV-7 decodes from a state the prefill graph never wrote — greedy output diverges from the oracle on the FIRST token. Every serving_scale number in this round had to be discarded and re-measured; the harness now refuses to run rather than drop a correctness switch."
status: CLOSED (2026-07-27) — all seven serving_scale legs re-measured under the repaired harness and gated (fp16 L2 greedy EXACT vs the oracle fixture; both quantized tiers token-identical across flag states; w8g64 greedy-lossless vs fp16 independently confirmed). Published artifacts are NOT affected — their recorded TTFT proves the switch was live when they were taken. Doc edits to README/§3 follow from this finding.
discovered_by: Opus 5 (1M), 2026-07-27
severity: medium (a benchmark harness silently measuring a wrong-output configuration)
related: [F0068, F0066, F0065, F0063, F0056]
machine: 5090 tower, one-shot containers, card idle, sky queue empty; same session as the F0068 sm120 close-out
---

# Finding F0069: the published numbers reproduce; the conventions and the harness did not

## 0. Why this round exists, and the mistake it started with

Handoff task #59 says the public numbers are stale and want re-measuring. The
first measurement of the round appeared to confirm it in the worst way:
`bench/mega_flag_matrix.sh` on the 1.5B returned a D leg of **497.6 tok/s**
against a published flagship of **514.5**, i.e. −3.3%, and *below* even the
F0065 rung of 502.3. Every leg passed its greedy-EXACT gate, so this looked like
a clean, well-gated regression.

It was not a regression. It was the wrong comparison, and it is worth recording
how close it came to being published, because the failure mode is not "measured
carelessly" — it is "measured correctly and compared to the wrong row".

`docs/BENCHMARKS.md` §7a-flagship states the ladder plainly:

| step | tok/s | finding |
|---|---:|---|
| megakernel PDL chain (MEGA+WKV_CUDA+PDL) | 493.0 | F0063 |
| + `add_ln` WIDE | 502.3 | F0065 |
| + sparse-path finalize | 509.0 | F0066b |
| + LoRA-gate epilogue (headline) | **514.5** | F0066c |

and `bench/results/f0066c/ab_summary.txt` names the headline leg's flags:
`F1 = full stack + WIDE + F0066b, RWKV_FUSED_LORA_GATED=1`, with
*"Default stays OFF pending the usual promotion review."*

`mega_flag_matrix.sh`'s D leg is `RWKV_MEGA=1 RWKV_WKV_CUDA=1 RWKV_PDL=1` and
nothing else — **it is the 493.0 row**, the ladder's bottom rung. Comparing it to
514.5 compares a config to a different config's number. The correct reading of
D = 497.6 is *+0.9% above its own published value*, and the +4.6 is itself
explained: `sparse_out_finalize` is launched unconditionally in today's tree
(`rwkv7_sparse_cmix.cu:163`, no env gate), so F0066b now rides along in every
leg including D, whereas F0063 measured D before it landed.

## 1. The ladder, re-measured same-session with liveness gates

Three rungs run back-to-back in one container, both models
(`bench/f0069_ladder.sh`, raw in `bench/results/f0069/c1_*.json`):

| rung | flags added to D | 1.5B | published | 7.2B | published |
|---|---|---:|---:|---:|---:|
| L0 | — | 497.8 | (493.0, pre-finalize) | 139.0 | — |
| L1 | `RWKV_ADDLN_WIDE=1` | 507.4 | 502.3 | 142.4 | 142.3 |
| **L2** | `+ RWKV_FUSED_LORA_GATED=1` | **513.3** | **514.5** | **142.8** | **142.8** |

**The 7.2B headline reproduces exactly. The 1.5B lands −0.23% off**, inside
run-to-run spread.

Two gates make this a measurement rather than an assertion, because the thing
that would otherwise ruin it is an env var that is silently ignored — the run
still produces a number, and the number is wrong:

- **Announce gate (hard, both directions).** L2 must print
  `[rwkv7] F0066c fused LoRA gate epilogue ENABLED`; L0 and L1 must not. All six
  legs matched their intent. A leg whose flag did nothing fails here instead of
  quietly reporting its neighbour's number.
- **WIDE liveness.** `RWKV_ADDLN_WIDE` has no announce line, so it is checked
  falsifiably instead: if L1 lands within noise of L0 the flag did nothing and
  the rung is void. Measured L1−L0 = **+9.6** (1.5B) and **+3.4** (7.2B), both
  far outside spread.
- Greedy fixture EXACT per leg, as in the source rounds.

An independent control on the card itself, on both models. The 1.5B albatross
baseline re-read **554.11** (median of three) against **553.9** on record — 0.04%;
the 7.2B read **155.82** (p50 6.417476 ms) against **155.75** on record — 0.05%. The
card is in the same state it was for the original numbers, so nothing here is a clock
artifact. Raw: `bench/results/f0069/albatross_1.5b_b1_3runs.log` and
`albatross_7.2b_b1_recovered.log`.

**Two corrections, in the order they happened, because the second is the instructive
one.** This finding originally cited the 7.2B control with no artifact behind it — the
run had happened, but nothing was committed, so an adversarial review of the
downstream PR correctly reported that 155.82 appeared nowhere in the repository. The
response to that review was to retract the number as a measurement that was never
made. That was wrong: the run *had* happened, and its raw output — SMOKE self-check
line, p10/p50/p90 — was recoverable and is now committed above. So the original defect
was a missing artifact, not a fabricated number, and the retraction was a second error
made by treating "I cannot find it" as "it does not exist" without first looking
where the evidence was actually kept.

## 2. The convention split

The 1.5B figure on the front page of `README.md` is **409.8 tok/s**, and it is
not in the ladder's convention at all. Its provenance,
`bench/results/ladder_full_5090.log`:

```
SERVING-SCALE  model=rwkv7-1.5b-fla  dtype=float16  mode=batch  cuda_graph=ON  radix=OFF
  decode=64tok steady-state (prefill-subtracted)  mem_fraction=0.85
 context |  bsz |  decode tok/s |   ms/step |    TTFT ms | peak VRAM MiB
    1024 |    1 |         409.8 |      2.44 |       36.0 |         16204
```

So there are two live conventions for "1.5B, single request":

| | harness | what it measures |
|---|---|---|
| **serving_scale** | `bench/serving_scale.py` | ctx-1024 steady-state decode, prefill subtracted, **offline Engine** |
| **bsz_throughput** | `bench/bsz_throughput.py` | wall-clock over 64-in/256-out at c=1, **served endpoint**, prefill included |

They are not interchangeable and the second necessarily reads lower. Dropping
514.5 (or 513.3) into the README's cell would have replaced a serving_scale
number with a bsz_throughput one — the exact error §7a-flagship's own framing
note warns about.

The published `409.8` also predates the megakernel line entirely: its announce
lines (`M6 fused fp16 GEMV`, `R2 fused paged shift+lerp`, `M9 fused LoRA`,
`M6 sparse channel-mix`) identify it as the F0056-era W1' set, with no
`RWKV_STATE_FP16`.

## 3. The harness was measuring a wrong-output configuration

The first pass of §4's table was thrown away. `serving_scale.py` built its
Engine like this:

```python
    disable_piecewise_cuda_graph=True,
    ...
    # keep the same invocation across sglang versions (e.g. main dropped
    # disable_piecewise_cuda_graph): only pass kwargs ServerArgs still accepts
    engine_kwargs = {k: v for k, v in engine_kwargs.items() if k in ServerArgs.__dataclass_fields__}
```

sglang has since renamed that switch to **`disable_prefill_cuda_graph`**. The
filter therefore dropped it in silence, and RWKV-7 ran with prefill CUDA graphs
**on** — a configuration the model does not support, because decode then starts
from a state the prefill graph never wrote.

The damage is not subtle and is not numerical drift. Greedy output diverges from
the oracle fixture on the **first** token, and stays fluent while doing it:

```
oracle : [37138, 45, 44312, 47, 11, 6699, ...]   " Paris, France.\nThe Eiffel ..."
measured: [46, 3448, 6699, 8412, 50727, 5383, ...]
```

Two controls pin the mechanism rather than merely correlating with it:

- With **no RWKV env flags at all**, the offline Engine produced *the same wrong
  tokens*. So the flag set is not the cause — the Engine invocation is.
- Adding `disable_prefill_cuda_graph=True` and changing nothing else made the
  same Engine reproduce the oracle **exactly, 24/24**.

`bench/bsz_throughput.py` is unaffected: it drives a served endpoint, and
`serve.sh` passes the switch on the CLI, where the old spelling still works.
That is why the §1 ladder — which runs through the server — reproduced cleanly
while everything measured through the Engine had to be redone.

**An inference this finding made and then retracted.** The filter landed in
`fee4608` (2026-07-05 19:04) and the published ladder artifacts in `015d93f`
(2026-07-05 19:54, message: *"full metric suites on sglang main"*) — 50 minutes
later, with the filter already in the tree. That ordering says the published
numbers were taken through the broken path. **They were not.** The recorded TTFT
settles it: a run with prefill graphs wrongly enabled reads ~23 ms, and with them
disabled ~37 ms. The published rungs recorded **36.0 / 37.3 / 40.4 ms**, matching
the corrected re-measurement (37.2 / 38.8 / 40.7) and not the broken one. The
sglang build in use that evening still carried the old field name, so the filter
passed it through. Commit ordering suggested a conclusion; a physical signature
in the artifact refuted it.

The harness now tries the known spellings for each **correctness** switch and
raises if none exists, instead of dropping it. Filtering unknown kwargs is right
for a tuning knob and fatal for a correctness one.

## 4. The flagship in the README's own convention

`bench/f0069_servingscale.sh` + `bench/f0069_quant.sh`, one session, ctx-1024
bsz-1, **repaired harness**, raw in `bench/results/f0069/ss_*.log`:

| tier | W1 (the published flag set) | published | **L2 (current flagship)** | L2 vs published |
|---|---:|---:|---:|---:|
| fp16 | 442.3 | 409.8 | **535.2** | **+30.6%** |
| fp16 + `STATE_FP16` | 442.8 | 447.3 | — | — |
| int8 `w8g64` | 498.1 | 461.9 | **596.8** | **+29.2%** |
| int4 `w4` | 597.7 | 548.8 | **742.6** | **+35.3%** |

TTFT per rung (37.2 / 38.8 / 40.7 ms) matches each published log's own TTFT
(36.0 / 37.3 / 40.4) — the cross-check from §3 that these are like-for-like.

Three things follow.

1. **The published rungs are stale-low by 7.8–8.9%**, uniformly across all three
   tiers. Their own flag sets re-read higher today because changes have landed
   unconditionally beneath them (the same mechanism as `sparse_out_finalize` in
   §0). The published values were valid measurements of their tree.
2. **`RWKV_STATE_FP16`'s separate benefit has been absorbed.** The published step
   was 409.8 → 447.3 (+9.2%); today the same step is 442.3 → 442.8 (+0.1%).
   Whatever the fp16 state bought in F0056, the current stack already gets by
   other means, so 447.3 should not be carried forward as an independent rung.
3. **535.2 replaces 409.8** — same harness, context, batch size and card, and
   30.6% above what the README publishes. The shipped stack is being *undersold*
   on its own front page.

Note that 535.2 (serving_scale) and 513.3 (bsz_throughput) are the same stack
measured two ways. Publishing both is fine; publishing either as the other is not.

The quantized checkpoints needed one thing worked out: they carry `qweight` +
`scale` and no `quantize_config.json`, so sglang cannot detect them and
`quantization="gptq"` fails outright — the format is *not* GPTQ (GPTQ has
`qzeros`/`g_idx` and int32 packing; these are symmetric, uint8/int8, `.scale`
singular). The shapes settle it without guessing: `qweight (2048, 1024) uint8`
with `scale (2048, 32)` is 4-bit packed two per byte at group_size 64. They are
served by our own `rwkv7_w4.cu`/`rwkv7_w8.cu` behind `RWKV_W4=1` / `RWKV_W8=1`,
both default OFF.

## 5. The gate, and the two ways it passed on nothing

`serving_scale.py` measures throughput and checks nothing, so every number above
arrived ungated. A quantized tier cannot be gated against the fp16 fixture — int4
is documented at −24pt MATH500, so a mismatch there would prove nothing. The
property the throughput claim actually rests on is narrower and testable:
*turning on the megakernel flag set must not change what the quantized model
emits*. So the gate compares each tier's greedy output under W1 against the same
tier under L2, token for token; w8g64 additionally gets an independent check
against fp16 (its §4 "greedy-lossless" claim), and fp16 L2 is checked against the
fixture's recorded oracle tokens.

Final verdict, on the repaired harness (`bench/f0069_quant_gate.sh`):

```
PASS  fp16_L2 == oracle fixture (greedy EXACT)
PASS  w8g64: megakernel flags do not change the output
PASS  w4:    megakernel flags do not change the output
PASS  w8g64 greedy-lossless vs fp16 (the §4 claim)
```

Getting there took two false greens.

The first cut reported **three passes**, all false. Every leg had died at engine
init — the generator was piped in on stdin, and sglang's scheduler subprocess
re-resolves the entry script by path, so it aborted with
`FileNotFoundError: <stdin>` before the model loaded. Each leg wrote a zero-byte
file, and `diff -q` on two empty files reports them identical. The gate was
structurally incapable of failing. It was caught only by reading the token lists
instead of the verdict line.

The repaired comparator requires both files to be non-empty and reports `VOID`
rather than `PASS` otherwise — and it paid for itself immediately. The next run
still produced nothing, for an unrelated reason: with the work at module level
each scheduler subprocess re-imports the entry module and re-enters
`sgl.Engine(...)`, which multiprocessing kills with `_check_not_importing_main`
(`serving_scale.py` avoids this only by having a `__main__` guard). The old
comparator would have called that four more passes. The new one reported four
`VOID`s and the second bug was visible at once.

That is three vacuous checks in one session — these two plus a CPU-only
adversarial suite in the transformers port, and a model-level history test that a
zero-init RWKV made degenerate. The shape is the same every time: **the assertion
was correct and the subject was absent**, so the check could only pass. A green
result is evidence only if the run can be shown to have done something — hence
the non-empty precondition here, the announce gate in §1, and the
falsifiable-magnitude check for the knob that has no announce string.

## 6. What this changes, and what it does not

Not stale, no edit needed: the §7a-flagship bsz1 table (142.8 / 514.5), correctly
labelled with its convention and date, and reproduced here.

Stale, replace in the serving_scale convention: `README.md` and
`README.zh-CN.md`'s RTX 5090 single-request cell, and the §3 ladder's final rows.
§3 is a *lineage* table — each row adds one kernel set to the previous — so the
historical rungs stay and the megakernel stack is appended rather than
overwritten. The 3090 column is not re-measurable this session and keeps its
existing "v0.5.10 historical ladder" label.

Not yet re-measured, and still carrying the pre-megakernel caveat already in the
docs: the peak/concurrency numbers (22,175 and the 7.2B peaks). Those come from
`bsz_throughput.py` and are therefore *not* affected by §3's harness bug, but
they remain pre-megakernel.

**Deliberately left alone: the Albatross int4 comparison** (§ "our int4 reaches
0.9908× of Albatross's fp16 on the author's own 5090 — 548.8 vs 553.9"). That
ratio is now stale in our favour: 742.6 against the same 553.9 would read
**1.34×**, flipping the claim from "almost matches" to "clearly exceeds". It is
not updated here, because it is a *matched pair* — both engines measured in one
session under one protocol (`bench/results/albatross_fleet_10cards.json`).
Dropping a number measured today against a competitor's number from a different
session is precisely the cross-session substitution this finding argues against,
and the fact that the error would flatter us is the strongest reason not to make
it. Re-running both sides together is the follow-up.
