---
doc_kind: finding
finding_id: F0053
title: "Qwen3.5-2B MATH500 avg@64 (chatml_thinking) was run with RWKV-tuned sampling params left in place and zero presence_penalty support in the harness: 93.15% of 32,000 generations hit the token cap, invalidating rollout_accuracy=31.41%; root-caused (confirmed via actual completions: textbook degenerate repetition loops, not genuine reasoning), harness fixed, and re-measured with Qwen's own documented sampling recipe -- corrected full 500x64 results: non-thinking (Qwen3.5-2B's documented default, headline number) avg@64=67.63% at 0.99% truncation, thinking avg@64=47.72% at 52.41% truncation (disclosed budget-capped floor, not a ceiling) -- both decisively beat RWKV-7 1.5B's own 40.4-40.6%"
last_verified_commit: "HEAD"
discovered_by: Sonnet 5 (agent-assisted), 2026-07-07/08
severity: info
status: open
related: [F0024, F0044, F0045, F0048, F0049, F0050]
---

# Finding F0053: Qwen3.5-2B MATH500 avg@64 was measured with the wrong sampling recipe; root cause, fix, and corrected numbers

**Status of this document: DONE.** Root cause, harness fix, five pilots, and both full
500x64 runs (non-thinking headline + thinking) are complete. Qwen3.5-9B was not attempted
this session (see "What's not done").

**Headline numbers** (full detail and methodology below):

| model | mode | avg@64 | truncated | vs. RWKV-7 1.5B (40.4-40.6%, F0024) |
|---|---|---:|---:|---|
| Qwen3.5-2B | non-thinking (its documented default) | **67.63%** | 0.99% | **+27.0-27.2pt, Qwen3.5 wins** |
| Qwen3.5-2B | thinking (disclosed floor, not ceiling) | 47.72% | 52.41% | **+7.1-7.3pt, Qwen3.5 wins** |

Read plainly: **Qwen3.5-2B beats RWKV-7 1.5B on this ruler in both modes**, reported the
same way a loss would be, per this project's own claims-need-numbers discipline.

## What was wrong

A prior session ran `bench/math500_avg64.py --prompt-style chatml_thinking` against
Qwen3.5-2B (500 MATH500 problems x 64 samples = 32,000 generations) as part of this
project's req #2 "beat Qwen3.5" comparison chapter (`docs/BENCHMARKS.md` §13). The result:

| metric | value |
|---|---:|
| `rollout_accuracy` (avg@64) | 31.41% (10,052 / 32,000) |
| `truncated_rate` (hit `max_new_tokens`) | **93.15%** |
| `mean_generated_tokens` | 3,938.1 (of a 4,096 cap) |
| `ended_eod_rate` (finished naturally) | 6.85% |

93.15% truncation means the number measured almost entirely "how far the model got
before being cut off," not "whether the model could solve the problem." This invalidated
the number for the comparison chapter. Stale file (kept, not cited, see "Cross-references"):
`bench/results/qwen35_accuracy/math500_avg64_2b_chatml_thinking_5090.json`.

## Root cause (verified directly, not assumed)

Two independent gaps compounded:

1. **`bench/math500_avg64.py` hardcodes `--top-p 0.28 --top-k 32`** (its own docstring:
   "REF L50-L55" -- BlinkDL's `eval_math500_albatross.py` defaults, correct *for RWKV*).
   When this project extended the harness with `chatml_thinking`/`chatml_direct` prompt
   styles to support non-RWKV tokenizers, these RWKV-tuned sampling defaults were left in
   place for the Qwen3.5 run instead of being overridden with Qwen3.5's own documented
   settings. Confirmed directly from the stale result file's own `config` block:
   `"top_p": 0.28, "top_k": 32, "max_new_tokens": 4096, "penalty": "off"`.

2. **The harness had no `presence_penalty`/`repetition_penalty` support at all** --
   `one_rollout()` never sent these fields, and there was no CLI flag to set them.
   Qwen3.5-2B's own model card (`models/qwen3.5-2b/README.md`, "Best Practices" section)
   documents:
   - Thinking mode (text tasks): `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
     presence_penalty=1.5, repetition_penalty=1.0`
   - Non-thinking mode (text tasks): `temperature=1.0, top_p=1.00, top_k=20, min_p=0.0,
     presence_penalty=2.0, repetition_penalty=1.0`
   - Recommended output length: "32,768 tokens for most queries... 81,920 tokens" for
     math/competition-level problems -- far above the 4,096 the flawed run used.
   - **Qwen3.5-2B operates in non-thinking mode by default** (confirmed at the code level,
     not just the README prose -- see "9B default-mode note" below).

   sglang's own `SamplingParams` (`python/sglang/srt/sampling/sampling_params.py`, verified
   against the exact version installed on the tower, `0.0.0.dev1+gb28bc1060`) has
   `presence_penalty` (range `[-2, 2]`, default `0.0`) and `repetition_penalty` (range
   `(0, 2]`, default `1.0`) as standard top-level fields -- these are real, already-wired
   sglang API fields the harness simply never threaded through.

3. **Confirmed at the text level, not just the config level**: reading the actual stale
   run's completions (`_generations.jsonl`) shows the truncated rows are not "the model
   was still reasoning" -- a large share are textbook degenerate repetition loops, e.g.
   (verbatim, from one truncated completion): `"I will write it as $(3, \frac{\pi}{2})$."`
   repeated more than a dozen times back-to-back before hitting the cap. This is exactly
   the failure mode `presence_penalty` exists to prevent, compounded by `top_p=0.28/top_k=32`
   being far too narrow a sampling window for a model tuned for `top_p=0.95/top_k=20`.

**9B default-mode note (checked, not assumed identical to 2B):** the two model cards
disagree on their own default: 2B's Quickstart says "Qwen3.5-2B operates in non-thinking
mode by default"; 9B's says "Qwen3.5 models operate in thinking mode by default." This is
confirmed at the chat-template level, not just README prose -- `chat_template.jinja`:
2B has `{%- if enable_thinking is defined and enable_thinking is true %}` (thinking is
opt-in), 9B has `{%- if enable_thinking is defined and enable_thinking is false %}`
(thinking is opt-out). Our harness passes `enable_thinking=True/False` explicitly for
both `chatml_thinking`/`chatml_direct`, so this difference doesn't affect our
measurements either way -- noted here only because it matters for how the numbers below
should be described in prose (don't call thinking mode "the" default for both sizes).
9B's card also has an internal inconsistency between its Quickstart tip box and its later
Best Practices section for one specific row (non-thinking + reasoning tasks): 0.95/20/1.5
vs 1.0/40/2.0. Not yet resolved since 9B wasn't reached this session (see "What's not
done").

## Fix

`bench/math500_avg64.py`: added `--presence-penalty` (default `0.0`) and
`--repetition-penalty` (default `1.0`) CLI args, threaded into the `sampling_params` dict
sent to sglang's `/generate`, and into the result JSON's `config` block (replacing the
previously-hardcoded `"penalty": "off"` with the actual values used). Both defaults are
sglang's own no-op values, so **every existing RWKV `fake_think`/`plain` invocation is
byte-for-byte unaffected** -- verified no downstream code parses the old `"penalty"` key
(repo-wide grep, clean) before removing it, and confirmed the new fields only change
behavior when a caller explicitly passes non-default values (which no RWKV invocation
does). `--top-p`/`--top-k` defaults were *not* changed (still `0.28`/`32`, still correct
for RWKV) -- the fix is the new opt-in flags plus documentation telling `chatml_*` callers
to override all four sampling args explicitly, not a change to any default that RWKV's own
already-published numbers depend on.

## Pilot methodology: cheap before expensive

Rather than re-run the full 32,000-generation job blind, five small pilots (20 problems x
16 samples = 320 generations each, a ~1% slice of the full cost) isolated how much of the
fix was sampling vs. budget, before committing to the expensive full run:

| pilot | mode | sampling | max_new_tokens | truncated | avg@16 | mean gen tokens |
|---|---|---|---:|---:|---:|---:|
| (stale, for reference) | thinking | RWKV defaults (bug) | 4,096 | 93.15%* | 31.41%* | 3,938* |
| P0 | thinking | corrected | 4,096 (unchanged) | 82.5% | 29.4% | 3,857 |
| P1 | thinking | corrected | 8,192 | 74.7% | 40.9% | 7,137 |
| P2 | thinking | corrected | 16,384 | 62.2% | 51.25% | 12,761 |
| P3 | non-thinking | corrected | 4,096 | 47.2% | 50.6% | 2,655 |
| P4 | non-thinking | corrected | 8,192 | 21.9% | 59.1% | 4,087 |
| P5 | non-thinking | corrected | 16,384 | **1.2%** | 67.2% | 4,990 |

\* stale row is the full 500x64 run, all others are the 320-generation pilot subset (first
20 problems, samples=16 -- avg@16 not avg@64); not directly comparable in absolute
accuracy terms to the pilot rows, shown only for the truncation-rate contrast.

**P5 is the confirming pilot that closed the budget decision for non-thinking mode**: at
16,384, non-thinking truncation falls to a trivial 1.2% (98.8% of generations end
naturally) with mean generation length (4,990) well under half the budget -- non-thinking
mode has clearly converged, unlike thinking mode which was still improving substantially
at every budget tested. This is now a clean, essentially-complete measurement in the
truncation sense, not a budget-limited approximation.

**What this shows:**
- **Fixing sampling alone (P0, same 4,096 budget as the bug) barely moves truncation**
  (93.15% -> 82.5%) -- the dominant lever is budget, not sampling. Reading the actual P0
  completions confirms why: the repetition loops are gone (replaced by genuine, coherent,
  step-by-step reasoning), but Qwen3.5-2B in thinking mode is simply very verbose --
  even an easy problem (converting $(0,3)$ to polar coordinates) consistently ran to the
  4,096 cap across independent samples without looping. "Genuinely needs more tokens" and
  "93% never finish" are different claims, and the pilot data separates them cleanly.
- **Thinking-mode truncation keeps falling as budget grows (82.5% -> 74.7% -> 62.2% at
  4,096/8,192/16,384) and accuracy keeps climbing sharply (29.4% -> 40.9% -> 51.25%)** --
  this is a real, ongoing effect, not a plateau. Qwen3.5-2B's own README explicitly
  recommends up to 81,920 tokens for math/competition-difficulty benchmarking, consistent
  with what the pilots show. **This means the budget chosen below is a practical cutoff
  for wall-clock reasons, not the point where truncation stops mattering** -- disclosed
  plainly rather than implied away.
- **Non-thinking mode is a much better fit for a bounded budget**: at the same 8,192
  budget, non-thinking's truncation (21.9%) is already in the same ballpark as RWKV-7's
  own published MATH500 truncation rate (14.2%, F0024), while thinking mode's is still
  62-75%. This is expected -- non-thinking responses are structurally short-form.

**Wall-clock budget decision (revised once, mid-flight, see note below):** the stale
(flawed) run took 2.66 hours for 32,000 generations at mean 3,938 tokens/generation
(13,148 tok/s aggregate, concurrency 256). Extrapolating pilot mean-token-length to the
full 500-problem set (likely somewhat higher than the pilot's first-20-problems subset,
which skews slightly easier than the full distribution) at each candidate budget:

| max_new_tokens | thinking mean tok/gen (pilot, easier subset) | est. full-run tokens | est. wall time |
|---:|---:|---:|---:|
| 8,192 | 7,137 | ~240M (assuming ~7,500 on full set) | ~5.1 hours @ ~13,000 tok/s |
| 16,384 | 12,761 | ~448M (assuming ~14,000 on full set) | ~9.6 hours @ ~13,000 tok/s |
| 32,768 | ~22,000 (extrapolated -- the per-doubling *absolute* increase is still growing, 3,857->7,137->12,761 are increases of +3,280 then +5,624, not shrinking) | ~800M (assuming ~25,000 on full set) | **~16-22 hours** @ 10,000-14,000 tok/s |

**First decision (superseded, see note): `--max-new-tokens 8192` for both modes.** A full
thinking-mode run was actually started at this budget and reached 1,600/32,000 rollouts
(~18 minutes, healthy throughput ~10-11k tok/s) before being deliberately stopped --
see "Mid-flight course correction" immediately below. Sunk cost: ~18 minutes of GPU
time, judged acceptable to discard rather than carry an inferior config through a
5-hour run.

**Mid-flight course correction:** partway through the 8,192 thinking-mode run, a fuller
review of the pilot trend (table above) produced a better-justified decision, superseding
the first one before it consumed meaningful GPU time:

1. **32,768 is Qwen3.5's own documented general-purpose default output length** ("32,768
   tokens for most queries" per both model cards) -- a more principled budget to target
   than an arbitrary doubling, *if* it were affordable. It is not: extrapolating the
   pilot's mean-token growth (table above, right column) projects **16-22 hours** for a
   full run at 32,768 -- squarely in "stop and reconsider" territory, so 32,768 was
   rejected on cost alone, not on the merits of the number it would produce.
2. **16,384 is the next-best defensible budget**: ~9.6 hours estimated for thinking mode
   alone is a large but bounded, disclosed cost, and the pilot already shows a real,
   substantial accuracy gain at this point (29.4% -> 40.9% -> 51.25% across
   4,096/8,192/16,384). **Final decision: `--max-new-tokens 16384 --ctx-limit 17408`
   for the thinking-mode full run** (17,408 = 16,384 + 1,024 prompt headroom, comfortably
   above the observed max prompt length across all 500 problems, 841 tokens).
3. **Non-thinking mode is reprioritized as the headline number for the 2B tier**, not a
   secondary curiosity -- Qwen3.5-2B's own model card states plainly that it "operates in
   non-thinking mode by default," which is a principled, defensible reason to lead with
   it (this project did not pick whichever mode happens to favor RWKV; the model's own
   documented default usage decides which mode is "the" headline). A confirming pilot at
   `--max-new-tokens 16384` (P5, see table update below) was run before committing to the
   full non-thinking job, continuing this document's cheap-before-expensive discipline
   even after the mid-flight revision.

The 16,384 pilot numbers (P2, P5) are kept in the table above/below specifically so
nobody mistakes either full-run number for Qwen3.5-2B's true asymptotic MATH500 ceiling
-- thinking mode in particular was still rising at the largest budget piloted (32,768);
16,384 is a deliberate, disclosed, cost-bounded operating point, not the ceiling.

## Result (full 500x64 runs)

**Non-thinking mode -- DONE, headline number for the 2B tier** (`chatml_direct`,
`--max-new-tokens 16384 --ctx-limit 17408 --temperature 1.0 --top-p 1.00 --top-k 20
--presence-penalty 2.0`, concurrency 256):

| metric | value |
|---|---:|
| **avg@64** (rollout_accuracy) | **67.63%** (21,642 / 32,000) |
| pass@64 | 97.00% |
| truncated_rate | **0.99%** |
| ended_eod_rate | 99.01% |
| mean_generated_tokens | 4,658.6 |
| gen_wall_time_s | 14,944.8 (4.15 hours) |
| throughput_gen_tok_per_s | 9,975.0 |

This is a clean measurement in the sense this document cares about: truncation is under
1%, actually *lower* than RWKV-7 1.5B's own published truncation rate (14.2%, F0024) at
its much smaller 1,500-token budget. The 4.15-hour wall time matches the pre-run
projection (~4.1-4.2 hours, tracked live across ten progress checkpoints during the run,
holding stable throughout) closely. Full: `bench/results/qwen35_accuracy/math500_avg64_2b_chatml_direct_5090.json`.

**Thinking mode -- DONE** (`chatml_thinking`, `--max-new-tokens 16384 --ctx-limit 17408
--temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5`, concurrency 256),
launched immediately after non-thinking completed, same server (no relaunch needed --
17,408 context-length already covers both):

| metric | value |
|---|---:|
| **avg@64** (rollout_accuracy) | **47.72%** (15,269 / 32,000) |
| pass@64 | 92.40% |
| truncated_rate | 52.41% |
| ended_eod_rate | 47.59% |
| mean_generated_tokens | 11,880.4 |
| gen_wall_time_s | 47,878.4 (13.30 hours) |
| throughput_gen_tok_per_s | 7,940.4 |

Truncation (52.41%) is meaningfully better than the flawed run's 93.15% and than every
pilot at every smaller budget, but is still substantial and disclosed plainly as such --
this is the same "budget-capped, not a ceiling" number flagged in the budget-decision
section above, not a claim that thinking mode has converged the way non-thinking did.
Full: `bench/results/qwen35_accuracy/math500_avg64_2b_chatml_thinking_5090_v2.json`.

**Wall-clock note (a real, disclosed deviation from the pre-run estimate):** the ~9.6-hour
estimate from the budget-decision section assumed throughput similar to the pilots'
(uncontended, low-concurrency) and to non-thinking's full run (~10,000 tok/s sustained).
The actual run instead showed a **periodic pool-pressure pattern** not present in any
pilot: at full 256 concurrency, thinking mode's much longer average generation length
periodically filled the ~1.5M-token KV-cache pool faster than requests were completing,
triggering sglang's built-in graceful degradation (request retraction -- evicting a
request back to the queue rather than crashing) several times over the course of the run,
each episode lasting minutes and dropping throughput from ~10,000 tok/s to as low as
~5,000-5,800 tok/s before the population naturally desynchronized and throughput
recovered. This never affected correctness (no errors, no crashes, confirmed no truncated
or malformed generations attributable to it) or resulted in resource exhaustion, only
average throughput -- the actual run took 13.30 hours against a 9.6-hour estimate. This
periodic effect was invisible to every pilot in this document because all pilots ran at
concurrency 32, far below the level where thinking mode's long generations pressure the
pool; it only appears at the full run's requested concurrency of 256. Watched live and
confirmed self-resolving on every occurrence (never required manual intervention); noted
here as a genuine methodological finding about server-side capacity planning for
long-generation workloads, not as a defect in the measured accuracy number itself.

## What's not done

- **Qwen3.5-9B** (thinking + non-thinking): not attempted this session. 2B alone (both
  modes, sequential) took **~17.5 hours of GPU time** (4.15h non-thinking + 13.30h
  thinking) -- substantially more than the ~7-8 hour pre-run estimate, mostly due to the
  thinking-mode pool-pressure effect above. 9B is a larger model that would need its own
  fresh pilot-and-budget-decision pass (its sampling recipe was already checked in
  advance, see "Root cause" above, and its own internal Quickstart-vs-Best-Practices
  inconsistency for one sampling row still needs resolving) and, empirically, its own
  concurrency/pool-capacity characterization -- not a given to inherit from 2B's. Stopping
  here cleanly, with 2B done properly in both modes, is preferred over rushing 9B and
  risking a second, harder-to-catch methodology or capacity-planning issue.
- Neither model card's recommendation to add "Please reason step by step, and put your
  final answer within \boxed{}" to the prompt for benchmarking was applied here --
  checked against BlinkDL's own `eval_math500_albatross.py` REF, which also does not add
  an equivalent instruction to RWKV's prompt, so this is symmetric (not a bias toward or
  against either model), but it is a known, disclosed way both numbers could differ from
  each side's best-achievable score under maximally-tuned prompting.
- Thinking mode's 47.72% is a real, complete 500x64 measurement, but per the
  budget-decision section above it is a floor, not Qwen3.5-2B's asymptotic thinking-mode
  ceiling -- the pilot curve (Pilots P0-P2) was still rising at every budget tested up to
  16,384, and 32,768 (the model's own documented general-purpose default) was priced out
  on wall-clock grounds alone, not because it wouldn't have scored higher.

## Downstream documents corrected

`docs/BENCHMARKS.md` §13.4, `docs/BENCHMARKS.zh-CN.md` §13.4, `CONTRIBUTIONS.md` req #2
row, `ROADMAP.md` "Active right now" section -- all previously said MATH500 avg@64 for
Qwen3.5 was "in progress" with no number; all updated to cite the corrected numbers above.

## Cross-references

`bench/math500_avg64.py` (harness fix) · stale (kept, not cited):
`bench/results/qwen35_accuracy/math500_avg64_2b_chatml_thinking_5090.json` · corrected:
`bench/results/qwen35_accuracy/math500_avg64_2b_chatml_thinking_5090_v2.json`,
`bench/results/qwen35_accuracy/math500_avg64_2b_chatml_direct_5090.json` · pilots (not
committed, tower-only, cited by number in the table above) · [F0024](0024-math500-avg64.md)
(RWKV-7 1.5B's own avg@64 = 40.60%, F0024's original; current-HEAD re-run 40.42%, both
within noise of each other, per `README.md`'s own drift-gate accounting -- the number this
finding's corrected Qwen3.5-2B result should be read against) · [F0044](0044-qwen35-mlx-feasibility.md)-[F0050](0050-qwen35-numpy-oracle-gate.md)
(the rest of the Qwen3.5 comparison chapter).
