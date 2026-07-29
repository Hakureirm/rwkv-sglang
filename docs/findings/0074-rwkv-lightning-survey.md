# F0074 — rwkv_lightning: the serving layer albatross points at, and what it does not publish

**Date:** 2026-07-29 · **Status:** SURVEY (public sources only; no source read) · **Prior:** F0007 (albatross baseline), F0023 (albatross kernel audit), F0032 (vllm-rwkv showdown)

## Why this one matters more than its star count suggests

`RWKV-Vibe/rwkv_lightning` is easy to file as one more RWKV inference repo. It is not.
**Albatross's own README designates it the full backend** — "Full backend:
https://github.com/RWKV-Vibe/rwkv_lightning". So it is not a peer of our speed
reference; it is the officially blessed serving layer *on top of* our speed reference,
which makes it the closest structural analogue to this project's sglang overlay in the
entire ecosystem: single GPU, hand-written kernels, continuous batching, prefill
admission, OpenAI-compatible HTTP.

Naming trap worth recording: **"lightning" is not PyTorch Lightning.** The sibling repo
`RWKV-Vibe/RWKV-LM-V7` is the Lightning one, and it is training. This one is inference.

## Method and its limits

Public sources only: README, GitHub API metadata, issue and PR discussion, licence.
**No source files, kernel implementations, or flag names were read** — ADR-0004's
no-competitor-source rule, which we hold whether or not the licence would permit it.
Everything below is either quoted from a public artifact or marked as inference.

The survey was first drafted from a delegated research pass, then the load-bearing
claims were re-checked directly before anything was written down here, on the principle
that a second-hand conclusion is not evidence. Verified against the GitHub API and the
raw READMEs on 2026-07-29: `license: null` with `contents/LICENSE` returning 404, zero
releases and zero tags, 37 stars / 8 forks, created 2025-10-14 and last pushed
2026-07-27; albatross's README carries the designation at line 75, verbatim; the
rwkv_lightning README is 453 lines of which exactly one is performance-related, and
that one is the GemLite warm-up caveat rather than a number. The two PR quotations
below were read from the PR bodies directly, which is also how the correction in
"what we should measure" was found.

## What it is

Serving backend, not a library and not a wrapper. On an albatross-lineage kernel base
it adds: dynamic batching with prefill admission, a session-keyed three-tier state
cache (VRAM → RAM → SQLite, persisted across restarts), two weight-only quantisation
paths, a Gradio UI, and seven endpoint families including OpenAI `/v1` chat completions,
a batch-completions endpoint and fill-in-middle. CUDA **and** HIP sources in tree,
JIT-compiled on first import, so an operator needs a full local toolchain — no wheels.

| | |
|---|---|
| First / last commit | 2025-10-14 / 2026-07-27 (active) |
| Commits · stars · forks | 117 · 37 · 8 |
| Issues | 0 open, 2 closed |
| PRs | **15 open**, 9 closed |
| Releases · tags · CI | **none · none · none** |
| Licence | **none — `license: null`, no LICENSE file** |
| Contributors | `Alic-Li` (72), `No-22-Github` (19), plus PR authors |

Dependencies as documented: torch from the CUDA 13.2 or ROCm 7.2 wheel channels,
fastapi, pydantic, ninja, numpy; optionally GemLite 0.6.0 and CUTLASS v3.9.2 headers.
**Not** flash-linear-attention, **not** transformers, **not** albatross-as-a-package —
it loads BlinkDL `.pth` directly and carries its kernels in tree. Triton only
transitively, through GemLite. Caveat that belongs with all of that: there is **no
`requirements.txt`, `pyproject.toml` or lockfile**, and the dependency-graph endpoint
404s, so the README install lines are the entire dependency declaration. That is
"documented deps", not "audited deps".

## The finding: it publishes no absolute performance number at all

No tok/s, no latency table, no memory table, anywhere in the repository — README,
docs, or committed artifacts. It **ships a benchmark script at the repo root and
commits no results from it.**

Every number that exists lives in PR discussion, and each is a **relative delta with
neither hardware nor model size stated**:

| Source | Claim (verbatim) | Hardware | Model |
|---|---|---|---|
| PR #3 | TTFT "82124 ms" → "1927 ms" at 1,915 input tokens | not stated | not stated |
| PR #19 | "~20.6% faster at bsz=96, ~41.2% faster at bsz=160" on staggered-finish batches | not stated | not stated |
| PR #17 | "no measurable speedup observed at bsz=32 (~0.3%, within noise)" | not stated | not stated |
| PR #26 | second queued batch admitted "typically 0.6-1.1s early" | not stated | not stated |

The numbers people associate with the project are **albatross's**, not its own: 145+
tok/s bsz1 and 11289 tok/s bsz1 prefill on RTX 5090 at 7.2B fp16. How much of that
survives its Python/FastAPI serving layer **has never been published**. Against a stack
in that state, our committed 5090 7.2B ladder with driver, build and flags recorded is
the decisive difference, and it is a difference in evidence rather than in speed.

Accuracy: none in repo. One external number exists — a sibling repo scores a 7.2B
checkpoint 21/30 on a 30-task agentic suite *assuming this stack as the server*,
hardware not stated. That scores the model, not the engine.

## Two weaknesses in their public record

- **Batched-decode CUDA graph capture is documented as unsafe.** PR #24 states capture
  "succeeds without any exception", verified against a bit-exact eager control at
  "diff 15-26, logit magnitude ~4-9 — not fp16 noise". Capture is confirmed safe only
  on their single-sequence path. Do not assume they have batched-decode graphs; they
  say they do not. Note that PR #24 is itself open and comment-only, so this is a
  documented-but-unfixed condition rather than a resolved one.
- **Sampler failure reported on sm_120 — our card's architecture.** Issue #9: on
  Blackwell sm_120 with a 13.3B checkpoint, load and forward work but the custom
  sampler returns invalid token ids while pure-torch sampling works. Closed without a
  confirmed fix; the one reply blamed a toolkit/driver mismatch. Cause unestablished.

Governance signal, recorded as observation not judgement: 14 of the 15 open PRs are
from a single contributor, all filed 2026-07-12→07-25, none merged — a substantial
unreviewed backlog against one-person review capacity, including security fixes and the
correctness bug above.

## What we should measure because of this

1. **Staggered-finish at high batch — a narrow gap, not the one it first looked like.**
   The first reading of their PR #19 was that we had an unmeasured 20-40% hole. That
   reading was wrong and is retracted here rather than quietly dropped. §7c already
   measures variable-length real load (ShareGPT, 500 prompts, neutral client) where we
   take peak output throughput 9,602 vs 8,865 tok/s and p99 inter-token 20.5 ms vs
   370.8 ms on the 5090; and retiring finished requests from a running batch is stock
   sglang continuous batching, not something we would have to add. Their PR is adding
   what our scheduler already does, on one endpoint (`/big_batch/completions`), and it
   is **still open and unmerged**, so the 20.6% / 41.2% are a proposed patch's claim
   about their own main rather than a property their engine ships.
   What is genuinely untested on our side is narrower: the *specific* pathology of a
   heavily skewed finish distribution (75/25) at bsz 96-160. ShareGPT's length mix is
   milder than that. Worth one run to confirm the scheduler holds, not worth treating
   as a known deficit.
   One detail from that PR worth carrying over on its own merits: they disclose that
   compaction changes output at `temperature>0` and is byte-identical only at
   `temperature=0`. Any batch-compaction change here needs the same statement.
2. **Warm state-resume as its own lane.** Their three-tier state cache targets the one
   axis where a recurrent stack beats an attention stack by orders of magnitude rather
   than percent: a constant-size state is cheap to persist, so a returning session
   should skip prefill entirely. We publish `serving_scale` and `bsz_throughput`; a
   "resume from cached state vs cold prefill" TTFT lane is absent.
3. **TTFT under queueing pressure.** Four of their PRs are admission-control and
   TTFT-tail work. We publish decode ladders; TTFT-under-concurrency is unmeasured here.
4. **Batch-size-dependent numerics, disclosed rather than discovered.** PR #19 states
   that on their unmodified main, "running the identical prompt at different batch sizes
   already produces bit-different logits". Most fp16 batched stacks share this and
   almost none say so. We should establish whether ours does and publish the answer
   either way — F0069's conventions are the right home for it.

## Consequence for reuse

**Nothing here can be vendored, forked or adapted.** No licence means all rights
reserved, and it is derived from Apache-2.0 albatross, which leaves the derivative's
status ambiguous rather than permissive. This is worth writing down so the option stops
being re-raised: even setting aside ADR-0004, the legal answer is no.
