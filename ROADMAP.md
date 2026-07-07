# Roadmap

**Purpose**: this file is forward-looking (what's active, what's queued, what's long-term).
[`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) is backward-looking (what's been delivered, with
evidence, for the bounty's contribution-scoring process). Read that one to see what's done;
read this one to see what's next. Last updated **2026-07-07**.

The underlying goal (see [`docs/adr/0001-scope-and-wedge.md`](docs/adr/0001-scope-and-wedge.md)):
make the sglang adaptation good enough that nobody — including BlinkDL — can point at an
obvious gap. Bo Peng's own bounty post (reposted 2026-07-06) is the actual acceptance bar;
the 7-item list below is quoted from it directly, not paraphrased, so status claims can be
checked against the source.

## Status against the 7 bounty requirements

| # | Requirement (Bo's own wording, condensed) | Status |
|---|---|---|
| 1 | Match new-Albatross/new-RWKV-LM perf/speed/accuracy/VRAM across bsz, all 4 frameworks + transformers training | **sglang only** (this repo's scope, see ADR-0001): close at low bandwidth (L4 0.90×), still behind at high bandwidth (H100 0.65× post-F0051, B200 untested post-fix) — **long-term iterative, see "Long-term" below**. Training / other 3 frameworks out of scope for this repo. |
| 2 | Beat Qwen3.5 at matched size+quant, across cloud/desktop/mobile/embedded | **Cloud + desktop tiers: done, RWKV wins peak concurrency at both size tiers on both GPUs.** Apple Silicon tier: speed done (split result, honest), accuracy partial (compression only, MATH500 blocked on this Mac's memory). Mobile/embedded: not attempted (no hardware). |
| 3 | transformers PEFT/RL trainability | Out of scope for this repo (sglang inference/serving track, not the HF training track). |
| 4 | vLLM/sglang dynamic batching + chunked prefill + state cache w/ good hit rate | **Done** for sglang (MambaRadixCache enabled, ~98% hit rate on realistic reuse load — see CONTRIBUTIONS.md req #3 row, note the numbering there needs a fix, see below). |
| 5 | Pascal+/AMD/domestic cards; PP+TP inference; zero2/3 training; autotune | PP+TP: **done**, verified on main under cuda-graph (F0036). Autotune: **done** (F0023/25/27, 11-card matrix). Pascal/AMD hardware validation: **not done** (task #9, pending — this is a real, named gap, not forgotten). Domestic cards: only reachable via the yuueang Ascend NPU community contribution channel, not directly built by us. |
| 6 | w8/w4 quant: VRAM down, faster than w16 on common cards (old cards too), near-Q4_K_M accuracy | w8: **done, lossless**. w4: real progress, not yet at the bar — symmetric GPTQ −3.34pt lambada, asymmetric refinement (F0043) closes 27-35% of the gap depending on metric, **but MATH500 avg@64 still shows a large reasoning-chain collapse at 1.5B** (0.40→0.22 even with the improvement). 7.2B's version of this question is running right now (see "Active now"). |
| 7 | Preliminary speculative decoding; DFlash etc. as follow-up | **Correctness done** (10/10 gate, Strategy B, F0046). Speed: real partial win (1.5-1.6× on the draft step), not yet net-positive overall (still 2.6-4.5× slower than spec-off). DFlash itself researched and explicitly deferred (ADR-0007) — consistent with Bo's own "as follow-up" framing. |

Full evidence trail for every ✅/◑ above lives in `docs/findings/` (50 numbered reports as of
this writing) and `CONTRIBUTIONS.md`. **Known documentation debt**: `CONTRIBUTIONS.md`'s
requirement numbering predates Bo's exact reposted list and doesn't have its own row for
requirement #2 (Qwen3.5) at all — that file needs a refresh pass; not done as part of this
roadmap write-up to avoid scope creep, tracked here so it isn't lost.

## Active right now (2026-07-07, updated after the history-scrub push)

Desktop-tier Qwen3.5-9B concurrency search (previously listed here) is **done** — F0049 closed
out with a confirmed-flat RWKV-7 peak beating Qwen3.5-9B's memory-ceiling-terminated peak by
+27.0% to +30.5%; see the requirements table above. Repo also had a PII/infra-identifier leak
(a dev box's real username and SSH alias, baked into committed benchmark JSON and a README)
found and fully remediated via `git filter-repo` + force-push this same day — see
`memory/feedback-scrub-infra-identifiers-precommit.md` if you have access to it; not repeated
here since ROADMAP is forward-looking, not an incident log.

Three independent workstreams running in parallel across the two GPU boxes and a rented
high-bandwidth GPU (see "a note on hardware access" below):

- **Tower (RTX 5090)**: Qwen3.5-2B/9B accuracy evaluation (MATH500 avg@64 + compression rate),
  matched methodology to RWKV-7's own numbers. Compression already in hand (Qwen3.5-2B 0.6729
  bpb vs RWKV-7 1.5B's 0.6085 — RWKV ahead). MATH500 avg@64 (2B, chatml_thinking) in progress,
  ~80% through its 32,000-rollout run as of this update; 2B non-thinking + 9B still to follow.
- **3090 box**: 7.2B int4-GPTQ MATH500 avg@64 (symmetric + asymmetric), the direct follow-up
  F0043 called for — decides whether a full K-quant rewrite (Stage 2) is worth building at all.
  Currently in the Hessian-calibration phase (prerequisite to quantizing); a Monitor is armed
  to pick the pipeline back up through quantize→eval once calibration finishes.
- **High-bandwidth GPU (rented per-job, not a standing box)**: F0052, continuing the Albatross
  bandwidth-gap investigation past F0051 — epilogue-fusing the FFN `relu(.)**2` activation
  directly into its preceding GEMV's store (F0051's identified next lever). A first attempt at
  this produced a complete-looking kernel+gate+model-wiring diff but the agent's session died
  before verifying it on real hardware; redispatched to actually build, gate, benchmark, and
  commit-or-roll-back on a rented card. Highest-blast-radius kernel work in this project, so
  gated extremely conservatively (default-OFF env flag either way — see F0051/F0052 discipline).
- **This Mac**: idle for GPU-heavy work by design — a parallel Qwen3.5-on-MLX accuracy pass ran
  into real memory pressure and was stopped on direct instruction; not being retried. The Apple
  Silicon tier's accuracy story stops at "compression rate only, honestly labeled" until this
  Mac has headroom to try MATH500 again, or a different Apple Silicon machine is available.

## Queued next (once the above lands)

1. **7.2B int4 MATH500 result → the actual go/no-go on w4 Stage 2** (K-quant mixed precision).
   This is a real decision point, not a formality — F0043's own data suggests the reasoning
   collapse may not be a pure bit-budget problem, so "measure first, don't build blind" applies.
2. **Qwen3.5 comparison → the actual BENCHMARKS chapter.** All the pieces (cloud speed, desktop
   speed, Apple Silicon speed, cloud accuracy, oracle-gate) need assembling into one coherent,
   cross-referenced chapter in `docs/BENCHMARKS.md`/`.zh-CN.md` rather than living scattered
   across 9+ findings docs and a memory log — this has been explicitly deferred multiple times
   in favor of finishing the underlying measurements first; it shouldn't be deferred again once
   the tower/3090 jobs above land.
3. **High-bandwidth kernel work, next increment** — whatever F0051's epilogue-fusion attempt
   concludes (a real fusion + measured delta, or a specific documented blocker) determines
   whether the 128-bit vectorized GEMV load idea gets tried next, or whether a different lever
   (per F0051's honest ceiling analysis) is more promising.
4. **Spec-decode speed profiling** — three unverified hypotheses on record (per-layer state
   `.clone()` cost, target-verify overhead, per-round Python orchestration) from F0046; nobody
   has profiled which actually dominates. Lower priority than the above three (this project's
   standing full-spectrum-over-single-stream doctrine puts concurrency/large-batch work ahead
   of single-stream spec-decode speed), but not dropped.

## Long-term / won't "complete" in the normal sense

- **Task #5 — raw-kernel parity with Albatross at matched precision.** Explicitly iterative;
  F0051's own honest ceiling analysis says even a fully-fused decode step only gets H100 to
  ~0.69× of Albatross, and Albatross itself isn't at its own ceiling yet either (per Bo directly)
  — this is a moving target pursued opportunistically, not a task with a finish line.
- **Task #9 — Pascal/AMD hardware validation + per-model sparsity re-verification.** Named,
  tracked, genuinely not started. Needs either borrowed/rented Pascal and AMD hardware or a
  clear-eyed decision to accept this as a scoped gap in the final write-up.
- **Upstream PR shepherding** (#30115 model adaptation, #30095 PP bugfix) — both open,
  maintainer-gated CI (a permissions issue on their end, not a code issue on ours, already
  diagnosed — see `memory/project-upstream-model-pr.md` if you have access to it). Nothing
  further to do here except respond quickly if a maintainer engages; do not re-attempt the
  `/tag-and-rerun-ci` self-service path, confirmed to be a no-op for external contributors.
- **Competitive monitoring** — standing practice, not a task with an end state: check
  vllm-rwkv / vkwr / hf-adapter / rwkv-mobile / the newer names from Bo's reposted reference
  list (shiroko98/vllm, MollySophia/nano-vllm, rwkv-rs/helicopter, RafaelUI's MLX-training
  repos, etc.) roughly weekly, or before anything gets published externally.

## A note on hardware access

This project uses three kinds of compute: two owned/leased boxes (a workstation-class GPU and
a desktop-class GPU, referred to elsewhere in this repo by their role, not a vendor name) held
long-term, this Mac for the Apple Silicon work, and short-lived rentals of specific GPU
architectures (e.g. a high-bandwidth datacenter card) when a question specifically needs
hardware this project doesn't own — always framed in committed material as "a high-bandwidth
GPU" or "each card's own real hardware," never naming the rental provider, per this project's
standing publication rule. If you're reading this file to understand *why* some findings cite
specific architectures (H100, B200) without a machine name attached to this project's usual
two-box setup, that's why.

## Known open questions (don't assume these are resolved)

- Does 7.2B's int4 MATH500 collapse the way 1.5B's did, or hold up? (Running now.)
- Is the Albatross reference itself using updated (faster) numbers than what this project's
  §7 comparison table cites? **Partially checked (2026-07-07):** `docs/BENCHMARKS.md` §7's
  actual published tables don't create a live contradiction — the main table is explicitly
  1.5B single-stream on matched cards (including a real RTX PRO 6000 row, 457.4 tok/s, and
  our own 5090 re-tune work), and the only 7.2B content there is a *relative* stock→re-tuned
  improvement table, not an absolute 7.2B-vs-Albatross large-batch number. The "10,250 tok/s
  batch decode / 11,289 tok/s bsz1 prefill / 5,848 tok/s bsz320" 7.2B figures this project has
  on record for Albatross live only in local memory (`project-rwkv-vllm-bounty.md`), never
  published here, so there is nothing in the public docs for Bo's newer 13,000/17,000 figures
  to contradict. What's still unresolved: whether to add Bo's newer number to our own docs at
  all — doing that responsibly needs the bounty repost's verbatim text (not a paraphrase-of-a-
  paraphrase; this project's own claims-need-numbers rule applies just as much to citing a
  third party's number as to citing our own) plus a judgment call on card-comparability that
  even Bo's own post reportedly hedges ("5090 would read lower, larger batch would read
  higher, roughly comparable" — again, from memory, not verified against source). Not
  resolving further until someone has the primary source in hand; deprioritized, not blocking.
- Is the epilogue-fusion-into-GEMV kernel work (in flight) safe to ship, or does the call-site
  inventory reveal it's riskier than F0051 hoped? Not yet known.
