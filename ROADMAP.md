# Roadmap

**Purpose**: this file is forward-looking (what's active, what's queued, what's long-term).
[`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) is backward-looking (what's been delivered, with
evidence, for the bounty's contribution-scoring process). Read that one to see what's done;
read this one to see what's next. Last updated **2026-07-08**.

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

## Active right now (2026-07-08, updated mid-run)

Desktop-tier Qwen3.5-9B concurrency search is **done** (F0049, RWKV-7 peak +27.0% to +30.5%
over Qwen3.5-9B's memory-ceiling-terminated peak). The PII/infra-identifier leak found during
the 2026-07-07 pre-push audit is **fully remediated** (`git filter-repo` + force-push, verified
zero occurrences across full history) — see `memory/feedback-scrub-infra-identifiers-precommit.md`
if you have access to it. The high-bandwidth epilogue-fusion kernel (F0052) is **done and
shipped**: byte-exact gate passed on both L4 and H100, real (if modest) win — H100 bsz1
+2.82% (393.2→404.3 tok/s), L4 +0.24% (noise-level, expected for a small fusion), F0051's H100
GPU-busy ceiling estimate updated 0.69×→0.71×, committed (`3f9053e`) and pushed with
`RWKV_FUSED_SQRELU` left default-OFF.

Two long-running jobs are the current critical path, both healthy and multi-hour by nature
(500 MATH500 problems × 64 samples = 32,000 generations each):

- **3090 box**: 7.2B int4-GPTQ MATH500 avg@64 pipeline. Calibration (192/192 Hessian shards,
  4-way sharded to fit GPU memory) and asymmetric quantization are **done** — a real blocker
  was found and fixed mid-pipeline (the pipeline's scratch directory had a stale pre-F0043
  copy of `gptq_w4.py` with no `--asym` support; refreshed from the canonical repo, see
  `memory/project-rwkv-w4-quant.md`). Oracle sanity check passed. **Currently running the
  fp16 baseline avg@64** (the reference point sym/asym get compared against) — this is the
  direct follow-up F0043 called for, and decides whether a full K-quant rewrite (Stage 2) is
  worth building at all. Not done yet; sym/asym runs still to follow after this baseline.
- **Tower (RTX 5090)**: Qwen3.5-2B MATH500 avg@64 — **DONE, both modes**. The first full run
  (chatml_thinking) came back with **93.15% truncated generations** — a real methodology bug,
  not a result: the harness inherited RWKV's own sampling defaults (`top_p=0.28`) instead of
  Qwen3.5's documented thinking-mode settings, and had no `presence_penalty` support at all
  (Qwen3.5's docs call for `presence_penalty=1.5` in thinking mode). Fixed (added
  presence_penalty support, corrected sampling params per-mode), re-piloted at several token
  budgets to right-size things before committing to a full re-run. **Non-thinking mode**: 67.63%
  accuracy, 0.99% truncated (Qwen3.5-2B's own documented default mode — a real,
  unfavorable-to-RWKV result vs RWKV-7 1.5B's published 40.4-40.6%, reported honestly).
  **Thinking mode**: 47.72% accuracy, 52.4% truncated, at the capped mnt=16384 budget (a
  deliberate fallback from the model's documented mnt=32768 default after pilot data showed a
  heavy-tailed truncation curve that made the full 32768 budget impractically expensive — a
  projected 16+ hours for one measurement). Both numbers beat RWKV-7 1.5B's own avg@64
  (40.4-40.6%) — reported the same way the concurrency wins above are, per this project's
  claims-need-numbers discipline. Full account, including the pool-pressure/retraction dynamic
  that made the actual thinking-mode run take 13.3h against a 9.6h pre-run estimate: F0053. This
  job is no longer active — Qwen3.5-9B was deliberately not attempted this session (2B alone
  took ~17.5 GPU-hours across both modes); see F0053 "What's not done" for the explicit
  disclosure.

This Mac is idle for GPU-heavy work by design (the Apple-Silicon Qwen3.5-MATH500 attempt hit
memory pressure and was stopped on direct instruction, not retried) — currently used for
orchestration/monitoring only, including a small local Rust dashboard
(`../rwkv-lab-dashboard/`, kept outside this repo) that watches both boxes' GPU/process/log
state live rather than requiring manual SSH checks.

## Queued next (once the above lands)

1. **7.2B int4 MATH500 result → the actual go/no-go on w4 Stage 2** (K-quant mixed precision,
   the real "close to Q4_K_M" work — super-block=256/sub-block=32, 6-bit scale+min, mixed
   4/6-bit precision by tensor role). **Not started yet** — this is a real decision point, not
   a formality: F0043's 1.5B data showed asymmetric alone closes only 27-35% of the fp16 gap
   (less on the metrics that matter most), and MATH500 avg@64 collapsed far more than lambada
   suggested it would, hinting the problem may not be purely a bit-budget one that a fancier
   encoding fixes. The fp16/sym/asym 7.2B avg@64 numbers currently running on the 3090 are
   exactly the data this decision needs — "measure first, don't build blind."
2. **Qwen3.5 comparison → the actual BENCHMARKS chapter.** All the pieces (cloud speed, desktop
   speed, Apple Silicon speed, cloud accuracy, oracle-gate, and now the corrected MATH500
   avg@64 numbers, both modes) have landed in `docs/BENCHMARKS.md`/`.zh-CN.md` §13.4 (F0053) —
   this item is **done** for the 2B tier. Qwen3.5-9B's own MATH500 avg@64 (thinking +
   non-thinking) remains not attempted (deliberately deprioritized this session, see F0053) and
   is the one piece still missing from a fully-complete chapter.
3. **High-bandwidth kernel work, next increment.** F0052 shipped a real but modest win; the
   next candidate per F0051/F0052's own ceiling analysis is 128-bit vectorized GEMV loads —
   not yet attempted, no agent currently assigned.
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
- ~~Is the epilogue-fusion-into-GEMV kernel work (in flight) safe to ship?~~ **Resolved
  2026-07-07**: yes — F0052 shipped, byte-exact gate, real modest win, see "Active right now."
- Does 7.2B int4's sym/asym MATH500 avg@64 pattern match 1.5B's (large collapse, asymmetric
  helps some but not enough), or does it hold up better at scale? Still running.
- ~~Does Qwen3.5-2B thinking mode's accuracy actually beat non-thinking's 67.6% once the capped
  mnt=16384 budget finishes?~~ **Resolved 2026-07-08**: no — thinking mode finished at 47.72%
  avg@64 (52.4% truncated on the full 500×64 run), well below non-thinking's 67.63% (0.99%
  truncated). Both still beat RWKV-7 1.5B's own 40.4-40.6%, but non-thinking is the stronger
  and cleaner (lower-truncation) of the two Qwen3.5-2B numbers, consistent with it being the
  model's better-converged mode at a bounded token budget — see F0053.
