---
doc_kind: finding
finding_id: F0077
title: "RWKV_SPEC completion push: the draft-state rollback was off by one token (accept-length 1.2 vs an independently measured alpha of 0.7 — every partial accept desynced the draft permanently, invisible to the correctness gate), plus a hand-rolled TARGET_VERIFY CUDA graph and a prefill merge-safety fix; after all three: gate 10/10 on 1.5B AND 7.2B, long-form accept 2.5-3.5, 1.5B round 35.4→12.5 ms, but net is still 0.78x (1.5B) / 0.85x (7.2B) median — only high-accept workloads (7.2B math, 3.52) cross 1x"
last_verified_commit: "sglang fork rwkv7-spec-decode @ b324c5bc3"
discovered_by: lead (spec completion push), 2026-07-31
severity: info
status: open
related: [F0029, F0030, F0046, ADR-0006]
---

# Finding F0077: the accept-length collapse was a one-token rollback bug, not a weak draft

## Context

The user's directive was to finish the speculative-decoding experiment properly before
saying anything publicly. F0046 left it at: correctness done (10/10), draft graphed,
net 3.5-4.5x SLOWER than plain. This session reconstructed the runtime (the tower main
container is gone — recipe below), profiled with per-phase timers, and attacked the
three dominant costs in order. Two were engineering; the third was a real logic bug
that had silently capped every measurement since the build.

## The runtime reconstruction (the tower recipe, now written down)

`lmsysorg/sglang:dev-cu13` + `PYTHONPATH=<fork worktree>/python` + **the delivery
repo's overlay `models/rwkv7.py` and `rwkv7_kernels/` grafted over the lean upstream
port** — the overlay model file is deliberately cross-version (0.5.10 AND main) and
carries the fast-linear stack (`RWKV_FAST_LINEAR=1 RWKV_GEMV_AUTOTUNE=1` etc.), whose
`gemv_mb` routing is what makes the verify's M=K projections bit-identical to M=1
decode. Without the graft the gate is 9/10 (the F0031 near-tie flip at a reduction-
order boundary); with it 10/10. Servers: `--speculative-algorithm RWKV_SPEC
--speculative-draft-model-path <0.1B> --speculative-num-draft-tokens 4
--disable-overlap-schedule --max-running-requests 32`.

## The bug: snapshots one token behind the committed sequence

The draft chain ran K-1 replays with state snapshots taken BEFORE each step:
`snaps[0]` was the state before consuming `t_last`, and the final state never
consumed the last proposal at all. So on EVERY rollback — and on full accepts, where
the code assumed "current state is already right" — the draft resumed one committed
token short. Once desynced it never resynced (nothing re-feeds missed tokens), so
after the first partial accept the draft was permanently proposing from a lagged
context. The correctness gate cannot see this: draft state shapes proposals, never
committed tokens. The tell was quantitative: measured accept-length sat at 1.2-1.6
while an independent probe (HF stack, 0.1B teacher-forced over the 1.5B's own greedy
trajectories, `bench/alpha_probe.py`) put per-token agreement at 0.685-0.955 on the
same prompts — the F0029 number, reconfirmed. Theory says accept ≈ 1+α+α²+α³ ≈ 2.5
at α=0.685; the worker measured 1.2. That factor-2 gap is the bug's signature.

Fix: K replays over `[t_last, d_0..d_{K-2}]`, snapshot AFTER each replay, so
`snaps[J]` is exactly the state for J accepted drafts, all J=0..K-1 including full
accept (the K-th replay's prediction is unused; its state is the point). Measured
accept on long-form prompts went 1.2-1.6 → 2.49-3.52, matching theory per-prompt.

## The other two fixes

- **Hand-rolled TARGET_VERIFY CUDA graph** (same idiom as F0046's draft graph;
  scratch pool slot freed after capture — holding it trips the scheduler's idle-time
  leak check). 1.5B verify 17.9 → 5.9 ms. The draft chain also now feeds itself
  on-GPU (argmax into the graph's static input buffer; one host sync per chain):
  draft phase 15.1 → 3.5 ms. Round total 35.4 → 12.5 ms.
- **Prefill merge-safety**: prefill results must carry `next_draft_input`
  (NGRAMWorker's contract) or a freshly prefilled request merging into an active
  decode batch dies in `NgramVerifyInput.merge_batch`. Latent until requests overlap.

## Measured (RTX 5090, fp16, greedy, 256-token long-form set, server e2e)

| | spec-off | spec-on K=4 | ratio | accept |
|---|---|---|---|---|
| 1.5B median (5 prompts) | 293.6 | 227.6 | 0.78x | 2.49-3.01 |
| 7.2B median (5 prompts) | 91.2 | 77.7 | 0.85x | 2.21-3.52 |
| 7.2B math (best case) | 89.9 | 93.7 | **1.04x** | 3.52 |

Gate: 10/10 token-identical at both sizes, graphs on. Per-phase (marginal, 7.2B):
verify 30.0 ms, draft 4.1 ms, tail 2.2 ms — the verify costs ~2.7 decode-steps'
time for ~2.8 committed tokens, which is the whole remaining story.

## What stands between here and a real net win

The verify forward costs ~1.7 (1.5B) to ~2.7 (7.2B) decode-steps per round even
graphed. Break-even needs accept ≈ round/step: 2.7 (1.5B) / 3.3 (7.2B). High-accept
workloads already cross (7.2B math). Candidate attacks, unprofiled: why the graphed
M=K verify reads more than one weights-pass' worth of bandwidth; K=6-8 on
high-accept workloads; a 0.4B draft (alpha up, draft cost x3). Not guesses to act
on without measuring first — the lesson of this whole finding is that the measured
number (accept 1.2) was screaming which layer was broken, and profiling phases
before attacking saved the effort from going to the wrong place twice.

## Cross-references

[[F0029]] (alpha, reconfirmed today) · [[F0030]] (HTTP prototype ruled out) ·
[[F0046]] (Strategy B build) · fork branch `rwkv7-spec-decode` @ b324c5bc3 ·
`bench/alpha_probe.py`, `bench/spec_speed_long.py` (committed beside this).
