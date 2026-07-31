---
doc_kind: finding
finding_id: F0077
title: "RWKV_SPEC completed to a real net win: the draft rollback was off by one token (accept 1.2 vs alpha 0.7, permanent desync invisible to the correctness gate), the verify's gemv_mb burned M weight passes for bit-invariance (fixed by a shared-weight variant, same bit contract, 2-3.9x on the kernel) — final: 7.2B long-form median 1.62x (K=6), math 2.44x (K=8), 1.5B 0.87x (draft overhead dominates small targets), gate 10/10 at every size and K"
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

## The second structural find: gemv_mb traded M weight passes for invariance

With the desync fixed, the graphed verify still cost ~2.7 decode-steps (7.2B:
30.0 ms marginal). The bandwidth arithmetic pointed at `gemv_mb` itself: its grid
puts the row index in `blockIdx.y`, so every one of the M=K rows re-reads the whole
weight matrix — 4 full passes over ~12 GB of projection weights ≈ 32 ms, which IS
the measured verify cost. The fix is `gemv_mbs` (same .cu): one block per output
tile, the weight fragment loaded once and applied to all M rows held in registers,
each row's fp32 reduction sequence unchanged — so row-for-row bit-identity with
`gemv_m1` is preserved (the harness now gates BOTH kernels across every autotuner
config; compute-sanitizer clean) while weight bandwidth drops M-fold. Kernel
microbench at M=4: 2.0-3.9x. `RWKV_GEMV_MB_SHARED=0` restores the per-row kernel.

One honesty note: a single illegal-memory-access crash was observed on the 7.2B
rig right after the first deployment (request #2 of a run), and then never again —
not under CUDA_LAUNCH_BLOCKING, not across 40+ subsequent gate requests at both
sizes. The kernel is standalone-clean under compute-sanitizer memcheck across all
shapes and configs. Treated as an open intermittent (possibly unrelated memory
pressure), recorded here rather than hidden; racecheck on the serving path is the
follow-up if it recurs.

## Measured (RTX 5090, fp16, greedy, 256-token long-form set, server e2e)

| | spec-off | spec-on K=4 final | ratio | accept |
|---|---|---|---|---|
| 1.5B median (5 prompts) | 293.6 | 255.8 | 0.87x | 2.49-3.01 |
| 1.5B best (math) | 293.5 | 286.4 | 0.98x | 3.01 |
| 7.2B median (5 prompts) | 91.2 | **143.7** | **1.58x** | 2.21-3.52 |
| 7.2B best (math) | 89.9 | **169.2** | **1.88x** | 3.52 |

Gate: 10/10 token-identical at both sizes, all graphs and the shared kernel on.
Marginal per-phase (7.2B): verify 30.0 → 13.1 ms, round 36.3 → 19.5 ms.

The shape of the result is exactly ADR-0006's economics: the draft costs the same
~4 ms/round regardless of target size, so the win concentrates where target steps
are expensive. At 1.5B the draft+orchestration overhead eats the acceptance profit
at alpha≈0.7; at 7.2B every prompt class clears 1.2x and reasoning/code clear 1.7x.

## K sweep (7.2B, measured; gate 10/10 at every K)

| K | story | explain | math | code | history | median |
|---|---|---|---|---|---|---|
| 4 | 1.23x | 1.35x | 1.88x | 1.77x | 1.57x | **1.58x** |
| 6 | 1.11x | 1.28x | 1.96x | 2.03x | 1.62x | **1.62x** |
| 8 | 1.04x | 1.18x | **2.44x** (219.7 tok/s, accept 6.29) | 2.01x | 1.54x | 1.54x |

Textbook workload dependence: bigger K trades low-accept prompts (story falls
toward 1x) for reasoning/code (math 2.44x at K=8). K=6 is the best fixed
median; adaptive K is the obvious follow-up. A 0.4B draft was considered and
rejected by arithmetic at these settings: its ~2.5x draft cost (+~13 ms/round
at K=8) cannot be repaid by the at-most (K+1 - 6.29) additional accepted
tokens per round. Draft-phase trim (~1 ms/step fill/alloc overhead) remains
open. The 1.5B case stays sub-1x at this alpha regardless of K.

## Cross-references

[[F0029]] (alpha, reconfirmed today) · [[F0030]] (HTTP prototype ruled out) ·
[[F0046]] (Strategy B build) · fork branch `rwkv7-spec-decode` @ b324c5bc3 ·
`bench/alpha_probe.py`, `bench/spec_speed_long.py` (committed beside this).
