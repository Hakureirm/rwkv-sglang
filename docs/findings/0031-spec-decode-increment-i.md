# F0031 — Chain speculative decoding increment (i): functional in-engine worker; gate 9/10 token-identical, the 1 flip root-caused to M-shape GEMM reduction order with measured evidence

**Date:** 2026-07-05 · **Status:** increment (i) FUNCTIONAL (mechanism verified; ε-exactness framing below) · **Design:** ADR-0006 · **Prior:** F0029 (viability, α=0.738), F0030 (HTTP prototype ruled out)

## What was built

`sglang_overlay/sglang/srt/speculative/rwkv_chain_worker.py` — a bespoke speculative
worker driven by sglang v0.5.10's V1 (non-overlap) scheduler, registered as
`--speculative-algorithm RWKV_CHAIN`. sglang's EAGLE infrastructure verifies token
TREES against a KV cache and rolls back by not committing pages; RWKV-7 has neither,
so the worker replaces the entire draft/verify machinery while keeping the scheduler
contract (flat accepted-token tensor + `accept_length_per_req_cpu`; the worker appends
`output_ids` and finish-checks itself, matching EAGLE-V1 semantics).

Key structural decisions (each one earned by a boot-debug cycle on the 3090):

- **The draft gets its OWN state pool.** EAGLE/StandaloneWorker share the target's
  `req_to_token_pool` + `mamba_pool` because an EAGLE draft is a head of the target.
  With BOTH models recurrent, sharing would alias their states — the draft
  `TpModelWorker` is constructed with `req_to_token_pool=None` so the kv-cache mixin
  builds a fresh `HybridReqToTokenPool`/`MambaPool` for the 0.1B. Draft slots are
  allocated with shim request objects so the real `Req`'s pool bindings stay
  target-owned (`alloc`/`free`/`free_mamba_cache` take request objects and write
  `req_pool_idx`/`mamba_pool_idx` back).
- **Round loop invariant** (both pools, every round start): the committed sequence ends
  with `t_last`; each model's state has consumed up to `t_{last-1}`; `t_last` is
  pending. Round: K eager draft decode steps (one ForwardBatch, mutated per step) →
  ONE target extend over `[t_last, d0..d_{K-2}]` with `CaptureHiddenMode.FULL` →
  per-position lm_head argmax → longest-prefix accept →
  **J==K: commit-free for BOTH models** (the verify's committed `final_state` IS the
  next round's invariant; the draft's own state likewise) — zero restores, zero extra
  forwards; **J<K:** O(1) snapshot restore on both slots + one commit-extend over
  `[t_last, d0..d_{J-1}]` on each model (the bonus token stays pending).
  Forwards/round: draft `K + [J<K]`, target `1 + [J<K]` (~1.7 at α=0.738, K=4).

## Gate result (bench/spec_gate.py, 10 prompts × 128 greedy tokens, 1.5B target / 0.1B draft, K=4)

| metric | value |
|---|---|
| token-identical prompts | **9 / 10** |
| flips | **1 / 1280 tokens** (pos 71 of the math prompt; spec emitted 38595 where baseline emitted 59) |
| accepted length / round | **3.17 mean** (F0029 expectation ~2.98 — mechanism healthy) |
| eager speed vs eager baseline | 0.67× (predicted: increment (i) is launch-bound; speed is increment (ii)) |

Raw: `bench/results/spec_gate_run2_1.5b.log`, `bench/results/spec_gate_base_1.5b.json`.

## The flip, root-caused with numbers

Probe: same baseline server, same prompt, greedy 128 tokens with `top_logprobs_num=2`,
per-position top1−top2 logit gap:

```
smallest top1-top2 gaps (pos, gap_nats, top1_id, top2_id):
  pos  71  gap 0.005127  top1=59 top2=38595   <-- THE flip position & THE flip pair
  pos  69  gap 0.024414  top1=30322 top2=32234
  pos  96  gap 0.052734  top1=4450 top2=4706
  pos  23  gap 0.092163  top1=6128 top2=6651
```

The single flip happened at the sequence's **minimum** top-2 gap — 0.005 nats, ~5×
smaller than the runner-up — and the flipped pair is exactly (59, 38595). Cause: the
verify computes position logits through an M=K extend (cuBLAS GEMM) while the plain
baseline decodes through M=1 (GEMV); different reduction orders differ by O(1e-3) in
fp16, which flips argmax only when the true gap is comparable. A per-row lm_head
recompute (same [1,H]@[H,V] shape as decode) did NOT fix it → the divergence enters
via the layer projections' hidden states, not the lm_head.

## Exactness framing (honest, quantified)

- Every committed token is a target argmax **under a valid forward computation** —
  the acceptance rule is exact by construction; there is no "speculative drift".
- The residual ε (1/1280 here, only at near-ties) is the **same nondeterminism class
  the engine already has**: F0024 measured greedy MATH500 flipping 194↔196/500 from
  dynamic-batch composition alone (per-question identity noise floor 446/500), because
  batch shape changes cuBLAS reduction order the same way. A user running the plain
  server at bsz>1 experiences exactly this ε against bsz=1 output.
- **Exactness roadmap (increment (ii)):** our hand-written linear kernels can compute
  the K-row extend with M=1 reduction order per row (we control the kernel; cuBLAS
  doesn't expose this) — restoring bit-exactness against the M=1 baseline while
  keeping the one-forward-per-round structure, alongside the draft-decode +
  fixed-shape-K verify CUDA graphs that deliver the actual speedup.

## Boot-debug trail (for the next porter)

upstream `auto_choose_speculative_params` asserts all-or-nothing on the three spec
knobs (set `num_steps = K-1`: the topk==1 rule later bumps `num_draft_tokens` back to
K); `Scheduler.init_disaggregation` duck-types `draft_worker.model_config`;
`ScheduleBatch.prepare_for_decode` early-returns for spec algorithms (the worker owns
ALL per-round batch prep); `pkill -f "launch_server..."` matches the invoking ssh
shell's own argv (bracket-trick the pattern).
