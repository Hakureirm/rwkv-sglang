---
doc_kind: adr
adr_id: 0003
title: "M1 scope & slicing into independently-gated increments"
status: accepted
date: 2026-06-30
last_verified_commit: (initial)
supersedes: []
superseded_by: []
---

# ADR-0003: M1 scope & slicing

## Context
The M1 design workflow produced a complete file-by-file plan
(`docs/design/m1-implementation-plan.md`). M1 = serve RWKV-7 0.1B in sglang
v0.5.10.post1 and pass the greedy-match-vs-numpy-oracle gate. Per ADSD Delta 11
("slice an ADR-phase to the smallest independently-gated increment"), M1 is sliced
so each increment has its own falsifiable gate and can be authored/audited alone.

## Options considered (key decisions)
1. **Weights**: (a) load BlinkDL `.pth` directly in sglang via a custom loader, vs
   **(b) fla-format + offline BlinkDL→fla converter** (CHOSEN). (b) matches PR#41060's
   no-remap fla loader and isolates conversion correctness from kernel correctness
   (oracle compares converted-weights sglang vs BlinkDL-.pth numpy).
2. **Correctness vs speed first**: (a) port albatross-grade kernels up front, vs
   **(b) correct-then-fast** (CHOSEN): use fla triton chunk/recurrent (correct) + torch
   for everything elementwise; defer albatross-parity kernel to M3.
3. **State plumbing**: (a) custom RWKV state pool, vs **(b) reuse sglang Mamba/hybrid-
   linear plumbing** with `full_attention_layer_ids=[]` (CHOSEN) — least new code,
   gets continuous batching for free.
4. **Prefix cache / CUDA graph**: defer both to M2 (per-request fresh state +
   `--disable-cuda-graph` in M1).

## Decision
Adopt the plan's §0 scope decisions. Slice M1 into four independently-gated increments:

| Inc | Deliverable | Gate (falsifiable) |
|---|---|---|
| **M1a** | Vendor 8 fla rwkv7/dplr triton files into a working sglang copy (import-fixed) | each kernel's output matches the fla original on random tensors (decode + chunk), within fp tol |
| **M1b** | `tools/convert_rwkv7_blinkdl_to_fla.py` + numpy-oracle baseline | converter emits all expected fla keys/shapes (transposes correct); `bench/oracle_numpy.py` on 0.1B .pth prints ref logits + greedy tokens |
| **M1c** | config + state params + `models/rwkv7.py` + `rwkv7_backend.py` + 4 wiring edits | sglang **boots** the model (dummy/converted weights), MambaPool allocates `temporal (12,64,64) fp32` + two `(768,1)` conv, backend selected; generates without crashing |
| **M1d** | `load_weights` + end-to-end correctness | single-seq greedy **matches numpy oracle** (max\|Δ\|/std<1e-2 + token-for-token continuation); batch-2 isolation; mixed prefill+decode both correct |

M1 is DONE when M1d passes. Sequencing: M1a ∥ M1b (independent) → M1c → M1d.

## Consequences
### Positive
Each increment is small, separately verifiable, and de-risks one failure class
(kernel port / weight conversion / wiring / numerics) in isolation. Correct-then-fast
means the oracle gate is reachable before any kernel-tuning effort.
### Negative / Risk
fla-format + converter adds a conversion step (mitigated: M1b gate). The fla kernels
carry an upstream "maybe-misaligned vs BlinkDL" warning → the oracle is numpy/BlinkDL,
and M1d will surface any kernel/numeric divergence as a hard fail (good).

## Outcome (2026-06-30, HEAD 700e554)
**M1 DONE — all increments passed.** M1a kernels gate (decode ~1.3e-6, chunk
≤8.1e-3), M1b converter (399 tensors), M1c boot, **M1d EXACT greedy-match** vs the
numpy/BlinkDL oracle (lead-verified, `bench/verify_m1d.py`). See [[F0005]].

## Cross-references
ADR-0001, ADR-0002 · `docs/design/m1-implementation-plan.md` · [[F0002]] [[F0003]]
[[F0005]] · gates: `bench/oracle_numpy.py`, `bench/verify_m1d.py`.
