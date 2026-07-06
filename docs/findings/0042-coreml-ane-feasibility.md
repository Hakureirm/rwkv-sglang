---
doc_kind: finding
finding_id: F0042
title: "CoreML/ANE feasibility probe for RWKV-7's WKV recurrence: FAIL — 0/47 ops ever prefer the Neural Engine at real model geometry (0.1B and 1.5B), even unrestricted; CoreML's own scheduler routes the whole recurrence to CPU, so the full converter build was stopped per the pre-registered gate"
last_verified_commit: "HEAD"
discovered_by: Sonnet 5 (agent-assisted), 2026-07-06
severity: info
status: open
related: [F0037, F0038]
---

# Finding F0042: CoreML/ANE feasibility probe — WKV recurrence never dispatches to ANE

## Context
The MLX port (F0037–F0041) is the complete Apple-Silicon GPU story. This was a probe into a
**second** Apple-Silicon path — CoreML targeting the Apple Neural Engine (ANE) — as a possible
"power-efficient/mobile-class" data point alongside GPU/CPU on the same M5 chip. `rwkv-mobile`
(`github.com/MollySophia/rwkv-mobile`, read-only reference clone, not part of this repo) was studied
for structure only: it compiles separate fixed-shape CoreML programs for prefill vs single-step
decode, threads WKV/token-shift state as explicit tensors (or CoreML `StateType`) in and out across
calls, and converts via `ct.convert(..., compute_units=ct.ComputeUnit.CPU_AND_NE)`. No code from it
was copied; the probe below is built from scratch in MIL (coremltools' IR builder — no torch, no
fla, matching `mlx_port/rwkv7_mlx.py`'s own "minimal deps, write it from scratch" discipline) using
this repo's own oracle math (`bench/oracle_numpy.py` `time_mixing`'s state update, replicated exactly
in the layout `mlx_port/rwkv7_mlx.py`'s `_wkv_scan_pure` already uses).

**The task's pre-registered stop condition:** convert just the WKV recurrence first and check with
CoreML's own device diagnostics whether it genuinely dispatches to ANE before building the full
per-layer converter, oracle-gating it, and reporting tok/s. "If it falls back to CPU for the core
recurrence, say so plainly and stop before building the full model." It fell back. This finding
documents the evidence and stops there, honestly, rather than building a converter whose headline
number would secretly be CPU-bound.

## Method
`coreml_port/probe_ane.py` builds the WKV delta-rule state update directly as a MIL program (no
surrounding Linear/LoRA projections — those are "any framework handles fine" per the task brief;
this isolates exactly the RWKV-specific numerics: sigmoid/exp decay, L2-normalize, and the
all-old-S-RHS state update `sa=-kk@S; S'=decay*S+(kk*a)*sa+k⊗v; y=r@S'`), at the **real** checkpoint
geometry (`mx.load(.../rwkv7-0.1b-fla)` → `attn.r_k.shape` = **H=12, D=64**; 1.5B = **H=32, D=64**),
fp16 (CoreML/ANE's native precision), converted with `compute_units=ct.ComputeUnit.CPU_AND_NE` (GPU
excluded, forcing the scheduler to pick between CPU and ANE only — the sharpest test of the
question). Ground truth comes from `coremltools.models.compute_plan.MLComputePlan` — CoreML's own
per-operation device-placement plan (`get_compute_device_usage_for_mlprogram_operation` →
`preferred_compute_device`, plus the theoretically-`supported_compute_devices` list), not a timing
proxy. Wall-clock (`CPU_AND_NE` vs `CPU_ONLY`, 200 reps) is recorded as corroborating, noisier
evidence on this shared M5 (load varies), not the primary signal — unlike wall-clock, the compute
plan's device assignment is a static compile-time decision and was verified identical across two
independent runs of the whole script.

Four probe shapes, plus adversarial positive controls to rule out "the probe methodology itself
is broken / this environment can't reach ANE at all":

1. `tokenshift` — the trivial lerp alone (sanity floor).
2. `wkv_step_T1` at 0.1B geometry (H=12,D=64) — the real bsz1 decode-step shape.
3. `wkv_chain_T4` at 0.1B geometry — 4 steps unrolled in one compiled function (a short
   sequential-dependency chain, the closest a fixed-shape ANE program can get to "prefill" for a
   recurrence this project has already ruled out reformulating as chunkwise-parallel algebra).
4. `wkv_step_T1` at 1.5B geometry (H=32,D=64) — does more heads change the scheduler's mind?
5. **Positive controls**: a big batched fp16 GEMM (1024×1024 @ 1024×1024 — the shape class of a
   *prefill*-chunk Linear projection), a fp16 conv2d (64ch, 56×56, 3×3 — historically THE canonical
   ANE op), and a **decode-shaped GEMV** (`[1,D]@[D,D]` — what RWKV's own r/k/v/o/ffn projections
   actually look like at bsz1) at both D=2048 (1.5B width) and D=768 (0.1B width).

## Results

Confirmed hardware first (this is a real M5, ANE genuinely enumerable — the negative result below is
not "no ANE present"): `MLComputeDevice.get_all_compute_devices()` → `MLNeuralEngineComputeDevice`
with `total_core_count=16` (matches Apple's published M5 16-core Neural Engine spec), plus GPU and
CPU devices, all real. Apple M5 (4P+6E, 10 cores), 32 GiB unified, macOS 27.0, coremltools 9.0.

**The WKV recurrence — every configuration, every op, CPU:**

| probe | ops (non-const) | preferred=ANE | preferred=CPU | under `ALL` too? | wall-clock ratio (CPU_ONLY / CPU_AND_NE) |
|---|---:|---:|---:|---|---:|
| tokenshift (lerp only) | 3 | 0 | 3 | same (3 CPU) | 0.98x |
| wkv_step_T1, 0.1B (H12,D64) | 30 | **0** | 30 | same (30 CPU) | 0.96x |
| wkv_chain_T4, 0.1B (H12,D64) | 105 | **0** | 105 | same (105 CPU) | 0.77x |
| wkv_step_T1, 1.5B (H32,D64) | 30 | **0** | 30 | same (30 CPU) | 1.02x |

Zero ops, across 168 total non-const operations spanning two model sizes and two chunk lengths,
ever get `preferred_compute_device = ANE` — including under **unrestricted** `compute_units=ALL`
(GPU and ANE both available; the scheduler still picks CPU for 100% of them). The wall-clock ratios
cluster around 1.0x (CPU_ONLY ≈ CPU_AND_NE, within this shared box's jitter) — consistent with
"CPU_AND_NE mode is, in practice, also just running everything on CPU," not contradicting it.

**Positive controls — the probe methodology is sound, and shows exactly the nuance that matters:**

| control | shape | preferred device | wall-clock ratio (CPU_ONLY/CPU_AND_NE) |
|---|---|---|---:|
| big batched GEMM | 1024×1024 @ 1024×1024 | **ANE** | **1.20x** (ANE genuinely faster) |
| conv2d | 1×64×56×56, 3×3 kernel | CPU | 0.98x |
| GEMV, decode-shaped | [1,2048]@[2048,2048] (≈1.5B width) | **ANE** (label) | **0.82x — CPU_AND_NE is *slower*** |
| GEMV, decode-shaped | [1,768]@[768,768] (≈0.1B width) | CPU | 1.02x |

Two things this establishes:
1. **The query methodology is trustworthy.** The big batched GEMM genuinely gets `preferred=ANE`
   *and* a real, corroborating 1.2x wall-clock win. If `MLComputePlan` always reported CPU
   regardless of reality on this machine, that control would have failed too — it didn't. So the
   30/30 and 0/30 splits above are a real, structural verdict about the WKV recurrence's op/shape
   economics, not a broken diagnostic.
2. **Even where the static plan says "ANE," bsz1 decode doesn't actually win.** The 2048-wide GEMV —
   the exact shape of a real r/k/v/o projection at bsz1 decode, i.e. the "surrounding linear layers"
   the task brief assumed "any framework handles fine" — is *labeled* `preferred=ANE`, but measured
   wall-clock is 18% *slower* through `CPU_AND_NE` than plain `CPU_ONLY`. A single, unbatched,
   unpipelined ANE dispatch doesn't amortize its own hand-off latency at this size. (Even the
   canonical conv op didn't clear the bar at a modest 56×56×64 shape.) This matters beyond WKV: it
   means a full decode step built from "ANE-eligible" Linears plus CPU-bound WKV would likely not
   even net out as a coherent ANE win on the *projections*, either.

## Why this is the honest read, not a probe artifact
The ops in question (`sigmoid`, `exp`, `matmul`, `reduce_l2_norm`, `mul`/`add` with broadcast) are
not unsupported by ANE — `supported_compute_devices` lists `ANE/CPU` for every one of them. This is
CoreML's **cost-based scheduler** choosing not to route them to ANE: the WKV state is `[H,K,V]` with
K=V=64 (a single head's slab is 64×64=4096 fp16 elements; 12–32 heads), and a per-token step is a
handful of tiny batched ops on that slab — nowhere near the size where ANE's fixed per-dispatch
overhead is worth paying, the same reason the 768-wide GEMV control also stayed on CPU while the
2048-wide one at least got the *label* (but still lost on wall-clock). This is the same
"decode is bandwidth/launch-bound, not compute-bound" story F0038 already measured on the MLX-GPU
side (bsz1 decode at 79% of the hard memory-bandwidth ceiling, many small per-layer launches costing
the other 21%) — a specialized batched-matmul accelerator has structurally little to offer a
workload whose fundamental unit is "read the weights once, do a little math per token." Sequential
recurrence at T=1 is the hardest possible shape for any accelerator built around large batched
compute; RWKV-7's own from-scratch design intentionally avoids reformulating the scan into
chunkwise-parallel matrix algebra (`mlx_port/README.md`: "out of scope for a correctness-first
port") specifically to keep summation order — and bit-exactness — pinned, so a large-single-tensor
version of the recurrence isn't on the table to try instead.

## Decision: stop here (feasibility gate = FAIL)
Per the task's pre-registered condition, the full per-layer CoreML converter (fixed-shape
decode/prefill programs, state threaded as explicit tensors, oracle-gating, tok/s measurement) was
**not built**. Building it would not change this structural verdict — it would spend real time to
arrive at a model whose dominant recurrent op still runs on CPU, wrapped in CoreML/ANE conversion
and runtime overhead the plain MLX-GPU port doesn't pay, with a real risk of an honest-looking but
substantively misleading "ANE tok/s" headline (mostly-CPU work wearing an ANE label). No tok/s
number is reported for CoreML/ANE — reporting one without a passing feasibility gate would violate
this project's own oracle-gate-before-speed discipline (`mlx_port/README.md`, `gate_oracle.py`).

```
ANE_FEASIBILITY_GATE: FAIL
  wkv_step_T1_01b   : 0/30 ops preferred=ANE (compute_units=CPU_AND_NE and ALL)
  wkv_step_T1_15b   : 0/30 ops preferred=ANE (compute_units=CPU_AND_NE and ALL)
  wkv_chain_T4_01b  : 0/105 ops preferred=ANE (compute_units=CPU_AND_NE and ALL)
  positive controls : big-GEMM=ANE(+real speedup), decode-GEMV@2048=ANE-label(but slower),
                       decode-GEMV@768=CPU, conv2d=CPU  [confirms methodology, not a broken probe]
DECISION: do not build the full converter; do not report ANE tok/s.
```

## What would change this (future work, not attempted)
The one shape class that *did* clear ANE's bar was a large **batched** GEMM (1024×1024, the
prefill-chunk-Linear shape) — consistent with F0037/F0038's own finding that this project's fused
Metal WKV kernel wins big on *prefill* (whole-chunk-per-dispatch) but barely moves *decode*
(bandwidth-bound, one token at a time). A speculative-decode or large-batch-decode workload
(bsz≫1, amortizing dispatch over many sequences at once) reshapes the recurrence into a genuinely
batched tensor and was not tested here — it is a different, open question from the bsz1 single-stream
shape this probe (and the rest of the Apple-Silicon work in this repo, F0037–F0041) targets. Not
pursued in this pass; flagged for whoever next touches Apple-Silicon serving-scale work.

## Honesty / scope
This is a **micro-op feasibility probe**, not a full-model benchmark — the numbers above are
op-dispatch counts and short (30–200 rep) wall-clock samples on a shared M5 (see F0038/F0041 for this
box's usual ±3–5% jitter), not a drift-cancelled multi-round A/B. That rigor wasn't needed here: the
compute-plan device assignment is a deterministic compile-time decision (confirmed identical across
two independent full runs of `probe_ane.py`), and the verdict (0/168 non-control ops ever prefer ANE)
has enough margin that noise doesn't change the conclusion. `rwkv-mobile`'s own converter also
targets `compute_units=ct.ComputeUnit.CPU_AND_NE` (a structural fact from reading their script); this
finding makes no claim about what their shipped numbers actually measure — that wasn't investigated
and isn't asserted either way.

## Cross-references
`coreml_port/probe_ane.py` (the probe, kept as a reusable artifact) · `mlx_port/rwkv7_mlx.py`
(`_wkv_scan_pure`, the math this probe replicates) · `bench/oracle_numpy.py` (`time_mixing`, the
ground-truth this repo gates everything against) · F0037 (MLX fused-Metal WKV, the Apple-Silicon
path that stands) · F0038 (M5 hardware ceilings — decode is bandwidth/launch-bound, the same
underlying reason ANE has nothing to add here) · `docs/BENCHMARKS.md` §12.6.
