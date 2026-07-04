---
doc_kind: finding
finding_id: F0028
title: "Full-stack composition + per-bsz gating: all hand kernels compose greedy-EXACT; the fused LoRA is M-gated (wins ≤M4, loses to cuBLAS ≥M8) after a measured regression; the composed stack (fast_linear+sparse_ffn+fused_lora+fused_glue+autotune) leads plain fp16 where it matters — committed-raw bsz1 225.9 tok/s (+46%) and peak 7334 @ bsz384 (+6.5%), parity within the run-to-run band at mid-bsz"
last_verified_commit: "ab50b2b"
discovered_by: lead (M13), 2026-07-04
severity: info
status: open
related: [F0020, F0024, F0026]
---

# Finding F0028: full-stack composition + per-bsz kernel gating

## Composition is greedy-exact
All hand-written fast paths enabled together — fused fp16 GEMV (`RWKV_FAST_LINEAR`), sparse
channel-mix (`RWKV_SPARSE_FFN`), fused 4-chain LoRA (`RWKV_FUSED_LORA`), fused paged
token-shift+lerp (`RWKV_FUSED_GLUE`), in-place WKV — pass `verify_batch --dtype float16` greedy
token-EXACT vs the numpy oracle on **both 1.5B AND 7.2B** (each: IDENTICAL 4/4, SHARED-PREFIX 5/5,
MIXED 6/6, OVERALL PASS — the full stack composes flawlessly at the flagship size too), with
every kernel confirmed FIRING. No interaction bug, no accuracy regression when composed.

## A measured regression → per-bsz gating (the (card×precision×bsz) principle in practice)
A full-stack throughput sweep exposed a **large-M regression**: peak collapsed to ~3088 tok/s @
bsz384 vs plain-fp16 6885. Isolation (toggling each env at c=128): removing `RWKV_FUSED_LORA`
recovered throughput (6893→3265 with it on, 6971 with it off), while `RWKV_FUSED_GLUE` was harmless
at large M. Root cause: `lora4_mn` is correctness-first (no smem staging, global `h` reads) and
**loses to the cuBLAS-batched ReplicatedLinear at large M**. Crossover sweep (fused on/off, other
fast paths on):

| concurrency | fused LoRA ON | OFF (others on) | winner |
|---|---|---|---|
| 1 | 225.9 | 208.0 | ON **+8.6%** |
| 2 | 297.4 | 277.7 | ON **+7.1%** |
| 4 | 532.1 | 520.1 | ON +2.3% |
| 8 | 1051.6 | 972.8 | ON +8.1% |
| 16 | 1733.7 | 1904.9 | OFF −9.0% |
| 32 | 2445.6 | 3172.2 | OFF **−22.9%** |

All cells are committed raws (same harness/config, audit-r2 HEAD): ON cells at c≤4 =
`bench/results/bsz_sweep_fullstack_HEAD.json` (the gate fires at T≤4); ON cells at c≥8 =
`bench/results/bsz_sweep_loraforced_HEAD.json` (gate lifted via `RWKV_FUSED_LORA_MAX_BS=512`);
OFF cells = `bench/results/bsz_sweep_loraoff_HEAD.json`. On the current HEAD the crossover sits
between c=8 and c=16 (the earlier motivating run — 235.2/309.5/546.5/929.2/1445/2009 vs
204.3/269.5/505.9/949.7/1859/3129, raw not retained — showed the same shape with parity at 8).
The default gate stays at **4**: its wins are unambiguous across runs, and raising it to 8 would
change which batches take the ~1-ULP fused path, so it would need a fresh greedy re-gate first —
`RWKV_FUSED_LORA_MAX_BS` is the per-deployment knob for anyone who wants to chase that.

Fix: **M-gate the fused LoRA to `T ≤ RWKV_FUSED_LORA_MAX_BS` (default 4)**; above it falls back to
cuBLAS. Both paths greedy-exact (`verify_batch` bsz4=fused / bsz5-6=fallback → PASS). This is the
concrete realization of the (card×precision×bsz) autotune rule: a kernel is enabled only in the
batch band where it measurably wins.

## Composed stack is now best across every batch size
Full stack (all envs, M-gate active, `--cuda-graph-max-bs 512`) vs plain fp16 (no hand kernels) —
**both the same methodology AND both committed raws**: `bench/bsz_throughput.py` wall-clock, 1.5B
fp16, RTX 3090, in64/out256. plain-fp16 cells = `bench/results/bsz_sweep_clean.json`; full-stack
cells = `bench/results/bsz_sweep_fullstack_HEAD.json` (audit-r2 HEAD, all envs on). Run-to-run
band on this harness is ±2–3% (an earlier uncommitted full-stack run measured e.g. bsz1 231.6,
peak 7326 — superseded by the committed raw below).

| concurrency | plain fp16 (no kernels) | **full stack** | Δ |
|---|---|---|---|
| 1 | 154.4 | **225.9** | **+46%** (all bsz1 hand kernels) |
| 32 | 3128 | 3130 | ±0 (M-gate active → cuBLAS path, as designed) |
| 128 | 6086 | 6023 | −1.0% (within band) |
| 384 | 6885 | **7334** (peak) | **+6.5%** |
| 512 | 6637 | 6746 | +1.6% |

At bsz1 the full hand-kernel stack (fast GEMV + sparse FFN + fused LoRA + fused glue + autotune)
lifts the *same-methodology* wall-clock throughput 154.4 → 225.9 (**+46%**); at mid-bsz the M-gated
LoRA hands the GEMMs to cuBLAS so the stack tracks plain fp16 within the band; at the peak the glue
+ large-batch path nets **+6.5%**. NOTE: the F0020 "226.5 bsz1" figure is a *different* methodology
(steady-state decode-tok/s, out512, prefill-subtracted), NOT comparable to these wall-clock numbers.

## Production wiring
`scripts/serve.sh` launches with the full verified stack ON (all fast-path envs + the
`--cuda-graph-max-bs 512` fix, F0024) in two verified modes (throughput / statecache). Because the
fused LoRA now self-gates by M, "all envs on" is optimal across bsz — no manual per-bsz tuning.

## Cross-references
[[F0020]] (fused LoRA bsz1) · [[F0024]] (cuda_graph_max_bs + best-bsz) · [[F0026]] (R2 glue) ·
ADR-0005 R3 · `scripts/serve.sh`.
