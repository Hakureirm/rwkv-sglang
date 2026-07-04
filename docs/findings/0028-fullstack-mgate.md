---
doc_kind: finding
finding_id: F0028
title: "Full-stack composition + per-bsz gating: all hand kernels compose greedy-EXACT; the fused LoRA is M-gated (wins ≤M4, loses to cuBLAS ≥M8) after a measured regression; the composed stack (fast_linear+sparse_ffn+fused_lora+fused_glue+autotune) is now best across every batch size — bsz1 231.6 tok/s and peak 7326 @ bsz384 (+6.4% over plain fp16)"
last_verified_commit: "HEAD"
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

| concurrency | fused LoRA ON | OFF | winner |
|---|---|---|---|
| 1 | 235.2 | 204.3 | ON +15% |
| 2 | 309.5 | 269.5 | ON +15% |
| 4 | 546.5 | 505.9 | ON +8% |
| 8 | 929.2 | 949.7 | OFF (parity) |
| 16 | 1445 | 1859 | OFF −22% |
| 32 | 2009 | 3129 | OFF −36% |

(dev-box A/B via the committed `bench/bsz_throughput.py` toggling only `RWKV_FUSED_LORA`; raw JSON
pending sync — the two isolation/crossover scripts and exact commands are in this session's run.)

Fix: **M-gate the fused LoRA to `T ≤ RWKV_FUSED_LORA_MAX_BS` (default 4)**; above it falls back to
cuBLAS. Both paths greedy-exact (`verify_batch` bsz4=fused / bsz5-6=fallback → PASS). This is the
concrete realization of the (card×precision×bsz) autotune rule: a kernel is enabled only in the
batch band where it measurably wins.

## Composed stack is now best across every batch size
Full stack (all envs, M-gate active, `--cuda-graph-max-bs 512`) vs plain fp16 (no hand kernels) —
**both the same methodology**: `bench/bsz_throughput.py` wall-clock, 1.5B fp16, RTX 3090,
in64/out256. plain-fp16 cells from `bench/results/bsz_sweep_clean.json`; full-stack cells are a
dev-box run of the same harness with all envs on (raw pending sync — re-run:
`RWKV_FAST_LINEAR=1 RWKV_FUSED_LORA=1 RWKV_SPARSE_FFN=1 RWKV_FUSED_GLUE=1 RWKV_GEMV_AUTOTUNE=1 ... bench/bsz_throughput.py`).

| concurrency | plain fp16 (no kernels) | **full stack** | Δ |
|---|---|---|---|
| 1 | 154.4 | **231.6** | **+50%** (all bsz1 hand kernels) |
| 32 | 3128 | 3312 | +5.9% |
| 128 | 6086 | 6499 | +6.8% |
| 384 | 6885 | **7326** (peak) | **+6.4%** |
| 512 | 6637 | 6705 | +1.0% |

At bsz1 the full hand-kernel stack (fast GEMV + sparse FFN + fused LoRA + fused glue + autotune)
lifts the *same-methodology* wall-clock throughput 154.4 → 231.6 (**+50%**); at large M the M-gated
LoRA avoids the cuBLAS-loss regime while glue + autotune net +6% — so the composed stack beats plain
fp16 at **every** concurrency. NOTE: the F0020 "226.5 bsz1" figure is a *different* methodology
(steady-state decode-tok/s, out512, prefill-subtracted), NOT comparable to these wall-clock numbers;
we do not claim 231.6 as "+X% over 226.5".

## Production wiring
`scripts/serve.sh` launches with the full verified stack ON (all fast-path envs + the
`--cuda-graph-max-bs 512` fix, F0024) in two verified modes (throughput / statecache). Because the
fused LoRA now self-gates by M, "all envs on" is optimal across bsz — no manual per-bsz tuning.

## Cross-references
[[F0020]] (fused LoRA bsz1) · [[F0024]] (cuda_graph_max_bs + best-bsz) · [[F0026]] (R2 glue) ·
ADR-0005 R3 · `scripts/serve.sh`.
