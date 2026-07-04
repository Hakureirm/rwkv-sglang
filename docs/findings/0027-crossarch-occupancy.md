---
doc_kind: finding
finding_id: F0027
title: "R4-B cross-arch occupancy (5 GPUs Turing→Blackwell, real cards): MEASURED validation of the launch-tuning thesis — albatross's fixed 64-thread linear configs hit a 66.7% ceiling on sm_86 (A10G/3090, block-count-cap) and mis-fit differently on every arch; corrects F0023 §5's over-broad 'transfers to all Ada' (L4 sm_89 reaches 100% via maxBlocks=24). Seeds the arch-aware autotune table."
last_verified_commit: "HEAD"
discovered_by: lead (M13), 2026-07-04
severity: info
status: open
related: [F0023, F0025]
---

# Finding F0027: cross-arch occupancy (R4-B) — measured launch-tuning validation

## Method
`bench/cuda_probe/occupancy_probe.cu` (verbatim copies of our `gemv_m1` candidate configs +
albatross's `row2_exact`/`rows_cfg` kernels) compiled per-arch and run via
the cross-arch harness on each card, using
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` (runtime API, **no profiler / no perf-counter
perms**). 5 real cards. Raw:
`bench/results/occupancy_crossarch.json`.

Device limits measured: T4 sm_75 (maxWarps/SM=32, maxBlocks/SM=16), A10G sm_86 (48, 16),
L4 sm_89 (48, **24**), H100 sm_90 (64, 32), RTX-PRO-6000 sm_120 (SMs=188).

## Result — occupancy of the fixed 64-thread linear configs

| kernel | T4 sm75 | A10G sm86 | L4 sm89 | H100 sm90 | Blackwell sm120 |
|---|---|---|---|---|---|
| albatross `row2_exact<64,2>` | 100% | **66.7%** | 100% | 75% | 83.3% |
| albatross `rows_cfg<64,3,4>` | 100% | **66.7%** | 83.3% | 62.5% | 66.7% |
| ours `gemv_m1<128,2>` | 100% | 100% | 100% | 100% | 83.3% |
| ours `gemv_m1<64,2>` | 100% | 66.7% | 100% | 100% | 83.3% |

## What it validates + what it corrects
- **Thesis CONFIRMED**: albatross's fixed 64-thread configs leave occupancy on the table and mis-fit
  per arch — the exact structural weakness arch-aware autotune removes. On **A10G/3090 (sm_86)** they
  hit **66.7%**, limiter = **block-count-cap** (16 resident blocks × 2 warps = 32 / 48 max), matching
  F0023 §5's analytical prediction (block-cap, not registers).
- **CORRECTION to F0023 §5** (benchmark-rigor: data over analysis): the earlier claim that the 67%
  ceiling "transfers to 4090/L4 (Ada sm_89, same 16-block limits)" is **WRONG** — L4 (sm_89) has
  **maxBlocks/SM = 24**, so the block-cap does not bind and `row2_exact<64,2>` reaches **100%**. The
  ceiling is **sm_86-specific** (A10G/3090), not "all Ada". F0023 §5 updated.
- **Hopper (H100)**: the block-cap is gone (32 blocks/SM), but the fat `rows_cfg<64,3,4>` still only
  reaches 62.5% (now reg/warp-limited) — so Hopper wants *different* (bigger-block) configs. Confirms
  "different sweet spot per arch".
- **Our default `gemv_m1<128,2>`** is already 100% on T4/A10G/L4/H100 and 83.3% on Blackwell — a good
  arch-blind default, with Blackwell the one place autotune should search harder.

## Autotune seed (feeds R4-B → `_select_config` table)
Per-arch rule for our `gemv_m1` M==1 path (occupancy-safe defaults; the warmup autotune refines by
wall-clock): **prefer 128-thread everywhere** (100% on sm_75/86/89/90); on **sm_86 avoid 64-thread**
(drops to 66.7%); on **sm_120 (Blackwell)** search {128,256}×tiles (128,2 is only 83.3%). This is the
data albatross's hand-frozen table lacks — it would need a manual per-GPU re-tune; we auto-select.

## Notes
- Occupancy is necessary-not-sufficient for throughput; the warmup autotune (`fast_linear.py`
  `_autotune_config`) still measures wall-clock per shape. This finding bounds the *search* (which
  configs can even reach full occupancy) + proves the cross-arch mis-fit is real and measured.
- Output-capture caveat: interleaved per-GPU prints mangled the live console on the first two runs;
  fixed by collecting all results and writing `occupancy_crossarch.json` + a single end-of-run block.

## Cross-references
[[F0023]] §5 (launch-tuning axis; corrected here) · [[F0025]] (GEMV autotune A-seg) · ADR-0005 R4 ·
`bench/cuda_probe/occupancy_probe.cu` · `bench/results/occupancy_crossarch.json`.
