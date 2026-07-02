---
doc_kind: design
title: "M6 phase-2 — hand-written sparse sqrelu FFN value-projection (+ fusion roadmap)"
date: 2026-07-01
last_verified_commit: pending
related: [F0013, F0014, F0015]
status: active
---

# M6 phase-2 — sparse channel-mix value-projection (and the fusion roadmap to surpass)

Goal: 极致优化甚至**超越**（精度保持 + 性能超越，对标 albatross 速度/VRAM + RWKV-LM 精度），
公平完整 benchmark，本项目为 sglang 推理集成，手写核。 This doc lands the ranked plan from the opt-design workflow
(wf_afbec306) + the key measurement that upgrades it.

## Key measurement (GATE 1 — PASSED, decisively)
Real-prompt sqrelu sparsity, instrumented on `Rwkv7FeedForward` (`RWKV_LOG_SPARSITY=1`),
eiffel fixture, bf16:

| model | sqrelu zero-fraction | nnz |
|---|---|---|
| 1.5B | **86.0%** (600 samples) | 14.0% |
| 7.2B | **90.2%** (288 samples) | 9.8% |

The design workflow assumed ~50-65%; the real value is **86-90%**. So the sparse
value-projection can skip ~86-90% of the value-weight reads. Value-proj ≈ 31% of 7.2B
decode bytes ⇒ potential ~25-35% decode speedup at bsz1 (well above the +13-16% the
workflow estimated at its lower sparsity assumption). This is a strong green light.

## Why this is the right first brick (rank 1, do-now)
- **Only lever past the dense-HBM ceiling** (dense caps ~67.5 tok/s @7.2B). Unlike the
  M6 gemv swap (F0015, kept byte-count constant → +5-9%), skipping zero-activation
  weight rows is a TRUE bandwidth saving.
- **Greedy-EXACT preserved**: relu(k)²=0 columns contribute exactly 0 (0·w=0, +0.0 exact);
  accumulating the surviving terms in **fp32** is the same rounding class as cuBLAS
  (precedent: gemv_m1 passed verify_m1d). So it keeps the accuracy crown.
- **cuda-graph SAFE** (I earlier doubted this mid-analysis; it is confirmed by reading
  albatross): cuda-graph captures kernel *launches* (static grid); a kernel that
  branches/loops on input data *inside* the block is fine. Albatross runs exactly this
  under `graph=True`. Static grid =
  (inter/TILE, H/TILE); all sparsity handled in-block via `__ballot_sync`/`__popc`
  shared-mem compaction. No host gather, no dynamic launch.
- **Hand-written** (per 手写), box-feasible (sm_86, CUDA 12.9, JIT arch 8.6).

## Design (port + adapt albatross, Apache-2.0)
Source to adapt: `refs/Albatross/faster3b_2606/cuda/rwkv7_mega_ops_260602.cu`
`cmix_sparse_down_relu_one_vtile_hfma2_split2_kernel` (lines ~246-321) +
`tile_cmix_value_weight` (`rwkv7_fast_v3b_b1t1_260602.py:72-76`).

1. **Load-time weight repack** (zero extra VRAM, in place): store `value.weight` so a
   skipped activation index skips a **contiguous coalesced** weight block. Keep a dense
   fallback reader for the M>1 / non-conforming path.
2. **Kernel** (`rwkv7_sparse_cmix.cu`, new): keep albatross's ballot/popc/warp-prefix
   compaction of nonzero `relu(k)²` indices; **replace half2 hfma2 accum with fp32
   register accum**; start with fp32 atomicAdd across inter-tiles. Compile flag for an
   fp16-accum A/B control (rank 3, lm-eval-parity only, NEVER shipped default).
3. **Wire** into `Rwkv7FeedForward.forward`: dispatch to sparse kernel ONLY for M==1
   (bsz1 decode) + conforming shapes (inter%128==0, H%256==0); M>1 / prefill fall back
   to dense `_proj_gemv` (batch-union sparsity collapses the saving).

## Gates — RESULTS (all PASSED; evidence in `bench/results/sparse_ffn/`)
- **G1 correctness — PASS**: `verify_m1d` greedy-EXACT with `RWKV_SPARSE_FFN=1`, fp16,
  cuda-graph ON: 0.1B 24/24, 1.5B 24/24, 7.2B 8/8.
- **G1b batch-invariance — PASS**: `verify_batch` (distinct-prompt batches) with sparse on:
  0.1B + 1.5B both PASS (all batches exact). So the fp32 atomicAdd order did NOT flip a
  knife-edge token on the fixtures → the deterministic fixed-split-K (rank 2) is NOT needed
  yet (keep it in reserve if a future model/prompt trips it).
- **G3 speed — PASS (big win)**: decode bsz1, fp16, cuda-graph ON, medianed:
  | model | base | sparse | sparse+`RWKV_FAST_LINEAR` | vs albatross-fp16 (faster3a) |
  |---|---|---|---|---|
  | 1.5B | 159.9 | 188.0 (+17.6%) | 194.1 | 0.52× → **0.63×** |
  | 7.2B | 45.9 | 59.1 (+28.8%) | **64.3** | 0.58× → **0.81×** |

  NOTE: this is the phase-2 A/B (sparse+fast, before the phase-3 in-place WKV). With phase-3
  the shipped best is higher — 7.2B bsz1 **65.7 (0.83×)**, 1.5B bsz1 202.9 — and the batched
  regime lifts too (see `bench/results/best2/` + comparison_clean.md's full table).
  The sparse value-proj (bandwidth win) and the gemv_m1 projections (F0015) are
  complementary and **stack**: 7.2B bsz1 45.9 → 64.3 tok/s (**+40%**). Standalone the
  7.2B value-proj kernel alone is 2.85× dense at ~96% sparsity (`standalone_verify.log`).
  ✅ re-baseline vs NEWEST albatross **faster3b_2606** RESOLVED: faster3b is **hardcoded
  C=4096 (7.2B ONLY)** and **B1T1-only** — it ValueErrors on 1.5B (C=2048) / 0.1B (C=768).
  At 7.2B bsz1 it is **79.05 tok/s ≈ faster3a** (79.6; sm120 tuning gives nothing on sm_86).
  ⇒ **faster3a remains the valid general baseline** (all other sizes + all bsz>1, since
  faster3b can't batch); the newest kernel adds no new target. So **ours 64.3 = 0.81× the
  newest** holds fairly (`bench/results/sparse_ffn/albatross_faster3b_72b.log`), and
  `comparison_clean.md` (vs faster3a) stays valid. No full re-baseline needed.
  ⬜ still TODO (G2): lm-eval parity confirmation with sparse on.
- **Known v1 cost**: the tiled value weight is a second copy (~+value-weight VRAM, e.g.
  +4.3 GB @7.2B). Fine on 24 GB (mem_fraction 0.92); the plan's in-place repack is the
  follow-up to reclaim it.

## Roadmap to actually SURPASS (this brick alone is necessary, not sufficient)
Honest: sparse value-proj alone ≈ +25-35% @7.2B bsz1 — closes much of the gap but the
full surpass needs compounding:
- **rank 4 — fuse the 8 LoRA GEMVs + gate math** into one kernel (they run ~1% peak BW =
  15.3% of the 1.5B step). Raises the DENSE fraction 42%→~70% @1.5B. greedy-EXACT (fp32).
- **rank 5 — whole time-mix mega-fusion** (lerp→proj→LoRA→WKV→norm→gate→o_proj on-chip).
  The only path to win **bsz8/32** (where sparsity collapses and we're at 0.57-0.85x).
  L-XL effort, highest payoff/risk. Stretch.

## Fair benchmark (must re-baseline)
Current `comparison_clean.md` used the OLD `faster3a_2605`. Must re-baseline vs the
NEWEST **`faster3b_2606`** (ships the sparse-FFN mega-kernel + `rwkv7_wkv_fp32io16_w0`,
whose fp32-io WKV is actually CLOSER to our fp32 state → cleaner head-to-head). Plus a
**Qwen3.5** comparison at matched quant, reporting decode/prefill/TTFT/
VRAM/state-size across bsz {1,8,32,64,128} + lm-eval. Report bsz8/32 too (no cherry-pick).

## Qwen3.5 comparison — OUT OF SCOPE for this project
The "beat Qwen3.5" head-to-head is out of scope for this project (an sglang inference
integration). **This project is measured only against albatross (speed/VRAM) + RWKV-LM
(accuracy).** (An exploratory Qwen3.5-2B run was dropped + the model deleted to reclaim disk.)

## bsz32 bottleneck — PROFILED (the production-regime target, user-chosen)
`bench/profile_components.py decode --bsz {1,32}` on 7.2B (bf16, graphed us/layer):

| component | bsz1 | bsz32 | note |
|---|---|---|---|
| rkv_proj / o_proj / ffn / lm_head | 137/49/355/669 | 137/48/396/684 | **flat** — weight-bound, batch for free |
| **wkv_recurrence** | 11.5 | **248** | **~21× — the ONLY batch-scaling component** |

At bsz32 the WKV recurrence is ~25% of the decode step and runs at only ~27% of the state
bandwidth floor (248us vs ~68us for 64MB/layer state traffic) → **~3.6× headroom**. This is
THE lever for the bsz32 gap (0.57×). The projections already batch efficiently (cuBLAS), so
the sparse/gemv M==1 kernels can't help bsz32 — but a faster batched WKV kernel can. Plan:
hand-optimize our triton WKV decode kernel (better V-tiling / occupancy / vectorized fp32
state I/O for bsz>1), referencing albatross `rwkv7_wkv_fp32io16_w0`; keep fp32 state +
greedy-EXACT.

## Remaining roadmap (M6 phase-3, active) — vs albatross/RWKV-LM only
1. **rank-4 hand-written LoRA fusion** (fuse the 8 LoRA GEMVs + gate math; they run ~1% peak
   BW = 15.3% of the 1.5B step). More bsz1 raw speed, greedy-EXACT, references albatross's
   `rkv_lowrank_pre_executor` / `lowrank_rank_out4_kk` (Apache-2.0).
2. **time-mix mega-fusion** (lerp→proj→LoRA→WKV→norm→gate→o_proj on-chip) — the path to
   bsz8/32 (where sparsity collapses; currently 0.57-0.85× albatross). L-XL, references
   albatross's mega kernels.
3. Refresh the full fair benchmark table (`comparison_clean.md`) with sparse+fast, vs albatross.

## Open risks (from the synthesis)
1. Insufficiency (alone < albatross ~77) — must compound. 2. atomicAdd nondeterminism →
verify_batch flips (mitigate: fixed-split-K). 3. Helps only bsz1 (bsz8/32 needs fusion).
4. fp32 accum could erode the byte-saving on a BW-bound kernel — measure, keep fp16 A/B.
5. Re-baselining vs faster3b may raise the target above 77.
