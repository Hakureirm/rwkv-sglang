---
doc_kind: finding
finding_id: F0060
title: "Megakernel Stage-A (#50): the 3090 bsz1 decode profile re-frames the endgame — the M==1 GEMVs already run at 94-99% of the 3090's achievable read BW per-kernel, so the bsz1 megakernel win is a FAST-CARD phenomenon (the 3090 is the structure/correctness gate, sm120 is the perf number); + the first fused-block increment: grouped r/k/v GEMV (gemv_rkv_m1, RWKV_MEGA=1), bit-exact (zero differing bytes kernel-level + byte-identical r/k/v at model level on real 1.5B/7.2B weights), 18->16 launches/layer, in-situ GPU-busy -6.5/-3.9us/layer (1.5B/7.2B), microbench graphed +3.17/+6.19us (~1-2% of the GPU-busy step; small on the bus-saturated 3090, honest), eager -44% (spec-draft-relevant), PDL griddepcontrol scaffolding guarded for sm90+ (inert on sm86)"
last_verified_commit: "HEAD (rwkv7_mega.cu + mega.py + rwkv7.py wiring + test_mega_rkv.py)"
discovered_by: Opus 4.8 (agent), 2026-07-16
severity: info
status: open - Stage-A increment landed (kernel + kernel/model bit-exact gates + microbench); sm120 flagship perf + full-server e2e before/after + verify_m1d under matched sglang staged
related: [F0051, F0058, F0056, F0023]
machine: 3090 (sm86), rwkvmain container, CUDA 12.9 / torch 2.11.0+cu129
---

# Finding F0060: RWKV-7 megakernel Stage-A — the 3090 bsz1 profile + the first PDL-ready fused-block increment

## 0. TL;DR

- **The task's "~144 ops/step" premise is stale** (F0051 already corrected it). Real 3090 bsz1
  fused stack: **18 kernel launches/layer, ~580/step (7.2B) / ~435/step (1.5B)**.
- **The decisive profile finding re-frames the endgame:** on the 3090 (measured achievable read
  BW **889.4 GB/s**), the M==1 decode GEMVs already run at **94–99% of that BW per-kernel**
  (r/k/v/o 835 GB/s = 93.9%, ffn.key 877 GB/s = 98.6%, 7.2B). The bus is **saturated by the
  individual launches**. So the "fatten-the-GEMV" lever (F0051 §5 #2, real on fast cards) is
  **not available on the 3090** — the recoverable bsz1 waste here is only the ~580 launch
  gaps/step + the ~15% non-GEMV GPU-busy. **The bsz1 megakernel win is a FAST-CARD phenomenon**
  (consistent with F0051's bandwidth-correlation: gap to albatross grows 3090 0.83× → 5090
  0.72× → H100 0.60×). **The 3090 gates structure + correctness; the flagship overlap number is
  an sm120 (5090) run.**
- **PDL is sm_90+ only, verified:** `griddepcontrol.wait/.launch_dependents` **fails to
  assemble on sm_86** ("requires .target sm_90 or higher") and compiles clean only under an
  `#if __CUDA_ARCH__ >= 900` guard. So on the 3090 the PDL overlap is structurally untestable;
  the fusion (launch merge + on-chip intermediates) is arch-independent and gates here.
- **Increment built: grouped r/k/v GEMV** (`gemv_rkv_m1`, `rwkv7_mega.cu`, env `RWKV_MEGA=1`,
  default OFF) — the three r/k/v projection GEMVs packed into ONE launch via a `blockIdx.y`
  role-split (Albatross's `rkv_lowrank_pre` multi-role kernel, competitor study §5; the canonical
  megakernel first-fusion). Reuses `gemv_m1`'s exact fp32 reduction + the SAME arch-aware
  `(threads, out_tile)` → **bit-exact by construction**.
- **Gates PASS:** kernel-level `torch.equal` vs 3× `gemv_m1` = **zero differing bytes**
  (shapes {1.5B, 7.2B, 0.1B, odd-N, small-K} × {uniform, heavy-tailed} × scales {0.5, 2, 8});
  model-level = **byte-identical r/k/v** with the real 1.5B + 7.2B projection weights. The
  increment is byte-identical to a `gemv_m1` path already under the production greedy-EXACT gate.
- **Measured.** In-situ deployed path (RWKV_MEGA on/off, profile_components): **launches/layer
  18 → 16 (−2), full step 7.2B ~580 → ~516 / 1.5B ~435 → ~387**; GPU-busy/layer **1.5B
  137.4 → 130.9 µs (−6.5), 7.2B 391.9 → 388.0 µs (−3.9)**. Isolated microbench graphed
  (production-relevant, pure kernel) **+3.17 µs (1.5B) / +6.19 µs (7.2B)** per r/k/v block.
  ≈ **1–2% of the GPU-busy step** ≈ ~1 point of the ceiling gap on the bus-saturated 3090 — a
  small, honest first recovery. Eager (spec-draft path, graphs don't hide launches): **−44%**
  on the 1.5B block (56.1 → 31.4 µs). The larger win is projected on sm120 (sub-peak GEMV +
  active PDL gap-hiding).

## 1. Method

- **Card constant.** 3090 achievable read BW measured with a read-only grid-stride `float4`
  reduction over a 2 GiB buffer (≫ the 6 MB L2), 3 warmup + 5×20 CUDA-event-timed iters:
  **889.4 GB/s = 95.0% of the 936.2 GB/s theoretical** (384-bit × 19.5 Gbps GDDR6X). This is
  the 3090 analog of ADR-0008 A0.1's 1691.7 GB/s for the 5090; ceiling math below uses 889.4,
  not the 936.2 theoretical the repo previously used for 3090 ceilings.
- **Per-op profile.** `bench/profile_components.py kernels` (torch.profiler CUDA activities,
  eager, 200 iters, random weights → value-independent kernel timing; builds the REAL deployed
  `Rwkv7DecoderLayer` so every fusion is exercised) with the shipping flag set
  (`RWKV_FAST_LINEAR/SPARSE_FFN/FUSED_LORA/FUSED_GLUE/GEMV_AUTOTUNE/FUSED_GATES/FUSED_SQRELU/
  FUSED_ADDLN/FUSED_GNGC`) **+ `RWKV_WKV_CUDA=1`**. `cuda_time_total` per kernel = GPU-busy
  (excludes inter-kernel gaps); the deployed overlay was verified current (has `rwkv7_wkv.cu`,
  `rwkv7_ln.cu`, GNGC in the model). Caveat: random weights make the sparse `ffn.value` ~50%
  sparse (vs ~90% on real prompts), so its per-layer time is not representative; the ceiling
  math uses the analytic 90.2%-sparse byte count instead.

## 2. The 3090 bsz1 decode profile (full fused stack + WKV-CUDA)

**7.2B fp16** (H=4096, L=32, nh=64): **18 launches/layer, 15 distinct, GPU-busy/layer 391.9 µs**;
lm_head 673.7 µs (2 launches); step ≈ 18×32 + 2 + 2 = **~580 launches**.

| kernel (per layer) | #/layer | µs/layer | what / BW efficiency |
|---|---|---|---|
| `gemv_m1<256,1>` (r/k/v/o) | 4.0 | **160.66** | 40.17 µs/launch; 33.55 MB → **835 GB/s = 93.9%** |
| `gemv_m1<256,2>` (ffn.key+sqrelu) | 1.0 | **153.00** | 134.2 MB → **877 GB/s = 98.6%** |
| `lora_stage2<8>` | 1.0 | 15.11 | LoRA up-proj |
| `sparse_cmix_f32acc` | 1.0 | 10.69 | ffn.value (random ~50% sparse — not representative) |
| `lora_stage1<128>` | 1.0 | 10.95 | LoRA down-proj |
| `add_ln<16>` | 1.0 | 10.28 | fused residual-add + LayerNorm |
| `vectorized_layer_norm` | 1.0 | 9.35 | the second norm boundary |
| `wkv_decode<float>` | 1.0 | 6.19 | **WKV CUDA kernel active** (F0058) |
| `gn_gatecorr` | 1.0 | 3.69 | fused GroupNorm + gate-corr |
| `lora_gates` / `kk_kmix` / `shift_lerp6` / `shift_lerp1` / 2×tiny | 6.0 | ~11.4 | fused glue/gates |

**The 5 big GEMVs = 313.66 µs = 80.0% of the 391.9 µs layer busy.** lm_head 536.9 MB / 673.7 µs
= 797 GB/s = 89.6%. Dense variant (sparse off): 16 launches/layer, GPU-busy **531.6 µs**, GEMVs
466 µs = **87.7%** — even more GEMV-dominated.

**1.5B fp16** (H=2048, L=24, nh=32): 18 launches/layer, GPU-busy/layer **137.4 µs**; lm_head
317.1 µs; ~435 launches/step. r/k/v/o `gemv_m1<128,2>` ×4 = 47.99 µs (12.0/launch; 8.39 MB →
**699 GB/s = 78.6%** — the smaller weight is more overhead-bound), ffn.key 39.79 µs (843 GB/s =
94.8%). GEMVs = 63.9% of layer busy.

## 3. Ceiling analysis (3090, 889.4 GB/s)

Byte counts from ADR-0008 §2.1/A0.2 (same models): 7.2B dense 13.94 GB/step, sparse-effective
(90.2% zero ffn.value) 10.07 GB; 1.5B dense 2.815 GB, sparse 2.123 GB.

| model | sparse ceiling | dense ceiling | ours bsz1 (measured*) | % of sparse ceiling | albatross | % |
|---|---|---|---|---|---|---|
| 7.2B | 88.3 tok/s | 63.8 | 65.7 | **74.4%** | 79.6 | **90.1%** ("~92% of 3090 BW") |
| 1.5B | 418.9 tok/s | 315.9 | 202.9→226.5 | 48–54% | 309.2 | 73.8% |

\*`bench/results/comparison_clean.md` (greedy 24/24 EXACT, cuda-graph ON, median of 7) — an
OLDER config that predates fused-LoRA/gates/addln/gngc/WKV-CUDA, so current is ≥ these. GPU-busy
floor from the §2 profile (upper bound on e2e): 7.2B ≈ 13.3 ms = **75 tok/s**, already only
**85% of the 88.3 sparse ceiling** — the ~15% between GPU-busy floor and BW ceiling is the
non-weight-streaming GPU work (norms / glue / gates / WKV-state r+w / LoRA overhead / sparse
imperfection), i.e. exactly the fusion target. **The ~16-point 7.2B gap to albatross on the
3090 mirrors the 5090's 79.4%→92.4%** — but its composition differs: on the 3090 the GEMVs
saturate, so the gap is dominated by launch gaps + non-GEMV busy; on the 5090 the sub-peak M==1
GEMV is a bigger share. **This is why the megakernel is prioritized for fast cards.**

## 4. Localized fusible waste (what the profile says to fuse)

1. **~580 launch gaps/step.** Even inside a captured graph, a ~1–2 µs producer-tail /
   consumer-prologue gap sits between adjacent kernels where no weight bytes stream. Over ~580
   launches this is the primary recoverable slice, and it is exactly what PDL
   `griddepcontrol.launch_dependents` removes (overlap the tail with the next prologue). **sm90+
   only.**
2. **~15% non-GEMV GPU-busy** (norms, glue, gates, WKV, LoRA overhead) — collapsible by keeping
   producer→consumer intermediates on-chip (arch-independent).
3. **NOT the per-kernel GEMV** on the 3090 (already 94–99% of BW). On the 5090 this flips (the
   M==1 GEMV can't saturate 1691 GB/s → sub-peak → the fatten-GEMV lever returns).

## 5. The increment: grouped r/k/v GEMV (`gemv_rkv_m1`, RWKV_MEGA=1)

`sglang_overlay/.../rwkv7_kernels/cuda/rwkv7_mega.cu` + `mega.py` loader + `models/rwkv7.py`
wiring (behind `RWKV_MEGA`, default OFF), `bench/test_mega_rkv.py` gate.

- **What.** The three r/k/v projections (three separate `gemv_m1` launches that share the
  `shift_lerp6` producer) packed into ONE grid: `dim3(N/OutTile, 3)`, `blockIdx.y ∈ {0,1,2}`
  selects the projection (per-proj x/weight/y). This is Albatross's `rkv_lowrank_pre_executor`
  multi-role single-grid technique (competitor study §5) — the clearest whole-block-fusion
  primitive and the megakernel's r/k/v stage. The three activations pass as **separate
  pointers** (not a `[3,K]` stack) so the caller never pays a gather launch — `xr/xk/xv` can
  point at their (non-adjacent) rows of the `shift_lerp6` output; the net effect is a clean
  −2 launches/layer (an earlier stacked variant regressed this to −1 via the gather kernel).
- **Bit-exact by construction.** Each output element's fp32 accumulation is `gemv_m1_kernel`
  verbatim (identical per-thread k-stride = Threads*4, identical warp-shuffle tree, identical
  serial cross-warp sum), launched with the SAME `(threads, out_tile)` `_select_config(N,K)`
  picks for the deployed path. No new numerics.
- **PDL scaffolding.** A guarded `griddepcontrol.launch_dependents` at the kernel tail,
  `#if __CUDA_ARCH__ >= 900` (verified: fails to assemble on sm_86 without the guard). It is
  currently **inert** — a real overlap only once the launch site sets
  `cudaLaunchAttributeProgrammaticStreamSerialization` and the downstream stage issues
  `griddepcontrol.wait` (the sm120 wiring step, §7). On the 3090 this file gates structure +
  correctness only, exactly like the WKV CUDA kernel (F0058).
- **Eligibility.** fp16, M==1 (bsz1 decode), plain `ReplicatedLinear` r/k/v (not W4/W8/int8,
  not tp>1), r/k/v share (N,K). Else the 3-launch path is untouched. Composes with all other
  fast-path flags.

## 6. Gates + measurement

**Kernel-level** (`bench/test_mega_rkv.py`, `torch.equal` vs 3× `gemv_m1_cfg`): **PASS — zero
differing bytes** across {1.5B r/k/v (N=K=2048, cfg 128,4), 7.2B (4096, cfg 256,4), 0.1B (768),
odd-N (6,2048 → OutTile=1), small-K (2048,8)} × {uniform, heavy-tailed} × scale {0.5, 2, 8}.

**Model-level** (real `Rwkv7Attention` r_proj/k_proj/v_proj weights + real `x_r/x_k/x_v`-lerped
activations, deterministic path): grouped output **byte-identical** to stacked `gemv_m1` on both
**1.5B and 7.2B**. (The full-forward eager cascade test is confounded by a pre-existing
harness-only nondeterminism — even RWKV_MEGA off-vs-off differs, from a nondeterministic
reduction in the eager WKV/torch path — so the model gate compares r/k/v directly, bypassing the
nonlinear WKV. Production greedy-EXACT is established by `verify_m1d` in the shipping stack; the
increment is byte-identical to a `gemv_m1` path already under it. A `verify_m1d` re-run was
blocked by a Mac-verifier vs container-sglang `ServerArgs` version skew — staged.)

**In-situ** (deployed layer forward, `profile_components kernels`, RWKV_MEGA off→on): launches/
layer **18 → 16 (−2)**, distinct 15 → 16 (`gemv_rkv_m1` appears; the r/k/v/o `gemv_m1` count
drops 4 → 1 = o_proj only); GPU-busy/layer **1.5B 137.4 → 130.9 µs (−6.5)**, **7.2B
391.9 → 388.0 µs (−3.9)**; full step **7.2B ~580 → ~516, 1.5B ~435 → ~387 launches**.

**Isolated microbench** (`bench/test_mega_rkv.py`, per r/k/v block, µs; graphed = production):

| shape | cfg | sep eager | grp eager | sep graphed | grp graphed | graphed Δ |
|---|---|---|---|---|---|---|
| 1.5B r/k/v | (128,4) | 56.10 | **31.41** | 33.22 | **30.05** | **+3.17** |
| 7.2B r/k/v | (256,4) | 124.58 | **116.03** | 121.07 | **114.88** | **+6.19** |

- **Graphed:** +3.17 µs/layer (1.5B) × 24 ≈ **76 µs/step ≈ +2%** of the ~3.7 ms GPU-busy step;
  +6.19 µs/layer (7.2B) × 32 ≈ **198 µs/step ≈ +1.5%** of the ~13.3 ms step ≈ ~1 point of the
  ceiling gap. Small — **because the 3090 bus is already saturated by the individual launches**
  (the 2 eliminated inter-kernel gaps are the only recovery; the weight bytes are unchanged).
- **Eager:** −44% on the 1.5B block (56.10 → 31.41) — the 3-launch→1 launch-latency win the
  graph hides in serving but that is real for the eager / spec-draft paths (cf. the WKV CUDA
  kernel's 3–6× eager win, F0058 §4).

Honest read: a **first increment that recovers ~1 point of the ~16-point 3090 gap** — exactly
the "2–3 of 13 points is a real result" bar, on the harder-to-move card. The lever it exercises
(launch merge + on-chip staging + PDL-ready structure) is the megakernel's, and it recovers
more on sm120.

## 7. sm120 execution plan (run the moment the 5090 frees; do NOT touch it during RL training)

1. **Rebuild + activate PDL.** Build `rwkv7_mega.cu` for `sm_120`; confirm the guarded
   `griddepcontrol` assembles (it will on sm90+), and set
   `cudaLaunchAttributeProgrammaticStreamSerialization` on the `gemv_rkv_m1` launch (host-side
   attribute confirmed present in CUDA 12.9). ADR-0008 A0.1 already proved coop+PDL launches
   capture into a CUDA graph on the 5090 stack — no fallback needed.
2. **Wire the PDL chain across the block.** `shift_lerp6` tail → `griddepcontrol.launch_dependents`;
   `gemv_rkv_m1` head → `griddepcontrol.wait`, tail → `launch_dependents` into the LoRA/kk stage.
   Capture the whole decode step into the existing graph. This is what turns the (currently inert)
   scaffolding into the continuous-weight-streaming win.
3. **Gate on sm120.** `verify_m1d` greedy-EXACT 24/24 (1.5B) + fixture 8/8 (7.2B) under a
   version-matched sglang; re-run `test_mega_rkv.py` (bit-exactness is arch-independent but
   re-confirm on the flagship silicon).
4. **Measure the flagship number.** bsz1 fp16 before/after (`RWKV_MEGA` off/on), 7.2B + 1.5B,
   report the new **% of the 1691.7 GB/s sparse ceiling** against the **79.4% → 92.4%** frame.
   On the 5090 the grouped kernel recovers the r/k/v **sub-peak GEMV** (unavailable on the
   saturated 3090) **and** the 2 gaps with active PDL — expected to move meaningfully more than
   the ~1 point measured here.
5. **Next Stage-A increments** (same file, same bit-exact-by-construction pattern):
   `gemv_rkv_m1` → add o_proj as a 4th role after the WKV block; fold the w/a/g/v LoRA
   compressed projections into the same grid (the full `rkv_lowrank_pre`); fuse the `shift_lerp6`
   producer into the r/k/v prologue (intermediate never leaves smem). Each is one PDL-chained
   stage toward the whole-block megakernel.

## 8. Honest ledger

- **Delivered:** the 3090 bsz1 profile + achievable-BW ceiling + the GEMV-saturation re-framing;
  the PDL-on-sm86 impossibility (verified) and its consequence; ONE gated bit-exact increment
  (kernel + model gates, zero differing bytes); its microbench (graphed + eager); the precise
  sm120 plan.
- **Staged (not done):** the sm120 flagship before/after + %-of-1691.7-ceiling; the full-server
  bsz1 e2e before/after under cuda-graph; `verify_m1d` under a version-matched sglang; the PDL
  chain wiring (steps 7.1–7.2) and the further Stage-A increments (7.5). None are blocked by a
  correctness or feasibility finding — they are the sm120-gated continuation.

## 9. Artifacts + cross-references

- Kernel/loader/wiring/gate: `sglang_overlay/.../rwkv7_kernels/cuda/rwkv7_mega.cu`,
  `.../rwkv7_kernels/mega.py`, `models/rwkv7.py` (`_MEGA` flag + grouped r/k/v block),
  `bench/test_mega_rkv.py`.
- Profile constants: 3090 read-BW probe (889.4 GB/s) + `bench/profile_components.py kernels`
  (7.2B / 1.5B, shipping flags + `RWKV_WKV_CUDA=1`).
- [[F0051]] (the launch-count profile this extends to the 3090 + the "~144 stale" correction +
  the F0051 §5 lever ranking this profile re-prioritizes for the slow card) ·
  [[F0058]] (the WKV CUDA kernel = the designated megakernel WKV component; same
  gate-on-3090 / perf-on-sm120 posture + eager-win-for-spec-draft framing) ·
  [[F0056]] (the W1' glue fusions already in the shipping stack) ·
  [[F0023]] (albatross layer-glue is the bsz1 gap) ·
  ADR-0008 (megakernel feasibility; A0.1 BW + coop/PDL-in-graph; this is its Stage-A) ·
  `rwkv-competitors/albatross-megakernel-study-2026-07-13.md` (PDL-chained structure; §5
  multi-role single-grid = this increment's blueprint) · ADR-0004 (zero-FLA; the kernel
  transcribes our own `gemv_m1`).
