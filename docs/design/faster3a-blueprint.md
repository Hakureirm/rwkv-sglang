---
doc_kind: design
title: "faster3a as blueprint: (A) large-M w8 tensor-core GEMM (task #11) and (B) fp16 single-stream last mile (task #5)"
status: proposed
author: kernel architect
date: 2026-07-03
related: [F0017, F0018, F0020]
sources:
  - blueprint: scratchpad/official_evals/albatross_rwkv7_fast_v3a.py   # faster3a_2605 (VKWR ported this)
  - ours: sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda/rwkv7_w8.cu
  - ours: sglang_overlay/sglang/srt/models/rwkv7.py
---

# faster3a blueprint → two implementation plans

## 0. What faster3a actually is (read this first — it reframes both tasks)

`albatross_rwkv7_fast_v3a.py` is a **standalone single-model fp16 inference driver**, not a
serving stack. Two facts dominate everything below:

1. **There is no int8/quant path anywhere.** `DTYPE = torch.float16` (line 14); every linear is
   fp16. `WKV_MODE` is `fp16` or `fp32io16` (line 51). So faster3a is **not** a blueprint for
   "how to keep int8 bytes alive into the compute-bound regime" — it never quantizes. What it *is*
   a blueprint for is the **weight-stationary fp16 GEMM tiling** and the **M-range dispatch
   philosophy** that we graft our int8-in-smem dequant onto. The int8 half is entirely ours; the
   blueprint supplies the tile geometry that wins at each (C, M) for these exact model widths.

2. **Its speed comes from (a) per-layer operator fusion and (b) an enormous hand-tuned fp16 GEMM
   dispatch table**, not from any single mega-kernel. faster3a chains several *fused-but-separate*
   ops per layer (it is NOT one monolithic kernel):
   - `tmix_mix6` (line 507): token-shift **+** all six lerps in one launch (reads `shift_state[0]`
     internally — the shift is fused into the lerp).
   - `tmix_kk_a_gate` (line 553): kk / L2norm / k-mix / a-gate in one launch.
   - `tmix_lnx_rkvres_xg` (line 575): `ln_x` (group-norm) **+** r·k·r_k residual correction **+**
     xg output-gate in one launch.
   - `tmix_vres_gate` (line 564): v-residual mix.
   - Cross-layer fusions, **gated to B==T==1 only** (`LN1_TMIX_FUSE`, line 382):
     `add_layer_norm_cmix_mix_f16` (line 374 — residual-add + ln2 + cmix mix) and
     `add_layer_norm_tmix_mix6_f16` (line 383 — residual-add + **next** layer's ln1 + **next**
     layer's mix6). These reach across the layer boundary into the next block's weights.
   - Linear projections: `linear_orig_layout` (line 619) dispatches into `linear_f16_orig` /
     `linear_f16_orig_lt_cfg(x, w, lt, cfg)` via a ~250-line switch keyed on **(group, C, rows)**
     with two hand-tuned tile knobs `lt ∈ {0,32,128}` and `cfg ∈ {0..6}` (lines 619–869). Two
     weight layouts coexist: **"orig" = (N,K) row-major weight** (weight-stationary, reused across
     the `rows`=M dimension) vs **transposed (K,N)**; the comment at line 55 states orig "slows
     tiny B*T", transpose "slows large B*T" — i.e. the orig/(N,K) layout is the **large-M
     weight-stationary** path. `linear_f16_m1_splitk` (line 609) is the M==1 split-K special case.
   - Weight prep: everything but the low-rank/orig-group weights is **transposed to (K,N) at load**
     (line 276); `att.r_k` flattened (line 279); optional `rkv` stacking for a batched bmm
     (line 308, `batched_rkv`, off by default, "consumes lots of VRAM" line 53).

The transferable lessons: **(A)** the (N,K) weight-stationary orig-layout GEMM with per-(C,M) tile
tuning is the shape of a large-M kernel that reuses weights across many rows — we keep the weight
**int8** in HBM and dequant-in-smem, which the blueprint never does. **(B)** the per-layer fusion
set tells us exactly which glue ops are worth folding together, and — critically — which fusions
faster3a itself only dares to do at B==T==1 (the cross-layer ones), i.e. the ones that don't
generalize to a serving batch.

---

## Part A — High-concurrency w8: large-M w8 tensor-core GEMM (task #11)

### The gap, precisely
`W8Linear.forward` (rwkv7.py:293–310): `M==1`→`gemv_w8_m1`; `2≤M≤8`→`gemm_w8_small`;
`8<M≤64`→`gemm_w8_tc`; **`M>64`→`dequant_w8`→`F.linear` (cuBLAS)** (line 308–309). The fallback
expands int8→fp16 in HBM *before* the GEMM (rwkv7_w8.cu:523 `dequant_w8_kernel`), so at bsz>64 the
int8 "half the bytes" advantage is **thrown away** — weights hit the tensor cores as fp16 exactly
like the fp16 baseline. Result: F0018's honest `bsz64 0.77×`, and bsz>64 merely ~matches fp16 with
no quant speedup (only the VRAM win survives).

The insight int8 has to exploit at large M: a GEMM tile that reads each weight element **once from
HBM as int8** (½ bytes) and reuses it across the whole M_tile of activation rows. As long as the
kernel stays weight-bandwidth-limited, int8 wins; once M is large enough that the GEMM is
compute-bound (tensor-core-FLOP-limited, and int8 gives **no** FLOP advantage because we MMA in
fp16 either way), the dequant→cuBLAS path is genuinely better and we should stay there.

### Why the current `gemm_w8_tc` cannot just have its M cap raised
Structural facts of `gemm_w8_tc_kernel` (rwkv7_w8.cu:230–448):
- `TC_M=16, TC_N=64, TC_K=64(=GROUP)`, **128 threads / 4 warps** (line 231–233, `__launch_bounds__(128,1)`).
- **"one block covers all M rows"** (line 475, 487): `mt = ceil(M/16)` m-subtiles, `1..4`, all
  driven by a single block via `acc[MT]` fragment array (line 276) and an `#pragma unroll for mt`
  MMA loop (line 328–332). At M>64, `MT>4` → the accumulator-fragment array and the per-kk unrolled
  MMA loop blow up register pressure and there is only **one** block per N-tile doing all the M
  serial work → grid is `N/64` blocks, far too few, and each block is enormous. Split-K (line
  479–515) exists precisely because the grid is starved at small N — it does not add M parallelism.
- The weight tile is dequanted **once per K-step to `smem_w[64][64+8]`** and reused across all m
  subtiles (line 391–407 / 306–321). This is the right amortization primitive — it just isn't
  tiled over M.

So task #11 is **not** "raise the cap"; it is a **new kernel** `gemm_w8_tc_large` with a proper 2-D
(M×N) block grid, reusing the current kernel's int8-in-smem-dequant primitive.

### Concrete spec for `gemm_w8_tc_large`

**Block/tile shape.** `M_tile × N_tile × K_tile = 64 × 64 × 64`, `K_tile == GROUP == 64` (keep the
one-scale-per-(n,k-tile) invariant that makes in-smem dequant trivial — same reason the current
kernel picks `TC_K==GROUP`). Grid = `(N/64, M/64)` (a real 2-D grid → GPU fills from the M axis, no
split-K needed). Each block computes a 64×64 output tile.

**Thread count: 256 (8 warps) — validated against the finding's "256-thread block rework" note.**
The finding (F0018 line 60, and the w4 README line 76) is right. Warp layout **4(M)×2(N)**: each
warp owns a 16×32 output sub-tile → `acc` = 1 (M) × 2 (N) = **2 fp32 wmma fragments/warp** (16 f32
regs for accumulators — vs the current kernel's `MT`=up-to-4 fragments *per warp*). This is the
core register-pressure fix: 8 warps each hold 2 acc frags (bounded, M-independent), instead of 4
warps each holding `ceil(M/16)` frags (unbounded in M). 256 threads also doubles the cp.async
issue width for the larger tiles.

**Weight-stationary vs activation-stationary → weight-stationary, unambiguously.** The int8 bytes
we are trying to save are the **weights**; the win only materializes if each int8 weight word is
read from HBM once and amortized across all 64 M-rows in the block. Concretely: stage the raw int8
weight tile `64(N)×64(K)` → dequant once to `smem_w` (fp16) per K-tile → every one of the 64
activation rows MMAs against it. Activations are the streaming operand. (faster3a's "orig" (N,K)
layout, line 55/276, is the fp16 analogue of exactly this choice — weight held stationary, rows
streamed.)

**Dequant interleave — keep dequant-once-to-smem-per-K-tile; do NOT go to int8-longer /
per-fragment dequant.** The current per-K-step dequant to `smem_w` (rwkv7_w8.cu:306–321) is
**more** correct at large M, not less. Reason: the whole economic argument is "dequant this tile
once, reuse across many rows." At M=64 the dequanted `smem_w` tile is reused by 64 rows instead of
16 → the dequant ALU cost is amortized 4× better than today. Dequant-per-fragment would re-dequant
the same weight for every m-subtile, destroying the amortization — the exact opposite of what large
M wants. Keeping weights int8 "longer" (into registers, dequant at MMA time) only helps if you were
smem-capacity-bound; we are not (see budget). **Decision: identical dequant primitive, just tiled
over a 2-D grid.**

**smem budget (sm86: 99 KB opt-in, 48 KB default; sm75 Turing: 64 KB max, 48 KB static).**
Per stage, for the 64×64×64 tile:
- `smem_a` fp16 `64×(64+8)` = 9.0 KB
- `smem_q` raw int8 words `64×16 u32` = 4.0 KB
- `smem_w` dequanted fp16 `64×(64+8)` = 9.0 KB
- `smem_c` epilogue f32 `16×(64+8)` = 4.6 KB
2-stage double-buffer of {a,q} + single w + c ≈ **(9.0+4.0)×2 + 9.0 + 4.6 ≈ 39.6 KB** → fits the
**48 KB default** (no opt-in needed) and even fits **Turing's 48 KB static** → the large-M path can
run on sm75 too (unlike a 128×128 tile, which would need ~71 KB → the 99 KB opt-in, sm80+ only). A
3-stage variant (≈53 KB) requires the sm86/sm80 99 KB opt-in
(`cudaFuncAttributeMaxDynamicSharedMemorySize`) and would be **sm80+ only** — offer it as a tuning
knob for the long-K ffn shapes (see below), not the default.

**Register pressure.** 2 acc frags/warp (16 regs) + a/b frag scratch + cp.async addressing. Target
≤128 regs/thread to keep 2 blocks/SM resident at 256 threads on sm86 (64K regs/SM). This is the
budget the current one-block-all-M kernel silently violates at M>64.

**cp.async pipeline depth.** 2-stage default (matches the proven sm80+ path in `gemm_w8_tc`,
rwkv7_w8.cu:281–335, and fits 48 KB → portable to Turing via the synchronous `#elif __CUDA_ARCH__
>= 700` fallback already in the file, line 358). 3-stage only behind the 99 KB opt-in for the
`4096×14336`-class ffn shapes where the M=64 crossover currently drags (F0018 line 50, w4 README
line 71) — deeper prefetch is exactly what those long-K compute-bound-ish shapes want.

**Split-K necessity: none for the large-M path.** With a 2-D `(N/64, M/64)` grid, at M>64 the grid
already has ≥`2·(N/64)` blocks and grows with M → the GPU is filled from the M axis. Drop split-K
here (it only ever existed to rescue the starved `N/64`-block grid at small M/small N). Keeping
`splits=1` also **simplifies the correctness contract** (single fixed-order K accumulation per
output element, no partial-reduce kernel).

### Exact dispatch thresholds (rwkv7.py `W8Linear.forward`, currently 293–310)

Replace the single `M>64 → dequant→cuBLAS` fallback with:

| M range | path | rationale |
|---|---|---|
| `M == 1` | `gemv_w8_m1` | unchanged (bandwidth-bound, greedy-EXACT) |
| `2 ≤ M ≤ 8` | `gemm_w8_small` | unchanged (one weight word feeds all rows, bit-identical to M1) |
| `8 < M ≤ 64` | `gemm_w8_tc` (current one-block-all-M) | unchanged — it wins here (F0018: 1.05–1.47× @M16) |
| **`64 < M ≤ M_CROSS`** | **`gemm_w8_tc_large` (new, sm80+; Turing uses the sync 700 path)** | **still weight-bandwidth-bound → int8 ½-bytes survives** |
| `M > M_CROSS` **or prefill** | `dequant_w8 → cuBLAS` (current fallback, kept) | **compute-bound: TC MMAs in fp16 regardless, int8 gives no FLOP win; weight read fully amortized** |

`M_CROSS`: start at **256** and tune per shape with `bench/verify_w8.py`. The crossover is where
arithmetic intensity makes the GEMM compute-bound; expect it in **256–512** for the 1.5B widths
(2048/2560) and *lower* for the long-K ffn (`inter≈4×C`, where cuBLAS's mature compute-bound
kernels pull ahead sooner — this is the same shape that already loses at M=64). Encode `M_CROSS` as
a per-(C, N) small table if a single constant leaves throughput on the table, but **do not** clone
faster3a's 250-line row-exact switch (see §3).

Guard exactly as the existing TC path does: `x.dtype==fp16 ∧ N%64==0 ∧ K%64==0 ∧ tc_supported()`
(rwkv7.py:306) — anything else degrades to `dequant→cuBLAS`, never crashes.

### Correctness contract and the risk the new tiling poses

Contract to preserve (identical to `gemm_w8_tc` today, rwkv7_w8.cu:12–18):
- **fp32 accumulation** — wmma `accumulator, float` fragments (line 276); unchanged in the 2-D tile.
- **Deterministic reduction order** — output element `y[m,n] = Σ_k` over K-tiles in **ascending k**
  with fp32 accum. In the 2-D no-split-K design this is *strictly simpler* than today: one block
  owns each output element and sums K in order → deterministic by construction, no split-K
  partial-reduce to get wrong. **If** a 3-stage/large-N variant ever reintroduces split-K, it MUST
  reuse the fixed-z-order `splitk_reduce_w8_kernel` (line 451) — never atomics.
- **greedy-EXACT gate.** w8 is greedy-EXACT today (F0018). Note the honest nuance: the **TC** path
  is not bit-identical to the M==1 GEMV (it MMAs fp16-rounded operands) — it matches the *dequant
  reference* at ~2.9e-4 rel and *that* is what clears the greedy gate. `gemm_w8_tc_large` inherits
  the **same** envelope, provided it keeps: same `__float2half_rn` dequant rounding (line 316–319),
  same ascending-K fp32 accum, same fp16 wmma inputs. Then it is numerically the same kernel with a
  different block-to-output mapping.
- **Risk:** the *only* thing changing is which (m,n) a block computes and the warp→sub-tile map.
  That does not touch per-element math → no new numerical risk **as long as K order and dequant
  rounding are byte-copied from the existing kernel.** The real risk is operational: (i) the 99 KB
  opt-in for a 3-stage/128-tile variant is sm80+ only — keep the default tile ≤48 KB so multi-arch
  JIT (Turing→Blackwell) is preserved; (ii) re-verify the greedy gate at M∈{96,128,192,256} in
  `bench/verify_w8.py` before flipping the dispatch threshold — the M=64 crossover shows shape
  sensitivity, so validate the new path on the actual ffn (`4096×14336`-class) shape, not just
  square shapes.

### Bottom line for Part A
One new kernel, `gemm_w8_tc_large`: **64×64×64 tile, 256 threads (8 warps, 4×2), weight-stationary,
dequant-once-per-K-tile-to-smem (the current primitive, tiled over a 2-D M×N grid), 2-stage
cp.async, no split-K, ≤48 KB smem (Turing-safe)**. Dispatch `64<M≤~256` to it; keep
dequant→cuBLAS for `M>~256`/prefill (genuinely compute-bound). This is the smallest change that
makes int8's ½-byte HBM advantage survive into the high-concurrency regime.

---

## Part B — fp16 single-stream last mile (0.73→~1.0× albatross) (task #5)

### Where we already stand (this decides the recommendation)
F0020: fp16 bsz1 decode = **226.5 tok/s**, already **> VKWR/albatross-faster3a's 224.6** on the
same card + checkpoint; w8 lossless **227.4**, w4 **259.1**. The per-component profile (F0020:44–49)
says the graphed step is **dominated by `lm_head` at 315.9 μs = 58.5%**, and lm_head is "268 MB
fp16 read ≈ **91 % of the 3090's bandwidth already** — the remaining fp16 wall is the head, not the
layers." **Every layer-glue op is small and launch/bandwidth-bound.** Already fused today:
`fused_lerp6` (rwkv7.py:647), `fused_kk_kmix` (701), `fused_gate_corr` (723), `lora4_m1` (685).
What remains *separate* that faster3a folds together:

| glue op (F0020 profile, μs/layer) | our state (rwkv7.py) | faster3a fuses it into | 
|---|---|---|
| token_shift **10.8** | separate `be.token_shift` (line 644), then `fused_lerp6` (647) | `tmix_mix6` — shift+lerp6 in one (blueprint line 507) |
| g_norm **4.4** | separate `self.g_norm(...)` (line 720) before `fused_gate_corr` (723) | `tmix_lnx_rkvres_xg` — ln_x+residual+gate in one (line 575) |
| norms **8.5** (ln1/ln2) | separate `attn_norm`/`ffn_norm` (line 833–835) | `add_layer_norm_cmix_mix_f16` / `add_layer_norm_tmix_mix6_f16` (lines 374/383) |
| kk/l2norm 11.8 | already `fused_kk_kmix` | `tmix_kk_a_gate` — parity |
| lerp 14.8 | already `fused_lerp6` | (part of `tmix_mix6`) |

### Staged fusion plan (cheapest-high-confidence first)

**Fusion 1 — token_shift + lerp6 (fold the shift into `fused_lerp6`).** *Save ≈10.8 μs/layer.*
Today `be.token_shift` (line 644) is a full separate launch that only produces `shifted`, then
`fused_lerp6(x, shifted, mix6)` (647) consumes it. faster3a's `tmix_mix6` reads `shift_state[0]`
and does the shift **inside** the lerp kernel (blueprint line 507). Mirror it: have `fused_lerp6`
read the shift-state and compute `shifted` internally → one launch instead of two.
- *Correctness risk:* **low.** Both are elementwise; `fused_lerp6` is already verified bit-identical
  to torch at fp16 (F0020 / rwkv7.py:639–641 comment). The shift is a copy/roll — no reduction, no
  new rounding. Stays greedy-EXACT.
- *Type:* **incremental** — one op absorbs another op's input; clean operator separation preserved.
- **Ship first.**

**Fusion 2 — g_norm + gate_corr.** *Save ≈4.4 μs/layer.* Today `self.g_norm(o)` (line 720) then
`fused_gate_corr(o, r, k, r_k, v, g, nh)` (723). faster3a does group-norm + r·k·r_k residual +
xg-gate in one op (`tmix_lnx_rkvres_xg`, line 575). Fold the per-head group-norm into
`fused_gate_corr`'s prologue.
- *Correctness risk:* **medium.** Group-norm has a per-head reduction → the fused kernel MUST keep
  the **fp32 reduction** and the same `eps`/order as `self.g_norm`, or it drifts off the greedy
  gate. Deterministic and self-contained (per-head, per-row), so achievable — just needs its own
  bit-exactness test vs the current two-op path before enabling.
- *Type:* **incremental.**
- **Ship second.**

**Fusion 3 — residual-add + ln2 + cmix-mix (intra-FFN prelude).** *Save part of the 8.5 μs norms.*
faster3a's `add_layer_norm_cmix_mix_f16` (line 374) fuses the residual add, ln2, and the ffn
token-mix. This stays **within the FFN module** (no cross-layer reach) → we can fuse `x = x +
attn_out` residual + `ffn_norm` + the ffn lerp into one op at the top of `Rwkv7FeedForward.forward`.
- *Correctness risk:* **medium** (layernorm fp32 reduction, same eps/order caveat as Fusion 2).
- *Type:* **incremental** — boundary stays at the module edge.
- **Ship third, only if 1+2 don't close enough of the gap.**

**Fusion 4 (NOT recommended) — cross-layer add + next-ln1 + next-mix6.** faster3a's
`add_layer_norm_tmix_mix6_f16` (line 383) reaches into **layer i+1's** weights, and faster3a itself
**gates this to B==T==1 only** (`LN1_TMIX_FUSE and B==1 and T==1`, line 382). This is the one that
*requires the mega-kernel structure*: fusing across the residual/layer boundary couples module i's
epilogue to module i+1's prologue → the two `nn.Module`s can no longer be independent, which breaks
sglang's per-layer dispatch, mixed prefill+decode batching, and tp/pp layer sharding.
- *Type:* **architectural** — sacrifices clean per-operator sglang integration. **ADR-gated.**

### The strategic call the task asks for
**Is beating albatross single-stream worth abandoning clean per-operator integration?** **No — and
we do not need to.** Three reasons:

1. **We already win single-stream.** 226.5 fp16 > 224.6 VKWR; 227.4 w8 lossless; 259.1 w4 (F0020).
   The "0.73→1.0×" framing predates the fused-LoRA + sparse-FFN + fast-GEMV work; the *current*
   engine steady-state already clears albatross-class fp16.
2. **The dominant cost is unfusable.** lm_head is 58.5 % of the step and already at **91 % of HBM
   bandwidth** (F0020:44–49). No amount of layer-glue fusion — not even the full mega-kernel —
   touches it, because it is a pure fp16 weight-read roofline. faster3a's mega-fusion optimizes the
   ~40 % that is layers; the incremental Fusions 1–3 capture the bulk of *that* (the launch
   overhead) **without** giving up sglang integration.
3. **The mega-kernel's marginal gain is small and its cost is large.** Fusion 4 buys a few μs/layer
   over Fusions 1–3, at the cost of module-boundary coupling that breaks dynamic batching, cuda-
   graph composability, and multi-arch JIT (see §3). Negative ROI for a serving stack.

**Recommendation:** Ship **Fusions 1 → 2 → 3** (all incremental, all keep clean operator
separation and the greedy-EXACT gate). **Stop before Fusion 4** (cross-layer mega-kernel). If we
want more single-stream headroom after that, spend it on the **actual wall**: the **gated int8
lm_head** (F0020 next-lever #3: ~150 μs, ≈+7 %, behind its own flag to protect argmax) attacks the
58.5 % head, which is a far better return than chasing faster3a's fp16 layer mega-kernel. Do **not**
open an ADR for the mega-kernel unless a customer workload proves the layer glue (not the head) is
the binding constraint — current evidence says it is not.

---

## 3. What NOT to copy from faster3a

Things that are correct for a standalone single-model fp16 script but would break sglang's dynamic
batching / cuda-graph / multi-arch-JIT / quant integration:

1. **The 250-line row-exact GEMM dispatch table** (`linear_orig_layout` / `linear_f16_orig_lt_cfg`,
   blueprint lines 619–869). It is overfit to specific `C ∈ {768,1024,2048,2560}` and *exact* row
   counts (`if path.rows >= 72 …`) on one GPU. sglang's M is **dynamic** (continuous batching), so a
   giant static switch on exact row counts is brittle and unmaintainable. Use a small principled
   threshold ladder (Part A) instead.
2. **`--use_fast_math` / `--extra-device-vectorization` build flags** (blueprint line 220). Our
   correctness contract is **IEEE, no fast-math** (rwkv7_w8.cu:16) — that is what makes w8/w4
   greedy-EXACT. Do **not** copy these flags into our kernel build; they would put the greedy gate
   at risk.
3. **Cross-layer fused ops** (`add_layer_norm_tmix_mix6_f16`, line 383) — reach into the next
   layer's weights and are B==T==1-only even in faster3a (line 382). Break module boundaries, mixed
   prefill/decode batches, and tp/pp sharding. (Part B Fusion 4.)
4. **CPU embedding + pinned-host `index_select`** (blueprint lines 340–358, `EMB_DEVICE="cpu"`).
   Fine for a bench that saves VRAM; introduces a host round-trip that cannot be captured inside
   sglang's full-step cuda-graph.
5. **Whole-forward CUDA-graph-per-(B,T)-shape + manual PP staging** (blueprint lines 904–990). sglang
   owns graph capture, batching, and parallelism; re-implementing them collides with the scheduler.
   Our kernels must be graph-*capturable*, not graph-*owning*.
6. **`batched_rkv` weight stacking + `bmm`** (blueprint line 308/519, off by default, "consumes lots
   of VRAM" line 53). Conflicts with per-projection quant (each `W8Linear` owns one weight) and our
   VRAM budget.
7. **Taking "faster3a has no int8" as evidence fp16 is enough.** It simply never implemented quant.
   Quant (w8 lossless 227.4, w4 259.1) is *our* edge and the whole point of Part A — don't let the
   blueprint's fp16-only shape talk us out of it.

---

## Appendix — source anchors
- Blueprint per-layer fusion: `tmix_mix6` L507, `tmix_kk_a_gate` L553, `tmix_vres_gate` L564,
  `tmix_lnx_rkvres_xg` L575; cross-layer L374/L383 (gated L382); GEMM dispatch L607–869; fp16-only
  L14/L51; weight transpose-at-load L276; fast-math flags L220; CPU emb L340–358; CUDA-graph bench
  L992–1027.
- Our w8 kernel: `gemm_w8_tc` L221–518 (TC_M/N/K L230–233, one-block-all-M L475/487, dequant-to-smem
  L306–321/L391–407, split-K L479–515, sync Turing path L358), `dequant_w8` L523–557, dispatch table
  registration L561–572.
- Our model dispatch: `W8Linear.forward` rwkv7.py L293–310; tmix glue L625–729 (token_shift L644,
  fused_lerp6 L647, fused_kk_kmix L701, g_norm L720, fused_gate_corr L723); FFN L732–.
- Profile + gap: F0018 L54–62 (bsz>64 gap, "256-thread block rework"), F0020 L44–49 (per-component
  profile, lm_head 58.5 % / 91 % bandwidth), w4 README L44–77 (TC kernel + M=64 crossover).
