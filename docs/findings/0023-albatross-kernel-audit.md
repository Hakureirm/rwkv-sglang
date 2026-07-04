---
doc_kind: finding
finding_id: F0023
title: "Albatross-vs-ours kernel audit (GEMV / GEMM / LoRA / layer-glue), line-by-line: tests Bo's 'GEMV/GEMM/LoRA under-optimized' hypothesis against source, verdict = partly true but never 'simply slow' — GEMV parity (we vendored its best kernel), GEMM overtake = w8a8 only (it has zero int8 path), LoRA hypothesis REFUTED (already 2-launch fused; we lag on batched M), layer-glue is the real F0007 bsz1 gap"
last_verified_commit: "HEAD"
discovered_by: lead (M13), 2026-07-03
severity: info
status: open
related: [F0007, F0018, F0020]
---

# Finding F0023: Albatross-vs-ours kernel audit (GEMV / GEMM / LoRA / layer-glue), line-by-line

Method: a bounded multi-agent audit (read → adversarial source-verify → synthesize; 25 agents,
0 errors). **Only inefficiencies that survived source verification (`verify.confirmed === true`)
are used below.** Two claims were refuted on inspection and are quarantined in §4 so we never ship
an unfounded "albatross is slow" statement. Prior throughput numbers (F0007 226.5 vs 309.2 tok/s)
come from F0007's own 3090 run, not from these source files, and are attributed as such.

Attribution: performance reference = Albatross faster3a / RWKV-LM v7 (not RWKV-CUDA). SGLang
integration + all kernels here designed by Fable. Albatross source read under `refs/Albatross/faster3a_2605/`.

## 1. Verdict on Bo's hypothesis, per operator

Bo's hypothesis ("albatross's GEMV/GEMM/LoRA are under-optimized") is **partly true, not uniformly,
and never in the 'albatross is simply slow' sense**:

- **GEMV** — *exact parity*. Our `gemv_m1_kernel` (`rwkv7_fast.cu:44-85`) is a byte-for-byte
  vendoring of albatross's best M==1 kernel `linear_orig_row1_exact4_f16_kernel`
  (`rwkv7_v3a_ops.cu:568-616`). So the F0007 gap is **not** a worse GEMV. Albatross's inner loop
  does carry confirmed headroom (32-bit `__half2` loads not 128-bit; serial thread-0 cross-warp
  reduce; `OutTile` pinned at 2) — but it is headroom *neither side exploits*. Overtake is open,
  not banked.
- **GEMM (large-M / high-concurrency)** — holds through *one* confirmed structural opening:
  albatross has **zero int8 compute path** (all fp16 `cublasGemmEx`/`cublasLtMatmul`), so **w8a8**
  is a lever it cannot follow. But on fp16 math both sides land on the same tensor-core cuBLAS, so
  our current weight-only-int8 does **not** beat it (parity-minus-dequant, confirming F0018).
- **LoRA** — **REFUTED for fusion structure**: albatross already fuses all 4 chains into exactly
  2 launches with tanh/sigmoid + v-residual gate inline; we are parity at bsz1 and actually
  *behind* for batched decode (our fused op is M==1 only).
- **Layer-glue** — albatross *legitimately leads* bsz1 latency (2 fused boundary kernels vs our
  ~7-8 launches; this is the F0007 0.73× mechanism); we lead *architecturally* via paged
  per-request state that albatross's dense state structurally forbids.

## 2. Per-operator comparison (confirmed inefficiencies only)

Albatross Python: `rwkv7_fast_v3a.py`; kernels: `faster3a_2605/cuda/rwkv7_v3a_ops.cu`,
`rwkv7_fast_ops_fp16.cu`. Ours: kernels `rwkv7_fast.cu` / `rwkv7_w8.cu` / `rwkv7_lora.cu`; Python
`rwkv7.py`, `rwkv7_backend.py`, `wkv_recurrent.py`, `lora_fused.py`.

| Operator (sub-item) | Albatross (file:line) | Confirmed inefficiency | Ours (file:line) | Verdict |
|---|---|---|---|---|
| **GEMV** M==1 loads | `linear_orig_row1_exact4`, 32-bit `__half2`, `rwkv7_v3a_ops.cu:580-592` | 4 halfs/iter via 2×32-bit; no 128-bit float4/uint4 on DRAM-bound weight stream | `gemv_m1_kernel` identical, `rwkv7_fast.cu:54-60` | **parity** (open both) |
| **GEMV** cross-warp reduce | serial thread-0 sum, `rwkv7_v3a_ops.cu:604-615` | `OutTile×4` dependent adds on thread 0, 127 idle; sync+single-thread tail/block | identical, `rwkv7_fast.cu:76-84` | **parity** |
| **GEMV** OutTile | fixed 2, `rwkv7_fast_v3a.py:627`; dispatcher only builds 2, `:2891-2894` | OutTile pinned at 2 regardless of N; grid=N/2 (win bounded by DRAM floor) | fixed `<128,2>`, `rwkv7_fast.cu:96-101` | **parity** |
| **GEMV** non-orig split-K | scalar 16-bit x load in hot K-loop, `rwkv7_v3a_ops.cu:155-159` | x one `__half`/K step (worst hot-path granularity) | never dispatch scalar-x GEMV, `rwkv7_fast.cu:54-60` | **we_lead** |
| **GEMV** lm_head | hand row-exact GEMV, `rwkv7_fast_v3a.py:612-627` | *n/a — albatross strength* (avoids cuBLAS, graph-capturable) | cuBLAS `ParallelLMHead`/`LogitsProcessor`, `rwkv7.py:76,80` | **opportunity** (we lag) |
| **GEMM** large-M compute | fp16 `cublasGemmEx`/`cublasLtMatmul`, `rwkv7_v3a_ops.cu:2724-2744,:2966-3003` | zero int8-TC path anywhere; Ampere int8-TC unused (~2× is estimate) | `dequant_w8(...); F.linear`, `rwkv7.py:325-326`; wmma+cp.async M≤32, `rwkv7_w8.cu:285-331` | **opportunity** (w8a8 headline) |
| **GEMM** cuBLASLt overhead | per-call desc create+heuristic+destroy, ws=0, `rwkv7_v3a_ops.cu:2961-2982,:3004-3008` | descriptor churn + algo search every call | torch `F.linear` cached handle/algo, `rwkv7.py:326` | **we_lead** |
| **GEMM** batching | static per-forward keyed on rows=B*T, `rwkv7_fast_v3a.py:156-178,332` | no scheduler/paged-KV; can't interleave/admit mid-decode | sglang `ForwardBatch`/scheduler | **we_lead** |
| **LoRA** fusion structure | 4 chains → 2 launches (`:981/3582`, `:1135/3677`); acts + v-gate inline `:1192-1230` | *n/a — Bo's "under-optimized" REFUTED here* | `lora4_m1` also 2 launches, `rwkv7_lora.cu:172,181` | **parity** |
| **LoRA** rank_in grid | `dim3(Rmax,M,4)`+early-return, `rwkv7_v3a_ops.cu:3563,:3582` | pads grid to max rank; wastes blocks when ranks differ | exact `Rtot`-packed grid, `rwkv7_lora.cu:172` | **we_lead** |
| **LoRA** rank_out loads | scalar `__half`, `rwkv7_v3a_ops.cu:1093,:1103` | 2-byte scalar loads halve throughput vs vectorized | vectorized `__half2`, `rwkv7_lora.cu:131-136` | **we_lead** |
| **LoRA** batched-M coverage | fuses M≤8 (`:3565`) / M≤4 (`:3664`) | *n/a — albatross strength; our gap* | fused op M==1 only, `lora_fused.py:75` | **behind** (we lag batched) |
| **Glue** tmix boundary | `add_layer_norm_tmix_mix6_f16` = 1 launch, `rwkv7_v3a_ops.cu:1745-1825`; gated `:382` | *n/a — albatross strength (bsz1)* | 4 launches (LN `rwkv7.py:849`, token_shift clone+scatter `rwkv7_backend.py:134-136`, lerp6 `rwkv7.py:663`, add `:850`) | **opportunity** (we lag bsz1) |
| **Glue** cmix boundary | `add_layer_norm_cmix_mix_f16` = 1 launch, `rwkv7_v3a_ops.cu:1623-1683`; gated `:373` (T==1, fires bsz>1) | *n/a — albatross strength* | LN + token_shift + xk lerp + add, `rwkv7.py:851,788`; `rwkv7_backend.py:132-149` | **opportunity** (we lag) |
| **Glue** state layout / serving | dense `[L,2,B,C]`/`[L,B,H,N,N]`, 1 graph per (B,T), `rwkv7_fast_v3a.py:326-330,992-1005` | dense contiguous per-batch state forbids request interleaving | paged `conv[cache_indices]` + `INDEXED_STATE`, `rwkv7_backend.py:135`; `wkv_recurrent.py:85-108` | **we_lead** |

## 3. Reverse-overtake roadmap (ranked by impact × confidence)

1. **w8a8 int8 tensor-core GEMM at large M (HEADLINE).** The int8-TC path albatross *structurally
   cannot match* (it has no int8 path anywhere: `rwkv7_v3a_ops.cu:2966,:2971`, fp16-in/fp32-compute).
   **CORRECTION (F0025): this is NOT unimplemented — sglang-native `--quantization w8a8_int8`
   (per-channel int8 weight + per-token dynamic int8 activation + sgl_kernel cutlass
   `int8_scaled_mm`) is already wired for RWKV-7 (`rwkv7.py:15-16,331`) and already delivers a
   decode speedup over bf16: 1.5B +15%/+34%/+19% at bsz 1/8/32; 7.2B +53%/+47% at bsz 1/8, VRAM −48%,
   greedy 8/8 EXACT (`bench/results/quant.md`).** So the int8-TC overtake is largely *realized*, not
   pending. Remaining work: (a) **measure it at LARGE M (bsz 64–512)** — the existing tables stop at
   32 and were taken with the low `cuda_graph_max_bs` cap (F0024), so the high-concurrency int8
   number, exactly the strategic-axis overtake, is unmeasured; (b) our hand-written **all-arch** w8
   path (`rwkv7_w8.cu`, group-wise-K) can extend int8 benefit to **sm<80** where cutlass
   `int8_scaled_mm` does not ship (the genuinely-new kernel work; the old "move group-wise-K →
   per-channel for s8 MMA" note applies only to that hand-written path, since sglang-native already
   does per-channel). *Effort: (a) small/measurement, (b) large/kernel. Confidence high that w8a8 is
   a real albatross-unreachable overtake; magnitude at large M is what (a) will quantify.*
2. **Paged-cache-aware fused layer-boundary kernels.** Write `add_ln_mix6_shift` +
   `add_ln_cmixmix_shift` mirroring albatross's 2 fused kernels (`rwkv7_v3a_ops.cu:1745-1825,:1623-1683`)
   but doing shift-store in-place into `conv[cache_indices]` to preserve serving. Collapses ~7-8
   glue launches/layer to ~2 + removes token_shift clone-gather+scatter. *Gain:* recovers most of
   the F0007 0.73× bsz1 gap **while** keeping paged state → wins bsz1 latency AND concurrency.
   *Effort large, confidence high* (feasibility proven by our `INDEXED_STATE` paged fusion).
3. **128-bit vectorized loads in the M==1 GEMV.** Stride K-loop by 8 halfs, load x + each weight
   row as one `uint4`/`float4` → 4 `__half2` (guard `K%8==0`). Both sides use only 32-bit today
   (`rwkv7_v3a_ops.cu:580-592`; `rwkv7_fast.cu:54-60`). *Gain:* halves LDG count / full 128-bit
   weight transactions (bounded by DRAM byte floor). *Effort small, confidence medium.*
4. **Extend fused LoRA (`lora4_m1`) to batched M.** Add M grid dim, reuse packed `d_cat`/`u_cat`
   across batch; today M==1 only (`lora_fused.py:75`) while albatross fuses M≤8/M≤4. *Gain:*
   removes M>1 decode fallback — the one place LoRA runs *against* us. *Effort medium, confidence high.*
5. **Route M==1 lm_head through `gemv_m1`** (+128-bit epilogue), validate logits vs cuBLAS to ULP.
   albatross keeps vocab proj in hand GEMV + static graph (`rwkv7_fast_v3a.py:612-627`); we defer
   to cuBLAS (`rwkv7.py:76,80`). *Gain:* removes a cuBLAS call, stays graph-capturable (bounded by
   cuBLAS large-N quality). *Effort medium, confidence medium.*

Lower-ROI confirmed items (do when convenient, don't headline): warp-parallel tree reduce for the
GEMV tail (mirror albatross's own `linear_f16_m1_splitk_reduce_warp_kernel`, `rwkv7_v3a_ops.cu:184-211`);
GEMV `OutTile` 4/8 selected by N to shrink grid.

## 4. Where albatross legitimately leads — and refuted claims we must NOT publish

**Albatross legitimately leads (do not overclaim against these):**
- **bsz1 decode layer-boundary latency** — 1 fused kernel/boundary vs our ~7-8 launches
  (`rwkv7_v3a_ops.cu:1745-1825,:1623-1683`); the confirmed F0007 0.73× mechanism. Open opportunity
  (§3 item 2), but until we ship the fused kernels, albatross is faster single-stream.
- **lm_head at M==1** — kept in hand GEMV inside the static graph (`rwkv7_fast_v3a.py:612-627`).
- **LoRA batched decode** — fuses M≤8/M≤4 (`:3565,:3664`); ours is M==1 only (`lora_fused.py:75`).
- **weight-only int8 at large M** — we are parity-minus-dequant (winning only VRAM) until w8a8
  lands; albatross stores fp16 and skips the dequant kernel. Honest self-assessment, not a flaw.
- **`__restrict__` / read-only cache usage is correct** (`rwkv7_v3a_ops.cu:568-573`) — verified as
  *not* an inefficiency. Declining tensor cores / cp.async for M==1 GEMV is also correct (no
  M-reuse for MMA, no weight reuse for double-buffering).

**Refuted claims — quarantined so they are never published:**
- **"Albatross doesn't epilogue-fuse the ffn relu-square in decode" — REFUTED.** In the default
  decode path (`CMIX_B1T1_SPARSE`), `cmix_sparse_up_one_kernel` fuses mix + key-GEMV + relu-square
  in one kernel (`rwkv7_fast_ops_fp16.cu:365-366`); the NOFC variant fuses relu-square into the
  down/value GEMV input (`:707-708`). The separate-elementwise pattern exists *only* on the
  `CMIX_DENSE` large-batch fallback (`rwkv7_fast_v3a.py:604-605`). **Note for our side:** our decode
  path still applies relu² *between* two `_proj_gemv` calls (`rwkv7.py:789-812`), so folding it into
  a `gemv_m1` epilogue is a **parity-closing** task for us, not an overtake.
- **"At bsz>1 albatross loses BOTH glue fusions" — PARTIALLY REFUTED.** Only the ln1+tmix fusion is
  bsz1-only (gated `B==1 and T==1`, `rwkv7_fast_v3a.py:382`). The add+LN2+cmix-mix fusion is gated
  on `T==1` alone (`:373`), so it **still fires at bsz>1 decode**. The confirmed publishable part is
  the serving-machinery gap: no scheduler, dense per-run state, 1 CUDA graph per (B,T)
  (`rwkv7_fast_v3a.py:326-330,992-1005`).

## 5. Launch-tuning axis (Bo: `linear_orig_layout` needs per-GPU tuning)

Focused source-verified deep-dive (its `map` agent died mid-stream on an API error; the occupancy +
design agents read source directly, and the two load-bearing claims below were re-verified by hand).

**Verdict — Bo is right, with a scope correction.** `linear_orig_layout` (`rwkv7_fast_v3a.py:619`) is
a frozen single-dev-GPU tuning table: a hand-written per-`(group, C, rows)` decision tree of literal
launch constants (`128/64` threads, `OutTile 1/2/4`, `RowTile`, `C==768/1024/2048/2560` special-cases,
`launch_bounds(...,1)`, and hand-picked cuBLASLt `(workspace_mb, algo_index)` pairs like `(0,5)`,
`(32,3)`), with **zero** runtime arch/device branching on either the Python or CUDA side (grep-clean;
the only `.cu` arch-like tokens are compile-time `wmma::fragment` decls). Per F0007 these are tuned on
albatross's **5090** and merely *run* un-retuned on our 3090. **But the ~309 tok/s bsz1 number is not
where the weakness shows** — that path is a pure weight-streaming GEMV, HBM-bandwidth-bound and already
at ~99% of the 3090 roofline (1.5B×2 B ≈ 3.0 GB/tok ÷ 936 GB/s ≈ 312 tok/s; measured 309.1 ≈ 99%). So
per-GPU *launch* tuning cannot move bsz1; occupancy is not the binding resource there. **The exploitable
per-GPU headroom is at M>1** (batched decode / prefill).

**Our side is the same weakness class, coarser.** `gemv_m1` (`rwkv7_fast.cu:96-102`) branches only on
`N%2` → `<128,2>`/`<128,1>`, threads hardcoded 128, no K/M/arch dependence; `fast_linear.py:56-61` is a
config-free pass-through. Albatross at least varies one knob per shape (the `use4` flag); ours is a
single frozen point. Neither is arch-aware.

**3090 occupancy by regime (GA102 sm_86: 16 resident blocks/SM, 48 warps/SM, 64K regs, 82 SMs):**
- *bsz1 decode* (`linear_orig_row1_exact_f16_kernel<128,2>`, `rwkv7_v3a_ops.cu:521,568`): ~zero
  headroom (roofline-bound, above).
- *M>1* — fixed small-block configs structurally forfeit occupancy: every **64-thread** config
  (`row2_exact<64,2>`, `rows_cfg<64,3,4>`, body `:426`) is capped by the 16-resident-block limit to
  16×2 = 32 warps = **67% ceiling** regardless of registers; `<64,3,4>` additionally carries a
  12-accumulator register tile + `launch_bounds(,1)`, plausibly landing **~42–50%** (worst offender);
  fat `rows_f16<3,4>/<4,2>` tiles ~67–83%. **Honest bound:** whether this occupancy deficit converts
  to tok/s is UNKNOWN from occupancy alone (at M>1 the op leaves the BW-bound regime) — needs `ncu`
  (`achieved_occupancy` + achieved DRAM/Tensor throughput) at target batch sizes; no % is asserted here.
- *cuBLASLt fallbacks*: the frozen `(workspace_mb, algo_index)` ints were picked on the 5090 and are
  not guaranteed good algos for sm_86 — a cross-arch algo-selection risk, quantifiable only by
  re-running `cublasLtMatmulAlgoGetHeuristic` on the 3090.

**Cross-arch (MEASURED — cross-arch probe, F0027, corrects the earlier analytical claim):** the 67%
ceiling is **sm_86-specific (A10G/3090: maxBlocks/SM=16)**, not "all Ada" as first written. Measured
occupancy of albatross's `row2_exact<64,2>` / `rows_cfg<64,3,4>`: A10G(sm_86) **66.7% / 66.7%**
(limiter=block-count-cap — 16×2warp=32/48, confirms the prediction); **L4(sm_89) 100% / 83.3%** —
L4's maxBlocks/SM is **24, not 16**, so the block-cap does NOT bind (earlier "transfers to L4" was
WRONG); H100(sm_90) 75% / 62.5% (block-cap gone, now reg/warp-limited); T4(sm_75) 100%/100%
(maxWarps=32, 16×2 fills it); Blackwell sm_120 83.3% / 66.7%. So the fixed 64-thread configs are
worst on **sm_86** and mis-fit differently on every arch — exactly what per-arch autotune fixes.
Our `gemv_m1<128,2>` measures 100% on T4/A10G/L4/H100, 83.3% on Blackwell.

**Our autotune overtake design.** Make `gemv_m1` pick `(Threads, OutTile)` from `(sm_arch, N, K)`:
(1) the kernel is already `template<int Threads,int OutTile>` — instantiate {64,128,256}×{1,2,4} and add
a `gemv_m1_cfg` op with a runtime dispatcher (mirror albatross's cfg dispatch `rwkv7_v3a_ops.cu:2817-2835`);
(2) one cached `cudaGetDeviceProperties` → `major*10+minor` as the arch key; (3) two-tier selection —
Tier A static heuristic table seeded from an offline 3090 sweep, closed-form fallback for unseen shapes
(pick OutTile so `grid=ceil(N/OutTile) ≥ ~2·numSM` and near a numSM multiple to bury the wave tail; pick
Threads so `K/(Threads*4) ≥ 2`; small-N → drop OutTile/Threads so SMs aren't starved), plus Tier B a
one-time **warmup-only** autotune cache persisted per-GPU; (4) **CUDA-graph safety** — all Tier-B timing
runs during sglang warmup *before* capture; capture emits a bare `gemv_m1_cfg` launch with no events/sync,
and a capture-time miss falls back to the Tier-A heuristic, never benchmarks. Why this can overtake and
not merely match: our `<128,2>` already reaches full GA102 occupancy, so the 226.5→~309 gap is
wave-quantization tails + no shape specialization (plus the §3-item-2 layer-glue gap), *not* an occupancy
wall — the lever is coalescing/tail/reuse + arch-portability (4090/5090/Hopper without a hand re-tune),
which albatross structurally lacks.

**Benchmark-fairness gate (PREREQUISITE before re-citing 309.2 for M>1 comparisons).** Smoking gun
(hand-verified): at bsz1 the entire decode is `linear_orig_rows_exact_f16(x, w, 128, 2, use4)`
(`rwkv7_fast_v3a.py:622-627`), and the row1 CUDA dispatch compiles ONLY `<128,2,false>` and `<128,2,true>`
(`rwkv7_v3a_ops.cu:2892-2893`; any other threads/out_tile → `TORCH_CHECK(false)` at `:2900`). So 309.1
was measured with threads/out_tile **locked by compilation**, `use4` the only swept knob — an essentially
un-swept row1 kernel. *Because bsz1 is roofline-bound, re-tuning is unlikely to move 309.1 much* (so
citing 309.1 as the **bsz1** target is fair), **but the gate matters for M>1**: add row1 instantiations
(threads {64,128,256}×out_tile {1,2,4}×use4), sweep the real 1.5B shapes and the already-compiled
`rows_cfg`/`rows_f16` sets + `lt_cfg` `algo_index`×`workspace_mb`; protocol per [[feedback-benchmark-rigor]]
(locked clocks, 50 warmup/1000 timed CUDA-event iters, median+p10/p90, sm_arch stamped); **report BOTH**
`albatross_stock_3090` (=309.1) and `albatross_retuned_3090` (=X) and target X. This is a GPU experiment,
queued after the MATH500 run frees the box.

**New roadmap actions (add to §3):**
- **6 — arch-aware autotuned `gemv_m1` launch** (`gemv_m1_cfg` + `sm_arch`+(N,K) two-tier selection).
  *Gain:* recovers wave-tail/shape-specialization losses (few-% to ~10%, shape-dependent) at M>1 +
  cross-GPU portability albatross lacks. *Effort medium, confidence medium.*
- **7 — benchmark-fairness re-tune gate** (establish `albatross_retuned_3090` before any "beat albatross"
  claim at M>1). *Effort medium, confidence high (prerequisite, not a speedup).*

[[project-albatross-launch-tuning]]

## Cross-references
[[F0007]] (albatross 3090 baseline 309.2 vs 226.5) · [[F0018]] (weight-only int8 no large-M win →
w8a8) · [[F0020]] (our fused 4-chain LoRA) · `docs/design/faster3a-blueprint.md`.
