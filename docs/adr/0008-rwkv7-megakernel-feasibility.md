---
doc_kind: adr
adr_id: 0008
title: "RWKV-7 persistent megakernel (single-launch decode step): feasibility study + design sketch — verdict: GO, scoped (0.1B draft/small-model first, 1.5B same-card albatross flip second, int4 variants third); spec-decode net win additionally gated on an unprofiled ~50 ms/round orchestration residual that no kernel can fix"
status: proposed
date: 2026-07-10
last_verified_commit: "f63ba89 (research spike, read-only; no code changed)"
supersedes: []
superseded_by: []
---

# ADR-0008: RWKV-7 persistent-megakernel feasibility

Research-only spike. No GPU run was performed for this document; every number below is
tagged **[measured]** (exists in this repo's bench/results or docs/findings, or in the
vendored KernelBench artifacts), **[computed]** (arithmetic from specs or from measured
inputs — the arithmetic is shown), or **[estimate]** (projection; stated assumptions).

## Context

Bo Peng shared kernelbench.com/mega in the community chat with "请发AI学习" (send this to
the AI to study). KernelBench-Mega scores AI agents on writing a **true single-launch
megakernel**: the entire per-token decode forward of a 4-layer Kimi-Linear-48B motif
(3×KDA gated-delta linear attention + 1×MLA, each + 64-expert MoE, W4A16 int4 weights)
fused into ONE kernel launch per `step()`. CUDA graphs / torch.compile / per-op loops are
explicitly rejected by a post-run authenticity judge — "hiding launches isn't fusing".

The signal matters to this project for three concrete reasons:

1. **The only authentic winning run on the whole board was produced on SM120** — the same
   architecture as our RTX 5090 tower. Its full solution source is public and now vendored
   locally (see Sources). It solves, at production quality, exactly the sub-problems an
   RWKV-7 megakernel needs: cooperative-launch work partitioning for bandwidth-bound
   GEMV chains, grid barriers + producer-consumer spin counters, fused int4 dequant-GEMV,
   and a **fixed-size fp32 recurrent-state update fused into the same launch** (KDA's
   `S[32,128,128]` delta-rule update is structurally the WKV-7 state update).
2. **Our own measured pain matches the problem's thesis.** On the 5090, our 0.1B fp16
   bsz1 decode is 260.6 tok/s eager vs 1452.4 tok/s graphed [measured, below] — a 5.6×
   launch-overhead tax on a small recurrent model — and the spec-decode draft is the one
   place where sglang's CUDA-graph machinery is architecturally blocked
   (`DecodeCudaGraphRunner` hard-codes EAGLE-style TARGET_VERIFY shapes; F0046). A
   persistent megakernel does not hide launches, it removes them, and it is the only
   escape hatch that doesn't require patching shared sglang spec infrastructure.
3. **Same-card bsz1 standing**: on the 5090 albatross does 554.0 tok/s (1.5B) and
   1779.7 tok/s (0.1B) bsz1 vs our 409.2 / 1452.4 [measured, below]. bsz1 is Bo's home
   turf; a megakernel is the strongest remaining lever on that axis.

Scope note: this is a **bsz1/latency-axis** proposal. It does not replace the W-series
concurrency work (w8a8 large-M on sm120 etc.); it is the complementary latency endgame.

## Part 1 — anatomy of the winning artifact

Sources (all vendored under
`a local clone of github.com/Infatoshi/kernelbench.com (outside this repo)`):
- problem contract: `benchmarks/mega/problems/02_kimi_linear_decode/{PROMPT.txt,reference.py,baseline.py,problem.yaml,benchmark.py}`
- winning solution (claude-fable-5, RTX PRO 6000 SM120, 2026-07-01):
  `public/runs/20260701_172615_claude_claude-fable-5_02_kimi_linear_decode_solution.py.txt` (1463 lines, read in full)
- judge annotation: `benchmarks/mega/results/annotations/20260701_172615_claude_claude-fable-5_02_kimi_linear_decode.yaml`
- the full board: 28 annotations for this problem (all read for verdict/score).

### 1.1 The scoreboard, honestly read

Score = baseline.py latency / solution latency, geomean over context 2048/8192/16384;
correctness = cosine ≥ 0.98 vs reference on output + both decode states, 3 seeds × ctx.

| run | score | verdict | approach |
|---|---|---|---|
| claude-fable-5 (2026-07-01) | **18.71×** | clean, **megakernel_authentic: true** | ONE cooperative CUDA kernel/step |
| claude-opus-4-8 (best of 3) | 19.35× | clean, but authentic=false | Triton fused int4 GEMVs + **CUDA-graph replay** |
| claude-opus-4-8 (earlier) | 14.40× | clean | 9-kernel Triton + graph |
| grok-4.5 | 18.94× | **contamination** | (disqualified) |
| glm-5.2 | 7.3–11.1× | clean | multi-kernel/graph |
| gpt-5.5 | 4.3–9.4× | clean | multi-kernel |
| claude-sonnet-5 | 4.03× | clean | multi-kernel |
| gpt-5.6-sol | 2.64× | clean, authentic=true | authentic single kernel, but slow |

Three honest takeaways [measured, from the annotations]:
- **A true megakernel matched — did not beat — the best CUDA-graph replay in absolute
  time** on this 4-layer problem (18.71× vs 19.35×; both ~0.29–0.31 ms/tok). The "+30%
  from true fusion" story only holds vs the *mid* graph run (14.4×). Fusion's absolute-time
  edge grows with layer count and with places graphs can't go — which is precisely our
  0.1B draft (12 layers × K steps, graphs blocked), not a 4-layer motif.
- **Single-launch alone is worth nothing** (gpt-5.6-sol: authentic megakernel, 2.64×,
  7× slower than the winner). The score lives in the fused dequant-GEMV quality and the
  work partitioning, not in the launch count.
- The winner ran at **~55–60% of the DRAM bandwidth floor** (312.6 µs/step measured vs
  ~185–210 µs floor at ~230–290 MB/tok on a ~1.8 TB/s card) [measured, annotation]. That
  is the realistic efficiency class for a first-generation megakernel on a hybrid model;
  we should not project above it without evidence.

### 1.2 The mechanisms (from the solution source, line-level)

**Launch skeleton.** `cudaLaunchCooperativeKernel`, grid = **188 blocks × 512 threads**,
75 KB dynamic smem/block (opt-in via `cudaFuncSetAttribute`), `__launch_bounds__(512, 1)`.
188 = the RTX PRO 6000's SM count (F0027 measured SMs=188 on that card) → exactly 1
co-resident block/SM; the host asserts
`cudaOccupancyMaxActiveBlocksPerMultiprocessor × SMs ≥ NBLK` before first use. All
weights/state/scratch pointers are packed into one `P` struct passed by value; a
`mk_setup` registry flattens the module tree once, `step()` is one C++ call.

**Grid barrier.** Hand-rolled sense-reversing barrier, NOT `grid_group::sync()`:
thread 0 of each block atomicAdd's a global counter; the last block resets it and flips a
global sense cell; others spin on the volatile sense; `__syncthreads()` + fences bracket
it. ~1 µs each [measured, solution docstring]; **14 barriers per token** (3 per KDA
layer, 5 for MLA+MoE). The block-local sense is re-seeded from the global cell at kernel
entry so the barrier survives across launches.

**Producer-consumer spin counters instead of extra barriers.** Within a stage, a `done[]
array of self-resetting atomic counters carries fine dependencies: the KDA state update
spins on per-head readiness (q/k/v/g tile epilogues + beta each arrive once → 5), the MoE
router result gates expert GEMVs (16 producer blocks arrive; consumers spin then run a
per-block redundant top-8 from smem — redundant compute instead of a barrier), expert
down-projections spin on per-expert `silu(gate)*up` tile completion. The last consumer
resets each counter for the next step — no zeroing pass.

**Work partitioning.** Every stage is a flat task loop
`for (t = blockIdx.x; t < NTASKS; t += NBLK)`; a task = (projection, 128-column tile,
split-K chunk). Split-K blocks write fp32 partial slices to global scratch;
`arrive_tile()` atomically counts arrivals and the **last-arriving block sums the slices
and runs the epilogue inline** — conv-window shift + SiLU, RoPE, cache append, residual
add, activation. Epilogues therefore cost no extra kernel and no extra barrier. Task
index remapping staggers block→task assignment so blocks that just produced (router rows)
don't immediately spin as consumers, and index-free tasks (shared expert) are scheduled
first to overlap producer latency.

**Fused int4 dequant (never materialized).** Weights stream straight from global/L2 as
`uint32` (4 columns × k-pair); activations are staged once per stage in smem as fp32.
Dequant is SIMD in bf16x2 lanes with zero CVT-pipe traffic: nibble n is splatted into a
bf16 lane as `0x4300 | n` (= 128+n exactly), `HSUB2` against a precomputed `(128+z)`
vector is exact, `HMUL2` by the bf16 scale applies the reference's single
round-to-nearest — **bit-matching the reference dequant**, so router logits come out
bit-identical and top-8 boundary flips are impossible. A second fp32-FMA path
(`w = rtb(fma(n, s, -z*s))` with an integer-trick bf16 rounding) serves the absorbed-MLA
stages. Group scale/zero pairs are loaded per 64-row segment and byte-permuted into lane
order — registers, not smem.

**Recurrent state in-kernel (the WKV-7 analog).** KDA's `S[32,128,128]` fp32 lives in
global memory (the state tensor the Python side owns) and is updated read-modify-write
inside the kernel: 128 tasks = 32 heads × 4 column-slabs; each 512-thread block loads the
head's q/k/v/decay into smem, each thread owns 8 rows × 1 column of S, computes
`S = S*exp(g)`, `pred = Σ S·k`, `delta = beta·(v−pred)`, `S += k⊗delta`, `out = Σ S·q`
with two smem reductions. State traffic = 2 MB/layer read+write — irrelevant next to
weights. **This is structurally our WKV-7 update** (decay → delta-rule write → readout);
ours is per-head 64×64 fp32 instead of 128×128.

**Everything stayed in the kernel.** MLA's growing latent cache is appended in-kernel
(the Model owns a capacity buffer; an `ingest` flag makes the kernel itself copy an
externally-fed cache before first use; Python only re-slices views). `step()` performs
zero eager torch ops. Note for our mapping: **the Kimi problem has NO embedding and NO
lm_head** — the winner never had to fuse a 65536-wide vocab GEMV. That part is new
engineering for us (see §4.3).

**Dev methodology worth copying** [measured, annotation]: per-stage `clock64()` stamps
written to a global array by block 0 (`p.stamps`, shipped in the final kernel), a
standalone GEMV unit test vs the reference quantizer, and a DRAM-rate sweep harness.
Trajectory within one session: 14.4× → 17.6× → 18.7×. The two same-model failures
elsewhere on the board: one session died at a provider rate limit after only the eager
correctness skeleton (0.48×, honest artifact), and gpt-5.6-sol shows "authentic but
unoptimized" scores 2.64×. Lesson: the megakernel must be built *fast-path-first* with
stage timing from day one, not correctness-first-then-hope.

## Part 2 — mapping onto RWKV-7

### 2.0 Ground truth about our decode path (read from source)

`sglang_overlay/sglang/srt/models/rwkv7.py` per layer at bsz1 fp16 (fast path fully on):
fused shift+lerp6 glue → r/k/v GEMVs (`gemv_m1`, 1 block per 8-row output tile, fp32
accum) → `lora4_m1` (all w/a/g/v LoRA chains in 2 launches) → fused gate activations (1)
→ `fused_kk_kmix` (1) → WKV recurrence (our Triton `wkv_recurrent`, fp32 state
`[size+1, H, 64, 64]` in the mamba pool, in-place indexed) → GroupNorm → `fused_gate_corr`
(1) → o_proj GEMV → ffn shift+lerp1 glue → ffn.key GEMV (+fused sqrelu epilogue) →
ffn.value GEMV (or sparse-cmix). Then final LN + lm_head GEMV + sampler.

Launch count [measured, F0051, H100, 1.5B]: **29.0 kernels/decoder-layer** (22.0 with
fused gates), ~699/step (~531 fused) at 24 layers. The "~144/step" figure in the task
brief is the stale note F0051 explicitly corrected. For the 0.1B: 12×29+3 ≈ **351/step**
(~267 with gates) [computed].

Correction to F0020's headline (matters for Amdahl): its "lm_head = 58.5% of the graphed
step" uses a **one-layer + head denominator**. Recompute from its own numbers [computed]:
per-layer components sum ≈ 219 µs (3090, 1.5B), step = 24×219 + 316 (head) ≈ 5.57 ms →
lm_head ≈ **5.7% of the full step**, consistent with the measured 203–226.5 tok/s and
with the 2.4 GB/step layer traffic at ~936 GB/s (2.56 ms floor — 268 MB in 316 µs is 91%
bandwidth for the head itself, that part stands). On the H100 profile the head is
95.8 µs of a ~2.4–2.5 ms step ≈ 4% [measured, F0051]. **There is no lm_head Amdahl
blocker at 1.5B**; at 0.1B the head is 100.7 MB of 287 MB/step = 35% of bytes — it must
simply be *inside* the megakernel (and for the draft we want its argmax in-kernel anyway).

### 2.1 Byte accounting and ceilings (RTX 5090)

Card constants: 1.792 TB/s theoretical GDDR7 (512-bit × 28 Gbps) [spec]; realistic
achievable stream ~88–92% ≈ 1.58–1.65 TB/s [estimate — measure with a memcpy/stream
probe at Stage A0]; 170 SMs [spec — verify at runtime; F0027 measured 188 on the RTX PRO
6000, same GB202 silicon]; ≥75 KB/block opt-in dynamic smem proven on GB202 by the winner
[measured].

Per-step bytes = layer weights (read once) + lm_head + emb row + fp32 state r/w +
activations. Weight totals from measured checkpoint bytes where available
(`bench/results/clean_ours_*.json` `weight_bytes`), else from dims.

**(a) 0.1B fp16** (d=768, L=12, 12 heads×64, inter 3072, vocab 65536; total weights
382.07 MB [measured]):
- emb = head = 65536×768×2 B = 100.66 MB → layers = 382.07 − 201.33 = 180.75 MB
- state: S = 12×12×64×64×4 B = 2.36 MB, ×2 (r+w) = 4.72 MB; shift states ≈ 0.15 MB; acts ~1 MB
- **per step ≈ 287 MB → floor 160 µs @1.792 → ceiling 6243 tok/s** (5580 @1.6) [computed]
- measured: ours graphed **1452.4 tok/s** (23% of ceiling), ours eager 260.6 (4.2%),
  albatross 1779.7 (28.5%) [all measured, 5090]
- megakernel at winner-class 45–60% of floor → **2800–3750 tok/s** [estimate]. Headroom
  is real and large: ~2–2.6× over albatross, ~1.9–2.6× over our graphed path.

**(b) 1.5B fp16** (d=2048, L=24, 32 heads×64, inter 8192; total 3054.81 MB [measured]):
- layers = 3054.81 − 2×268.44 = 2517.9 MB; head 268.44 MB read per step
- state 24×32×64×64×4 = 12.58 MB ×2 = 25.2 MB; shift ~1.6 MB
- **per step ≈ 2815 MB → floor 1.571 ms → ceiling 636.5 tok/s** (568 @1.6) [computed]
- measured: ours **409.2** (64% of ceiling), albatross **554.0** (87% of ceiling!) [5090]
- megakernel at 85–92% → **541–585 tok/s** [estimate]. Honest reading: at 1.5B a
  megakernel ≈ *catching albatross, +0–6% beyond it*. Albatross proves 87% is reachable
  WITHOUT single-launch (per-layer glue fusion); the megakernel is one way to get there
  and the only obvious way past ~90%, but the absolute upside over a fully-glued
  multi-kernel path is modest. (Consistent with Part 1: fable-megakernel ≈ opus-graph on
  4 layers.)

**(c) 1.5B int4 (our g64 symmetric)**: quantized r/k/v/o+ffn = 50.33 M params/layer ×24
= 1208 M × 0.53125 B/param (0.5 nibble + fp16 scale /64) = 641.7 MB; LoRA/norms fp16
102.5 MB; head fp16 268.4 MB; state 25.2 MB:
- **per step ≈ 1040 MB → floor 580 µs → ceiling 1723 tok/s** (1538 @1.6) [computed]
- measured: 259.1 tok/s on the **3090** [F0020 context]; 3090 ceiling = 900 → 29%.
  No 5090 w4 bsz1 measurement exists yet [gap].
- megakernel at 55–70% → **950–1200 tok/s (5090)** [estimate]. **This is the largest
  relative headroom of the mid sizes (~3×)** — int4 shrank the bytes 2.7× but the launch
  structure didn't shrink, which is exactly the Kimi problem's core thesis (int4 only
  pays if the *whole step* shrinks around it).

**(d) 7.2B int4 (g64 sym)**: quantized 201.3 M/layer ×32 = 6442 M × 0.53125 = 3422.6 MB;
fp16 rest ≈ 439 MB; head 536.9 MB; state 67.1 MB:
- **per step ≈ 4470 MB → floor 2.494 ms → ceiling 401 tok/s** (358 @1.6) [computed]
- measured: 102.8 tok/s on the 3090 (49% of that card's 209 ceiling) [F0017]; no 5090 run.
- megakernel at 60–75% → **240–300 tok/s (5090)** [estimate] vs albatross-fp16 147.0
  (but see audit flag below). Accuracy tax of 7.2B w4 (−3.1 pt symmetric) is a separate,
  already-documented tradeoff.

**(e) 7.2B fp16** for reference: per step ≈ 13.93 GB → floor 7.77 ms → **ceiling
128.7 tok/s** [computed]. ⚠️ Albatross's measured 147.0 tok/s (6.80 ms p50)
**exceeds this theoretical ceiling** (implies ≥2.05 TB/s effective). Before citing that
number anywhere, audit it: either its harness excludes part of the step, the card's
memory is clocked above 28 Gbps, or the run needs re-verification. Flagged per house law
(numbers must survive adversarial review); do not build comparisons on it until resolved.

### 2.2 The spec-decode draft application — necessary, but not sufficient

Current state [measured, F0046]: correctness done (`bench/spec_gate.py` 10/10
token-identical, 128 and 256 gen-len). Speed: spec-on **53.7 tok/s median** with the
hand-rolled draft CUDA graph (33.2 eager) vs spec-off **240.7** → 3.5–4.5× slower.
α = 0.738 → ~2.98 target-tokens/round at K=4 [measured, F0029].

Do the round arithmetic [computed]: 2.98 tokens / 53.7 tok/s = **55.5 ms per round**.
Known components: 3 draft steps (graphed, sub-ms to ~1 ms class each) + 1 target verify
(M=4 extend, low-single-digit ms class) ≈ **5–6 ms**. The other **~50 ms/round** is
python orchestration / per-layer state `.clone()` snapshots / worker plumbing — F0046
explicitly lists these as unprofiled hypotheses. This residual dominates so hard that
**even a zero-cost draft forward would only move spec-on from ~53.7 to ~57 tok/s**
[computed: 2.98/(55.5−3)]. A megakernel cannot fix python.

What the megakernel *does* uniquely fix on the draft side:
- It replaces the entire K-step draft loop with **one launch**: emb-row gather → 12
  layers → lm_head → argmax → next emb row, K times, grid-barriered between steps, state
  carried in registers/smem/global. This kills (i) all ~350×K kernel launches/gaps,
  (ii) the DecodeCudaGraphRunner blocker permanently, and (iii) the *inter-step python*
  — including writing the per-step `intermediate_ssm` / `intermediate_conv_window`
  snapshots in-kernel (the capture loop added in F0046 becomes kernel stores).
- Weight strategy: re-stream per step. 287 MB/step IS the bandwidth floor; nothing fits
  residency (170 SMs × ~100 KB smem ≈ 17 MB total on-chip [computed] vs 181 MB of layer
  weights — pinning is fantasy, L2 (~96 MB class on GB202 [spec-estimate]) will
  naturally serve the hot loras/norms).
- Floor for a K=4 round's draft side: 4 × 160 µs = 0.64 ms; at 50% efficiency ~1.3 ms
  [computed/estimate] vs the current graphed ~3 ms and eager ~10+ ms class.

End-state math if — and only if — the orchestration residual is separately fixed to
~1 ms/round [estimate]: round = 1 (orch) + ~2.5 (verify) + ~1.3 (mega draft at 50%) ≈
4.8 ms → 2.98/4.8 ms ≈ **620 tok/s vs spec-off 409 → net ≈ 1.5×**; at 60–75% draft
efficiency and a 2 ms verify: **~700–745 tok/s, net ≈ 1.7–1.8×** — the F0029 viability
(~2×) with realistic frictions. So: the megakernel is the *draft-side endgame piece*,
but the **binding constraint today is the 50 ms residual**, and the first action item is
a profile, not CUDA (Stage A0).

### 2.3 Feasibility constraints, honestly

1. **Grid-barrier overhead scales with layer count and bites the small model.** Winner:
   14 barriers × ~1 µs over a 313 µs step = 4.5%. Us, 0.1B: ~3–4 barrier-equivalents per
   layer × 12 + head ≈ 40–50 µs vs a 160 µs floor = **25–31%** if done naively
   [computed]. Mitigation is exactly the winner's: producer-consumer spin counters for
   intra-layer deps (WKV state update spins on per-head r/k/v/w readiness — the direct
   analog of KDA-update-spins-on-epilogues, proven pattern), barriers only at true global
   reconvergence. Target ≤2/layer. This is why the 0.1B projection above says 45–60% of
   floor, not 85%. At 1.5B/7.2B barriers are 4–6% — a non-issue [computed].
2. **Cooperative co-residency forces the tile budget.** 1 block/SM at 512 threads +
   ~75 KB smem (winner-proven on GB202). ≤170 blocks on the 5090 [spec]. Activation
   staging in smem: xn fp32 = 3 KB (0.1B) / 8 KB (1.5B) / 16 KB (7.2B d=4096); the worst
   K is 7.2B ffn.value K=16384 → 64 KB fp32 (or 32 KB fp16) — fits. All our projection
   widths are multiples of 128 (768…65536) → clean 128-col tiles; lm_head = 512 tiles ×
   split-K, ~6–18 waves over 170 blocks — plenty parallel [computed].
3. **lm_head + argmax in-kernel is new territory** (the winner had no vocab head at
   all). 65536×768 fp16 GEMV is embarrassingly tileable; argmax = per-block max/idx +
   one barrier + final reduce. For the draft we only need the argmax token (greedy
   chain), not the logits — sampler bypass is exactly what the spec worker wants. For
   the serving path (Stage B/C) the megakernel can stop at hidden-out and let sglang's
   LogitsProcessor run lm_head unfused first (worth 35% of 0.1B bytes staying unfused —
   acceptable for stage 1, fuse later).
4. **fp32 state changes nothing vs Kimi**: their S was fp32 in global too, RMW'd
   in-kernel. Ours is smaller per layer (12×64×64 vs 32×128×128). Our conv (token-shift)
   state is a trivial 1-row fp32 buffer. State bytes are ≤67 MB/step even at 7.2B —
   noise next to weights [computed].
5. **Bit-exactness is the house gate and it is achievable but laborious.** Precedent:
   `gemv_mb` and `lora4_m1` both reproduce torch/gemv_m1 reduction orders bit-for-bit;
   the winner's LOP3 dequant bit-matches its reference's rounding. The megakernel must
   reproduce (i) `gemv_m1`'s per-tile fp32 reduction order (same skeleton → doable),
   (ii) the Triton `wkv_recurrent` update order (port its loop structure literally),
   (iii) LoRA/gate/GroupNorm elementwise orders. Where an order can't be matched, fall
   back to the F0031-class near-tie argmax audit and say so. Gate battery per stage:
   `bench/spec_gate.py` 10/10 (draft), `verify_m1d` greedy-EXACT 24/24 + `verify_batch`
   (serving), fixture 8/8 (7.2B int4).
6. **sglang integration is pluggable, with one open question.** The natural shape is an
   opt-in `RWKV_MEGAKERNEL=1` fast path dispatched at `Rwkv7ForCausalLM.forward` when
   (decode ∧ bs==1 ∧ fp16 ∧ tp=1 ∧ pp=1 ∧ eligible weights): flatten weights once into a
   `P`-struct registry (winner's `mk_setup` pattern), read state via the mamba pool
   tensors + `mamba_cache_indices[0]` (pad-slot row-0 convention already exists), fall
   back to the normal path otherwise — identical governance to the existing seven
   RWKV_* flags. Open question [unverified]: whether `cudaLaunchCooperativeKernel` is
   capturable inside CUDA-graph stream capture on our CUDA 12/13 stack. If not: run the
   megakernel outside the graph — at 1 launch/step that costs microseconds, which is the
   whole point (the draft variant never wanted a graph anyway).
7. **Occupancy tension int4-side**: the dequant ALU (LOP3/HSUB2 or our fp32-FMA w4
   inner loop) at 512 thr/SM single-block occupancy needs the winner's SIMD trick to
   stay bandwidth-bound; ours is symmetric g64 (no zeros!) — strictly simpler than
   Kimi's asymmetric g128 (one HSUB2 disappears; scale layout [N, K/64] already serves
   `gemv_w4_m1`).
8. **Host overhead bounds serving gains** (not draft gains): 0.1B graphed serving wall
   is 689 µs of which the GPU step is maybe 300–450 µs [estimate — A0 profiles it];
   sglang's per-step host cost (~0.24–0.39 ms) would then cap a 250 µs megakernel at
   ~2000–2900 tok/s served unless the overlap scheduler hides it. Engine-loop numbers
   (albatross-comparable) don't pay this tax.

### 2.4 Staged build plan (gated, smallest-increment)

- **Stage A0 — profile + probe (0.5–1 day).** (i) Profile one spec round end-to-end on
  the tower: split the 55.5 ms into draft-forward / verify / clones / python (F0046's
  open question). (ii) 5090 ground truth: deviceQuery SM count, memcpy/stream achievable
  GB/s, coop-launch-in-graph-capture yes/no, and re-audit the albatross 7.2B 147 tok/s
  anomaly. **Exit criteria**: a table that says where the 50 ms lives, and the real
  bandwidth constant for all ceiling math. Riskiest unknown: none — this is pure
  measurement, and it can *downgrade* Stage B's priority if verify/orchestration
  dominates beyond repair.
- **Stage A — 0.1B single-step megakernel, standalone (2–4 days).** One cooperative
  kernel: 12 layers (shift+lerp, r/k/v GEMVs, LoRA chains, WKV fp32 update, GroupNorm,
  gates, o_proj, ffn) + optional lm_head+argmax, weights re-streamed, state in global
  fp32. Built winner-style: GEMV unit test vs `gemv_m1` bits → per-stage `clock64`
  stamps → stage-by-stage fusion. Gate: **bit-exact hidden/state vs the current engine
  path on fixed states** (and if bit-exact fails, quantified ULP + argmax-flip audit),
  plus tok/s on the 5090 vs the 1452/1780 standings. Riskiest unknown: barrier/spin
  overhead at 12-layer scale (abort/redesign threshold: >35% of step in
  sync at the stage-timing readout).
- **Stage B — K-step draft variant + spec integration (1–2 days after A).** Loop K
  steps in-kernel (emb gather → … → argmax → next token), write `intermediate_ssm` /
  conv-window snapshots per step in-kernel, wire into `rwkv_spec_worker.py` (lives in
  the sglang-upstream fork branch `rwkv7-spec-decode`, not this repo). Gate:
  `bench/spec_gate.py` **10/10 token-identical** + non-spec regression suite, then A/B
  vs the draft-graph. Net spec headline additionally requires the A0-identified
  orchestration fix (likely python-side, separate workstream). Riskiest unknown: the
  50 ms residual staying dominant — in which case B still ships as "draft forward
  solved, round overhead is the remaining work", with the round math shown.
- **Stage C — 1.5B fp16 single-step (2–3 days).** Same kernel dim-templated (d=2048,
  L=24). Target: ≥554 tok/s — **flip the same-card albatross bsz1 standing**. Gate:
  greedy-EXACT 24/24 + `verify_batch` untouched-fallback proof. Riskiest unknown:
  matching albatross's measured 87%-of-ceiling in one launch (their glue is per-layer
  hand-fused; we must at least tie before the single-launch structure adds anything).
- **Stage D — int4 variants (2–3 days, opportunistic).** Reuse g64-sym format +
  `gemv_w4_m1` inner loop inside the megakernel; targets from §2.1(c)/(d): 1.5B-int4
  ~950–1200 tok/s (biggest headroom, ~3×), 7.2B-int4 ~240–300 tok/s on the 5090. Gate:
  fixture-EXACT 8/8 (7.2B), lambada within the already-published w4 deltas. Riskiest
  unknown: dequant ALU pressure at 1-block/SM occupancy (winner's SIMD trick is the
  mitigation, and ours is simpler).

Total: ~1.5–2 weeks of focused work for A0→C, D optional. Each stage is independently
publishable and independently abortable.

## Verdict: GO (scoped), first step = Stage A0

- **0.1B (draft + small-model showcase): clear GO.** 2–2.6× measured-headroom over the
  best existing path on the target card, the one place CUDA graphs are architecturally
  blocked, the winner's SM120 playbook covers every hard sub-problem except the (easy)
  vocab head, and it doubles as the first true single-launch RWKV-7 decode step in the
  ecosystem (albatross is per-op launches with fused glue, F0023; verify competitors
  again at publish time per house law).
- **1.5B fp16: GO as the albatross same-card flip** (409 → target ≥554), with the honest
  caveat that the ceiling grants only ~+6% beyond albatross — the win is parity+structure,
  not a blowout.
- **int4 (1.5B/7.2B): the largest headroom (~2–3×)** and the purest transplant of the
  Kimi problem's thesis; do after C on the same skeleton.
- **Spec-decode net win: NOT promised by this ADR.** The megakernel fixes the draft
  side; the measured round budget says ~50 ms/round lives outside any kernel. A0's
  profile decides that workstream; the ADR's math shows net ~1.5–1.8× once both pieces
  land.
- **Do not reallocate W1 (sm120 w8a8 large-M) capacity to this** — different axis;
  this is the bsz1/latency and spec-draft lever.

Single highest-EV first step: **Stage A0** — one day, zero risk, converts the two
load-bearing unknowns of this document (the 50 ms spec residual; the 5090's real
achievable bandwidth + coop/graph capture support) into measurements, and re-audits the
one competitor number that currently violates physics.

## Cross-references

Winning artifact + board: vendored at
`a local clone of github.com/Infatoshi/kernelbench.com (outside this repo)` (clone of
`github.com/Infatoshi/kernelbench.com`, 2026-07-10; run pages browsable at
kernelbench.com/runs). Problem contract: `benchmarks/mega/problems/02_kimi_linear_decode/`.

This repo: `sglang_overlay/sglang/srt/models/rwkv7.py` ·
`.../layers/attention/linear/rwkv7_backend.py` ·
`.../rwkv7_kernels/cuda/{rwkv7_fast.cu,rwkv7_w4.cu,rwkv7_lora.cu,rwkv7_glue.cu}` ·
`bench/spec_gate.py` · `bench/results/{clean_ours_0.1B_5090main.json,`
`clean_ours_1.5B_5090main.json,bsz_sweep_fullstack_5090.json,albatross_5090/retuned_summary.json,`
`throughput_{0.1b,1.5b}_fp16_5090.log}` · findings [[F0017]] [[F0020]] [[F0023]] [[F0027]]
[[F0029]] [[F0046]] [[F0051]] [[F0052]] · ADR-0004 (no-FLA law) · ADR-0006 (spec-decode) ·
ADR-0005 (reverse-overtake roadmap).
