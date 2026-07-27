---
doc_kind: finding
finding_id: F0068
title: "Stage-B cut 3 (#57): before building the banked inline-lerp GEMV, decompose what the boundary cluster actually costs. Measured on sm86 at decode T=1: add_ln = graph-node dispatch floor (1.07us) + a width-INDEPENDENT residual + ~0.21us per element-per-thread; the deployed (32,16) WIDE tier is the OPTIMUM of the thread ladder (bracketed both ways — (32,32) WIDER is SLOWER, so one block already saturates its SM and adding threads buys nothing). Two results: (1) FastTree — replay the aten inter-warp Welford schedule inside warp 0 with __shfl_down instead of 4 rounds of smem ping-pong, 9 block-wide syncs -> 2, BIT-IDENTICAL by construction, measured -6.5% (N=4096) / -7.1% (N=2048) on add_ln; (2) the banked F0066 §5 inline-lerp boundary kernel's -230us/step projection does NOT survive the arithmetic — its boundary kernel is a strict superset of add_ln, and F0066a's own failed J=6 kernel already measured the single-block marginal cost (~20 GB/s) that makes it a NET LOSS; (3) the cross-block row SPLIT this finding nominated as the next lever was then BUILT and measured in the same round — its parallelisation works (work term -26%) but sm86's second dispatch floor eats all of it, making it the one arm whose verdict genuinely requires sm120 (where PDL runs 96.8% overlapped with negative inter-kernel gaps); (4) cut 3 then measured the two largest UNEXAMINED items in the addressable pool — sparse_cmix (atomics REFUTED as the bound at 2.8-6.1%; it runs at 81-87% of the 91.3%-of-peak this kernel class actually achieves) and the LoRA pair (71.8% of achievable, the worst ratio in the kernel set, ~131us/step of headroom) — which CLOSES the budget: every headroom identified, taken in full and to unreachable ceilings, sums to ~240us/step against the -560us needed, so the remaining 8% to Bo is NOT in the per-kernel efficiency of this decode structure."
status: PARTIAL (2026-07-27) — sm86 mechanism + gates GREEN, sm120 step-level effect UNMEASURED (the 5090 box was offline for this round; three independent reachability probes all failed). FastTree gates: bit-identity 96/96 torch.equal across both launcher tiers (bench/test_addln_fasttree.py) + the torch transcription contract preserved at the parity tier, 0 differing bytes, at FastTree=0 AND =1 (bench/test_addln_numerics.py). WIDER (32,32) tier landed in-tree, default OFF, published as an honest NEGATIVE; SPLIT (cross-block, tier 3) landed in-tree, default OFF, CONDITIONAL negative pending sm120. Cut 3 closes the #57 budget: the -560 us/step needed to pass Bo is not available in per-kernel efficiency (~240 us total headroom to unreachable ceilings) — the remaining gap is structural (Bo ~13 kernels/step vs our 469), which is a scope call for the user, not a kernel task. Both new knobs default OFF; nothing promoted to serve.sh defaults until a 5090 A/B runs
discovered_by: Opus 5 (1M), 2026-07-27
severity: info
related: [F0066, F0065, F0064, F0063]
machine: authored on the Mac tree; ALL measurements on the 3090 box (sm86, CUDA 12.9) in container rwkvmain — the 5090 tower was offline for this round
---

# Finding F0068: what the add_ln boundary actually costs, and why the banked inline-lerp projection does not survive it

## 0. Why this round exists

[[F0066]] §5 banked the "inline-lerp GEMV" as the successor to its failed J=6
boundary kernel, with an estimated **−230 us/step**: a compact boundary kernel
(`~2us`, one block/row) does add + LN + conv-scatter and writes only `(y, d)`
where `d = round_fp16(prev − y)` is the role-independent lerp delta, and every
consumer recomputes its own `x_role` in registers during its x-load.

House law after [[F0064]] and [[F0066]]a is explicit: *nominate no lever before
doing the per-kernel byte-floor arithmetic, store side included.* Two
projections have already died for skipping it. So this round did the
arithmetic first — and it killed the projection.

## 1. The measured boundary budget (5090, F0066-E0 per-kernel table, 7.2B, 32 layers)

BUSY/step 7202.5 us, span/step 7078.4 us (141.28 tok/s kernel-loop framing).

| kernel | calls/step | us/step | us/call | grid at T=1 |
|---|---:|---:|---:|---|
| gemv_grouped_m1 (rkv + o) | 64.56 | 2658.02 | 41.17 | BW wall |
| gemv_m1 (ffn.key) | 32.26 | 2602.94 | 80.69 | BW wall |
| **add_ln** | **64.56** | **255.09** | **3.95** | **ONE block** |
| shift_lerp6 | 32.28 | 56.83 | 1.76 | 16 blocks |
| shift_lerp1 | 32.28 | 47.07 | 1.46 | 16 blocks |

- The two GEMVs are **5261.0 us/step = 73.0% of BUSY**, already at 95.6–97.7%
  of achievable bandwidth ([[F0064]]) — a floor, not a lever.
- The cuBLAS `gemvx` lm_head is 324.08 us/step in **one** call: vocab 65536 ×
  H 4096 × 2 B = 537 MB ÷ 1691.7 GB/s = **318 us**. It is at ~100% of DRAM
  bandwidth — also a floor (corroborates [[F0064]] §10's "lm_head 98%").
- ⇒ **77.5% of BUSY is a bandwidth wall.** The entire addressable pool is the
  remaining **1617 us/step**. Reaching Bo's 155.2 tok/s needs span 7003 → 6443
  = **−560 us/step**, i.e. **~35% of everything that is not a bandwidth wall**.
- The whole boundary cluster (add_ln + both shifts) is **359.0 us/step = 5.0%
  of BUSY**. Even deleting it outright cannot close the gap to Bo.

## 2. Where add_ln's per-call time actually goes (3090, measured)

`bench/bench_addln_configs.py` + `bench/bench_kernel_floor.py` — CUDA-graph
replay (the deployed condition: same-stream nodes serialize, so N replays = N ×
kernel time), T=1, medians over 50 reps.

**Graph-node dispatch floor** (`relu_sq` on 4 elements = ~no work):

| probe | us |
|---|---:|
| null, 1 block | **1.073** |
| null, 16 blocks | 1.102 |
| add_ln (WIDE, N=4096) | 4.137 |

Two things fall out immediately: the dispatch floor is **1.07 us and does not
care about grid size** (1 → 16 blocks is +0.03 us), and it is **26% of add_ln's
whole per-call cost**. Fitting `t = f + k·N` across N ∈ {2048, 4096} at fixed
512 threads gives `k ≈ 0.21 us per element-per-thread` and a width-independent
residual of ≈1.6 us on top of dispatch.

## 3. Result 1 — FastTree: 9 block-wide syncs → 2, bit-identical (WIN)

The transcribed aten inter-warp tree spends **2 `__syncthreads()` per level** —
9 block-wide syncs at the deployed (32,16) WIDE config — to move ONE partial
per warp through shared memory. FastTree lands every warp's partial in smem
once (1 sync), then replays the **identical halving schedule** inside warp 0
with `__shfl_down` (lane *y* holds warp *y*'s partial; `offset = NW/2..1`
combines the same operand pairs in the same order), then publishes (1 sync).

**Bit-identity is by construction** — same combine sequence, same operands, no
reassociation — and is gated, not assumed:

| gate | result |
|---|---|
| `bench/test_addln_fasttree.py` — torch.equal on x_new **and** y, FastTree=1 vs =0, tiers (32,4)/(32,16)/(32,32), N ∈ {2048, 4096, 8192}, T ∈ {1,4}, normal/large/tiny/mixed rows | **PASS, 96/96** |
| `bench/test_addln_numerics.py` — parity tier still byte-exact vs torch add+LayerNorm, at FastTree=0 **and** =1 | **PASS, 0 differing bytes** |

Measured (3090, T=1, us/call, median of 50):

| tier | FastTree=0 | FastTree=1 | Δ |
|---|---:|---:|---:|
| parity (32,4), N=4096 | 7.113 | 7.088 | −0.4% |
| **WIDE (32,16), N=4096** | 4.428 | **4.138** | **−6.5%** |
| **WIDE (32,16), N=2048** | 3.559 | **3.305** | **−7.1%** |

The parity row is the **mechanism self-check**: parity has `blockDim.y=4` = 2
tree levels = 5 syncs, so FastTree should buy ~nothing there — and it buys
−0.4%. The gain appears only where the tree is deep, which is what "the cost
removed is sync-chain latency" predicts.

⚠ **What this does NOT establish.** −6.5% is a per-kernel number on sm86 in a
warm-cache loop. Carried naively to the 5090's 255.09 us/step it is ≈ −16.6
us/step ≈ **+0.23%** end-to-end — a *projection*, explicitly not a result.
[[F0064]] and [[F0066]]a both died at exactly this step. It must be measured on
the 5090 before any claim or default flip.

## 4. Result 2 — WIDER (32,32) is SLOWER (NEGATIVE, published)

Prediction under the width model: N=4096 at 1024 threads = 4 elements/thread =
the same per-thread load as N=2048 at 512 threads = **3.30 us**. **Refuted.**

| tier, N=4096 | FastTree=0 | FastTree=1 |
|---|---:|---:|
| WIDE (32,16) | 4.428 | **4.138** |
| WIDER (32,32) | 4.907 | 4.162 |

WIDER is worse in both arms (and needs FastTree just to claw back to parity
with WIDE, since it adds a 5th tree level). Mechanism: **at 512 threads the
single block already saturates its SM's issue capacity** — adding threads
inside one block does not add throughput, it only deepens the reduction. The
earlier N-halving gain came from halving total work, not from parallelism.

⇒ **The thread ladder is now bracketed in both directions and (32,16) is the
optimum.** The only remaining way to spend more of the GPU on this kernel is
*more blocks* (cross-block row split), not more threads. WIDER stays in-tree,
default OFF, as the evidence for that boundary.

Numerics for both new tiers were gated to the [[F0065]] bar (no farther from
fp32 truth than the parity tier): **11 of 12 cases are byte-identical to
parity**, the 1 case that differs is not farther from truth (worst max-error
delta 0.0). fp16 absorbs the partition change almost everywhere.

## 5. Result 3 — the banked −230 us/step projection does not survive (REFUTED, by arithmetic)

The banked design's boundary kernel does add + LN + conv-scatter and writes
`(y, d)`. That is a **strict superset of today's add_ln**, which measures
**3.95 us/call as a single block** — so it cannot be the design's assumed
`~2us`. And the marginal cost of adding bytes to exactly this single-block
kernel has already been measured, by [[F0066]]a's own failed experiment:

- `add_ln_shift<2,6>` = 10.43 us/call vs add_ln 3.95 ⇒ **+6.48 us for +128 KB**
  from one block ⇒ **~20 GB/s single-block marginal throughput**.

The `(y, d)` boundary kernel adds conv-read 16 KB + conv-write 16 KB + d-store
8 KB = **40 KB** ⇒ **+2.0 us/call** ⇒ ~5.95 us/call ⇒ **384 us/step**, against
today's composed 359.0 ⇒ **+25 us/step, a net loss** — the same single-block
store wall that killed J=6, reached by the same route.

The variant that *can* win keeps add_ln unfused and instead slims
`shift_lerp6` into a `shift_d` that writes only d (8 KB) instead of 6 planes
(48 KB), **retaining the 16-block grid**: estimated 1.76 → ~1.0 us/call,
boundary ≈ **−30 us/step ≈ +0.4%**, minus whatever the converted consumers pay
back in extra L2 reads. Honest expected value: **~+0.5%, not the banked +3–4%.**

Note also that `lora_stage1` is a **poor** conversion target: its own header
records it as latency-bound (6.4 us vs a ~1.8 us byte floor), and its x is 50%
of its per-block bytes, so inline-lerp would *double* its reads — the opposite
regime from the DRAM-bound GEMVs. Convert the rkv GEMV; leave stage1 alone
unless measured otherwise.

## 5b. Result 4 — SPLIT (cross-block row split) is a CONDITIONAL negative: the parallelism works, sm86's second dispatch eats it

§4 concluded the only remaining structural lever is *more blocks*. That was
built this round rather than left as a projection (`RWKV_ADDLN_WIDE=3`,
`RWKV_ADDLN_SPLIT_NB`, default OFF): p1 = grid (T, NB) does the residual add,
writes the x_new slice and one raw Welford partial per block into a persistent
scratch ([[F0066]]b's allocate-once precedent, capture-stable address); p2 =
grid (T, NB) folds ALL NB partials in the same fixed order in every block (so
every block derives bit-identical mean/rstd) and applies to its own slice.

NB sweep at N=4096 (us/call, one vec per thread): **NB=4 4.417 · 8 4.590 ·
16 5.669 · 32 7.835** — monotonically worse, and all above WIDE+FastTree's
4.159. Full ladder at the best NB=4:

| tier | N=2048 | N=4096 |
|---|---:|---:|
| parity (32,4) + FastTree | 4.567 | 7.092 |
| **WIDE (32,16) + FastTree** | **3.308** | **4.159** |
| WIDER (32,32) + FastTree | 3.490 | 4.149 |
| SPLIT (NB=4) | 4.017 | 4.401 |

But the decomposition says the idea is not wrong, the *card* is:

| | dispatch | work | total |
|---|---:|---:|---:|
| WIDE + FastTree | 1.07 | 3.07 | 4.159 |
| SPLIT NB=4 | ~2.15 (2 nodes) | **2.27** | 4.401 |

**The parallelisation delivered — the work term drops 26% — and sm86's second
1.07 us dispatch floor consumes all of it.** That floor is exactly what the
5090 does not have in the same form: with PDL armed it runs **96.8% overlapped
transitions and NEGATIVE inter-kernel gaps** ([[F0066]]b). If the second node
is even half as expensive there, SPLIT flips positive (3.07 → 2.27 on the work
term would be ≈ −66 us/step ≈ +0.9%). This is the same card-dependence class
as [[F0051]] (identical fusion: +9.24% on H100 vs +1.22% on L4).

⇒ SPLIT is **not rejected**, it is **unresolvable on sm86**. It stays in-tree,
default OFF, as a ready-to-measure arm for the first 5090 window. Numerics
gated the same way: 10/12 cases byte-identical to parity, the other 2 with
worst max- and mean-error deltas of exactly 0.0.

One implementation trap worth recording: the first SPLIT build sized blocks for
2 vecs/thread, which made NB×64 = 512 total threads — **the same thread count
as WIDE**, i.e. the same work on the same resident threads plus an extra
launch. It measured 5.04 us and looked like a clean refutation of the whole
idea. It was a harness bug in the launcher, not a result. Sizing for 1
vec/thread is what produced the numbers above.

## 5c. Cut 3 — the rest of the addressable pool, measured: the −560 us/step is NOT there

§1 established that only 1617 us/step of the 7202.5 us BUSY is not already at a
DRAM wall, and that beating Bo needs −560 us — ~35% of that pool. This cut
measures the two largest unexamined items in it, at the **real 7.2B geometry
read from the checkpoint config** (hidden 4096, layers 32, intermediate 16384,
LoRA low-rank dims decay/a/gate/v = 128/128/480/96 ⇒ R_total = 832).

**A floor must mean ACHIEVABLE bandwidth, not peak.** The sparse_cmix dense
case below tops out at **91.3% of peak** on the 3090; quoting peak-relative
numbers overstates headroom, which is the error this section exists to avoid.

### sparse_cmix (418.17 us/step, 25.9% of the pool) — atomics REFUTED, near its ceiling

Every one of the inter/FFN_TILE = 128 tiles atomicAdds into the same [H] fp32
buffer: 524k atomic f32 adds per call, ~2 MB of atomic traffic into 16 KB. That
is the obvious suspect, so it was priced directly (`_probe_cmix_noatomic`, a
probe op that swaps the atomicAdd for a plain store — **wrong results by
construction**, unreachable from any model path):

| density | atomic | plain-store probe | atomic cost | atomic share | % of peak BW |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 19.384 | 18.202 | 1.183 | 6.1% | 74.0 |
| 0.12 | 22.192 | 21.176 | 1.016 | 4.6% | 77.5 |
| 0.14 | 25.246 | 24.535 | 0.711 | 2.8% | 79.5 |
| 1.00 | 156.984 | 156.767 | 0.218 | 0.1% | **91.3** |

**Refuted:** in the real 86–90%-zero band the atomics are 2.8–6.1% of the
kernel. sparse_cmix is a weight-stream kernel running at 74–80% of peak, i.e.
**81–87% of the 91.3% this kernel class actually achieves.** What the sparsity
costs is coalescing (scattered 512 B row reads vs a dense burst), not the
combine. Headroom to a ceiling it cannot reach: **≤63 us/step.**

### the LoRA pair (463.4 us/step, 28.7% of the pool) — the largest real headroom

`lora4_m1` (stage1 + stage2, two launches), same harness, 3090:

| | value |
|---|---:|
| us/call | 24.412 |
| dispatch floor × 2 | 2.19 (9.0%) |
| work | 22.221 |
| DRAM (d_cat + u_cat) | 13.63 MB |
| achieved on work | 613.5 GB/s = **65.5% of peak = 71.8% of achievable** |

**~28% off its own ceiling — the worst efficiency ratio in the kernel set, and
the largest remaining headroom (~131 us/step if it could be closed).** A named
suspect for the next round, not yet tested: stage1 launches ONE block per
down-row, so the xs row is re-read by all 832 blocks — 6.82 MB of redundant xs
traffic, exactly equal to d_cat's own 6.82 MB. Row-tiling (R rows per block)
would amortise it, at the cost of dropping to 832/R blocks (on the 5090's 170
SMs, R=4 leaves ~1.2 blocks/SM — the trade-off has to be measured, not argued).

### The budget, closed

| item | us/step | state |
|---|---:|---|
| GEMVs + lm_head | 5585 | bandwidth wall — floor, not a lever |
| LoRA pair | 463 | 72% of achievable → ~131 headroom |
| sparse_cmix | 418 | 81–87% of achievable → ≤63 |
| add_ln | 255 | ladder bracketed; FastTree ≈ −17, SPLIT card-dependent |
| shift_lerp6/1 | 104 | the inline-lerp arm → ~30 |
| gn_gatecorr / wkv / kk_kmix / finalize / misc | 320 | unexamined |

**Every headroom identified so far, taken IN FULL and to ceilings that are by
definition unreachable, sums to ~240 us/step against the −560 us needed.** Even
adding an optimistic share of the 320 us not yet decomposed, kernel-level
optimisation of the current decode structure does not get from 142.8 to 155.2.

This does not say the gap is unclosable — it says **it is not in the per-kernel
efficiency of this structure.** Bo runs ~13 kernels/step against our 469; the
remaining difference is structural (how the step is decomposed into kernels at
all), not a matter of tightening the kernels we have. Whoever picks up #57
should treat "find another 560 us in these kernels" as measured-and-refused,
and price a structural change instead.

⚠ Scope: percentages are 3090 measurements used as a proxy for sm120, and the
"91.3% achievable" ceiling is taken from one kernel's dense case. Both should be
re-measured on the 5090 before this budget is quoted publicly.

## 6. Disposition + what is next

- FastTree: in-tree, `RWKV_ADDLN_FASTTREE=1`, **default OFF** pending a 5090
  A/B. Bit-identical + gated, so it carries no numerics risk of its own.
- WIDER: in-tree, `RWKV_ADDLN_WIDE=2`, **default OFF**, negative published.
- SPLIT: in-tree, `RWKV_ADDLN_WIDE=3` + `RWKV_ADDLN_SPLIT_NB` (default 4),
  **default OFF**, conditional negative — the one arm whose verdict genuinely
  needs sm120. Its scratch is allocated on first call; production use needs
  that first call to be an eager warmup (outside capture), which the model's
  warmup forwards already provide, but a SPLIT default-flip should assert it.
- Nothing promoted to `scripts/serve.sh` defaults this round.
- The `add_ln_shift` path is untouched (it keeps the default `FastTree=false`
  template arg and the matching `blockDim.y*3/2` smem allocation — the FastTree
  layout needs `blockDim.y*3` and the launchers size shared memory per config).

**The first 5090 window should run, in this order:** (1) FastTree A/B — gates
are green, only the step-level number is missing; (2) SPLIT A/B at NB ∈ {4, 8}
— the decisive card-dependent question of §5b; (3) only then consider the
`shift_d` + rkv-GEMV inline-lerp arm of §5, which is worth ~+0.4% and costs a
multi-kernel surgery.

**Scope discipline for whoever picks this up:** every number in this finding is
sm86, T=1, CUDA-graph, warm-cache, single-kernel. No step-level or tok/s claim
is made, and none may be published from this finding alone.

## 7. Artifacts

`bench/results/f0068/`: `addln_sweep.jsonl` (16 configs × 50 reps),
`split_nb_sweep.jsonl` (SPLIT NB ∈ {4,8,16,32}),
`sparse_cmix_floor.json` (density × atomic-probe A/B), `lora_floor.json`,
`kernel_floor.json` (dispatch-floor probe), `gate_fasttree.json`,
`gate_numerics.json`. Harness: `bench/bench_addln_configs.py`,
`bench/bench_kernel_floor.py`, `bench/addln_sweep.sh`,
`bench/bench_sparse_cmix.py`, `bench/bench_lora_floor.py`. Gates:
`bench/test_addln_fasttree.py`, `bench/test_addln_numerics.py` (both
self-contained — they load `rwkv7_ln.cu` directly and do not need the installed
sglang package, so they can gate a kernel without mutating a deployed tree).

[[F0065]] (the opener this continues) · [[F0066]] §5 (the design this
re-prices) · [[F0064]] (the byte-floor law both invoke) · task #57.
