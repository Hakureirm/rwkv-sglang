---
doc_kind: finding
finding_id: F0058
title: "task #54 hand-CUDA WKV decode kernel: bit-exact (zero differing bytes) vs the Triton kernel on BOTH state dtypes with fp32 in-register accumulation preserved; in-situ 6.54 ms/step at bs=320 fp16-state (target <=6.5 met within trace resolution) - but the campaign's decisive result is measurement archaeology: the task's 7.41 ms premise was stale (pre-glue-fusion regime; F0056's fusions had already bought the Triton kernel its drain window, 6.64 in-situ today, nobody re-measured), the competitor kernel is IDENTICAL to our Triton standalone (232.1 vs 232.4 us/layer, same ~1.52 TB/s in-place r+w wall) and its true equal-conditions bar is 6.19 ms/step (-5% vs ours); e2e c=320 is a wash by construction (+0.30% scheduler gauge, inside client-wall noise), real kernel wins land at bs<=128 (1.1-1.8x device-time) and eager bsz1 (3-6x launch-to-launch); kernel is the designated WKV component for the megakernel line (#50)"
last_verified_commit: "12d5a5f (kernel + loader + dispatch + gate)"
discovered_by: Fable 5 (agent), 2026-07-14
severity: info
status: closed - stage 1 landed (kernel + gates + microbench + e2e attribution closure); remaining -5% kernel-level gap vs competitor documented with a concrete next design (warp-split)
related: [F0056, F0047]
---

# Finding F0058: hand-CUDA WKV decode (task #54) - the kernel, its gates, and what the measurements actually said

## 0. TL;DR

- **Deliverable**: `rwkv7_kernels/cuda/rwkv7_wkv.cu` + loader + dispatch, env
  `RWKV_WKV_CUDA=1` (default OFF), decode T==1 indexed-pool path, both state
  dtypes. **Bit-exact vs the Triton kernel: zero differing bytes on o AND the
  state pool, both dtypes, pads included** - the stronger contract option (no
  tolerance fallback needed even for fp16 state). Batch-invariant by
  construction.
- **Where it wins**: bs=32 1.6-1.8x, bs=128 1.1-1.6x device-time vs the Triton
  kernel; eager launch-to-launch bsz1 3-6x (thin C++ op vs the heavy python
  launcher; captured graphs mask this in serving, real for eager/spec paths).
  bs>=320: +1.3-1.5% device-time (drain regime), in-situ 6.544 vs 6.638
  ms/step.
- **Why bs=320 doesn't flip serving**: the task premise (Triton = 7.41
  ms/step in-situ) was measured BEFORE the F0056 glue fusions became default.
  Un-fused glue (~15 memory-bound torch kernels right after WKV) used to eat
  the L2 drain window, pinning the WKV kernel at its standalone wall (232
  us/layer = 7.42 ms/step). The fusions replaced that tail with fused kernels
  + the compute-bound GEMM block, and the SAME Triton kernel now retires its
  state writes under the GEMMs: today it measures **6.638 ms/step in-situ**.
  Nobody had re-measured the kernel after the fusions landed. This finding
  closes that attribution gap with an apples-to-apples in-situ pair.
- **Competitor reality check** (kills a narrative, keeps a bar): their CUDA
  WKV (`wkv_fp16_v1_clone_kernel`, PR#8 binary) measures **232.1 us/layer
  standalone - identical to our Triton kernel (232.4)**; both sit at the same
  ~1.52 TB/s in-place scattered-row r+w wall. There was never "0.9 ms of
  hand-CUDA kernel skill". BUT in the drain regime their kernel does hold a
  real edge: 193.3 vs our 203.5 us/layer (**-5%**), structural (their fp16
  hfma2 arithmetic halves state register pressure -> ~11-12 resident
  blocks/SM vs our 9; our fp32-accumulation contract deliberately refuses
  that trade). Concrete next design to close it is documented (S6).
- e2e (this session, same protocol as F0056): shape A c=320 anchor 9,394.2
  (recorded 9,406.1 reproduces to 0.13%) vs +RWKV_WKV_CUDA **9,392.1** - a
  wash at the client wall, **+0.29% scheduler-gauge p50** (10,234.6 vs
  10,205.0), exactly the kernel-level delta. c=1 shape B: [S5].

## 1. What the profiling campaign established BEFORE writing the kernel

All numbers RTX 5090 (sm120), 7.2B shapes (H=64, D=64), bs=320 decode, fp16
activations, in-place indexed state pool (serving-hot configuration),
CUDA-event timing over 200 iters unless noted. Scripts:
`scratch/wkvcuda/*.py` on the box, Mac copies `scratch_wkvcuda/`.

1. **The Triton number reproduces standalone**: 232.4 us/layer fp16-state,
   453.4 fp32-state (eff-BW 1523 GB/s); config sweep re-confirmed (32,4)
   within noise of best.
2. **The in-place scattered-row r+w wall is ~1506-1522 GB/s, NOT the 1691.7
   GB/s read wall** (F0037): a minimal Triton kernel that only round-trips
   the same state rows (no math, no vectors) measures 222.8 us = 1506 GB/s.
   The WKV kernel was already within ~4% of its access-pattern wall
   standalone; mixed r+w streams pay bus-turnaround vs the read-only wall.
3. **The competitor's CUDA kernel is NOT faster standalone**: their own PR#8
   binary at the same shapes/methodology: 232.1 us/layer == ours.
4. **The gap lives in the serving DRAIN regime**: with each WKV launch
   followed by a compute-bound GEMM (as in a real layer), state writes retire
   from L2 during the next kernel and WKV wall time becomes read-shaped. A
   stable three-way paired probe (one process/profile, identical
   clock/thermal/L2 conditions, state pools rotated cold; `bench_triple2.py`,
   repeats +-0.5 us):

   | bs | Triton (ours) | hand-CUDA (this work) | competitor |
   |---|---|---|---|
   | 32 | 17.4 | **10.2** | 16.5 |
   | 128 | 64.0 | **54.3** | 55.5 |
   | 320 | 205.5-206.6 | 202.4-204.3 | **193.1-193.3** |
   | 512 | 347.1 | 344.8 | **331.4** |

   (us/layer device time. Solo back-to-back at bs>=320 is wall-locked ~232
   for all three - the drain regime is the serving-relevant one.)

## 2. The kernel (rwkv7_kernels/cuda/rwkv7_wkv.cu, RWKV_WKV_CUDA=1, default OFF)

One 64-thread block per (request, head); thread t owns state column t.
Decode-step only (T==1, indexed pool); varlen prefill and the non-indexed API
stay on the Triton kernel. Both pool dtypes (fp32 = bitwise-oracle tier, fp16
= RWKV_STATE_FP16 tier), fp32 in-register accumulation in both.

- **Bit-exactness is a transcription, not an accident**: the Triton kernel's
  compiled PTX was extracted for both state dtypes (they compile DIFFERENT
  reduction layouts: fp32-state = 16 serial-4 leaves over {g,g+16,g+32,g+48},
  fp16-state = 32 serial-2 leaves over {g,g+32}) and the exact float
  association trees, the FMA fusion pattern (`m = mul(b,sa);
  m = fma(S,decay,m); S' = fma(k,v,m)`), `ex2.approx.f32` decay (tl.exp is
  NOT expf), the `zeros+load` -0.0 normalization (`__fadd_rn(x, 0.0f)`), the
  `0.0f - kk` negation, and every cvt.rn rounding site are reproduced
  serially per column (trees in the kernel header; PTX kept under
  `scratch_wkvcuda/wkv_fp{32,16}state.ptx`). Only the association ORDER pins
  the bits - not thread geometry - so the kernel keeps the trees while
  choosing its own layout, and is batch-invariant by construction (no
  cross-request reduction).
- **Memory shape**: state staged HBM->smem via cp.async (16B units with an
  `L2::evict_first` cache-hint policy - single-use inbound lines must not
  displace outbound dirty lines), issued directly after the cache-index load
  and overlapped with the gate-vector f32 precompute (r*scale, ex2 decay,
  kk*a, -kk, k staged once per (n,h) in 1.25 KB smem; the 2-program Triton
  tiling read every vector twice); v needs no smem (thread t only reads
  v[t]). All state stores are back-loaded into one 32B-unit burst
  (`st.global.L2::evict_last.v4.b64`) after the math, maximizing what L2 can
  retire under the following kernels. Pad slots (-1 / >=size) read S=0,
  skip the store, still write the (discarded) o row - byte-identical to the
  Triton s_mask semantics under padded cuda-graph replay.
- **Perf archaeology kept honest**: a low-liveness combine-order rewrite
  (~8 live regs) measured 5% SLOWER than the P[]-array form (serial chains
  starved the memory pipeline - ILP beats occupancy here); register caps via
  launch_bounds minBlocks produced 24-32B spills and lost 4-6%. Final: 107
  regs (fp16 variant), 0 spills, 9 blocks/SM, smem 9.25 KB/block.

## 3. Gates - all green (bench/test_wkv_cuda.py)

- **Zero differing bytes vs the Triton kernel** on o AND the state pool, per
  state dtype, across B in {1,2,3,24,320,512} x H in {32,64} x input families
  {uniform, heavy-tailed, exact-zero-kk / w=0 edge, subnormal states}, pad
  sentinels (-1 and >=pool-size), and a 64-step chained recurrence
  (carried-state drift). Re-run green after every kernel iteration, including
  the final cp.async/evict variant.
- **Batch-invariance probe**: rows of a bs=320 launch == the same request at
  bs=1 (torch.equal on o and the touched pool slot).
- The fp16-state gate is ALSO bit-exact-vs-Triton - the stronger option in
  the task contract; no mirrored-rounding fallback was needed. Community
  outer ruler for context (BlinkDL, RWKV main channel, 2026-07-13): kernel
  error standard 0.004, 0.006 = FLA-level and "harms long-text performance".
  Our contract is stricter by construction, and the fp32-accumulation +
  storage-rounding discipline it transcribes is exactly what that line
  endorses (zero-FLA stance per ADR-0004).
- Engagement is verified, not assumed: the dispatch prints a one-time
  `[rwkv7_wkv] hand-CUDA WKV decode ACTIVE (state=..., H=...)` stderr line;
  both e2e legs' server logs carry it. Interplay: `RWKV_WKV_FP16_CFG`
  (Triton-only tile hook) is inert when the CUDA path is eligible; the CUDA
  kernel always carries the pinned (32,4)-config trees.

## 4. Microbench (deliverable #3) - final kernel

Two vantage points, both on the landed bytes. (a) PAIRED same-process
device-time (drain regime; `bench_paired.py`/`bench_triple2.py` - the stable
methodology; the sequential "+gemm" mode of bench_ab.py shows +-10% ordering
effects and is not citable):

| bs | state | Triton us/layer | hand-CUDA | speedup |
|---|---|---|---|---|
| 32 | fp16 | 17.4 | **10.2** | 1.71x |
| 128 | fp16 | 64.0 | **54.3** | 1.18x |
| 320 | fp16 | 206.4 | **204.0** | 1.012x |
| 512 | fp16 | 347.1 | **344.8-345.3** | 1.006x |
| 32 | fp32 | 26.9 | **18.5** | 1.45x |
| 128 | fp32 | 144.4 | 151.7 | 0.95x (documented regression cell: L2-resident regime, 17.25KB smem residency; fp32 mid-batch is not a serving configuration) |
| 320 | fp32 | 426.8 | **425.7** | 1.00x |
| 512 | fp32 | 698.3 | **699.7** | 1.00x |

(b) SOLO launch-to-launch wall (back-to-back loop, includes launcher overhead;
`bench_ab.py` solo columns, stable):

| bs | state | Triton us | hand-CUDA | speedup |
|---|---|---|---|---|
| 1 | fp16 | 13.4 | **3.7** | 3.6x |
| 1 | fp32 | 13.6 | **4.6** | 3.0x |
| 32 | fp16 | 13.1 | **9.1** | 1.45x |
| 128 | fp16 | 46.2 | **28.2** | 1.64x |
| 320 | fp16 | 231.5 | 234.8 | 0.99x (both wall-locked; the drain regime above is the serving-relevant one) |
| 512 | fp16 | 369.5 | 374.6 | 0.99x |

bsz1 kernel device time is 2.5-2.7 us for both kernels (parity); the solo 3-6x
is the thin C++ op vs the heavy eager Triton launcher - masked by cuda-graph
capture in serving, real for eager/spec-draft paths.

## 5. e2e serving (deliverable #4) - same protocol/flags as F0056 legG1

| leg | config | tok/s | note |
|---|---|---|---|
| w2A_anchor | full W1' stack (Triton WKV) | **9,394.2** | reproduces the recorded 9,406.1 to 0.13% |
| w2B_wkvcuda | + RWKV_WKV_CUDA=1 | **9,392.1** | wash at client wall; scheduler-gauge p50 10,234.6 vs 10,205.0 = **+0.29%** |
| w2C_c1_anchor | shape B 64/256 c=1 | **133.6** | reference 133.4 (F0056) reproduces |
| w2D_c1_wkvcuda | shape B 64/256 c=1 + CUDA | **133.3** | -0.2% = inside the +-0.8% same-config band; no bsz1 regression (kernel device time parity at bs=1; the eager-launch win is graph-captured away here) |

**In-situ kernel closure** (12-step /start_profile mid-wave at c=320, both
configs, 352 wkv launches = 11 complete steps each, `scratch/wkvcuda/prof*/`):
Triton `_wkv_recurrent_kernel` **6.638 ms/step** vs hand-CUDA
`wkv_decode_kernel` **6.544 ms/step** (-0.094 ms = the +0.29/0.30% gauge delta
exactly; three independent measurement layers agree). The task target
"<=6.5 ms/step contribution" is met within trace resolution (drain-probe
device time 6.51; in-situ 6.54); the task's baseline premise (7.41) belongs
to the pre-fusion-default regime, not today's stack. GEMM block confirmed at
~20 ms/step = 64.6% of GPU busy - the W1 lever remains w8a8 V2, not WKV.

## 6. Honest ledger vs the bar + next design

- Equal-conditions kernel bar (competitor, drain regime): 193.3 us/layer =
  6.19 ms/step vs our 203.5/6.54. **-5%, structural**: their fp16 arithmetic
  (hfma2 on half2 state in registers) halves state register pressure ->
  ~11-12 blocks/SM residency vs our 9 (104-107 regs; regs are our occupancy
  cap, smem/SM=100KB caps at 10). Our fp32-accumulation bit-exactness
  contract refuses the fp16-arithmetic trade by design.
- Documented next design (not attempted, est. ~+1% serving at c=320):
  warp-split pairs - two adjacent lanes co-own a column, each computes half
  the leaf set (P[16] live instead of P[32]), one shfl_xor(1) per reduction
  preserves the exact tree ((W0+W2)+(W1+W3) = E+O with E,O per-lane;
  addition commutativity keeps bits) -> est. ~80 regs at 128 threads/block =
  24 warps/SM, same memory shape.
- Strategic position: this kernel is the designated WKV component of the
  megakernel line (#50, PDL+graph): bit-exact fp32-accumulation core,
  back-loaded evict-hinted burst I/O (the drain design PDL chaining
  composes), and a 3-6x cheaper eager launch for the draft/spec paths.

## 7. Artifacts

- Kernel/loader/dispatch: `sglang_overlay/.../rwkv7_kernels/cuda/rwkv7_wkv.cu`,
  `.../wkv_cuda.py`, `.../wkv_recurrent.py` (dispatch + ACTIVE line)
- Gate: `bench/test_wkv_cuda.py`
- Probes (box `scratch/wkvcuda/` (repo-relative), Mac
  `scratch_wkvcuda/`): bench_wkv, probe_wall, bench_theirs, probe_interleave,
  bench_ab, bench_paired, bench_triple, bench_triple2 (.py), the two
  extracted PTX trees, prof/ + prof2/ in-situ traces
- e2e legs: `scratch/w1prime/w2{A,B,C,D}_*.json` + server logs (ACTIVE line)

## Cross-references

[[F0056]] (the W1' ledger this extends; its glue fusions turn out to have
already bought the Triton WKV its drain window - the un-re-measured 7.41 is
this finding's cautionary tale: **re-measure downstream kernels after landing
upstream fusions**) - [[F0047]] - ADR-0004 (zero borrowed code; the kernel
transcribes OUR OWN Triton kernel's compiled semantics) - ADR-0008 / #50.
