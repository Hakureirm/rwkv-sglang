---
doc_kind: finding
finding_id: F0025
title: "Serving eval: PD-mixed (open-loop Poisson) tail latencies + arch-aware GEMV launch autotune (A-segment): gemv_m1_cfg parametrized {64,128,256}×{1,2,4} with (sm_arch,N,K) selection, token-exact vs the fixed kernel; 3090 gains small (1.01–1.05×, BW-bound as F0023 predicted) — the win is cross-arch portability (no per-GPU hand-tune)"
last_verified_commit: "HEAD"
discovered_by: lead (M13), 2026-07-03
severity: info
status: open
related: [F0023, F0024, F0016]
---

# Finding F0025: PD-mixed serving tail latencies + arch-aware GEMV autotune (A-seg)

## Part A — PD-mixed serving benchmark (prefill+decode mixed, open-loop)
"PD-mixed" = requests arrive open-loop (Poisson at rate λ), so a new request's **prefill**
interleaves with in-flight requests' **decode** in the same scheduler step — the realistic online
regime that a closed-loop static batch hides. `bench/pd_mixed.py` (streaming `/generate`, TTFT =
time-to-first-token, TPOT = (total−TTFT)/(tokens−1); direct client because the box is
modelscope-only and sglang `bench_serving --dataset-name random` needs an HF corpus download).
1.5B fp16, 512-in/256-out, 300 prompts, `--cuda-graph-max-bs 512`:

| arrival rate | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
|---|---|---|---|
| 2 req/s | 520 | 123 / 293 ms | 9.2 / 11.6 ms |
| 4 req/s | 1023 | 178 / 340 ms | 13.7 / 19.7 ms |
| 8 req/s | 1976 | 228 / 322 ms | 36.8 / 47.4 ms |
| 16 req/s | 2610 | 302 / 1449 ms | 70.8 / 107.5 ms |
| ∞ (burst) | 3195 | 6592 / 12539 ms | 66.2 / 91.1 ms |

Clean latency-vs-load tradeoff: at ≤8 req/s TTFT p99 stays <350 ms and TPOT p99 ~12–47 ms; the
tail grows with load; ∞ (all 300 arrive at once) blows the TTFT tail to 6.6 s (300 queued prefills,
expected). Complements the best-bsz peak (F0024 §speed): F0024 reports peak throughput, this reports
tail latency under realistic arrival. `bench/results/pd_mixed.json`.

## Part B — arch-aware GEMV launch autotune (F0023 §5 roadmap #6, A-segment)
F0023 §5 showed albatross's linear dispatch is a hand-frozen per-GPU (5090) table, and **our own
`gemv_m1` had the same weakness, coarser** (fixed `<128,2>`/`<128,1>` by N-parity only, no arch
awareness). Fix (this finding):
- **`gemv_m1_cfg(x, w, threads, out_tile)`** (`rwkv7_fast.cu`): the same kernel, parametrized over
  Threads∈{64,128,256}×OutTile∈{1,2,4} via a 9-way switch. Occupancy of these is compile-time
  (regs/smem), so the key is purely `(sm_arch, N, K)`.
- **`_select_config(N,K)`** (`fast_linear.py`): `torch.cuda.get_device_capability` → arch key;
  in-process + on-disk cache (`~/.cache/rwkv7_fast/gemv_autotune_<gpu>.json`); a one-time **warmup
  autotune** (CUDA-event micro-bench of valid configs) or a closed-form heuristic. **cuda-graph
  safe**: never benchmarks while `torch.cuda.is_current_stream_capturing()` (falls back to
  heuristic), so timing happens in eager warmup and is frozen before capture.

**Correctness gate (3090):** `gemv_m1_cfg(...,128,2)` is **token-exact vs the fixed `gemv_m1`**
(`torch.equal`) on all RWKV-7 1.5B M==1 shapes, rel-err vs fp32 ~3e-4 (same as before). No accuracy
change — it is the same kernel, only the launch config varies.

**3090 sweep (best config per shape, `bench/autotune_gemv.py`):**

| shape | N×K | fixed `<128,2>` | best cfg | speedup |
|---|---|---|---|---|
| att r/k/v/o | 2048×2048 | 16.86 µs | (256,2) | 1.04× |
| ffn key | 8192×2048 | 40.79 µs | (128,1) | 1.01× |
| ffn value | 2048×8192 | 42.34 µs | (256,1) | 1.05× |

**3090 gains are small (1.01–1.05×) — and that is the predicted result, not a disappointment.**
F0023 §5 established bsz1 GEMV is HBM-bandwidth-bound (~roofline), so the launch config can't move it
much *on the arch it happens to fit*. The autotune's real value is **cross-arch portability**: the
same fixed `<128,2>` that is ~optimal on the 3090 mis-fits other archs (F0023 §5: 64-thread configs
hit a 67% occupancy ceiling on sm_86/89, different sweet spots on sm_90/120), and albatross forces a
manual per-GPU re-tune there while we auto-select. **B-segment** (task #14, per-card L4/A10G/H100/
RTX-PRO-6000) will seed + validate the other-arch rows and quantify the portability win.

## Part C — w8a8 large-M throughput (ADR-0005 R1): the high-concurrency int8 overtake
F0023 §3 headline: int8 tensor-core GEMM is the path albatross structurally cannot follow (fp16
only). sglang-native w8a8 already delivers it at small M (quant.md: +15–53% decode ≤bsz32), but the
**large-M** regime — exactly the high-concurrency strategic axis — was never measured (tables stop
at 32, and were taken under the low `cuda_graph_max_bs` cap, F0024). Measured now on the clean
single-GPU with `--cuda-graph-max-bs 512` (`rwkv7-1.5b-w8a8` pre-quantized model,
`bench/results/bsz_sweep_w8a8.json`), vs the fp16 clean sweep (F0024):

| concurrency | fp16 tok/s | **w8a8 tok/s** | int8 vs fp16 |
|---|---|---|---|
| 1 | 154.4 | 174.5 | +13% |
| 8 | 968.0 | 1192.4 | +23% |
| 32 | 3128.4 | 3895.5 | +25% |
| 64 | 5036.2 | 5697.0 | +13% |
| 128 | 6086.4 | 7005.7 | +15% |
| 256 | 6742.7 | 8752.9 | **+30%** |
| 384 | 6884.8 | 8755.2 | +27% |
| **512** | 6637.1 | **9152.5** | **+38%** |

**Integrated accuracy (measured 2026-07-04, F0024 addendum):** full-corpus uncheatable compression **0.6161** (+0.0076 bpb vs fp16 0.6085) and greedy MATH500 **199/500 (39.8%)** (statistically = fp16 39.2%) — the w8a8 throughput path's cost is small and now quantified on the decreed metrics (w8g64 remains the lossless int8 at 0.6086).

**w8a8 peak ≈ 9152 tok/s @ 512 vs fp16 peak 6885 @ 384 = +33% peak throughput**, and +30–38% in the
high-concurrency band (256–512). This is the concurrency overtake albatross cannot match on two
counts at once: no scheduler/dynamic-batching AND no int8 path. Accuracy caveat (honest, per
quant.md): 1.5B w8a8 free-running greedy diverges at token 12/24 (small-model int8 drift, a known
near-tie cascade), while **7.2B w8a8 is greedy 8/8 EXACT** (int8 noise absorbed at scale) — so the
w8a8 throughput win is unqualified at 7.2B and comes with a small-model accuracy note at 1.5B. VRAM
−41–48%. This closes ADR-0005 R1.

**Cross-precision composition — w8a8 + R2 fused glue (the highest throughput ceiling):** the R2 paged
glue is byte-exact and operates on the fp16 normed hidden (identical in w8a8), so w8a8+glue output ≡
w8a8-alone (same accuracy), while the glue's HBM-round-trip saving stacks on top of int8-TC. Measured
(`bench/results/bsz_sweep_w8a8_glue.json`, all fast paths + glue on the w8a8 model): peak **9686 tok/s
@ bsz512** (vs plain w8a8 9152 = +6%; vs plain fp16 6885 = **+41%**), c256 8753→9646 (+10%), bsz1
174.5→195.1 (+12%). So the two techniques albatross structurally lacks — int8 tensor-core GEMM AND
whole-layer paged mega-fusion — **compose**, and the composed ceiling is +41% over plain fp16.

## Addendum (2026-07-05) — autotune on/off A/B quantified on the 3090 (task: per-card deltas)

The question "how much does autotune buy per card" now has a clean measured answer for the
tuning-baseline card. Method: identical full-stack serving (1.5B fp16, F0028 flags, cuda-graph
max-bs 512, `bench/bsz_throughput.py` in64/out256), the ONLY variable `RWKV_GEMV_AUTOTUNE`
(0 = closed-form heuristic; 1 = default OutTile-only scope, logits-invariant), disk cache cleared
before the tuned side. THREE runs committed — off (cold), on, off again (hot) — because the first
pass showed a consistent −2..−3.5% on the tuned side at mid/high concurrency, which is
architecturally impossible for this kernel (M-gate routes c>1 decode GEMMs to cuBLAS; the tuned
kernel only serves M==1). The hot re-run of OFF landed ABOVE the cold OFF (+3.5% @ c128, +4.9% @ c384) —
the OFF side's own same-config spread is of the same magnitude as the ON-side deficits,
though ON sat lowest of the three runs at every point:

| c | off (cold) | on | off (hot) | verdict |
|---|---|---|---|---|
| 1 | 231.3 | 230.4 | — | par (−0.4%, in band) |
| 4 | 568.8 | 557.6 | 571.4 | in noise |
| 32 | 3386.5 | 3267.4 | 3443.4 | in noise |
| 128 | 6698.0 | 6473.8 | 6934.9 | in noise (spread ±3.5%) |
| 384 (peak) | 7425.8 | 7255.0 | 7788.5 | in noise |

Raw: `bench/results/autotune_ab_3090_{off,on,off_hot}.json` (table abridged — c2 −0.2% and
c8 −3.2% omitted for width; full raws committed). Because the first round's on-side deltas sat
at the edge of the observed spread, two decisive follow-up rounds were run and committed
(`autotune_ab2_*` = same-session back-to-back OFF→ON; `autotune_ab3_*` = ORDER REVERSED ON→OFF):

| round | order | c128 first / second | c384 first / second |
|---|---|---|---|
| ab | off → on (separate boots) | 6698.0 / 6473.8 | 7425.8 / 7255.0 |
| ab2 | off → on (back-to-back) | 6955.2 / 6685.8 | 7824.0 / 7523.5 |
| ab3 | **on → off** (back-to-back) | **7008.0** / 6772.8 | **7792.0** / 7565.6 |

The FIRST server boot of a session measures ~3-3.5% higher at c>=128 **regardless of which side
it is** — with the order reversed, ON lands high and OFF lands low by the same margin. The
mid/high-concurrency deltas are therefore a boot-position artifact of the harness session
structure, not an autotune effect (consistent with the M-gate argument: the tuned kernel serves
M==1 only, with sub-0.1% incidental M==1 work inside c>1 levels). Verdict, now two-sided:
**no measurable autotune effect at any operating point on the 3090** — bsz1 par (231.3 vs
230.4), c>=4 differences fully explained by boot position. Methodology note for future A/Bs on
this harness: alternate boot order or interleave sides; never compare sides across different
boot positions within a session.

Kernel-level (default class-locked scope, `bench/autotune_gemv.py`): att_rkvo's apparent 1.10x is a
measurement artifact — the "best" config IS the fixed reference (128,2), timed twice across a
clock-state change (the tool itself thereby demonstrates ~10% single-shot drift, which also
sub-noises its historical 1.01-1.05x cells); ffn_key 1.00x; ffn_value's (256,1) is IN the default
scope (the closed-form heuristic already picks the 256 class for K>=4096), so its 1.04x is a
heuristic-vs-legacy-fixed-reference gain present on BOTH A/B sides — not an autotune win.

**Conclusion:** on the card our closed-form heuristic was derived on, autotune's measured value is
confirmation, not speed — the heuristic is already optimal (and autotune has no room to inflate).
The quantitative per-card story therefore rests where the cross-arch selection evidence already
pointed (5 cards picked different winners): the pending sm120 (RTX 5090, albatross's home-turf
constants) and the per-card L4/A10G/H100 on/off legs. Until those numbers exist, materials must cite
the 3090 result as "par by design on the baseline card" — never extrapolate a per-card gain.

Reconciliation note: this session's absolute levels run above F0028's committed raws (c128
6474-6935 vs 6023; peak 7255-7789 vs 7334) — run bands are session- and concurrency-dependent
(driver/clock state), so peaks and deltas must only be compared within one session's discipline;
cross-session comparisons need a same-session re-baseline. F0028's "±2-3%" band statement is
superseded by this session's observed +4.9% same-config spread at c384.


## Addendum 2 (2026-07-05) — per-card quantification across 9 GPU SKUs (sm75..sm100)

Kernel-level on/off A/B for `gemv_m1_cfg` on nine cards, audit-hardened methodology
(interleaved 4-pass per-config measurement, median across passes, 10-iter warmup + 50-iter
CUDA-event round per pass — kills the first-measured clock-ramp bias documented in Addendum 1).
Scope: class-locked (threads fixed at the heuristic class = the logits-invariant subspace);
gains are heuristic-config vs best-in-class-config on the RWKV-7 1.5B GEMV shapes.
Raw: `bench/results/autotune_ab_9cards.json` (per-config medians + pass spreads included);
RTX 5090 row added 2026-07-05 from the same harness run standalone on the workstation
(`bench/results/autotune_ab_5090.json`).

| card | arch | SMs | gain att_rkvo / ffn_key / ffn_value | max gain (shape) |
|---|---|---|---|---|
| T4 | sm75 | 40 | +7.6% / +5.6% / +2.5% | +7.6% (att_rkvo) |
| L4 | sm89 | 58 | +0.1% / +11.3% / +24.1% | +24.1% (ffn_value) |
| A10G | sm86 | 80 | +0.1% / +0.3% / +2.1% | +2.1% (ffn_value) |
| A100-40GB | sm80 | 108 | +0.0% / +0.0% / +4.9% | +4.9% (ffn_value) |
| A100-80GB | sm80 | 108 | +0.6% / +1.6% / +0.0% | +1.6% (ffn_key) |
| L40S | sm89 | 142 | +0.0% / +9.2% / +2.6% | +9.2% (ffn_key) |
| H100 | sm90 | 132 | +0.0% / +0.0% / +0.0% | +0.0% (att_rkvo) |
| H200 | sm90 | 132 | +2.0% / +0.0% / +0.0% | +2.0% (att_rkvo) |
| B200 | sm100 | 148 | +0.5% / +0.0% / +0.0% | +0.5% (att_rkvo) |
| RTX 5090 | sm120 | 170 | +0.0% / +3.2% / +5.0% | +5.0% (ffn_value) |

Reading: the closed-form heuristic is already optimal on H100 (0.0% on all three shapes) and
near-par on A10G/A100-80/B200 (<=2.1%) — consistent with the 3090 serving A/B (Addendum 1).
The value concentrates where the heuristic misses: **L4 +24.1% (ffn_value), L40S +9.2%
(ffn_key), T4 +7.6% (att_rkvo)** — and the winning out_tile differs by card AND shape
(1 vs 2 vs 4), which is precisely the per-card launch-selection effect hardcoded-constant
kernels cannot express. Consumer Blackwell repeats the pattern: on the RTX 5090 (170 SMs)
the heuristic's out_tile=4 loses to out_tile=1 on both FFN shapes (+3.2%/+5.0%) — with
170 SMs there are already enough blocks without output tiling. sm100 (B200) runs the kernel unmodified at 4.1-6.2us/shape.
All numbers are per-card real-hardware measurements; serving-level deltas on any given card
follow only where M==1 decode dominates (see the M-gate scope note above).

## Cross-references
[[F0023]] (§5 launch-tuning axis, the overtake design) · [[F0024]] (best-bsz peak + cuda_graph_max_bs)
· [[F0016]] (serving-scale wedge) · ADR-0005 (roadmap) · `bench/pd_mixed.py` · `bench/autotune_gemv.py`
· `bench/bsz_throughput.py` · `bench/results/bsz_sweep_w8a8.json`.
