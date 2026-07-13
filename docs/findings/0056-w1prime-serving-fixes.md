---
doc_kind: finding
finding_id: F0056
title: "W1' serving fixes: internal step-time profiling on the 7.2B fp16 decode step (bs=320, shape A 128in/1280out) found step_p50=39.27ms with the fp32 WKV state kernel at ~70-74% of the non-GEMM ('improvable') GPU-busy budget and unfused glue kernels the remainder (GEMM already compute-bound at ~51% of total); RWKV_STATE_FP16 (opt-in, temporal state only) plus five byte-exact glue fusions land step_p50 39.27->31.31ms and shape-A c=320 serving 7,603.5->9,406.1 tok/s (+23.7%); state-fp16 gate ladder (1.5B lambada/compression, 7.2B MATH500 avg@64) all green, worst movement -0.32pt on MATH500 (inside sampling noise); state pool halves (7.2B 33->17 MB/req); glue fusions promoted to serve.sh default, state-fp16 stays a documented opt-in throughput switch"
last_verified_commit: "0bf9e27 (glue fusions promoted to default); 55e12b7 (RWKV_STATE_FP16); 8819cc0 (VRESGATE + fp16-state WKV tile hook)"
discovered_by: Fable 5 (agent), 2026-07-13
severity: info
status: closed — landed and default where byte-exact, documented opt-in where lossy
related: [F0047, F0051, F0052]
---

# Finding F0056: W1' — closing the 7.2B fp16 serving gap (state bytes + glue fusions)

## 0. Context

Internal profiling of the 7.2B fp16 decode step (shape A, 128in/1280out, bs=320, RTX 5090)
found a step time of **39.27ms** with real headroom: a torch-profiler kernel trace of our own
runner (12 measured decode steps, competitor-free — no external code or data involved) put
total GPU-busy at 35.88ms/step, of which the two large compute-bound projection GEMMs already
account for ~51% (18.17ms/step) — a wall this work does not address. Of the remaining
"improvable" budget (~16.8-17.7ms/step depending on how the two small LoRA wmma kernels are
bucketed), the fp32 **WKV recurrent-state kernel is 70-74%** (12.44ms/step) and the **unfused
glue-kernel cluster is the other 26-30%** (gate-corr, LayerNorm, residual adds, token-shift
gather/index_put/copy, sigmoid, lerp, GroupNorm, clamp, pow — ~15 stock torch/Triton launches
per layer). The WKV kernel itself already runs near its HBM wall (F0047-adjacent finding, not
restated here): the state's carried **bytes**, not the kernel's arithmetic, were the cost.

This finding documents the two fixes that followed from that attribution, the gate ladder that
cleared them, three incidental bugs fixed en route, and the full serving ledger. "W1'" is this
project's internal name for this work item; it is unrelated to any external benchmark or PR
comparison and none is cited here — see `feedback-avoid-fla-dependency` / project discipline on
not disclosing infra comparisons publicly. All numbers below are reproduced from landed raws
this session (fetched from the 5090 tower's `scratch/w1prime/` working directory and, for the
kernel trace, `scratch/gapattr/`), not carried over from memory.

## 1. Fix 1 — `RWKV_STATE_FP16`: the temporal WKV state in fp16 (commit `55e12b7`)

Storage-only change: the MambaPool temporal state buffer allocates fp16 under the env gate.
`wkv_recurrent` was already dtype-polymorphic (fp32 in-register accumulation, casting only at
the HBM load/store boundary), so **no kernel arithmetic changes** — only the bytes the state
occupies in the pool and round-trips through HBM each step. Token-shift/conv state stays fp32
unconditionally. Halves per-request state: 7.2B **33→17 MB/req**; 1.5B **12.98→6.68 MB/req**
(512-slot pool 6.01→3.01 GB) — independently confirmed this session via the served processes'
own reported free-memory at matched batch size (7.2B bs=344: 5.93 GB free anchor vs 11.32 GB
free with state-fp16 + all fixes on; 1.5B bs=1: 20.81 GB vs 23.83 GB), not merely computed from
byte-widths.

Default **OFF**: the fp32 bitwise-oracle tier (§1's correctness gate, `docs/BENCHMARKS.md`) is
untouched by construction — the flag departs from bitwise-exactness by design, so it is gated on
the project's lossy-tier rulers instead, the same way quantization tiers are (§5 below).

## 2. Fix 2 — five byte-exact glue fusions (commits `706e968`, `1de2f512`, `8819cc0`, promoted
default in `0bf9e27`)

Each fusion transcribes the exact torch/Triton rounding chain it replaces (verified
`torch.equal` / zero-differing-bytes, not tolerance-based) and is individually env-gated:

| flag | what it fuses | replaces |
|---|---|---|
| (H-tiled grid, no new flag) | paged shift+lerp grid, tiled for large batch | the existing R2 shift-lerp kernels' small-batch grid shape |
| `RWKV_FUSED_ADDLN` | `x_new = x + delta` and LayerNorm in one kernel at every norm boundary | torch add + `nn.LayerNorm` (transcribes aten's `vectorized_layer_norm_kernel`) |
| `RWKV_FUSED_GNGC` | GroupNorm + gate-correction (`(r·k·r_k).sum·v`) + output gate in one kernel | torch `RowwiseMoments` + `GroupNorm1d` + a separate Triton `_gate_corr` launch |
| `RWKV_FUSED_RELUSQ` | `relu(ffn.key(x))**2` epilogue-fused into the key-projection GEMV | the standalone GEMV + 2 elementwise kernels (relu, pow) |
| `RWKV_FUSED_VRESGATE` | batched LoRA-gate activations (3 sigmoids + neg/mul + sub/mul/add) for w_log/a/v-residual mix | ~8 stock elementwise kernels per layer |

Gates, all green: `bench/test_ln_fused.py` / `bench/test_glue.py` zero differing bytes across
H=768/2048/4096 × T=1..4096 × uniform/heavy-tailed/subnormal inputs (plus a dedicated
1,572,864-row summation-tree isolation probe for the GroupNorm reduction); greedy 24/24 EXACT
per fusion alone and with the full stack; `bench/verify_batch.py --cuda-graph` OVERALL PASS
(identical / shared-prefix / mixed batches) with everything on. All five are byte-exact and are
now **default in `scripts/serve.sh`** (commit `0bf9e27`) — this is not a lossy tier.

## 3. Three incidental bugs fixed en route

Not the point of this work item, but real and worth a paper trail:

1. **`verify_batch.py` kwarg drift** (`46371ac`): sglang main removed
   `ServerArgs.disable_piecewise_cuda_graph`; the harness passed it unconditionally and died at
   `Engine()` construction on the main-lineage stack. Fixed by filtering engine kwargs against
   `ServerArgs.__dataclass_fields__`, the same pattern `greedy_check.py` already used.
2. **Comment-marker corruption in a Mac checkout of `models/rwkv7.py`**: the committed
   `RWKV_FUSED_GATES`/`RWKV_FUSED_SQRELU` comment blocks had lost their leading `# ` on wrapped
   lines, so the file did not compile as committed on at least two independent checkouts (fixed
   in-line during `1de2f512`; recurred and was fixed again on the 3090 box's checkout, commit
   `6c9cce3`, "restore comment markers dropped in the F0051/F0052 promotions").
3. **Stubbed R2 glue on a deployed main-port backend**: the paged shift+lerp kernels already
   shipped in this repo, but one deployment's backend tree had them stubbed out; the real
   implementation was restored there and byte-gated against `bench/test_glue.py` on that target
   stack (`1de2f512`).

## 4. Gate ladder — `RWKV_STATE_FP16`, all green

Two fast, cheap rulers at **1.5B** (lambada, compression) plus the decisive, expensive ruler at
**7.2B** (MATH500 avg@64) — the standard project pattern of gating cheap first, spending GPU-hours
on the decisive number second. Raws landed this session; every number below is read directly
from them, not transcribed from a summary.

**Correctness (both sizes, prerequisite to any of the below):** greedy 24/24 EXACT + a
256-token zero-divergence probe, flag ON — the lossy departure from the fp32 oracle tier is
confined to state-fp16's own storage format, not a decoding-correctness regression.

**Lambada** (1.5B, `lm-eval local-completions`, n=5153, `bench/results/lambada_1.5b_fp16_state{off,on}_5090.json`):

| leg | acc | perplexity |
|---|---|---|
| state OFF | 0.67126 | 4.7474 |
| state ON | 0.67145 | 4.7476 |
| Δ | **+0.0002** (+0.02pt) | +0.0002 |

(Cross-check: an independent numpy/PyTorch-oracle 1.5B lambada reference, `bench/results/clean/acc_ref_1.5B_lambada.json`, reads 0.67107 — both W1' legs land within 0.04pt of that independent number, the expected serving-stack-vs-reference noise band.)

**Compression** (1.5B, `uncheatable_eval`, N=300 pooled bpb, ctx 4000,
`bench/results/uncheatable_1.5b_fp16_state{off,on}_n300_5090.json`):

| leg | pooled bpb |
|---|---|
| state OFF | 0.5892803 |
| state ON | 0.5892813 |
| Δ | **+0.0000010** (~1e-6) |

**MATH500 avg@64** (7.2B, the decisive ruler, full final stack — state-fp16 + all five glue
fusions — served on the deployed fast path per the harness log's own startup banner;
`bench/results/math500_avg64_7.2b_fp16_stateon.json` vs the already-landed
`bench/results/math500_avg64_7.2b_fp16.json` baseline):

| leg | avg@64 | truncated | mean generated tokens |
|---|---|---|---|
| baseline (pre-W1', landed 2026-07-09 — no state-fp16, no glue fusions) | **64.18%** (20537/32000) | 6.27% | 480.6 |
| state ON (+ all five glue fusions, same deployed fast path) | **63.86%** (20436/32000) | 6.45% | 483.1 |
| Δ | **−0.32pt** | +0.18pt | +2.5 |

The ON leg bundles state-fp16 with the five glue fusions (that is what the deployed fast path
runs); this does not confound the attribution, because the five fusions are separately gated
byte-exact (§2 — `torch.equal`, zero differing bytes) and therefore cannot themselves move an
accuracy metric. The entire −0.32pt is attributable to state-fp16. −0.32pt on a 32,000-rollout
avg@64 run is well inside sampling noise for this protocol (F0055 §5 derives a ±0.6pt-at-2σ band
for this exact harness from binomial scatter; this delta is roughly half that). Truncation and
mean length both move by trivial amounts (contrast this with F0055's genuine RED signature on
the w4a8 kernel: truncation more than doubling, mean length +49% — nothing like that pattern
appears here). **The fp16-state gate ladder is clean.**

## 5. Serving ledger (RTX 5090, 7.2B unless noted)

**Shape A (128in/1280out), c=320 — the concurrency this work targeted:**

| checkpoint | tok/s | Δ vs previous | raw |
|---|---|---|---|
| baseline (nothing on) | 7,603.5 | — | `w1prime_legA_anchor_7.2b_5090.json` |
| + `RWKV_STATE_FP16` | 8,931.3 | +17.5% | `w1prime_legB_state_7.2b_5090.json` |
| + shift-lerp H-tile (isolated) | 9,273.4 | +3.8% over state-fp16 | `w1prime_legC1_shiftlerp_7.2b_5090.json` |
| + addLN (isolated) | 9,000.7 | +0.8% over state-fp16 | `w1prime_legC2_addln_7.2b_5090.json` |
| + GNGC (isolated) | 9,036.5 | +1.2% over state-fp16 | `w1prime_legC3_gngc_7.2b_5090.json` |
| + relu² (isolated) | 8,941.9 | +0.1% over state-fp16 | `w1prime_legC4_relusq_7.2b_5090.json` |
| combo recovery pass | 9,344.9 | +4.6% over state-fp16 | `w1prime_legF_combo_7.2b_5090.json` |
| **final sweep (all 4 above + state-fp16)** | **9,334.7** | — | `w1prime_legFinal_A_7.2b_5090.json` |
| **+ VRESGATE, focused re-measurement** | **9,406.1 (peak)** | +0.65% | `w1prime_legG1_vres_7.2b_5090.json` |
| chunked-prefill-size=8192 variant (tried) | 9,374.6 | −0.3% vs peak, not adopted | `w1prime_legG2_cps8k_7.2b_5090.json` |

**Net: 7,603.5 → 9,406.1 tok/s = +23.7%.** The four isolated glue fusions individually sum to
+5.9% over the state-fp16 baseline, matching the isolated measurements above (commit `0bf9e27`'s
own contemporaneous message independently reports the same shape: "H-tiled shift+lerp glue
+3.8%; the other four +0.1-1.1% loop each, sub-additive stacked"). The first naive combined
stack (`w1prime_legC_full_7.2b_5090.json`, 8,811.2) landed *below* the state-fp16-only number
(8,931.3) — more than merely sub-additive, net-negative — which the commit message attributes to
the four fusions competing for occupancy/registers when stacked rather than to measurement
noise; this finding does not have an independent kernel-occupancy measurement to confirm that
mechanism specifically, so it is reported as the contemporaneous explanation, not re-derived
here. Whatever the mechanism, it was transient: the very next checkpoint (combo recovery,
9,344.9) and every measurement after it are consistently above the state-fp16 baseline, and
`legFinal_A` (9,334.7) and `legG1_vres` (9,406.1, same flag set, a focused single-point
re-measurement) agree to within 0.8%, the same band as other same-config reruns in this ledger.

**Full final-sweep concurrency points, shape A** (`w1prime_legFinal_A_7.2b_5090.json`):
c=64 **4,999.1**, c=128 **7,755.6**, c=320 **9,334.7** (9,406.1 with VRESGATE additionally on,
measured at c=320 only — VRESGATE was not independently re-swept at c=64/128).

**Shape B (64in/256out)** — interim (addLN+GNGC only) → final (all 5 fusions + state-fp16),
`w1prime_legD_shapeB_interim_7.2b_5090.json` → `w1prime_legFinal_B_7.2b_5090.json`:

| concurrency | interim | final | Δ |
|---|---|---|---|
| c=1 | 126.5 | **133.4** | +5.5% |
| c=32 | 2,468.0 | **2,636.4** | +6.8% |
| c=128 | 6,593.9 | **7,087.3** | +7.5% |

**1.5B single-stream (c=1)** — baseline → interim (addLN+GNGC only, batch-oriented fusions,
expected near-zero at bsz1) → final (all 5 + state-fp16):

| leg | tok/s |
|---|---|
| baseline (`w1prime_legE0_1.5b_5090.json`) | 421.2 |
| interim, addLN+GNGC only (`w1prime_legE_1.5b_5090.json`) | 419.0 (−0.5%, within noise — confirms these two are batch-oriented, not single-stream levers) |
| **final, all 5 + state-fp16** (`w1prime_legEf_1.5b_5090.json`) | **447.3 (+6.2% vs baseline)** |

**Step-time attribution, reproduced this session** (`w1prime_step_attribution_7.2b_5090.json`,
via `scratch/w1prime/steptime.py` against the landed server logs — the scheduler's own
steady-state per-iteration throughput, a different vantage point from the wall-clock sweep
numbers above): step_p50 **39.27ms → 31.31ms** at bs=320 (anchor → final/VRESGATE), with the
state-fp16-only intermediate at 33.07ms — i.e. state-fp16 alone closes 6.20ms of the 7.96ms
total step-time reduction (77.9%), the five glue fusions the remaining 1.76ms (22.1%), a
close (not pixel-identical — different measurement, coarse two-point deltas vs the finer
per-kernel trace in §0) corroboration of the kernel-level attribution this finding opened with.

## 6. Positioning decision

`serve.sh`'s default fast-path combo now includes all five glue fusions (byte-exact — no
tier change). **`RWKV_STATE_FP16` stays a documented, opt-in throughput switch**, not a default
— the same treatment this project gives its quantization tiers (`docs/BENCHMARKS.md` §4): a
named flag, an accuracy cost disclosed next to the speed number, left to the deployer to choose.
Given the gate ladder above (§4: +0.02pt lambada, ~0 compression, −0.32pt MATH500 avg@64, all
inside noise for their respective protocols) the honest recommendation is that most deployments
concurrency-bound on state-pool capacity should turn it on — but the default stays conservative
so the bitwise-oracle tier (§1 of `docs/BENCHMARKS.md`) remains reachable with zero flags.

## Cross-references

[[F0047]] (the corrected 7.2B fp16 concurrency ceiling this ledger's baseline builds on) ·
[[F0051]] / [[F0052]] (the bsz1-decode epilogue-fusion program this extends to the large-batch
serving axis) · `docs/BENCHMARKS.md` §4 (quantization-tier disclosure pattern this finding's
positioning decision mirrors) · `docs/findings/0055-w4a8-large-m-tc.md` (the sibling task#52
kernel work landed the same day, unrelated mechanism, same gate-ladder discipline).
