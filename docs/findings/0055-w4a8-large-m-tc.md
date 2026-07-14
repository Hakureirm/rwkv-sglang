---
doc_kind: finding
finding_id: F0055
title: "w4a8 large-M tensor-core path (task#52): kills the w4 M=64 concurrency cliff (c66 622.8->931.4 tok/s, peak 1407->1468.5 moving c64->c128) at a per-token-int8 activation tax that is w8a8-class on compression/lambada (+0.0042 bpb pooled, −0.35pt) but RED on MATH500 avg@64 TWICE — unrestricted M>64 dispatch 57.66% (−3.42pt, truncation 36.7% vs 14.0%) AND the RWKV_W4_TC_MAX_M=512 decode-only cap 58.07% (−3.00pt, truncation 33.0%): the cap recovered only +0.42pt, refuting the prefill-poisoning hypothesis — decode-side a8 inside the generation loop is the dominant tax; final positioning = experimental opt-in for throughput-tolerant workloads only, default OFF permanent, no third iteration without new evidence; cross-cert with F0056: activations, not state storage, are RWKV-7's precision-sensitive axis"
last_verified_commit: "0fd63e8 (kernel, this repo's history); dispatch cap follow-up 70336b6"
discovered_by: Fable 5 (agent, Stage 1 + Stage 3, 3090 box), 2026-07-13; capped re-gate landed 2026-07-13/14
severity: info
status: closed — RED twice (unrestricted −3.42pt; capped −3.00pt); prefill-scope hypothesis refuted; kernel = experimental opt-in for throughput-tolerant workloads, RWKV_W4_TC_LARGE_M default OFF permanent; no third iteration without new evidence (pre-registered)
related: [F0017, F0043, F0024, F0056]
---

# Finding F0055: w4a8 large-M tensor-core GEMM — cliff before/after + Stage-3 accuracy certification

## 0. Context

The w4 (int4 weight-only, GROUP=64) tier's dispatch had a hole above M=64: M==1 GEMV,
2≤M≤8 small-GEMM, 8<M≤64 fp16-wmma tensor-core GEMM — then **M>64 fell back to
dequant→HBM→cuBLAS**, whose ~36 bits/element effective weight traffic produced a measured
concurrency cliff on 3090 (7.2B GPTQ decode c=64→66 = 1429.5→719.7 tok/s in the cliffmap;
1407.0→622.8 in the Stage-1 re-run). Task#52 Stage 1 (commit `3f0f0b3` + `6c9cce3`) filled the
hole with `gemm_w4a8_tc` (`rwkv7_kernels/cuda/rwkv7_w4.cu`): packed int4 weights × **per-token
int8 activations** (sglang's own `per_token_quant_int8`, the op the w8a8 tier uses) on the proven
s8-wmma pipeline — cp.async staging, K_TILE=GROUP=64 order-exact per-group int32 sums folded in a
contraction-proof fp32 chain, no atomics/split-K, batch-invariant by construction, 32/64-row
tiles auto-selected (`RWKV_W4A8_ALGO=-1`).

The semantics change vs everything below M=64 is **w4a16 → w4a8**: activations are quantized to
s8 per token instead of staying fp16. Kernel-level act-quant tax measured 8.3e-3 relative
(`verify_w4a8.py` report; re-confirmed this session: 8.26e-03/8.71e-03 on 2048²/4096² vs the
w4a16-dequant reference). Bit-exactness vs the integer reference is gated (all M/N/K combos +
ragged-N + K-pad + batch-invariance + cross-algo: **GATE: PASS**, re-run twice this session on
the DEPLOYED .cu). Because the semantics change, the path shipped env-gated
`RWKV_W4_TC_LARGE_M=1`, default OFF, pending this Stage-3 e2e accuracy certification on the
project's rulers (feedback-benchmark-rigor: compression + MATH500 avg@64 are the C-position
rulers; lm-eval lambada as the historical w4-tier reference point).

Box/config for everything below: RTX 3090 24GB (sm86), long-lived `rwkvmain` container
(sglang main flavor), Stage-1-deployed overlay verified byte-identical before runs (md5 sweep:
all rwkv7_kernels/* + models/rwkv7.py match repo HEAD `6c9cce3`; the 7 infra-glue diffs are the
container flavor's own files, as Stage 1 left them). Model: `rwkv7-7.2b-w4gptq` (the tier where
int4 accuracy is the selling point). Canonical `scripts/serve.sh` throughput mode (7 fast-path
envs), `--dtype float16 --mem-fraction-static 0.85 --cuda-graph-max-bs 384
--max-running-requests 384 --chunked-prefill-size 4096 --disable-piecewise-cuda-graph
--disable-radix-cache --page-size 1 --attention-backend triton`, port 30010. OFF and ON legs are
byte-identical boots except the flag.

## 1. What the kernel buys (Stage-1 speed evidence, this box)

Decode concurrency sweep, 7.2B GPTQ, in64/out256 (`bsz_sweep_7.2b_w4gptq_3090_cliff_stage1_*.json`):

| concurrency | base (dequant→cuBLAS) tok/s | w4a8 tok/s | Δ |
|---|---|---|---|
| 48 | 1303.0 | 1323.0 | +1.5% |
| 64 | **1407.0** (peak) | 1360.5 | −3.3% |
| 66 | 622.8 | 931.4 | **+49.5%** |
| 72 | 653.5 | 996.4 | +52.5% |
| 80 | 713.4 | 1077.8 | +51.1% |
| 96 | 817.8 | 1225.5 | +49.9% |
| 112 | 912.0 | 1346.8 | +47.7% |
| 128 | 998.4 | **1468.5** (peak) | +47.1% |

The base curve never recovers its c=64 peak by c=128; the w4a8 curve is monotonic through the
old cliff and sets a **higher peak (+4.4%) at 2× the concurrency**. (c≤64 differences are
run-to-run band: the flag only changes M>64.)

Microbench (this container, `verify_w4a8.py --bench --iters 30`, includes the per-token quant in
"ours"; 7.2B shapes):

| shape | M=66 | M=128 | M=256 | M=384 | M=512 |
|---|---|---|---|---|---|
| attn 4096×4096 vs dq | 1.34× | 1.20× | 1.36× | 1.17× | 1.08× |
| ffn.k 16384×4096 vs dq | 1.89× | 1.90× | 1.57× | 1.15× | 1.18× |
| ffn.v 4096×16384 vs dq | 1.71× | 1.56× | 1.27× | 1.16× | 1.07× |
| (same three, vs plain fp16 cuBLAS) | 0.60–0.81× | 0.52–0.81× | 0.67–0.93× | 0.76–0.79× | 0.80–0.87× |

**Honest framing:** the win is *intra-w4-tier* — it replaces the w4 fallback, it does not beat
fp16-weight cuBLAS at these M (0.5–0.9×). The w4 tier's value at M>64 remains VRAM (4.6 GB
weights) + the now-uncliffed concurrency curve, not raw speed vs fp16. On the 1.5B attn shape
(2048×2048) w4a8 *loses* to the dequant fallback until M≈512 (0.49–0.68×) — small-N tiles
under-fill the GPU; the flag should stay per-tier/per-shape opt-in there (see §5).

## 2. Certification design — the path must provably fire

A gate that never exercises the new kernel is vacuous. Three instruments, all container-local
and temporary:

1. **Dispatch counter** (python-side `W4Linear.forward` M>64 branch; module-level dict +
   stderr print at n≤5 then every 20k, marked `[w4a8-cert]`): patched into the container's
   deployed `models/rwkv7.py` for the session, **restored byte-identical after** (md5
   `2097a350c0d277e66e43a12f78cb9351` before-patch == after-restore; patch md5 was
   `5713cca6…`). Zero effect on OFF legs (branch short-circuits on the flag; verified: 0
   counter lines in the OFF server log).
2. **Capture-time evidence**: decode CUDA-graph capture at bs ∈ {72,80,…,384} ran the branch
   576× per size (192 W4Linear projections × 3 runs) — every captured decode graph above
   bs=64 *contains* `gemm_w4a8_tc`. First fires logged at M=384 (descending capture order).
3. **Run-time evidence**: counter prints during evals show live fires at M=4096 (compression
   chunks), M=3858 (lambada mixed extend batch), i.e. the scored prefill traffic itself went
   through w4a8; for MATH500 the scheduler log shows steady `Decode batch, #running-req: 383,
   cuda graph: True` — decode replays of the w4a8-containing graphs at M>64 throughout.

Structural honesty note: compression and lambada are **logprob-scoring harnesses — the flag's
effect reaches them via prefill GEMMs (M≈4096), not decode batches** (their decode is 1
token/request at client concurrency 32, i.e. decode bs≤32 stays on the unchanged w4a16 kernels).
That is the correct exercise for these rulers: every scored token's logits pass through the w4a8
GEMMs. Decode-side M>64 is exercised (and measured) by MATH500 at client concurrency 384.

Both interim rungs ran OFF then ON against fresh boots, GPU idle-verified between servers.
`verify_w4a8.py` gate re-run on the deployed .cu before anything: PASS. Boot-config identity is
verifiable, not asserted: the `server_args=` lines of all three server logs hash identical after
normalizing the auto-generated `random_seed` (md5 9c1f567e… for OFF/ON/MATH500 alike). The seed
is irrelevant to rungs 1–2 (greedy logprob scoring); for rung 3 it differs from the published
baseline's process anyway — inherent to the avg@64 band framing.

## 3. Rung 1 — compression rate (uncheatable, N=300 pooled bpb, ctx 4000)

Same 15-corpora × 20-docs on-disk set as F0043's N=300 reruns; `uncheatable_eval.py
--ctx-len 4000 --concurrency 32`; 4dp:

| leg | pooled bpb | Δ | wall |
|---|---|---|---|
| flag OFF (w4a16 fallback) | **0.5438** | — | 182 s |
| flag ON (w4a8) | **0.5479** | **+0.0042** | 225 s |

Per-corpus: uniform +0.0030…+0.0066 across all 15 (worst: ao3_nonenglish +0.0066,
wikipedia_english +0.0061, github_other +0.0058) — no outlier corpus, a flat ~0.8% relative tax.

Position curve (mean −log₂p per bucket, OFF→ON): +0.0248 [0-64), +0.0195 [64-128),
+0.0187 [128-256), +0.0186 [256-512), +0.0161 [512-1024), **+0.0147 [1024+)** — the tax
*shrinks* with position. The WKV state does **not** compound the activation-quant error over
long context; if anything the recurrence absorbs it.

Calibration against the accepted precedent: the published w8a8 tier costs **+0.0041** pooled bpb
at 7.2B (BENCHMARKS §2, old N=7500 corpus, vs fp16) — the same per-token-int8 activation
mechanism. w4a8's +0.0042 (vs w4a16, N=300) is the same magnitude: **the a8 tax is real,
visible at 4dp, and exactly w8a8-class.** It is NOT invisible; it is the known, already-shipped
cost of int8 activations, now measured cleanly in isolation (weights held identical).

Wall-time observation (directional): ON is +23% slower on this prefill-dominated workload —
at M≈4096 w4a8 loses to dequant+cuBLAS (consistent with the microbench trend toward large M).
See §5.

## 4. Rung 2 — lambada full (lm-eval local-completions, 5153 docs, greedy loglikelihood)

Same protocol as the F0017/F0043 rows (`num_concurrent=32, batch_size 32`, local parquet):

| leg | acc | Δ | ppl |
|---|---|---|---|
| flag OFF | **0.7297** ± 0.0062 | — | 3.5720 |
| flag ON | **0.7262** ± 0.0062 | **−0.35pt** | 3.6432 |

−0.35pt is inside the ±0.62pt 1σ stderr — statistically indistinguishable. Protocol sanity: the
OFF leg reproduces the published 7.2B GPTQ figure **exactly** (historical: bf16 0.7425, GPTQ
−1.28pt ⇒ 0.7297; measured OFF: 0.7297) — same-protocol comparability is established, this is
not a re-derived baseline. Perplexity +0.071 (+2.0% relative) is the same small uniform logprob
tax rung 1 measured.

## 5. Rung 3 — MATH500 avg@64 (the decision ruler) — RED, twice

Protocol: `math500_avg64.py` (faithful albatross-port; fake_think prompt, temp 1.0 / top_p 0.28 /
top_k 32, max_new 1500, ctx 8192), 500×64, client concurrency 384 — the config of the published
flag-OFF baseline `bench/results/math500_avg64_7.2b_sym.json` (**61.075%**, wall 26882s, this
card class). Leg 1 launched 2026-07-13 08:18 UTC, unrestricted `RWKV_W4_TC_LARGE_M=1`
dispatch (every M>64, decode and prefill alike). Leg 2 (capped re-gate) launched 13:16 UTC the
same day on the `RWKV_W4_TC_MAX_M=512` build (commit `70336b6` deployed to the container as a
scoped patch — deployed-file diff vs the pre-cert baseline shows exactly the cap hunks and
nothing else, md5 `04674263…` recorded), identical server config and client invocation.

| leg | avg@64 | pass@64 | truncated | mean generated tokens |
|---|---|---|---|---|
| flag OFF (published sym baseline) | **0.61075** (19544/32000) | 0.782 | 13.97% | 536.4 |
| flag ON, unrestricted M>64 dispatch | **0.57656** (18450/32000) | 0.876 | 36.74% | 796.5 |
| flag ON, capped 64<M≤512 (decode-only) | **0.58072** (18583/32000) | 0.854 | 32.97% | 756.4 |
| **Δ unrestricted vs OFF** | **−3.42pt** | +9.4pt | **+22.8pt** | +260 |
| **Δ capped vs OFF** | **−3.00pt** | +7.2pt | **+19.0pt** | +220 |

Raw: `bench/results/math500_avg64_7.2b_w4gptq_w4a8full_3090.json` and
`math500_avg64_7.2b_w4gptq_w4a8capped_3090.json` (+ companion `.log`s).

**Leg-2 routing evidence** (the counter instrumentation was gone by then — §2's restore —
so a fresh temporary marker session ran on a separate boot BEFORE the re-gate, then the
deployed file was restored to the clean capped build, md5-verified, for the scoring run
itself): CUDA-graph capture fires the w4a8 branch at M=384 (inside the cap; capture covers
bs 72–384, all ≤512); a live 4096-token prompt probe routes to the w4a16 dequant+cuBLAS
fallback (`fallback-branch fired M=4096`); a c=96 load decodes on `cuda graph: True` replays
of the capped-dispatch graphs. `verify_w4a8.py` re-run on the deployed artifact: PASS.
Corroboration from the scoring run itself: wall 15293.6s → 14056.0s (−8.1%) and client-side
gen throughput 1666.6 → 1722.0 tok/s (+3.3%) vs leg 1 — prefill measurably moved back to the
faster path, exactly as the cap intends.

Against the pre-registered decision bands in the draft of this section (±0.6pt = noise;
0.6–2pt = small real effect; >2pt = do not default-ON): **both legs clear the "do not
default-ON" threshold — −3.42pt unrestricted, −3.00pt capped; neither is a close call.** The
failure signature matches F0043's int4-collapse class exactly, not a generic accuracy wobble:
truncation more than doubles (13.97% → 36.74% / 32.97%) and mean generated length grows
41–49% (536 → 796 / 756 tokens) — the model is losing the thread mid-derivation and rambling
to the token cap, the same loses-thread-and-rambles pattern F0043 root-caused for 1.5B int4
and the asymmetric-7.2B collapse. `pass@64` actually rises slightly (0.782→0.876/0.854) —
with 64 samples per problem, more truncated-but-not-wrong rollouts still contain at least one
lucky pass; this is consistent with degraded-but-not-random generation, not a correctness bug
in the kernel itself (rung 0's bit-exact gate stands unchanged).

**Root-cause hypothesis (pre-registered after leg 1, REFUTED by leg 2):** the leg-1 reading
was that the unrestricted dispatch was the problem — every M>64 dispatched to w4a8, so the
entire prompt prefill (M up to ~4096 on this protocol) ran through per-token int8 activation
quantization before generation even started, on top of the kernel being measurably *slower*
than the dequant fallback at those M (§1, +23% wall on the compression workload). The
`RWKV_W4_TC_MAX_M=512` cap removes prefill from the path entirely (leg-2 routing evidence
above proves it), and the speed side of the hypothesis held (−8.1% wall). **The accuracy side
did not: only +0.42pt of the −3.42pt came back.** Prefill-side a8 was a minor contributor;
the dominant damage is decode-side a8 in the generation loop itself — at client concurrency
384, every decode step above bs=64 quantizes every projection input to int8, and the error
feeds back through the WKV state autoregressively across hundreds of generated tokens. This
also squares rung 1's position curve with the RED: within a single forward pass the
recurrence absorbs the tax (the curve shrinks with position), but generation re-injects it at
every step — a feedback loop the logprob rulers structurally cannot see (their decode is 1
token/request at client concurrency 32, i.e. decode bs≤32 never touches this kernel; under
the cap the compression/lambada workloads now exercise w4a8 *nowhere at all*). The F0043
lesson stands, sharpened: perplexity-adjacent metrics cannot stand in for avg@64, and for
decode-batch kernels only a decode-batch generative ruler exercises the path being certified.

## 6. Verdict — RED twice; final positioning: experimental opt-in, default OFF permanent

- The kernel does its job at the thing it was built for: the M=64 cliff is gone (+47–52%
  across c=66–128, peak +4.4% at 2× concurrency), bit-exactness gates all green on the
  deployed artifact, and rungs 1–2 confirm the pure per-token-a8 tax is real but small and
  w8a8-class (+0.0042 bpb pooled, −0.35pt lambada). The cap (`70336b6`) additionally makes it
  strictly faster end-to-end (leg 2 wall −8.1% vs leg 1) and confines the semantics change to
  the decode range it was built for.
- **But the decision ruler is RED in both configurations: −3.42pt unrestricted, −3.00pt
  capped, both with the truncation/rambling collapse signature.** The prefill-poisoning
  hypothesis is refuted (§5): the dominant cost is decode-side per-token-int8 in the
  generation loop, i.e. it is intrinsic to what the kernel *is* (w4a8 semantics at decode
  batch), not to how it was dispatched. There is no dispatch-shaped fix left to try.
- **Final positioning (per the pre-registered rule: no third iteration without new
  evidence):** `RWKV_W4_TC_LARGE_M=1` (+ `RWKV_W4_TC_MAX_M`, default 512) stays an
  experimental, default-OFF opt-in for throughput-tolerant workloads — bulk/draft generation,
  data pipelines, anything where +47–52% decode throughput above the c=64 cliff (peak +4.4%
  at 2× concurrency, in 4.6 GB of weights) is worth a ~3pt MATH500-class accuracy cost that
  rungs 1–2 show is invisible on perplexity-family metrics. It is NOT for reasoning
  workloads; accuracy-critical high-concurrency serving on a VRAM budget should use the
  near-lossless w8 (g64 weight-only) tier instead, or run w4 at c≤64 where the unchanged
  w4a16 kernels serve everything.
- **Cross-cert conclusion (with F0056, same day, same 7.2B MATH500 protocol):** storing the
  recurrent WKV *state* in fp16 moved the ruler −0.32pt (inside noise — GREEN, F0056 §4);
  quantizing *activations* to int8 moved it −3.42/−3.00pt (RED twice, here). RWKV-7's
  precision-sensitive axis under serving is the activations entering the projections, not the
  state's storage width: halving state bytes is nearly free, int8-quantizing the generation
  loop's GEMM inputs is not. That asymmetry is now measured, not assumed, on this project's
  decision ruler — and it generalizes the F0043 lesson: the cheap rulers stay blind to
  exactly the mechanism (autoregressive re-injection) that the decision ruler punishes.
- The flag stays default OFF permanently; `scripts/serve.sh` does not export it. Anyone
  turning it on gets the §1 speed table and owns the §5 accuracy table — both are disclosed
  side by side in `docs/BENCHMARKS.md` §4b.
- Artifacts: `/tmp/w4a8_cert/` in the 3090 container (uncheat_72b_w4gptq_{OFF,ON}.json +
  curves, math500 out JSONs for both legs, serve/smoke logs, the cap/smoke patch scripts,
  `rwkv7.py.capped_clean` reference build), `~/w4a8_cert_stage/lambada_{OFF,ON}.log` on the
  box host — all landed in `bench/results/` as
  `uncheatable_7.2b_w4gptq_w4a8{off,on}_n300_3090.json` (+ curves),
  `lambada_7.2b_w4gptq_w4a8{off,on}_3090.log`,
  `math500_avg64_7.2b_w4gptq_w4a8full_3090.json` and
  `math500_avg64_7.2b_w4gptq_w4a8capped_3090.json` (+ `.log`s), and the Stage-1 speed sweep
  as `bsz_sweep_7.2b_w4gptq_3090_cliff_stage1_{base,w4a8}.json`. Full 32k-rollout generations
  JSONLs (~90 MB/leg) are archived in the 3090 box checkout's `bench/results/`, not committed.
  Counter and smoke instrumentation reverted byte-identical (md5-verified) after their
  sessions; the deployed container build is the clean capped one (md5 `04674263…`).
