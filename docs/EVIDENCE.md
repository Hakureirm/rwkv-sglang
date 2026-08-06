# How to trust — and re-run — any number here

The evidence chain, shortest path first: what gates correctness, how the rulers are
defined, which timing convention a number uses, and where the raw data sits. Full
detail stays in [BENCHMARKS.md](BENCHMARKS.md) and the dated
[findings](FINDINGS.md); this page is the map.

## 1. The correctness gate

Everything stands on one gate: **greedy output must match a pure-numpy fp32 reference
token-for-token** — 24/24 on 0.1B / 1.5B / 7.2B (CUDA) and Apple Silicon (MLX), and
unchanged under dynamic batching, chunked prefill, CUDA graphs, and TP/PP 2/4/8
([§1](BENCHMARKS.md#1-correctness-the-gate-everything-else-stands-on),
[F0036](findings/0036-pp-cudagraph-vfirst-fix.md)).

The reference is numpy on purpose: it shares no code with the served implementation,
so a bug both sides agree on has nowhere to hide. Every fast-path kernel ships only
after passing this gate byte-exact.

## 2. The two timing conventions

Two windows, both current, never interchangeable
([F0069](findings/0069-public-number-conventions.md) — the finding that caught a
phantom −3.3% "regression" that was pure convention mixing):

| | steady-state | wall-clock |
|---|---|---|
| includes prompt reading (TTFT)? | no | yes |
| tool | `bench/serving_scale.py` | `bench/bsz_throughput.py` (64-in/256-out) |
| example (same card, same stack) | 1.5B fp16 **535.2** tok/s | 1.5B fp16 **514.5** tok/s |

Every table in BENCHMARKS.md names its window. All comparison tables run
**cuda-graph ON** (the production decode path); the eager baseline exists only for
kernel development and is never quoted against the graph numbers.

## 3. The accuracy rulers

| ruler | definition | why this form |
|---|---|---|
| compression bpb | official RWKV evaluation, bits per byte on the fixed corpus; lower is better | the community-standard quality number — comparable with published RWKV results |
| MATH500 avg@N | 500 problems × N sampled rollouts, mean accuracy; paired differences use a **cluster bootstrap over problems** | rollouts within one problem are correlated, so naive binomial error bars overstate confidence; avg@64 shrinks within-problem variance enough to resolve ~1-pt effects |
| oracle 24/24 | greedy tokens vs the numpy reference | correctness, not quality — it gates whether a change is the *same model* |

A quantization tier is reported with **both** bpb and MATH500, because they disagree:
int4 costs 1.5B **−24 pt** of MATH500 while moving bpb only 0.6085→0.6514 — compression
hides reasoning damage
([§4](BENCHMARKS.md#4-quantization-what-you-trade-and-what-you-get),
[F0082](findings/0082-gptq-loses-to-rtn-on-math500.md)).

## 4. Where the raw data is

| kind | location |
|---|---|
| every benchmark's raw output | [`bench/results/`](../bench/results/) — committed, per-run JSON/JSONL |
| per-card fleet runs | [`bench/results/fleet_main_10cards.json`](../bench/results/fleet_main_10cards.json), [`albatross_fleet_10cards.json`](../bench/results/albatross_fleet_10cards.json) |
| ROCm W8/W4 prefill operator + all-size serving | [`rocm_gfx1100_quant_prefill_e2e.json`](../bench/results/rocm_gfx1100_quant_prefill_e2e.json), [F0084](findings/0084-rocm-quant-prefill.md) |
| the scripts that produced them | [`bench/`](../bench/) — each table in BENCHMARKS.md names its script |
| claim → raw-log map | [`CONTRIBUTIONS.md`](../CONTRIBUTIONS.md) |
| dated methodology + negative results | [`docs/findings/`](findings/) via the [index](FINDINGS.md) |

Engine versions: since 2026-07-05 everything new runs on **sglang main**, from
[`sglang_mainline/`](../sglang_mainline/) — the tree that produced the numbers, now
committed. Rows marked "(v0.5.10)" came from an earlier line that has been removed from
this repository; they are kept as record but are not reproducible from it. If a number
here matters to you, check which of the two it is before trying to re-run it.

## 5. Re-running

```bash
# correctness gate (any box with the model):
python bench/oracle_check.py --model <dir>          # expects 24/24

# single-request ladder / conventions:
python bench/serving_scale.py   --model <dir>       # steady-state
python bench/bsz_throughput.py  --model <dir>       # wall-clock 64-in/256-out

# MATH500 avg@N with the cluster bootstrap:
python bench/math500_avg64.py --model <dir> --samples 64 --out out.json
python bench/math500_compare.py --baseline a=... --arm b=...   # paired CIs
```

Launch flags for the production configuration are in
[`scripts/serve.sh`](../scripts/serve.sh); if a re-run disagrees with a committed
number, open an issue with the JSON — that is what the raw logs are committed for.

## 6. What "verified" does not mean

Findings record the misses as well: predictions registered before runs and then
refuted ([F0081](findings/0081-int4-layer-protection.md),
[F0083](findings/0083-grid-and-group-size.md)), a checker that crashed and was
misread as passing, GPU-gated tests that had never run on a GPU. The practice the
misses converge on: **silence is not success — a check counts only when it visibly
executed and could have failed.**
