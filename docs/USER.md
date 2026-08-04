# Find your number

Navigation by need. Every figure here is copied from [BENCHMARKS.md](BENCHMARKS.md),
where its full conditions and raw log live — this page only routes you to the right
section. Model is RWKV-7 1.5B fp16 on RTX 5090 unless the row says otherwise.

## I want one stream to be fast

| what | number | detail |
|---|---:|---|
| 7.2B fp16, one request | **142.8 tok/s** — 92.0% of Bo's official Albatross (155.2) | [§7a-flagship](BENCHMARKS.md#7a-flagship-single-stream-megakernel-ladder-the-flagship-bsz1-story-vs-bo) |
| 1.5B fp16, one request | **514.5 tok/s** (wall-clock) / **535.2** (steady-state) | [§7a-flagship](BENCHMARKS.md#7a-flagship-single-stream-megakernel-ladder-the-flagship-bsz1-story-vs-bo) · [§3](BENCHMARKS.md#3-single-request-speed-ladder-steady-state-15b-fp16) |
| 1.5B int4, one request | **742.6 tok/s** — 0.9908× Albatross's *fp16* on the same card | [§4b](BENCHMARKS.md#4b-int4-serving-speed-gptq-vs-rtn-vs-fp16-measured-2026-07-0910) |

Those two 1.5B numbers are **different conventions, both current**: wall-clock includes
reading the prompt, steady-state subtracts it. Quoting one as the other is the exact
mistake [F0069](findings/0069-public-number-conventions.md) exists to prevent.

## I want total throughput under load

| config | peak | detail |
|---|---:|---|
| 1.5B fp16, RTX 5090 | **29,533 tok/s** (c=320, plateau c256–512) | [§5](BENCHMARKS.md#5-serving-throughput-rwkv-7-15b-wall-clock-64-in256-out-concurrency-sweep) |
| 7.2B fp16, RTX 5090 | **8,277 tok/s** (c=320) | [§5](BENCHMARKS.md#5-serving-throughput-rwkv-7-15b-wall-clock-64-in256-out-concurrency-sweep) |
| 1.5B int8 w8a8, RTX 3090 | **9,851 tok/s** (c=256) | [§5](BENCHMARKS.md#5-serving-throughput-rwkv-7-15b-wall-clock-64-in256-out-concurrency-sweep) |
| real workload (ShareGPT) | 9,845 output tok/s peak · 32 ms median TTFT at 16 req/s | [§7c](BENCHMARKS.md#7c-real-workload-comparison-sharegpt-variable-length-conversations) · [§9](BENCHMARKS.md#9-latency-under-real-load) |

Why this architecture likes load: the recurrent state is a **constant size**, so 1→256
concurrent sequences or 64× longer context each cost under 0.2 GB extra —
[§10](BENCHMARKS.md#10-the-structural-advantage-constant-size-state).

## I want it smaller (quantization)

| tier | what it costs | what it buys | detail |
|---|---|---|---|
| **w8g64** (int8 weights) | greedy-lossless, 24/24 oracle-exact | half the weight bytes | [§4](BENCHMARKS.md#4-quantization-what-you-trade-and-what-you-get) |
| **w8a8** (int8 end-to-end) | −2.3 pt MATH500 (avg@64) | tensor-core int math; on 7.2B/32 GB: 1.86× the concurrency fp16 can reach | [§4](BENCHMARKS.md#4-quantization-what-you-trade-and-what-you-get) |
| **w4** (int4 weights) | 1.5B: **−24 pt** MATH500 — compression bpb hides it | 742.6 tok/s bsz1; smallest files | [§4b](BENCHMARKS.md#4b-int4-serving-speed-gptq-vs-rtn-vs-fp16-measured-2026-07-0910) |

Two findings worth reading before picking a tier: at 7.2B the whole 4-bit accuracy cost
is ~1.5–3 pt and quantization is nearly free
([F0082](findings/0082-gptq-loses-to-rtn-on-math500.md),
[F0083](findings/0083-grid-and-group-size.md)); and on the small models our shipped
GPTQ int4 measured *worse* than plain RTN on the reasoning ruler
([F0082](findings/0082-gptq-loses-to-rtn-on-math500.md)).

## I want it on my hardware

| platform | status | detail |
|---|---|---|
| 11 CUDA GPUs — T4 (2018) → A10/A10G/A100/L4/L40S/H100/H200/B200/RTX 3090/5090 | same code, per-card numbers committed | [§6](BENCHMARKS.md#6-the-10-gpu-fleet-same-code-same-recipe-every-card) |
| multi-GPU TP 2/4/8 · PP 2/4/8 | greedy 24/24 == single GPU, cuda-graph ON | [§6b](BENCHMARKS.md#6b-multi-gpu-tp--pp-verified-on-main-cuda-graph-on) |
| Apple Silicon (MLX + Metal kernel) | gated by the same numpy oracle | [§12](BENCHMARKS.md#12-apple-silicon-mlx) · [`mlx_port/`](../mlx_port/) |
| launch autotune | constants re-selected per card at warmup — nothing hardcoded travels | [§8](BENCHMARKS.md#8-launch-autotune-across-cards-why-hardcoded-constants-dont-travel) |

## I want to compare it with something

| against | one-line result | detail |
|---|---|---|
| **Albatross** (Bo's official runtime) | bsz1 7.2B at 92.0%; our int4 at 0.9908× its fp16; T4-class cards only we serve out of the box | [§7](BENCHMARKS.md#7-comparison-with-albatross-blinkdls-official-speed-reference) · [§7a](BENCHMARKS.md#7a-albatross-at-large-batch-same-code-on-a-single-rtx-5090-72b-fp16) |
| **vllm-rwkv** (community vLLM fork) | trades blows on microbench; real-workload (ShareGPT) reverses it — read both | [§7b](BENCHMARKS.md#7b-comparison-with-vllm-rwkv-the-community-vllm-fork) · [§7c](BENCHMARKS.md#7c-real-workload-comparison-sharegpt-variable-length-conversations) |
| **HuggingFace port** (this project's transformers implementation) | three-way on one 5090, same grid | [§7d](BENCHMARKS.md#7d-the-huggingface-port-on-the-same-grid--three-way-one-rtx-5090) |
| **Qwen3.5** (same engine, matched size) | the architecture comparison, same-precision | [§13](BENCHMARKS.md#13-comparison-with-qwen35-same-engine-same-precision-matched-size) |

## Accuracy, if that is the question

| ruler | 1.5B | 7.2B | detail |
|---|---:|---:|---|
| compression bpb (official; lower = better) | 0.6085 | **0.5413** | [§2](BENCHMARKS.md#2-accuracy-rulers-official-rwkv-evaluation-definitions) |
| MATH500 | avg@64 **0.4042** | greedy **0.6320** | [§2](BENCHMARKS.md#2-accuracy-rulers-official-rwkv-evaluation-definitions) |
| greedy vs numpy fp32 oracle | 24/24 | 24/24 | [§1](BENCHMARKS.md#1-correctness-the-gate-everything-else-stands-on) |

How the rulers are defined, why avg@N, and how to re-run any of this:
[EVIDENCE.md](EVIDENCE.md).
