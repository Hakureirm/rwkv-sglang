# RWKV-7 (Goose) × sglang

**English** · [简体中文](README.zh-CN.md)

Production serving for **RWKV-7** on [sglang](https://github.com/sgl-project/sglang):
token-exact against the reference implementation, quantized (int8/int4), and running on
11 platforms — 10 CUDA GPU models (2018's T4 through B200 and RTX 5090) plus Apple Silicon.
Every number below has its raw log committed in [`bench/results/`](bench/results/).

**➡ Full benchmark reference: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)** — every measured
axis in readable tables (correctness gates, accuracy rulers, per-GPU speeds, the Albatross
comparison, quantization trade-offs, latency under load), each linked to its raw log.

**Runs on sglang `main` and on v0.5.10** — one code base, the version difference is detected
at runtime. The model-support core is submitted upstream:
[sglang PR #30115](https://github.com/sgl-project/sglang/pull/30115).

## Why RWKV-7 for serving

RWKV-7 is a recurrent model: its per-sequence state is a **fixed size**, no matter how long
the context — a Transformer's KV cache grows with every token. Measured effect: going from
1 to 256 concurrent sequences, or growing the context 64×, each costs **less than 0.2 GB**
of extra VRAM. High concurrency and long context are where this architecture wins.

## What works (2026-07-06)

| | |
|---|---|
| **Correctness** | Greedy output is token-exact vs a numpy fp32 reference — 24/24 tokens on 0.1B / 1.5B / 7.2B (CUDA) and on Apple Silicon (MLX); also exact under dynamic batching, chunked prefill, CUDA graphs, and tensor/pipeline parallel (TP 2/4/8, PP 2/4/8; TP=2 & PP=2 re-verified on main under cuda-graph ON, greedy 24/24 == 1-GPU — [F0036](docs/findings/0036-pp-cudagraph-vfirst-fix.md)) |
| **Accuracy rulers** | MATH500 avg@64 (the low-variance ruler): **0.4042** on main, matching v0.5.10's 0.4060 within noise — no regression. Compression rate: fp16 0.6085, int8 w8g64 0.6086 (lossless), w8a8 0.6161, int4-GPTQ 0.6514. Quantization on the reasoning ruler (avg@64): w8a8 −2.3pt, int4 −24pt — [§4](docs/BENCHMARKS.md#4-quantization-what-you-trade-and-what-you-get) |
| **Serving features** | Dynamic batching, chunked prefill, recurrent-state prefix cache (~98% hit rate under high-reuse load) |
| **Quantization** | Two int8 tiers + int4 (GPTQ), all hand-written CUDA. **w8g64** (weight-only): greedy-lossless (24/24 oracle-exact). **w8a8** (tensor-core): compression == cutlass (0.6161), a measured −2.3pt on MATH500 avg@64, for large-batch/VRAM wins. On sm120/Blackwell — where upstream has no int8 GEMM at all — our s8-wmma kernel serves w8a8 and beats fp16 cuBLAS on the GEMM (1.03–1.55× at batch ≥512). On **7.2B / one 32 GB 5090, int8 serves 1.86× the concurrency and 13.1% higher peak than fp16 can reach** (fp16's real ceiling is ≥344 concurrent, corrected 2026-07-07 from an undertested 221) — details in [BENCHMARKS §4](docs/BENCHMARKS.md#4-quantization-what-you-trade-and-what-you-get) / [F0047](docs/findings/0047-fp16-72b-concurrency-correction.md) |
| **Speculative decoding** | Phase 1 working: a draft model proposes, the target verifies in one pass, rejected tokens roll back via an O(1) state snapshot. 9/10 test prompts token-identical to normal decoding; the single difference is a benign floating-point rounding-order effect, fully analyzed in [F0031](docs/findings/0031-spec-decode-increment-i.md) |
| **Apple Silicon** | Native MLX implementation with a custom Metal kernel, gated by the same numpy reference — see [`mlx_port/`](mlx_port/) |
| **Upstream work** | Model PR [#30115](https://github.com/sgl-project/sglang/pull/30115), verified on RTX 3090 and RTX 5090; also found and fixed a silent pipeline-parallel data-corruption bug: issue [#30015](https://github.com/sgl-project/sglang/issues/30015) → fix PR [#30095](https://github.com/sgl-project/sglang/pull/30095) |

## Speed

1.5B model, one GPU, sglang main. "Single request" = one stream, sustained decode speed.
"Peak" = best total throughput across a concurrency sweep (64-token prompts, 256-token outputs).

| GPU (1.5B) | single request | peak serving throughput |
|---|---|---|
| RTX 3090 | 230.7 tok/s | 7,205 tok/s fp16 · **9,851 tok/s int8** |
| RTX 5090 | **409.8 tok/s** fp16 · **548.8** int4 | **22,175 tok/s** |

**7.2B, one RTX 5090 (32 GB):** single request 123.7 tok/s (fp16). Peak serving:
**6,709 tok/s (fp16 @ c320, safe to ≥344 concurrent)** vs **7,587 tok/s
(int8, 640 concurrent)**. At bsz 1 fp16 is faster; int8's win is the VRAM headroom that
lets 7.2B scale to 1.86× the concurrency — the full story (and the correction to a
previously-published 5,983/221 figure that undertested the concurrency grid) is in
[BENCHMARKS §4](docs/BENCHMARKS.md#4-quantization-what-you-trade-and-what-you-get).

- The same stack runs on T4, L4, A10G, A100 (40/80GB), L40S, H100, H200, B200 —
  per-card results in [`fleet_main_10cards.json`](bench/results/fleet_main_10cards.json).
- Real-workload sample (ShareGPT, RTX 5090): 9,845 output tok/s at peak; at 16 requests/s,
  median time-to-first-token is 32 ms.
- **Comparison with BlinkDL's Albatross** (the official speed reference — note it is a
  benchmark loop without request scheduling or an API): our single-request speed is
  0.9004× (L4) to 0.5129× (B200) of its decode speed — the higher the GPU's memory
  bandwidth, the more its fused-layer design gains. On the author's own RTX 5090 our int4
  reaches **0.9908×** of its fp16 speed. On T4-class GPUs the *stock* Albatross kernel
  doesn't compile (it uses sm80+ `cp.async` — a removable limit; BlinkDL notes a patched
  kernel runs on T4), so out-of-the-box only this stack serves T4. Per-card data:
  [`albatross_fleet_10cards.json`](bench/results/albatross_fleet_10cards.json).

## Quickstart

**On sglang main** (e.g. inside the `lmsysorg/sglang:dev-cu12` container):

```bash
cd /sgl-workspace/sglang
git apply <this-repo>/sglang_main_port/upstream_edits.patch   # 7 small wiring edits
# then copy the RWKV-7 files (model, backend, kernels, config):
#   file list and destinations in sglang_main_port/README.md
python -m sglang.launch_server --model-path <rwkv7-model-dir> --trust-remote-code \
    --attention-backend triton --dtype float16 --disable-radix-cache
```

**On sglang v0.5.10** (pip install): `BOX=<host> SP=<site-packages> bash scripts/deploy.sh`
— rsyncs the overlay and applies two one-line patches.

The hand-written fast-path kernels are opt-in environment flags, all greedy-exact; the
recommended production set is in [`scripts/serve.sh`](scripts/serve.sh). Models: any
fla-format RWKV-7 checkpoint (`fla-hub/rwkv7-*`), or our prequantized int8/int4 checkpoints
on ModelScope (`Hakureirm/rwkv7-g1-*`).

**On a Mac**: [`mlx_port/README.md`](mlx_port/README.md).

## Layout

```
sglang_overlay/    the implementation: model, state backend, CUDA/Triton kernels, spec-decode worker
sglang_main_port/  the same code as applied to sglang main (patch + file list)
mlx_port/          native Apple Silicon implementation (MLX + Metal kernel)
bench/             every benchmark and correctness-gate script; raw outputs in bench/results/
docs/              findings (numbered measurement reports) and design decisions — the evidence chain
scripts/           deploy.sh (v0.5.10 deploy) · serve.sh (recommended launch flags)
```

## Where every claim comes from

[`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) maps each headline number to its raw log.
[`docs/findings/`](docs/findings/) are dated measurement reports with methodology, including
negative results. If you re-run a script in `bench/` and get a different number, please open
an issue.
