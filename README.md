# RWKV-7 (Goose) × sglang

**English** · [简体中文](README.zh-CN.md)

Production serving for **RWKV-7** on [sglang](https://github.com/sgl-project/sglang):
token-exact against the reference implementation, quantized (int8/int4), and running on
11 platforms — 10 CUDA GPU models (2018's T4 through B200 and RTX 5090) plus Apple Silicon.
Every number below has its raw log committed in [`bench/results/`](bench/results/).

**How this was built.** Most of the code, kernels, measurements and documents in this
project were produced by AI agents (Claude — Opus, Sonnet and Fable models) working
under human direction; each finding under `docs/findings/` names the model that
produced it in its `discovered_by` field. This is stated here rather than left in the
findings' front-matter because it should change how you read the claims: nothing here
asks to be taken on the author's word. The benchmark scripts, the baselines they are
quoted against, the raw logs and the numpy oracle are committed, and the correctness
claims are checked against implementations this project did not write — BlinkDL's own
runtime and reference above all.

## Three numbers, then the map

| | | detail |
|---|---|---|
| **Speed** | 7.2B fp16 **142.8 tok/s** single-request on one RTX 5090 — 92.0% of Bo's official Albatross; 1.5B **514.5** | [USER.md](docs/USER.md) |
| **Scale** | 1.5B **29,533 tok/s** peak serving at 320 concurrent — constant-size state, <0.2 GB extra for 1→256 streams | [USER.md](docs/USER.md) |
| **Trust** | greedy output **24/24 token-exact** vs a pure-numpy fp32 oracle, on every platform and under TP/PP | [EVIDENCE.md](docs/EVIDENCE.md) |

Find your number by what you need:

| you want | go to |
|---|---|
| one stream fast · throughput under load · smaller models (int8/int4) · your GPU · comparisons (Albatross / vllm-rwkv / Qwen3.5 / HF port) | **[docs/USER.md](docs/USER.md)** |
| how any number was measured, the two timing conventions, the accuracy rulers, how to re-run | **[docs/EVIDENCE.md](docs/EVIDENCE.md)** |
| every measured axis in one page (the full reference) | **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** · [中文](docs/BENCHMARKS.zh-CN.md) |
| dated measurement reports, methodology, negative results | **[docs/FINDINGS.md](docs/FINDINGS.md)** (76 findings) |
| interactive charts (hover / zoom / toggle tiers) | **[hakureirm.github.io/rwkv-sglang/interactive/](https://hakureirm.github.io/rwkv-sglang/interactive/)** |
| each headline claim → its raw log | [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) |

**Runs on sglang `main`**, from [`sglang_mainline/`](sglang_mainline/) — the tree that
produced the numbers below, committed as such. The model-support core is submitted upstream:
[sglang PR #30115](https://github.com/sgl-project/sglang/pull/30115).

## Why RWKV-7 for serving

RWKV-7 is a recurrent model: its per-sequence state is a **fixed size**, no matter how long
the context — a Transformer's KV cache grows with every token. Measured effect: going from
1 to 256 concurrent sequences, or growing the context 64×, each costs **less than 0.2 GB**
of extra VRAM. High concurrency and long context are where this architecture wins.

## What works

| | |
|---|---|
| **Correctness** | Greedy output token-exact vs the numpy fp32 reference — 24/24 on 0.1B / 1.5B / 7.2B (CUDA) and Apple Silicon (MLX); exact under dynamic batching, chunked prefill, CUDA graphs, TP/PP 2/4/8 ([F0036](docs/findings/0036-pp-cudagraph-vfirst-fix.md)) |
| **Accuracy rulers** | MATH500 avg@64 **0.4042** (1.5B) · compression bpb 0.6085 (1.5B) / **0.5413** (7.2B). Quantization measured on both rulers, because they disagree — [USER.md](docs/USER.md#i-want-it-smaller-quantization) |
| **Serving** | Dynamic batching, chunked prefill, recurrent-state prefix cache (~98% hit under high-reuse load), TP/PP |
| **Quantization** | w8g64 greedy-lossless · w8a8 tensor-core int8 (on 7.2B/32 GB: 1.86× the concurrency fp16 reaches) · int4 with the honest accuracy bill — [USER.md](docs/USER.md#i-want-it-smaller-quantization) |
| **Speculative decoding** | draft-verify with O(1) state rollback — [F0031](docs/findings/0031-spec-decode-increment-i.md) |
| **Apple Silicon** | native MLX + Metal kernel, gated by the same oracle — [`mlx_port/`](mlx_port/) |
| **Upstream** | model PR [#30115](https://github.com/sgl-project/sglang/pull/30115); found and fixed a silent PP data-corruption bug upstream: [#30015](https://github.com/sgl-project/sglang/issues/30015) → [#30095](https://github.com/sgl-project/sglang/pull/30095) |

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


The hand-written fast-path kernels are opt-in environment flags, all greedy-exact; the
recommended production set is in [`scripts/serve.sh`](scripts/serve.sh). Models: any
fla-format RWKV-7 checkpoint (`fla-hub/rwkv7-*`), or our prequantized int8/int4 checkpoints
on ModelScope (`Hakureirm/rwkv7-g1-*`).

**On AMD ROCm**: source [`scripts/rocm_env.sh`](scripts/rocm_env.sh), then use
[`scripts/serve_rocm.sh`](scripts/serve_rocm.sh). Contributor-owned `gfx1100`
evidence covers the correctness-first ROCm path across the complete public
0.1B / 0.4B / 1.5B / 2.9B /
7.2B / 13.3B matrix; exact revisions, gates, and raw logs are in
[`docs/ROCM.md`](docs/ROCM.md). A standalone HIP W8/W4 decode kernel now makes
both quantized modes faster than dense at bsz1 and bsz8 across the measured
size matrix. Fused ROCm quantized prefill covers M=9..256: W4 prefill improved
1.31-2.89x at bsz1 and 2.06-3.15x at bsz8 across 0.1B-13.3B; W8 uses a
measured shape gate. RTN W4 model quality and physical CDNA validation remain
open gates.

**On a Mac**: [`mlx_port/README.md`](mlx_port/README.md).

## Layout

```
sglang_mainline/   the implementation that runs: model, state backend, CUDA kernels, spec-decode worker
sglang_main_port/  the wiring edits to sglang's own files (upstream_edits.patch)
mlx_port/          native Apple Silicon implementation (MLX + Metal kernel)
bench/             every benchmark and correctness-gate script; raw outputs in bench/results/
docs/              USER.md · EVIDENCE.md · BENCHMARKS.md · FINDINGS.md + findings/ — the evidence chain
scripts/           serve.sh (recommended launch flags)
tools/             doc generators (gen_findings_index.py)
```

If you re-run a script in `bench/` and get a different number, please open an issue —
that is what the raw logs are committed for.
