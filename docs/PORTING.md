# Porting this stack to new hardware (NPU / other accelerators)

A guide for anyone porting the RWKV-7 sglang integration to a new platform — written for
the Ascend NPU effort, applicable to any target. Everything referenced here is in this
repository; nothing else is required.

## The one rule: the oracle gate defines "working"

A port is correct when greedy decoding matches the numpy fp32 reference token-by-token:

1. Reference implementation: [`bench/oracle_numpy.py`](../bench/oracle_numpy.py) — pure
   numpy, fp32, no dependencies. This is the ground truth for every backend we ship
   (CUDA/Triton, Apple Metal, and any future port).
2. Fixtures: [`bench/fixtures/`](../bench/fixtures/) — pinned prompt tokens + the expected
   24 greedy tokens for 0.1B and 1.5B. Gate runner: [`bench/greedy_check.py`](../bench/greedy_check.py).
3. Target: **24/24 exact on 0.1B AND 1.5B** before any performance number is published.
   Both our CUDA stack and the MLX port ([`mlx_port/`](../mlx_port/)) passed this gate; the
   MLX port is the best template for "same math, new backend" — one file, loader + model +
   generation, gated before benched.

Precision policy that passed everywhere: weights may be bf16/fp16, but keep **fp32 for the
recurrence state, residual stream, norms, LoRA chains, and token-shift**. GroupNorm epsilon
is `head_dim * norm_eps` (see the numpy oracle — copy its semantics, not our CUDA code).

## What the port actually consists of

- **The WKV recurrence** (the only custom kernel that matters first): sequential over T,
  parallel over heads and V-columns; fp32 state. Reference mapping:
  [`sglang_overlay/.../rwkv7_kernels/wkv_recurrent.py`](../sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/wkv_recurrent.py)
  (Triton) and `mlx_port/rwkv7_mlx.py` (Metal — one threadgroup per head, one thread per
  V column). A first port can be a pure-ops loop (that alone passed 24/24 on MLX); the
  fused kernel is a speed increment, not a correctness requirement.
- **Everything else is standard ops** (matmuls, norms, sigmoid/lerp gates, sqrelu FFN) —
  if the platform runs PyTorch (e.g. torch-npu), the model file runs eager with only the
  WKV loop replaced.
- **Serving integration** (sglang scheduler, state pool) is platform-neutral Python; the
  state pool allocates plain tensors. The known port risk: pool allocation on a non-CUDA
  device is unverified — start with the standalone model (MLX-port style), integrate
  serving second.

## Known difficulty map (stated up front)

| item | difficulty | note |
|---|---|---|
| standalone greedy inference, eager | low | MLX port took one day incl. gates |
| WKV as a custom kernel | medium | static shapes, no atomics — friendly to static-graph compilers; RWKV state is O(1), no paged KV |
| Triton kernels as-is | high on NPU | expect a rewrite (AscendC / platform DSL) or the eager fallback |
| sglang serving on the device | unverified | state-pool device allocation + graph capture are the open questions |
| bsz=1 latency | platform-dependent | small-kernel launch overhead dominates; batch throughput is the friendlier first target |

Community context worth knowing: recurrent models are a good fit for static-graph
accelerators (fixed-size state, no dynamic KV paging) — the llama.cpp RWKV maintainer has
said as much publicly. The honest caveat from NPU practitioners: single-stream latency is
the hard part; start with batch.

## Prequantized checkpoints

int8 (lossless on our rulers) and int4-GPTQ checkpoints are published on ModelScope
(`Hakureirm/rwkv7-g1-*`). Format: `qweight` (int8/int4-packed) + `scale` per group-64 —
the quantizer is [`bench/quant_w4.py`](../bench/quant_w4.py) (`--bits 8` for int8). A port
can consume these directly (dequant-then-matmul first, fused kernels later). One measured
warning: int4 hurts multi-step reasoning far more than perplexity metrics suggest (see
BENCHMARKS §4) — prefer int8 for quality-sensitive use.

## Contributing back

Run the gate, then the benchmark scripts in `bench/` (they are HTTP clients — engine- and
device-neutral). Numbers submitted with the gate log + raw output attached get added to the
coverage tables with attribution. Preferred shape for a port: an overlay or PR against this
repo rather than a long-lived fork — the MLX port (`mlx_port/`) shows the layout.

Questions: open an issue.
