# RWKV-7 native MLX port (Apple Silicon)

A self-contained, single-stream RWKV-7 (Goose) inference implementation on
[MLX](https://github.com/ml-explore/mlx), correctness-anchored to this repo's
ground truth: every generated token must match `bench/oracle_numpy.py` (the
pure-numpy fp32 oracle) exactly before any number is reported. Zero external
model deps: no fla, no torch, no transformers — just `mlx` and the fla-format
safetensors checkpoint.

## Files

- `rwkv7_mlx.py` — model + weight loader + greedy `generate()`. Two WKV
  paths: `pure` (vectorized-over-heads MLX ops, sequential over T) and
  `metal` (fused `mx.fast.metal_kernel` scan: one threadgroup per head, one
  thread per V-column, whole chunk per dispatch — the Triton
  `wkv_recurrent.py` mapping ported to Metal). Select per instance
  (`load_model(..., wkv=...)`) or via `RWKV_MLX_WKV`.
- `gate_oracle.py` — the 24-token greedy oracle gate (consumes
  `bench/fixtures/oracle_rwkv7_{01b,15b}_eiffel.json` exactly like
  `bench/greedy_check.py`). Also carries a standalone World-vocab tokenizer
  (trie-equivalent greedy longest-match; no transformers).
- `bench_mlx.py` — bsz1 decode + 1024-token prefill throughput (`--quant w8|w4`
  supported). Re-runs the oracle gate in-process before timing; for fp16 it
  aborts on any mismatch (the exact default), for quant it reports the
  greedy-vs-oracle match without aborting (quant is graded by compression).
- `compression_mlx.py` — direct-call **uncheatable-eval compression rate**
  (bits/byte), the accuracy ruler; same methodology as `bench/uncheatable_eval.py`
  so numbers are comparable to the CUDA column. Works for fp16 and `--quant`
  (F0040). No server needed (MLX has none).
- `sharegpt_mlx.py` — **real-workload** single-stream (bsz1) bench over real
  ShareGPT prompts: TTFT + inter-token-latency distribution and prefill/decode
  throughput over the true prompt-length mix (F0041).
- `bench_mlx_qwen35.py` — **not part of the RWKV-7 port**: benchmarks Qwen3.5-2B
  via `mlx_lm` (the opponent's own native MLX implementation) with the exact
  same protocol as `bench_mlx.py`, for a same-machine matched comparison
  (F0045). This is the one file in this directory that intentionally uses
  `mlx_lm`/`transformers` — the "zero fla/torch/transformers" policy above
  governs what this project ships as its own RWKV-7 implementation, not the
  yardstick used to benchmark a competitor's model.
- `results/` — committed compression + ShareGPT + Qwen3.5 comparison result
  JSONs (reproducibility).

## Correctness gate (oracle-exact)

`GATE_ALL_PASS` — greedy continuation matches the numpy fp32 oracle
**24/24 token-exact** for every (model, WKV path) pair, at bf16 weights
(fp16 fallback never needed):

| model | wkv=pure | wkv=metal | step-fed-prompt cross-check |
|---|---|---|---|
| RWKV-7 0.1B (g1d, 12L·768) | 24/24 | 24/24 | PASS |
| RWKV-7 1.5B (g1g, 24L·2048) | 24/24 | 24/24 | PASS |

The cross-check re-runs the gate feeding the prompt token-by-token through
the compiled decode step (oracle-style) instead of the chunked vectorized
prefill; both prompt paths must produce identical continuations.

Precision policy that passes: **bf16 weights for the big projections**
(emb/head/r/k/v/o/ffn; MLX GEMM accumulates fp32, like cuBLAS bf16) and
**fp32 for everything else** — residual stream, LayerNorms, LoRA chains,
token-shift lerps, kk L2-norm, WKV recurrence + state (fp32 state matches
all backends in this repo), GroupNorm (eps = head_dim·norm_eps = 64e-5, the
oracle's constant), and the r·k·r_k bonus.

## Measured numbers — Apple M5, 32 GB unified, MLX 0.31.2

macOS 27.0, Python 3.13.13, bf16 weights + fp32 state. Decode = bsz1 greedy,
128 steady-state tokens after prompt + 16-token warmup, async-pipelined
(same `greedy_loop` the gate validates), pipeline drained before the clock
stops; prefill = 1024-token prompt, end-to-end including final state
materialization. Median of 3 runs.

| model | WKV path | decode bsz1 (tok/s) | prefill 1024 (tok/s) | peak mem (GiB) |
|---|---|---:|---:|---:|
| 0.1B | pure  | 291.0 | 1274.8 | 0.40 |
| 0.1B | metal | 290.3 | 10399.3 | 0.91 |
| 1.5B | pure  | 32.0 | 384.8 | 3.40 |
| 1.5B | metal | 36.4 | 1947.5 | 6.68 |

Reading the numbers: decode at these sizes is dominated by the per-token
weight read + per-step launch chain, so the fused scan barely moves bsz1
decode at 0.1B (within noise) and gives ~+14% at 1.5B; prefill is where the
Metal kernel pays off (whole-chunk scan in one dispatch per layer vs a
Python-level loop): **8.2x at 0.1B, 5.1x at 1.5B**. Peak memory differs
between paths mainly because the metal path prefills with chunk=256 vs 32
(bigger transient activation/scan buffers), not because of the state.

## How to run

```bash
# weights (fla-format safetensors) — relayed from the LAN box:
# fetch the fla-format model dirs (e.g. from your GPU host or the HF fla-hub mirrors)
scp -r <gpu-host>:/path/to/rwkv7-0.1b-fla /tmp/mlx_models/
scp -r <gpu-host>:/path/to/rwkv7-1.5b-fla /tmp/mlx_models/

# gate (both models x both WKV paths; exits nonzero unless GATE_ALL_PASS)
python mlx_port/gate_oracle.py

# bench (re-gates in-process, then times)
python mlx_port/bench_mlx.py

# generate
python -c "
from mlx_port.rwkv7_mlx import load_model
from mlx_port.gate_oracle import WorldTokenizer
tok = WorldTokenizer('/tmp/mlx_models/rwkv7-1.5b-fla/rwkv_vocab_v20230424.txt')
m = load_model('/tmp/mlx_models/rwkv7-1.5b-fla', wkv='metal')
out, _ = m.generate(tok.encode('\nThe capital of Japan is'), 64)
print(tok.decode(out))"
```

## Scope & honesty notes

- **Single-stream inference port**, deliberately: greedy bsz1 decode +
  chunked recurrent prefill. It is NOT the sglang serving stack — no
  continuous batching, no paged state pool, no server.
- **Optional weight quantization** (`load_model(..., quant="w8"|"w4")` or
  `RWKV_MLX_QUANT`, mirroring CUDA w8g64 / w4-g64): fp16 stays the bit-exact
  default; w8 is greedy-lossless and speeds bsz1 decode +28–68% at −20–39%
  peak memory; w4 goes further on memory/decode at an int4 accuracy cost. Quant
  is gated by the compression ruler, not the oracle gate (F0039/F0040).
- Prefill runs the exact recurrence (chunked sequential scan; state carries
  all cross-chunk context, so chunking is mathematically exact). No
  chunkwise-parallel DPLR algebra — that would change summation order and is
  out of scope for a correctness-first port.
- Numbers above are this machine only (Apple M5, 32 GB). Other Apple chips
  will differ; re-run `bench_mlx.py` to get honest local numbers.
- `num_heads` is derived from the checkpoint's `r_k` shape, not config.json
  (the fla-hub 0.1B config says 32; the checkpoint has 12 — same trap the
  sglang `Rwkv7Config` documents).
- The 0.1B fixture's `prompt_tokens` encode a literal backslash-n (fixture
  quirk, documented in its `_comment`); the gate therefore always feeds the
  fixture's pinned `prompt_tokens`, and tokenizer-encode agreement is
  reported informationally (it MATCHes on the 1.5B fixture).
- Gate scope: 24-token greedy match on the Eiffel fixture per model, the
  same bar every backend in this repo is held to (`bench/greedy_check.py`);
  it pins the full layer math but is not a long-context or sampling test.
