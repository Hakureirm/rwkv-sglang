# AMD ROCm bring-up

This document tracks the ROCm port of the RWKV-7 SGLang mainline tree. The first
validation target is a 48 GiB `gfx1100` device with ROCm 7.2.1, PyTorch 2.9.1,
and Triton 3.5.1.

The implementation now lands directly under [`sglang_mainline/`](../sglang_mainline/).
The committed measurements were produced by the contributor on the same gfx1100
machine while this work still targeted the retired v0.5.10 layout; they are kept
as contributor-owned raw evidence. The model, quantization dispatch, and kernel
code have since been rebased onto mainline without carrying over the retired
backend's `extend_prefix_lens` state-reset logic. Mainline clears fresh recurrent
slots upstream instead.

## Current path

The ROCm lane uses the existing Triton WKV recurrence and standard PyTorch
projections for the dense model. CUDA C++ extensions that depend on PTX, WMMA,
`cp.async`, or NVIDIA warp semantics remain disabled. Weight-only W8/W4 decode
now has a separate HIP extension for batches 1 through 8; it shares checkpoint
layouts with CUDA but contains no NVIDIA-only code. BF16 and FP16 are supported.
Triton remains the build-failure fallback, and larger quantized prefill batches
use a fixed-tile fused-dequant Triton GEMM through the stable 256-row window.
W4 covers the complete M=9..256 range. W8 is shape-gated and retains
dequantization plus rocBLAS wherever the fused path did not win end to end.

Source the environment helper before deploying manually:

```bash
PYTHON=/path/to/rocm/python source scripts/rocm_env.sh
```

It verifies that PyTorch reports a HIP runtime, derives `PYTORCH_ROCM_ARCH` from
the active card, disables AITER by default for consumer RDNA, and disables the
NVIDIA-only optional kernels. Existing environment values are respected.

ROCm also defaults `RWKV_ROCM_PREFILL_TILE=256`. rocBLAS can select different
GEMM reduction algorithms for a full prompt and a scheduler chunk; the small
rounding difference is recurrently amplified and can eventually change a greedy
token. The stable row micro-tile makes both paths execute the same GEMM shapes.
It affects only ROCm projections with more than 256 rows; decode and CUDA are
unchanged. Set the value to `0` to disable it. On the validated 1.5B / 2048-token
prefill probe, the tile did not add a performance tax: median wall time improved
from 0.2400 s to 0.2243 s (1.07x); both raw logs are included with the all-size
results below.

Quantized prefill is enabled by default with `RWKV_ROCM_QUANT_PREFILL=1`.
Set it to `0` to return every M>8 W8/W4 projection to dequantization plus
rocBLAS without changing checkpoint format or decode dispatch.

For serving, apply `sglang_main_port/upstream_edits.patch` and copy
`sglang_mainline/` into an SGLang `main` checkout as described by the repository
quickstart, then launch with:

```bash
MODEL=/path/to/rwkv7 PYTHON=/path/to/rocm/python \
  bash scripts/serve_rocm.sh
```

PyTorch intentionally keeps `torch.cuda` and SGLang keeps `cuda-graph` naming
on ROCm; those names do not imply an NVIDIA runtime.

## Core numerical gate

The portable recurrence can be tested without installing SGLang:

```bash
/path/to/rocm/python bench/test_rocm_wkv.py
```

The gate covers batched decode and packed variable-length prefill. It compares
Triton output and final recurrent state with a direct fp32 PyTorch expression.

## Serving gates

After the server reports ready, verify dynamic batching against independent
batch-1 requests:

```bash
python bench/verify_rocm_serving.py --url http://127.0.0.1:30000
```

For the recurrent-state cache, launch `MODE=statecache` and include the cache
gate:

```bash
MODEL=/path/to/rwkv7 MODE=statecache bash scripts/serve_rocm.sh
python bench/verify_rocm_serving.py --state-cache
```

The gate flushes the cache, records an uncached continuation, warms a complete
prefix state, and then requires both a non-zero `cached_tokens` result and exact
greedy continuation after restore. This result belongs to the original gfx1100
deployment. It is retained as evidence, not presented as a fresh validation of
mainline's current cache wiring.

To verify chunk boundaries, serve the same model first with
`--chunked-prefill-size 4096` and then with a value larger than the test prompt.
The generated token IDs must be identical. A 5000-token test on `gfx1100` matched
exactly between a `4096 + 904` prefill and a single 5000-token prefill.

### Post-rebase mainline smoke

After moving the implementation to `sglang_mainline/`, the contributor repeated
a focused gfx1100 smoke on a ROCm-compatible SGLang runtime:

- dense G1d 0.1B BF16 matched the independent NumPy oracle for 24/24 tokens;
- HIP graph capture and identical/shared-prefix/mixed batches through batch 8
  were exact;
- a 2048-token prompt split into eight 256-token chunks matched single-shot
  continuation for 24/24 tokens;
- the portable WKV gate passed decode and packed-varlen prefill;
- the rebased W8/W4 HIP and Triton operator gate passed M=1/8/64/256 on
  representative hidden and FFN shapes.

This smoke validates the new landing path and the absence of the retired
`extend_prefix_lens` workaround. The complete all-size performance tables below
remain the original contributor measurements rather than a second full sweep.

## Original gfx1100 result

The initial 1.5B fp16 validation on ROCm 7.2.1 completed the following gates on
the pre-mainline deployment:

- model load and HTTP generation;
- decode graph capture at batch sizes 1, 2, 4, and 8;
- dynamic batch size 8, exact against eight batch-1 references;
- 5000-token chunked prefill, exact against single-shot prefill;
- state-prefix restore, 2048 of 2052 prompt tokens cached (99.805% hit),
  with exact greedy continuation;
- standalone decode and packed-varlen WKV numerical checks.

The machine-readable result is committed as
[`bench/results/rocm_gfx1100_15b.json`](../bench/results/rocm_gfx1100_15b.json).

## Official all-size matrix

The ROCm gate covers every size in the public `BlinkDL/rwkv7-g1` ModelScope
series, using exact checkpoint revisions rather than size aliases:

| Size | Checkpoint | Context | gfx1100 result |
|---:|---|---:|---:|
| 0.1B | `rwkv7-g1d-0.1b-20260129-ctx8192.pth` | 8192 | PASS |
| 0.4B | `rwkv7-g1d-0.4b-20260210-ctx8192.pth` | 8192 | PASS |
| 1.5B | `rwkv7-g1h-1.5b-20260710-ctx10240.pth` | 10240 | PASS |
| 2.9B | `rwkv7-g1h-2.9b-20260710-ctx10240.pth` | 10240 | PASS |
| 7.2B | `rwkv7-g1h-7.2b-20260710-ctx10240.pth` | 10240 | PASS |
| 13.3B | `rwkv7-g1h-13.3b-20260710-ctx10240.pth` | 10240 | PASS |

`tools/convert_rwkv7_blinkdl_to_fla.py` infers layer count, hidden width,
head geometry, all four low-rank dimensions, FFN width, vocabulary, weight
dtype, and `ctxNNNN` from each checkpoint. It also writes both model and
generation configs, so no size-specific source edit is required.

Run the complete matrix with:

```bash
PYTHONPATH=/path/to/sglang/python \
python bench/verify_rocm_all_sizes.py \
  --model-root /models/rwkv7/fla \
  --fixture-root bench/fixtures \
  --output-dir bench/results/rocm-all-sizes
```

For every size, this requires:

1. exact greedy output against an independent pure-numpy fp32 oracle;
2. graph capture through batch size 8;
3. identical, shared-prefix, and mixed dynamic batches matching batch-1;
4. 2048-token prefill split into eight 256-token chunks matching single-shot
   prefill token-for-token.

The exact model/fixture mapping is machine-readable in
[`bench/rocm_model_matrix.json`](../bench/rocm_model_matrix.json).
The consolidated environment, settings, results, and raw logs are in
[`bench/results/rocm_gfx1100_all_sizes/`](../bench/results/rocm_gfx1100_all_sizes/summary.json).
All six revisions passed all four gates on the 48 GiB `gfx1100` target in that
deployment. The raw outputs are attributed to the contributor and are not a
claim that the maintainer independently reproduced them.

## W8/W4 on ROCm

`RWKV_W8=1` and `RWKV_W4=1` use group-size-64 symmetric weight-only
quantization. For decode batches up to 8, the HIP kernel reads packed weights,
applies the group scale, accumulates in fp32, and writes BF16/FP16 output without
materializing a dense weight matrix. The same kernel is used during HIP graph
capture.

Run the standalone operator gate with:

```bash
PYTHONPATH=/path/to/sglang/python \
PYTORCH_ROCM_ARCH=gfx1100 \
python bench/verify_rocm_quant.py \
  --batches 1,8,9,16,32,64,128,256 \
  --output bench/results/rocm_gfx1100_quant.json
```

The committed gfx1100 run covers W8 and W4, BF16 and FP16, batches 1 and 8,
and four real projection shapes. Every row was batch-exact against independent
batch-1 calls. Relative error against an offline dequantized reference remained
below `3.3e-4` for FP16 and `2.7e-3` for BF16. Single-stream kernels were
`1.30x` to `2.93x` faster than dense projections in that matrix.

The first end-to-end gate used G1d 0.1B, BF16, HIP graphs, and radix cache off:

| Mode | bsz1 decode | vs dense | bsz8 decode | vs dense |
|---|---:|---:|---:|---:|
| Dense | 219.0 tok/s | 1.000x | 1772.3 tok/s | 1.000x |
| W8G64 | 248.4 tok/s | **1.134x** | 1924.0 tok/s | **1.086x** |
| W4G64 RTN | 270.2 tok/s | **1.234x** | 1995.8 tok/s | **1.126x** |

Total checkpoint-directory size fell from 382,111,842 bytes dense to
299,839,157 bytes W8 and 257,371,829 bytes W4. These total ratios are smaller
than the projection-only theoretical reductions (51.56% and 26.56% of FP16),
because embeddings, the output head, normalization, and low-rank tensors remain
at checkpoint precision.

The follow-up matrix covered the remaining five official sizes with the same
BF16/HIP-graph/radix-off decode protocol (`64` generated tokens, prefill length
`256`). Decode ratios are against the dense checkpoint on the same process and
GPU. Checkpoint percentages include every file in the model directory. Peak
VRAM is whole-process `rocm-smi` usage at batch 8, so the static SGLang pool
makes the small-model memory reductions look deliberately conservative.

| Size | W8 checkpoint | W4 checkpoint | W8 decode b1 / b8 | W4 decode b1 / b8 | W8 / W4 peak-VRAM reduction at b8 |
|---:|---:|---:|---:|---:|---:|
| 0.4B | 67.6% | 50.8% | **1.192x / 1.091x** | **1.260x / 1.109x** | 0.9% / 0.9% |
| 1.5B | 61.7% | 41.9% | **1.469x / 1.266x** | **1.636x / 1.342x** | 3.9% / 4.8% |
| 2.9B | 58.6% | 37.3% | **1.551x / 1.338x** | **1.694x / 1.334x** | 5.8% / 7.7% |
| 7.2B | 56.7% | 34.3% | **1.722x / 1.497x** | **2.060x / 1.534x** | 10.6% / 14.9% |
| 13.3B | 55.2% | 32.0% | **1.707x / 1.510x** | **2.031x / 1.551x** | 17.7% / 26.2% |

Run this matrix with:

```bash
PYTHONPATH=/path/to/sglang/python \
python bench/verify_rocm_quant_models.py \
  --model-root /models/rwkv7/fla \
  --quant-root /models/rwkv7/quant \
  --fixture-root bench/fixtures \
  --sizes 0.4B,1.5B,2.9B,7.2B,13.3B \
  --dtype bfloat16 --mem-fraction 0.75 \
  --skip-w4-batch \
  --output-dir bench/results/rocm-quant-all-sizes
```

Correctness is intentionally split into independent claims:

- The standalone HIP W8/W4 projections passed an independent dequantized
  reference and were bit-exact between M=8 and eight M=1 calls.
- W8 matched the dense NumPy greedy oracle for 24/24 tokens on all six official
  sizes and passed HIP-graph identical/shared-prefix/mixed dynamic batches.
- W8 and W4 single-shot versus chunked-prefill continuations matched for all six
  sizes.
- RTN W4 is **not** certified as model-quality or batch-exact. The 0.1B fixture
  matched only 1/24 dense tokens. On 0.4B, eight identical requests diverged
  from the checkpoint's batch-1 continuation after token 9, with graphs both on
  and off, even though the projection kernel itself remained batch-exact. An
  fp32 output-head diagnostic removed that divergence but was too slow to ship;
  this points to near-tie logit amplification in the lossy checkpoint rather
  than a hidden W4 kernel mismatch. The broad RTN matrix therefore records this
  gate as explicitly skipped, not passed. Quality-preserving GPTQ/K-quant or
  mixed-precision protection remains required.

Machine-readable evidence is in
[`rocm_gfx1100_quant.json`](../bench/results/rocm_gfx1100_quant.json) and
[`rocm_gfx1100_quant_e2e.json`](../bench/results/rocm_gfx1100_quant_e2e.json).
The five-size summary, raw logs, throughput JSON, and the retained 0.4B W4
known-failure log are in
[`rocm_gfx1100_quant_all_sizes/`](../bench/results/rocm_gfx1100_quant_all_sizes/README.md).

### Fused quantized prefill

The M>8 implementation now fuses group-64 dequantization into the Triton GEMM.
It uses a fixed `64x32x64` output/reduction tile so M=64/128/256 executes the
same row reduction layout. The W4 configuration is enabled for every valid
projection through M=256. W8 uses fewer warps and a conservative shape gate;
non-winning shapes keep the previous fallback.

The operator matrix covers every unique hidden/FFN width in the six official
models, BF16, M=9/16/32/64/128/256, independent dequantized references, 64-row chunk
invariance, and fused-versus-fallback timing; a representative FP16 shape matrix
also passed. All selected rows passed. Across
the measured real shapes, W4 was `2.24x-14.91x` faster than the previous
dequantize-then-rocBLAS projection at M=9..256. W8 is dispatched only on the
winning rows recorded by the same gate.

End-to-end W4 prefill under the existing BF16/HIP-graph/radix-off protocol:

| Size | bsz1 before -> after | speedup | vs dense | bsz8 before -> after | speedup | vs dense |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1B | 8,446.5 -> 11,036.5 | **1.31x** | 0.954x | 17,552.4 -> 36,196.0 | **2.06x** | 0.793x |
| 0.4B | 4,243.7 -> 5,743.5 | **1.35x** | 0.959x | 7,672.3 -> 16,126.5 | **2.10x** | 0.804x |
| 1.5B | 2,399.4 -> 4,241.9 | **1.77x** | 0.889x | 3,100.5 -> 6,448.6 | **2.08x** | 0.625x |
| 2.9B | 1,334.6 -> 2,596.7 | **1.95x** | 0.852x | 1,527.8 -> 3,631.2 | **2.38x** | 0.593x |
| 7.2B | 479.1 -> 1,376.4 | **2.87x** | 0.962x | 516.9 -> 1,628.0 | **3.15x** | 0.776x |
| 13.3B | 251.7 -> 726.5 | **2.89x** | 0.962x | 271.9 -> 853.3 | **3.14x** | 0.774x |

The same dispatch raises W8 prefill by about 3-4% at 1.5B, 2-3% at 2.9B,
and 23-26% at 7.2B/13.3B. The 0.1B/0.4B M=256 shapes remain on fallback
because enabling their individually plausible square kernel did not improve
the complete model.

A 1.5B 512-token prompt split into 64-token chunks matched single-shot greedy
continuation 24/24 for both W8 and W4. The large-M kernel can be disabled with
`RWKV_ROCM_QUANT_PREFILL=0` as a deployment kill switch.

The rejected path is recorded too: a numerically correct rocWMMA prototype was
only `0.17x-0.33x` dense and became slower than fallback on several M=256/512
shapes. It was removed rather than hidden behind an optimistic default. See
[F0084](findings/0084-rocm-quant-prefill.md) and the consolidated
[`rocm_gfx1100_quant_prefill_e2e.json`](../bench/results/rocm_gfx1100_quant_prefill_e2e.json).

Remaining quantization limits are quality and cross-architecture evidence, not
the old W4 prefill implementation: RTN W4 still needs GPTQ/K-quant or
mixed-precision protection, and CDNA targets still require physical-card runs.

## Acceptance sequence

1. `bench/test_rocm_wkv.py` passes on every target architecture.
2. Model startup and 24-token greedy fixture pass for 0.1B and 1.5B.
3. Dynamic batches equal independent batch-1 requests.
4. Chunked prefill equals single-shot prefill.
5. State-cache hit, abort, compaction, and slot reuse preserve continuation.
6. Decode graphs pass for batch sizes 1, 2, 4, and 8.
7. Dense prefill/decode/VRAM results are recorded with raw logs.
8. W8 and W4 pass kernel-reference, memory, decode-speed, and model-level
   batch/graph gates; W4 quality and quantized-prefill performance are reported
   separately rather than hidden by the kernel gate.

## HIP fused-kernel roadmap

- Port the WKV decode kernel first; retain the Triton recurrence as fallback.
- Port sparse SqReLU FFN with wave32/wave64-safe ballot and reduction code.
- Keep the completed M=9..256 fused W8/W4 prefill path gated by operator and
  end-to-end speed evidence; explore activation quantization for W8 rows that
  still fall back.
- Validate W8 and quality-preserving W4 checkpoints over all six model sizes.
- Re-enable glue and normalization fusions independently after byte/numerical
  equivalence tests.
- Tune `gfx1100` first, then validate CDNA targets without hard-coded wave size.
