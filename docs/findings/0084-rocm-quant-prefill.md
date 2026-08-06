---
doc_kind: finding
finding_id: F0084
title: "ROCm fused-dequant prefill closes the W4 implementation gap: 1.31-2.89x at bsz1 and 2.06-3.15x at bsz8 across all six RWKV-7 sizes; the first rocWMMA design was correct but slower and was removed"
last_verified_commit: "HEAD"
discovered_by: gfx1100 48 GiB, ROCm 7.2.1, 2026-08-05
severity: info
status: closed
related: [F0017, F0018, F0055]
---

# F0084 — fused ROCm W8/W4 prefill

## The gap

The ROCm quantized decode path already read packed W8/W4 weights directly and
beat dense at batches 1 and 8. Prefill did not. Every M>8 projection expanded
the complete packed matrix to BF16/FP16 and then called rocBLAS. On the larger
models, dequantization dominated TTFT and made W4 prefill roughly one third of
dense even though the checkpoint occupied far less memory.

The production projection tile is M=256 because `RWKV_ROCM_PREFILL_TILE=256`
also keeps rocBLAS reduction shapes stable across scheduler chunks. The kernel
therefore had to win on real `(M,K,N)` projection shapes rather than on a square
microbenchmark.

## First implementation: correct, rejected

The first design used rocWMMA with a 64x64x64 workgroup tile, eight wave32s,
FP32 accumulation, and group-64 weights unpacked once into LDS. It passed the
dequantized numerical reference in FP16 and BF16.

It did not pass the performance gate. Depending on shape it achieved only
`0.17x-0.33x` the dense GEMM, lost to the existing fallback at several M=256
and M=512 points, and emitted unsupported register-layout-transform warnings
for the selected fragment layout. A correct kernel that slows production is
not a fast path, so the rocWMMA code was removed rather than shipped disabled.

## Shipped implementation

The retained implementation extends the existing Triton fused-dequant kernels:

- fixed `BLOCK_M=64`, `BLOCK_N=32`, `BLOCK_K=64`;
- FP32 accumulation, BF16/FP16 input and output;
- W4 uses eight warps, W8 uses four;
- checkpoint layout stays unchanged (`uint8 [N,K/2]` for W4,
  `int8 [N,K]` for W8, FP16 group scales);
- no full dense weight is materialized;
- one row layout is used for M=64/128/256, preserving exact output across a
  64-row split for every selected operator row;
- W4 covers every valid M=9..256 projection;
- W8 is dispatched only on model shapes where the fused path won. Other W8
  shapes deliberately retain dequantization plus rocBLAS.

`RWKV_ROCM_QUANT_PREFILL=0` is the deployment kill switch. Decode M<=8 keeps
the separate HIP kernel and is unchanged.

## Operator gate

The matrix includes all unique hidden and FFN dimensions used by the public
0.1B, 0.4B, 1.5B, 2.9B, 7.2B, and 13.3B checkpoints. BF16 was tested at
M=9/16/32/64/128/256 against an independently dequantized PyTorch reference and against
the previous dequantize-then-rocBLAS timing. A representative FP16 matrix over
the small, 1.5B, and 7.2B/13.3B shape classes passed the same gate.

- 90 W4 large-M rows selected, all passed;
- 70 W8 large-M rows selected, all passed;
- every selected row was exact when recomputed as 64-row chunks;
- W4 speedup over the model's actual fallback: `2.24x-14.91x`;
- W8 speedup over the model's actual fallback on dispatched rows: `1.09x-4.00x`;
- unsupported W8 rows are recorded as fallback rather than silently counted as
  passes.

Raw machine-readable gates:

- `bench/results/rocm_gfx1100_quant_prefill.json`
- `bench/results/rocm_gfx1100_quant_prefill_extra.json`
- `bench/results/rocm_gfx1100_quant_prefill_fp16.json`

## End-to-end W4 result

Protocol: BF16, HIP graphs enabled through batch 8, radix cache disabled,
64 generated tokens, 256-token prefill, same gfx1100 machine and model files as
the previously committed all-size fallback baseline.

| Size | bsz1 prefill before -> after | speedup | after / dense | bsz8 prefill before -> after | speedup | after / dense |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1B | 8,446.5 -> 11,036.5 | **1.31x** | 0.954x | 17,552.4 -> 36,196.0 | **2.06x** | 0.793x |
| 0.4B | 4,243.7 -> 5,743.5 | **1.35x** | 0.959x | 7,672.3 -> 16,126.5 | **2.10x** | 0.804x |
| 1.5B | 2,399.4 -> 4,241.9 | **1.77x** | 0.889x | 3,100.5 -> 6,448.6 | **2.08x** | 0.625x |
| 2.9B | 1,334.6 -> 2,596.7 | **1.95x** | 0.852x | 1,527.8 -> 3,631.2 | **2.38x** | 0.593x |
| 7.2B | 479.1 -> 1,376.4 | **2.87x** | 0.962x | 516.9 -> 1,628.0 | **3.15x** | 0.776x |
| 13.3B | 251.7 -> 726.5 | **2.89x** | 0.962x | 271.9 -> 853.3 | **3.14x** | 0.774x |

W8 gains are intentionally smaller because fewer shapes are selected: about
3-4% at 1.5B, 2-3% at 2.9B, and 23-26% at 7.2B/13.3B. The small-model M=256
W8 kernels were removed from dispatch after the end-to-end run failed to show a
model-level win despite positive isolated square-projection timings.

The consolidated result is
`bench/results/rocm_gfx1100_quant_prefill_e2e.json`.

## Correctness and remaining limits

A 1.5B 512-token prompt split into eight 64-token chunks produced the same
24-token greedy continuation as single-shot prefill for both W8 and W4. Decode
throughput did not regress in the end-to-end runs because M<=8 still uses the
existing HIP path.

This closes the **ROCm W4 prefill implementation** gap, not every quantization
claim. RTN W4 quality and its near-tie batch sensitivity remain open; they need
GPTQ/K-quant or mixed-precision protection. The optimization is validated on
physical gfx1100 only. CDNA and other RDNA generations still need physical-card
correctness and crossover measurements before receiving architecture-specific
performance claims.

The measurements above were produced by the contributor before the repository
retired `sglang_overlay/`. The same kernels and dispatch have been moved to
`sglang_mainline/`; the obsolete `extend_prefix_lens` backend workaround was not
moved because mainline zeroes fresh recurrent slots upstream. The raw numbers
remain contributor-owned evidence rather than maintainer-reproduced results.
