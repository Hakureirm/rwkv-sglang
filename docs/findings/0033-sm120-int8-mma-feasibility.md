# F0033 — sm120 int8 tensor-core MMA: feasible with standard wmma, 1.9933× fp16 throughput

**Date:** 2026-07-06 · **Status:** PROBE COMPLETE (go decision for the real kernel) · **Prior:** F0018 (w8 kernels), F0032 (§ int8 gap)

## Question

The 5090 (sm120) currently has no int8 tensor-core path: upstream `int8_scaled_mm` is
sm80–90 only (explicit `NotImplementedError` on sm120), and our own `gemm_w8_tc` does its
MMAs in fp16 after in-smem dequant — weight-byte savings, no FLOP advantage. Before
committing to writing an int8×int8 MMA kernel (#28), two unknowns needed settling on real
sm120 silicon: (a) does the standard sm80-era `signed char` wmma fragment API even
compile/run on sm120, or does Blackwell demand a different intrinsic path? (b) what is the
actual int8:fp16 tensor-core throughput ratio — is there real FLOP headroom?

## Probe

`bench/probes/int8_mma_probe.py` — two 30-line kernels, identical structure, register-resident
fragments, 200k back-to-back `mma_sync` iterations per block, 8 blocks/SM × 170 SMs:

- `f16_mma_loop`: `wmma::fragment<…, __half>` × fp32 accumulator, m16n16k16
- `s8_mma_loop`: `wmma::fragment<…, signed char>` × int32 accumulator, m16n16k16

Pure MMA-issue-rate measurement (no memory traffic in the loop) — an upper bound, not an
achievable-GEMM number.

## Result (RTX 5090, sm120, 170 SMs, JIT `TORCH_CUDA_ARCH_LIST=12.0`)

| kernel | throughput |
|---|---|
| fp16 wmma (fp32 accum) | 256.4 TFLOPS |
| **s8 wmma (int32 accum)** | **511.2 TOPS** |
| ratio | **1.9933×** |

Both answers are clean:

1. **The plain sm80+ wmma int8 syntax compiles and runs on sm120 unmodified.** No cutlass,
   no Blackwell-specific intrinsics, no arch-conditional code needed — one kernel source
   covers sm80 through sm120, which matches how the rest of `rwkv7_w8.cu` ships.
2. **The FLOP headroom is the full theoretical 2×.** The int8 tensor-core rate is not
   fused/emulated on this part; a w8a8 GEMM that issues s8 MMAs has 2× the compute ceiling
   of the current fp16-MMA path on top of the 2× weight-byte saving.

## What this de-risks (and what it doesn't)

Go for #28: the kernel work is now purely an engineering task (activation quant epilogue,
s8 fragment loading from group-64 quantized weights, int32→fp16 rescale) with no
platform-support risk. It does NOT promise end-to-end 2×: real GEMMs are partly
bandwidth-bound and the L2-illusion lesson (F0007) applies — the probe number is a ceiling,
and only e2e serving numbers count. Accuracy budget also applies: int8 activations are the
lossy part (w8a16 is our lossless tier; w8a8 is the throughput tier).

## Cross-references

`bench/probes/int8_mma_probe.py` (rerunnable, prints the table above) · F0018 §large-M gap ·
F0032 §secondary notes · BENCHMARKS §4.
