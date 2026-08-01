# F0079 — the large-batch step is GEMM-bound, and the megakernel line does not reach it

**Status:** CLOSED — the lead this finding named was measured and **does not exist**; see the
retraction at the end. The profile itself stands and its first two conclusions are the
useful ones.
**Date:** 2026-08-01 · 5090 (sm120), 7.2B fp16, sglang main with the F0078 repairs
**Method:** the same torch-profile → `bench/step_span_from_trace.py` route F0078 used at bsz1,
re-pointed at a c=320 load so the shapes are the ones large-batch serving actually runs.

## Why look

The bsz1 profile is the only one this project had, and every lever built on it — the
megakernel line, the PDL chain, the fused glue — targets a step made of many small kernels.
Nothing said those levers reach the large-batch step, and the peak sweep gives a reason to
doubt it: at c320 the 7.2B reads 8,277 tok/s, about 38.7 ms per forward, while the traffic
that step must move (14 GB of weights plus ~10.7 GB of fp16 state read+write at that
concurrency) is ~14 ms against this card's 1.79 TB/s. Two thirds of the step is something
else, and the bsz1 histogram cannot say what.

## What the profile says

| | bsz1 (F0078, repaired) | c=320 |
|---|---:|---:|
| kernels/step | 464.4 | 1043.8 |
| overlapped transitions | 95.6% | **19.8%** |
| gap/step | −30.3 us | +857.8 us |

Top of the per-kernel table at c=320, per step:

| kernel | count | us | share |
|---|---:|---:|---:|
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3` | 232.7 | **26,493** | **58%** |
| `wkv_decode_kernel<half>` | 31.9 | 6,596 | 14% |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x256_32x4` | 1.3 | 3,153 | 7% |
| `_wkv_recurrent_kernel` | 1.6 | 1,579 | 3% |
| everything else (gn_gatecorr, kk_kmix, add_ln, relu_sq, …) | — | ~9,800 | 21% |

Three things follow.

**The megakernel line is a bsz1 story.** Overlapped transitions fall from 95.6% to 19.8%.
Programmatic dependent launch buys back launch latency between small dependent kernels; at
c=320 the step is a few large GEMMs and there is no launch latency left to hide. That is not
a defect — it is the honest scope of that work, and it belongs next to the +7% ladder so the
ladder is not read as a serving-wide number.

**The step is GEMM-bound, on library kernels.** 58% of it is one cutlass GEMM. Our own
hand-written kernels serve M==1 and do not participate here at all, so the large-batch path
is, in effect, stock cuBLAS plus our glue.

**And that GEMM is leaving a lot on the table.** The projections are 2·320·(4·4096² +
4096·16384 + 16384·4096) ≈ 4.12 TFLOP per step across 32 layers; 26.5 ms of GEMM time puts
the achieved rate near 155 TFLOP/s, roughly a third of this card's dense fp16 tensor-core
peak. Two smells in the trace point the same way: the selected tiles are `cutlass_80_*`,
i.e. Ampere-era shapes dispatched onto Blackwell, and there are 132 `splitKreduce_kernel`
launches per step, which is the split-K path paying a reduction pass.

## Retraction: there is no GEMM headroom, and the "third of peak" was my error

The section above said to measure the shapes standalone before writing any kernel. Doing
that immediately killed the lead:

| shape | achieved |
|---|---:|
| r/k/v/o projection, M=320, K=N=4096 | **223.2 TFLOP/s** |
| ffn key, M=320, K=4096, N=16384 | 223.2 |
| ffn value, M=320, K=16384, N=4096 | 228.5 |
| square 4096³ — the friendliest shape there is | 225.0 |
| square 8192³ | 227.8 |

**The skinny shape runs at the same rate as the friendliest one.** There is no shape penalty
and no dispatch problem; toggling `allow_fp16_reduced_precision_reduction`, the one split-K
knob torch exposes, changes nothing (220.8 vs 221.9, noise). The `cutlass_80_*` tile names
and the splitK launches were real observations that turned out to mean nothing — the library
picks Ampere-lineage tiles because they are the right ones here, not because it failed to
notice the architecture.

The "roughly a third of peak" claim was a bad comparison, and the bad part was mine: I
divided by the fp16-accumulate marketing number. PyTorch accumulates fp16 matmuls in fp32,
and this card's dense fp32-accumulate rate is in the 210–225 band — which is exactly where
every one of these GEMMs already sits. They are at the ceiling, not a third of it.

What survives is the profile's first two findings, and they are the ones worth carrying:
the megakernel/PDL ladder is a bsz1 lever (95.6% → 19.8% overlap), and 58% of the
large-batch step is library GEMM that none of our kernels touch. The correct reading of
that is now the opposite of what this finding first suggested: the large-batch path spends
most of its time in code that is already running at hardware speed, so effort aimed there
should go at *removing work* — fewer or smaller GEMMs, quantisation, sparsity — rather than
at making the existing GEMMs faster. The int8 tier winning at 7.2B (8,756 vs 8,277, F0078)
is exactly that kind of lever, and it is already in the tree.
