# F0079 — the large-batch step is GEMM-bound, and the megakernel line does not reach it

**Status:** OPEN — a lead with numbers, no optimisation attempted yet
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

## What this does not establish

It does not show the GEMM *can* be made faster here — a third of peak is not automatically
recoverable, the shapes are skinny (M=320 against K=N=4096), and cuBLAS may already be
choosing well for them. It names where the time is and what to measure next: a standalone
sweep of these exact shapes against cuBLAS defaults, cuBLASLt heuristics with split-K
disabled, and the sm120-native tile set, before any kernel is written. The int8 tier already
has a hand-written s8-wmma GEMM in this repo, so the question of whether an fp16 counterpart
pays is answerable with the machinery that exists.
