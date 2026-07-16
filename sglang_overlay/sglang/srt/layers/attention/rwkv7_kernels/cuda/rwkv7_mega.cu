// Copyright 2025-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
//
// RWKV-7 megakernel line (task #50, ADR-0008) — Stage-A fused-block increment.
//
// This file is the seed of the bsz1 decode "megakernel" — NOT a cooperative
// single-launch monolith, but (following the 2026-07-13 Albatross structural
// study) a PDL-chained + CUDA-graph-captured micro-kernel pipeline whose fused
// stages keep intermediates on-chip and stream weights continuously.
//
// Stage-A increment: gemv_rkv_m1 — the r/k/v projections (three independent
// [N,K]·[1,K]^T GEMVs that today are three separate launches) packed into ONE
// grid via a blockIdx.y role-split (proj = blockIdx.y in {0,1,2}). This is the
// "multi-role single-grid" primitive from the Albatross study §5
// (rkv_lowrank_pre_executor) — the clearest real example of whole-block fusion,
// and the megakernel's r/k/v stage.
//
// BIT-EXACTNESS (house law): every output element's fp32 accumulation is the
// SAME code as rwkv7_fast.cu::gemv_m1_kernel (identical per-thread k-stride =
// Threads*4, identical warp-shuffle tree, identical serial cross-warp sum), and
// each proj is launched with the SAME (Threads, OutTile) the deployed
// gemv_m1_cfg path picks for (N,K). => y3[p] is byte-identical to
// gemv_m1(x3[p], w_p). Gated by bench/test_mega_rkv.py (torch.equal vs 3x
// gemv_m1) + greedy verify_m1d. No new numerics are introduced.
//
// PDL (Programmatic Dependent Launch) scaffolding: griddepcontrol.launch_dependents
// requires .target sm_90+ (verified: it FAILS to assemble on sm_86/3090 without
// the guard), so it is compiled out on Ampere and active only on Hopper/Blackwell
// (our sm120 5090). It is currently INERT — becoming a real overlap win only once
// (a) the launch site sets cudaLaunchAttributeProgrammaticStreamSerialization and
// (b) the downstream stage issues griddepcontrol.wait. Both are the documented
// sm120 wiring step (see F0060 / ADR-0008 Stage-A). On the 3090 this file gates
// STRUCTURE + CORRECTNESS only; the flagship overlap number is an sm120 run.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/library.h>

namespace {

using dtype = at::Half;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

// PDL forward-signal: tell the runtime the dependent (next PDL stage) may begin
// its prologue (weight-load) before this kernel's tail retires — hides the
// kernel-to-kernel gap that otherwise stalls weight streaming ~13x/layer even
// inside a captured graph. sm_90+ only; inert (compiled out) on sm_86.
__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// gemv_rkv_m1: y3[3,N] where y3[p] = x_p @ W_p[N,K]^T, p in {r=0,k=1,v=2}.
// Grid = dim3(N/OutTile, 3); blockIdx.y = proj. Body is gemv_m1_kernel verbatim
// (per-proj select of x/weight/y), so each row is bit-identical to gemv_m1. The
// three activations are passed as SEPARATE pointers (not a [3,K] stack) so the
// caller never pays a gather/stack launch — xr/xk/xv can point at wherever they
// live (e.g. non-adjacent rows of the shift_lerp6 output).
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void gemv_rkv_m1_kernel(
    int K, int N,
    const dtype* __restrict__ xr,     // [K]
    const dtype* __restrict__ xk,     // [K]
    const dtype* __restrict__ xv,     // [K]
    const dtype* __restrict__ wr,     // [N, K]
    const dtype* __restrict__ wk,     // [N, K]
    const dtype* __restrict__ wv,     // [N, K]
    dtype* __restrict__ y3) {         // [3, N]
  const int proj = blockIdx.y;        // 0=r, 1=k, 2=v
  const dtype* __restrict__ x = (proj == 0) ? xr : (proj == 1) ? xk : xv;
  const dtype* __restrict__ weight = (proj == 0) ? wr : (proj == 1) ? wk : wv;
  dtype* __restrict__ y = y3 + static_cast<int64_t>(proj) * N;

  const int n0 = blockIdx.x * OutTile;
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) acc[j] = 0.0f;
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const dtype* wj = weight + static_cast<int64_t>(n0 + j) * K + k;
      const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(wj));
      const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(wj + 2));
      acc[j] = fmaf(x0.x, w0.x, acc[j]);
      acc[j] = fmaf(x0.y, w0.y, acc[j]);
      acc[j] = fmaf(x1.x, w1.x, acc[j]);
      acc[j] = fmaf(x1.y, w1.y, acc[j]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) partial[warp][j] = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) sum += partial[w][j];
      y[n0 + j] = __float2half_rn(sum);
    }
  }
  pdl_launch_dependents();  // sm_90+ only; inert on sm_86 (see header)
}

#define RWKV7_RKV_LAUNCH(T, OT)                                                \
  gemv_rkv_m1_kernel<T, OT><<<dim3(static_cast<int>(N) / (OT), 3), (T), 0,     \
                              stream>>>(                                       \
      static_cast<int>(K), static_cast<int>(N), xr.data_ptr<dtype>(),         \
      xk.data_ptr<dtype>(), xv.data_ptr<dtype>(), wr.data_ptr<dtype>(),       \
      wk.data_ptr<dtype>(), wv.data_ptr<dtype>(), y3.data_ptr<dtype>())

at::Tensor gemv_rkv_m1(at::Tensor xr, at::Tensor xk, at::Tensor xv,
                       at::Tensor wr, at::Tensor wk, at::Tensor wv,
                       int64_t threads, int64_t out_tile) {
  const int64_t K = wr.size(1);
  const int64_t N = wr.size(0);
  TORCH_CHECK(xr.numel() == K && xk.numel() == K && xv.numel() == K,
              "gemv_rkv_m1 requires xr/xk/xv numel==K (M==1)");
  TORCH_CHECK(xr.is_contiguous() && xk.is_contiguous() && xv.is_contiguous(),
              "gemv_rkv_m1 requires contiguous xr/xk/xv");
  TORCH_CHECK(wr.size(1) == K && wk.size(1) == K && wv.size(1) == K,
              "gemv_rkv_m1 weight [N,K] mismatch");
  TORCH_CHECK(wk.size(0) == N && wv.size(0) == N,
              "gemv_rkv_m1 requires r/k/v same N");
  TORCH_CHECK((K % 4) == 0, "gemv_rkv_m1 requires K%4==0");
  TORCH_CHECK((N % out_tile) == 0, "gemv_rkv_m1 requires N % out_tile == 0");
  auto y3 = at::empty({3, N}, xr.options());
  if (N == 0) return y3;
  if (K == 0) return y3.zero_();
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (threads * 100 + out_tile) {
    case 64 * 100 + 1:  RWKV7_RKV_LAUNCH(64, 1);  break;
    case 64 * 100 + 2:  RWKV7_RKV_LAUNCH(64, 2);  break;
    case 64 * 100 + 4:  RWKV7_RKV_LAUNCH(64, 4);  break;
    case 128 * 100 + 1: RWKV7_RKV_LAUNCH(128, 1); break;
    case 128 * 100 + 2: RWKV7_RKV_LAUNCH(128, 2); break;
    case 128 * 100 + 4: RWKV7_RKV_LAUNCH(128, 4); break;
    case 256 * 100 + 1: RWKV7_RKV_LAUNCH(256, 1); break;
    case 256 * 100 + 2: RWKV7_RKV_LAUNCH(256, 2); break;
    case 256 * 100 + 4: RWKV7_RKV_LAUNCH(256, 4); break;
    default: TORCH_CHECK(false, "gemv_rkv_m1 unsupported (threads,out_tile)=(",
                         threads, ",", out_tile, "); use {64,128,256}x{1,2,4}");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y3;
}
#undef RWKV7_RKV_LAUNCH

}  // namespace

TORCH_LIBRARY(rwkv7_mega, m) {
  m.def("gemv_rkv_m1(Tensor xr, Tensor xk, Tensor xv, Tensor wr, Tensor wk, "
        "Tensor wv, int threads, int out_tile) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_mega, CUDA, m) {
  m.impl("gemv_rkv_m1", TORCH_FN(gemv_rkv_m1));
}
