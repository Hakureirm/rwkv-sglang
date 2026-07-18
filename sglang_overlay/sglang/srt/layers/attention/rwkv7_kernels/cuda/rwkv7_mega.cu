// Copyright 2025-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
//
// RWKV-7 megakernel line (task #50, ADR-0008) — Stage-A fused-block increments.
//
// This file is the seed of the bsz1 decode "megakernel" — NOT a cooperative
// single-launch monolith, but (following the 2026-07-13 Albatross structural
// study) a PDL-chained + CUDA-graph-captured micro-kernel pipeline whose fused
// stages keep intermediates on-chip and stream weights continuously.
//
// ROLE-GENERIC GROUPED DECODE GEMV (gemv_grouped_m1_kernel): y[G,N] where
// y[p] = x_p @ W_p[N,K]^T for role p in [0,G). Grid = dim3(N/OutTile, G) and
// blockIdx.y selects the role. This is the "multi-role single-grid" primitive
// from the Albatross study §5 (rkv_lowrank_pre_executor) — the clearest real
// example of whole-block fusion. The public ops layer the projection stages of
// the RWKV-7 time-mix block onto it:
//   gemv_rkv_m1  (G=3)  r/k/v stage        — Stage-A1 increment (F0060 §5)
//   gemv_o_m1    (G=1)  output projection  — Stage-A2 (F0060 §7.5 "add o_proj")
//   gemv_rkvo_m1 (G=4)  whole-block r/k/v/o stage the sm120 megakernel chains
// o_proj is another M==1 [N,K]·[1,K]^T GEMV structurally identical to r/k/v (same
// (N,K)=(H,H), hence the SAME _select_config), so it slots in as a 4th role for
// free. On the 3090 (no PDL persistent grid) o_proj still launches on its own —
// it is post-WKV, so it cannot share the r/k/v launch here; gemv_rkvo_m1 is the
// bit-exact-gated PREFAB the 5090 whole-block grid assembles (see F0060 §7.5).
//
// BIT-EXACTNESS (house law): every output element's fp32 accumulation is the
// SAME per-thread k-stride = Threads*4 partition, the SAME warp-shuffle tree, and
// the SAME serial cross-warp sum as rwkv7_fast.cu::gemv_m1_kernel, and each role
// is launched with the SAME (Threads, OutTile) the deployed gemv_m1_cfg path
// picks for (N,K). The F0064 BW rewrite (below) changes only how the weight/x
// bytes are FETCHED (one 64-bit int2 load per 4-half chunk, optional K-unroll x2
// load-hoist), never which fp32 terms are summed or in what order => y[p] stays
// byte-identical to gemv_m1(x_p, W_p). gemv_m1_kernel itself is left UNCHANGED as
// the bit-exact reference, so bench/test_mega_rkv.py (torch.equal vs G x gemv_m1)
// + greedy verify_m1d prove the rewrite against an independent golden. No new
// numerics are introduced. Roles are passed by value in a
// small pointer pack (param space, no device indirection / gather launch), so a
// role can point at wherever its input lives (non-adjacent shift_lerp6 rows, the
// post-WKV o input, ...).
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

#include "rwkv7_pdl.cuh"  // wait/launch_dependents + armed launch (sm120 step)

namespace {

using dtype = at::Half;

// Max roles a single grouped launch packs (r/k/v/o = 4).
constexpr int kMaxRoles = 4;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

// gemv_grouped_m1: y_all[G,N], y_all[p] = x_p @ W_p[N,K]^T, p = blockIdx.y.
// Grid = dim3(N/OutTile, G). Body is gemv_m1_kernel's accumulation with the F0064
// wide-load rewrite (per-role select of x/weight/y); same fp32 terms in the same
// order, so each row is byte-identical to gemv_m1. The up-to-4 role
// activations/weights are SEPARATE __restrict__ scalar params (not a by-value
// pointer pack) — a dynamic-indexed pack defeats the aliasing/register analysis
// and measured ~1.5x SLOWER than the 3-pointer Stage-A1 kernel; the ternary on
// __restrict__ params keeps the exact codegen of the original (proj = blockIdx.y
// is uniform per block, so the select is a single resolved pointer). Unused
// slots (G<4) mirror role 0's pointers and are never dereferenced (proj < G).
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void gemv_grouped_m1_kernel(
    int K, int N,
    const dtype* __restrict__ x0, const dtype* __restrict__ x1,
    const dtype* __restrict__ x2, const dtype* __restrict__ x3,
    const dtype* __restrict__ w0, const dtype* __restrict__ w1,
    const dtype* __restrict__ w2, const dtype* __restrict__ w3,
    dtype* __restrict__ y_all) {
  const int proj = blockIdx.y;
  const dtype* __restrict__ x =
      (proj == 0) ? x0 : (proj == 1) ? x1 : (proj == 2) ? x2 : x3;
  const dtype* __restrict__ weight =
      (proj == 0) ? w0 : (proj == 1) ? w1 : (proj == 2) ? w2 : w3;
  dtype* __restrict__ y = y_all + static_cast<int64_t>(proj) * N;

  const int n0 = blockIdx.x * OutTile;
  // PDL: x is produced by the stream predecessor (shift_lerp6 / gn_gatecorr);
  // block until its stores are visible. No-op on a plain launch / sm<90.
  rwkv7_pdl_wait();
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) acc[j] = 0.0f;
  // BW LEVER (F0064): the weight stream is the bsz1 bottleneck (flagship BUSY/step
  // is GEMV-dominated at ~81% of peak BW vs Bo's ~92%). Two bit-exact-preserving
  // load-path rewrites, both keeping the EXACT 4-half-per-thread partition and the
  // EXACT per-acc FMA order (=> torch.equal vs gemv_m1 still holds; only how bytes
  // are FETCHED changes, never the fp32 accumulation):
  //   V1 (default): one 64-bit int2 load (4 contiguous halves) per weight row,
  //     replacing the two STRIDED 32-bit __half2 loads (offsets +0 and +2). Across
  //     a warp each thread now reads a contiguous 8-byte span => one 256B/warp
  //     coalesced stream instead of two half-populated sector streams, and the
  //     per-row load-instruction count halves.
  //   V2 (RWKV7_GEMV_KUNROLL2): additionally issue BOTH consecutive k-chunks'
  //     loads before either chunk's FMAs => 2x memory-level parallelism to cover
  //     DRAM latency (Little's law: more bytes in flight = higher achieved BW),
  //     with the FMA order still chunk(k) fully then chunk(k+kstride) fully — the
  //     same sequence the scalar loop produces, so byte-identical.
  const int kstride = Threads << 2;
#if defined(RWKV7_GEMV_KUNROLL2)
  int k = threadIdx.x << 2;
  for (; k + kstride < K; k += kstride << 1) {
    const int2 xp0 = *reinterpret_cast<const int2*>(x + k);
    const int2 xp1 = *reinterpret_cast<const int2*>(x + k + kstride);
    const float2 xa0 = __half22float2(*reinterpret_cast<const __half2*>(&xp0.x));
    const float2 xb0 = __half22float2(*reinterpret_cast<const __half2*>(&xp0.y));
    const float2 xa1 = __half22float2(*reinterpret_cast<const __half2*>(&xp1.x));
    const float2 xb1 = __half22float2(*reinterpret_cast<const __half2*>(&xp1.y));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const dtype* wj = weight + static_cast<int64_t>(n0 + j) * K + k;
      const int2 wp0 = *reinterpret_cast<const int2*>(wj);
      const int2 wp1 = *reinterpret_cast<const int2*>(wj + kstride);
      const float2 wa0 = __half22float2(*reinterpret_cast<const __half2*>(&wp0.x));
      const float2 wb0 = __half22float2(*reinterpret_cast<const __half2*>(&wp0.y));
      const float2 wa1 = __half22float2(*reinterpret_cast<const __half2*>(&wp1.x));
      const float2 wb1 = __half22float2(*reinterpret_cast<const __half2*>(&wp1.y));
      acc[j] = fmaf(xa0.x, wa0.x, acc[j]);
      acc[j] = fmaf(xa0.y, wa0.y, acc[j]);
      acc[j] = fmaf(xb0.x, wb0.x, acc[j]);
      acc[j] = fmaf(xb0.y, wb0.y, acc[j]);
      acc[j] = fmaf(xa1.x, wa1.x, acc[j]);
      acc[j] = fmaf(xa1.y, wa1.y, acc[j]);
      acc[j] = fmaf(xb1.x, wb1.x, acc[j]);
      acc[j] = fmaf(xb1.y, wb1.y, acc[j]);
    }
  }
  for (; k < K; k += kstride) {  // tail: 0 or 1 remaining chunk (V1 body)
    const int2 xp = *reinterpret_cast<const int2*>(x + k);
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(&xp.x));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(&xp.y));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const dtype* wj = weight + static_cast<int64_t>(n0 + j) * K + k;
      const int2 wp = *reinterpret_cast<const int2*>(wj);
      const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(&wp.x));
      const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(&wp.y));
      acc[j] = fmaf(x0.x, w0.x, acc[j]);
      acc[j] = fmaf(x0.y, w0.y, acc[j]);
      acc[j] = fmaf(x1.x, w1.x, acc[j]);
      acc[j] = fmaf(x1.y, w1.y, acc[j]);
    }
  }
#else
  for (int k = threadIdx.x << 2; k < K; k += kstride) {
    const int2 xp = *reinterpret_cast<const int2*>(x + k);
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(&xp.x));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(&xp.y));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const dtype* wj = weight + static_cast<int64_t>(n0 + j) * K + k;
      const int2 wp = *reinterpret_cast<const int2*>(wj);
      const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(&wp.x));
      const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(&wp.y));
      acc[j] = fmaf(x0.x, w0.x, acc[j]);
      acc[j] = fmaf(x0.y, w0.y, acc[j]);
      acc[j] = fmaf(x1.x, w1.x, acc[j]);
      acc[j] = fmaf(x1.y, w1.y, acc[j]);
    }
  }
#endif
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
  rwkv7_pdl_launch_dependents();  // sm_90+ only; inert on sm_86 (see header)
}

#define RWKV7_GROUPED_LAUNCH(T, OT)                                            \
  rwkv7_launch_maybe_pdl(                                                      \
      pdl, gemv_grouped_m1_kernel<T, OT>,                                      \
      dim3(static_cast<unsigned>(N / (OT)), static_cast<unsigned>(G)),         \
      dim3(T), 0, stream.stream(),                                             \
      static_cast<int>(K), static_cast<int>(N),                               \
      xs[0], xs[1], xs[2], xs[3], ws[0], ws[1], ws[2], ws[3],                 \
      y_all.data_ptr<dtype>())

// Shared launcher: allocates y_all[G,N] and dispatches the (threads, out_tile)
// the deployed gemv_m1 path uses for (N,K). Every public op below funnels here.
// xs/ws are host-side 4-arrays (unused slots mirror role 0, never dereferenced
// since blockIdx.y < G); they only expand into the kernel's scalar params.
at::Tensor gemv_grouped_launch(const dtype* const xs[kMaxRoles],
                               const dtype* const ws[kMaxRoles], int64_t G,
                               int64_t K, int64_t N,
                               const at::TensorOptions& opts,
                               int64_t threads, int64_t out_tile) {
  TORCH_CHECK(G >= 1 && G <= kMaxRoles, "gemv_grouped: G must be 1..", kMaxRoles);
  TORCH_CHECK((K % 4) == 0, "gemv_grouped requires K%4==0");
  TORCH_CHECK((N % out_tile) == 0, "gemv_grouped requires N % out_tile == 0");
  auto y_all = at::empty({G, N}, opts);
  if (N == 0) return y_all;
  if (K == 0) return y_all.zero_();
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool pdl = rwkv7_pdl_enabled("mega");
  switch (threads * 100 + out_tile) {
    case 64 * 100 + 1:  RWKV7_GROUPED_LAUNCH(64, 1);  break;
    case 64 * 100 + 2:  RWKV7_GROUPED_LAUNCH(64, 2);  break;
    case 64 * 100 + 4:  RWKV7_GROUPED_LAUNCH(64, 4);  break;
    case 128 * 100 + 1: RWKV7_GROUPED_LAUNCH(128, 1); break;
    case 128 * 100 + 2: RWKV7_GROUPED_LAUNCH(128, 2); break;
    case 128 * 100 + 4: RWKV7_GROUPED_LAUNCH(128, 4); break;
    case 256 * 100 + 1: RWKV7_GROUPED_LAUNCH(256, 1); break;
    case 256 * 100 + 2: RWKV7_GROUPED_LAUNCH(256, 2); break;
    case 256 * 100 + 4: RWKV7_GROUPED_LAUNCH(256, 4); break;
    default: TORCH_CHECK(false, "gemv_grouped unsupported (threads,out_tile)=(",
                         threads, ",", out_tile, "); use {64,128,256}x{1,2,4}");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y_all;
}
#undef RWKV7_GROUPED_LAUNCH

// ---- public ops: each validates M==1 + same (N,K) per role, then funnels ----

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
  const dtype* xr_p = xr.data_ptr<dtype>();
  const dtype* wr_p = wr.data_ptr<dtype>();
  const dtype* xs[kMaxRoles] = {xr_p, xk.data_ptr<dtype>(),
                                xv.data_ptr<dtype>(), xr_p};
  const dtype* ws[kMaxRoles] = {wr_p, wk.data_ptr<dtype>(),
                                wv.data_ptr<dtype>(), wr_p};
  return gemv_grouped_launch(xs, ws, 3, K, N, xr.options(), threads, out_tile);
}

at::Tensor gemv_o_m1(at::Tensor xo, at::Tensor wo,
                     int64_t threads, int64_t out_tile) {
  const int64_t K = wo.size(1);
  const int64_t N = wo.size(0);
  TORCH_CHECK(xo.numel() == K, "gemv_o_m1 requires xo numel==K (M==1)");
  TORCH_CHECK(xo.is_contiguous(), "gemv_o_m1 requires contiguous xo");
  const dtype* xo_p = xo.data_ptr<dtype>();
  const dtype* wo_p = wo.data_ptr<dtype>();
  const dtype* xs[kMaxRoles] = {xo_p, xo_p, xo_p, xo_p};
  const dtype* ws[kMaxRoles] = {wo_p, wo_p, wo_p, wo_p};
  return gemv_grouped_launch(xs, ws, 1, K, N, xo.options(), threads, out_tile);
}

at::Tensor gemv_rkvo_m1(at::Tensor xr, at::Tensor xk, at::Tensor xv,
                        at::Tensor xo, at::Tensor wr, at::Tensor wk,
                        at::Tensor wv, at::Tensor wo,
                        int64_t threads, int64_t out_tile) {
  const int64_t K = wr.size(1);
  const int64_t N = wr.size(0);
  TORCH_CHECK(xr.numel() == K && xk.numel() == K && xv.numel() == K
                  && xo.numel() == K,
              "gemv_rkvo_m1 requires xr/xk/xv/xo numel==K (M==1)");
  TORCH_CHECK(xr.is_contiguous() && xk.is_contiguous() && xv.is_contiguous()
                  && xo.is_contiguous(),
              "gemv_rkvo_m1 requires contiguous xr/xk/xv/xo");
  TORCH_CHECK(wr.size(1) == K && wk.size(1) == K && wv.size(1) == K
                  && wo.size(1) == K,
              "gemv_rkvo_m1 weight [N,K] mismatch");
  TORCH_CHECK(wk.size(0) == N && wv.size(0) == N && wo.size(0) == N,
              "gemv_rkvo_m1 requires r/k/v/o same N");
  const dtype* xs[kMaxRoles] = {xr.data_ptr<dtype>(), xk.data_ptr<dtype>(),
                                xv.data_ptr<dtype>(), xo.data_ptr<dtype>()};
  const dtype* ws[kMaxRoles] = {wr.data_ptr<dtype>(), wk.data_ptr<dtype>(),
                                wv.data_ptr<dtype>(), wo.data_ptr<dtype>()};
  return gemv_grouped_launch(xs, ws, 4, K, N, xr.options(), threads, out_tile);
}

}  // namespace

TORCH_LIBRARY(rwkv7_mega, m) {
  m.def("gemv_rkv_m1(Tensor xr, Tensor xk, Tensor xv, Tensor wr, Tensor wk, "
        "Tensor wv, int threads, int out_tile) -> Tensor");
  m.def("gemv_o_m1(Tensor xo, Tensor wo, int threads, int out_tile) -> Tensor");
  m.def("gemv_rkvo_m1(Tensor xr, Tensor xk, Tensor xv, Tensor xo, Tensor wr, "
        "Tensor wk, Tensor wv, Tensor wo, int threads, int out_tile) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_mega, CUDA, m) {
  m.impl("gemv_rkv_m1", TORCH_FN(gemv_rkv_m1));
  m.impl("gemv_o_m1", TORCH_FN(gemv_o_m1));
  m.impl("gemv_rkvo_m1", TORCH_FN(gemv_rkvo_m1));
}
