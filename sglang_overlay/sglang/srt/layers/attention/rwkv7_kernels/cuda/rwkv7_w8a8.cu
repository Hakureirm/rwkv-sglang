// Copyright 2025-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
//
// rwkv7_w8a8.cu — w8a8 int8×int8 tensor-core GEMM (per-token activation scale ×
// per-channel weight scale), the sm120 stand-in for sgl_kernel's cutlass
// `int8_scaled_mm` (which is compiled for sm80–90 only and raises
// NotImplementedError on Blackwell consumer parts).
//
//   out[m,n] = (Σ_k x_q[m,k] * w_q[n,k]) * x_scale[m] * w_scale[n]
//
// Same operand contract as the cutlass op so it drops into sglang's
// `--quantization w8a8_int8` linear method unchanged:
//   x_q     [M,K]  int8, row-major contiguous  (from per_token_quant_int8)
//   w       [K,N]  int8 — the .t() view of the loader's contiguous [N,K] tensor
//   x_scale [M]/[M,1] fp32,  w_scale [N]/[N,1] fp32
//   out     [M,N]  fp16 or bf16
//
// Design (probe-backed, F0033: s8 wmma issues at 1.9933× the fp16 rate on sm120):
//   * standard sm80+ `signed char` m16n16k16 wmma fragments + int32 accumulators —
//     one source covers sm80 through sm120, no cutlass, no arch-special intrinsics.
//   * per-channel × per-token scales mean the int32 accumulation runs over the FULL
//     K extent and is rescaled once in the epilogue — no per-group fragment surgery.
//   * int32 sums are order-exact, so the result is bit-identical for a given (m,n)
//     regardless of grid/batch shape: batch-invariant by construction (stronger than
//     the fp16 cuBLAS path it replaces).
//   * block tile 32(M)×128(N)×64(K), 256 threads = 8 warps (2×4), each warp one
//     16×32 sub-tile; 2-stage cp.async double buffer (same idiom as rwkv7_w8.cu's
//     sm80 path); no split-K in V1 (grid is full at the large-M shapes this op is
//     dispatched for; small M stays on the w8a16 GEMV/small kernels).
//
// Built WITHOUT --use_fast_math. cuda-graph safe (no allocs, current stream).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdint>

namespace w8a8 {

constexpr int BM = 32;   // rows per block
constexpr int BN = 128;  // cols per block
constexpr int BK = 64;   // K-extent staged per iteration
constexpr int CPAD = 4;  // int32 epilogue staging pad (132*4B rows, 16B-aligned)

template <typename OutT>
__device__ __forceinline__ OutT to_out(float v);
template <>
__device__ __forceinline__ __half to_out<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 to_out<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ float from_out(__half v) { return __half2float(v); }
__device__ __forceinline__ float from_out(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// One block: out tile [m0, m0+32) × [n0, n0+128), int32 accum over all of K.
// w points at the [N,K] row-major storage (the torch-side [K,N] is its .t() view).
template <typename OutT>
__global__ __launch_bounds__(256, 2) void gemm_w8a8_tc_kernel(
    const int8_t* __restrict__ x,   // [M,K]
    const int8_t* __restrict__ w,   // [N,K] storage
    const float* __restrict__ xs,   // [M]
    const float* __restrict__ ws,   // [N]
    const OutT* __restrict__ bias,  // [N] or nullptr; added in fp32 pre-round
    OutT* __restrict__ y,           // [M,N]
    int M, int N, int K) {
// s8 wmma fragments + cp.async need sm80+ (JIT builds for the local arch only;
// the python side gates dispatch to sm100/120, so older-arch builds never run this).
#if (!defined(__CUDA_ARCH__)) || (__CUDA_ARCH__ >= 800)
  using namespace nvcuda;
  const int n0 = blockIdx.x * BN;
  const int m0 = blockIdx.y * BM;
  const int tid = threadIdx.x;   // 0..255
  const int warp = tid >> 5;     // 0..7
  const int wm = warp >> 2;      // 0..1  -> rows [m0+wm*16, +16)
  const int wn = warp & 3;       // 0..3  -> cols [n0+wn*32, +32)

  __shared__ __align__(16) int8_t s_a[2][BM][BK];
  __shared__ __align__(16) int8_t s_b[2][BN][BK];
  __shared__ __align__(16) int32_t s_c[BM][BN + CPAD];

  const int a_rows_live = (M - m0 < BM) ? (M - m0) : BM;
  const int b_rows_live = (N - n0 < BN) ? (N - n0) : BN;

  // Rows beyond M/N never change across k-chunks: zero them once in both buffers
  // (their MMA products then contribute exact zeros; epilogue stores are guarded).
  for (int r = a_rows_live; r < BM; ++r)
    for (int c = tid; c < BK; c += 256) {
      s_a[0][r][c] = 0;
      s_a[1][r][c] = 0;
    }
  for (int r = b_rows_live; r < BN; ++r)
    for (int c = tid; c < BK; c += 256) {
      s_b[0][r][c] = 0;
      s_b[1][r][c] = 0;
    }
  __syncthreads();

  // Stage k-chunk [k0s, k0s+64) into buffer `buf`: 16B cp.async per op.
  // A: 32 live rows × 4 chunks = ≤128 ops; B: 128 live rows × 4 chunks = ≤512 ops.
  auto stage = [&](int k0s, int buf) {
    const int a_ops = a_rows_live * 4;
    for (int t = tid; t < a_ops; t += 256) {
      const int r = t >> 2, c = (t & 3) * 16;
      __pipeline_memcpy_async(&s_a[buf][r][c],
                              x + static_cast<int64_t>(m0 + r) * K + k0s + c, 16);
    }
    const int b_ops = b_rows_live * 4;
    for (int t = tid; t < b_ops; t += 256) {
      const int r = t >> 2, c = (t & 3) * 16;
      __pipeline_memcpy_async(&s_b[buf][r][c],
                              w + static_cast<int64_t>(n0 + r) * K + k0s + c, 16);
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[2];
  wmma::fill_fragment(acc[0], 0);
  wmma::fill_fragment(acc[1], 0);

  stage(0, 0);
  __pipeline_commit();

  const int nk = K / BK;
  for (int i = 0; i < nk; ++i) {
    if (i + 1 < nk) {
      stage((i + 1) * BK, (i + 1) & 1);
      __pipeline_commit();
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();
    const int buf = i & 1;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
      wmma::load_matrix_sync(a, &s_a[buf][wm * 16][kk], BK);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        // s_b holds B as [n][k] rows; a col_major K×N fragment with ldm=BK reads
        // element (k, n_local) at n_local*BK + k — exactly this layout.
        wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
        wmma::load_matrix_sync(b, &s_b[buf][wn * 32 + h * 16][kk], BK);
        wmma::mma_sync(acc[h], a, b, acc[h]);
      }
    }
    __syncthreads();  // s_a/s_b[buf] free for the stage after next
  }

  // Epilogue: park int32 tiles in smem, rescale, store guarded.
  wmma::store_matrix_sync(&s_c[wm * 16][wn * 32], acc[0], BN + CPAD,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(&s_c[wm * 16][wn * 32 + 16], acc[1], BN + CPAD,
                          wmma::mem_row_major);
  __syncthreads();

  for (int t = tid; t < BM * BN; t += 256) {
    const int r = t >> 7;          // /BN
    const int c = t & (BN - 1);    // %BN
    const int gm = m0 + r, gn = n0 + c;
    if (gm < M && gn < N) {
      // Explicit rounding ops pin the epilogue bit-exactly across compilers:
      // no-bias = two rn-muls; bias = rn-mul then a SINGLE fused mul-add
      // (one rounding — the more accurate form, and what nvcc contracts to).
      const float v1 = __fmul_rn(static_cast<float>(s_c[r][c]), xs[gm]);
      const float v = (bias != nullptr)
                          ? __fmaf_rn(v1, ws[gn], from_out(bias[gn]))
                          : __fmul_rn(v1, ws[gn]);
      y[static_cast<int64_t>(gm) * N + gn] = to_out<OutT>(v);
    }
  }
#endif  // __CUDA_ARCH__ >= 800
}

at::Tensor gemm_w8a8_tc(at::Tensor x, at::Tensor w, at::Tensor x_scale,
                        at::Tensor w_scale, at::ScalarType out_dtype,
                        c10::optional<at::Tensor> bias) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "w8a8: CUDA tensors required");
  TORCH_CHECK(x.dtype() == at::kChar && w.dtype() == at::kChar,
              "w8a8: int8 operands required");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "w8a8: x must be [M,K] contiguous");
  const int64_t M = x.size(0), K = x.size(1);
  TORCH_CHECK(w.dim() == 2 && w.size(0) == K, "w8a8: w must be [K,N]");
  const int64_t N = w.size(1);
  // The [K,N] operand must be the .t() view of contiguous [N,K] storage.
  TORCH_CHECK(w.stride(0) == 1 && w.stride(1) == K,
              "w8a8: w must be a transposed view of a contiguous [N,K] tensor");
  TORCH_CHECK(K % BK == 0, "w8a8: K must be a multiple of ", BK);
  TORCH_CHECK(x_scale.dtype() == at::kFloat && w_scale.dtype() == at::kFloat,
              "w8a8: fp32 scales required");
  TORCH_CHECK(x_scale.numel() == M && w_scale.numel() == N,
              "w8a8: scale shapes must be [M] and [N]");
  auto xs = x_scale.contiguous();
  auto wsc = w_scale.contiguous();
  TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kBFloat16,
              "w8a8: out_dtype must be fp16 or bf16");

  const at::Tensor* b = bias.has_value() ? &bias.value() : nullptr;
  if (b != nullptr) {
    TORCH_CHECK(b->is_cuda() && b->is_contiguous() && b->numel() == N &&
                    b->scalar_type() == out_dtype,
                "w8a8: bias must be contiguous [N] of out_dtype");
  }

  auto y = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0) return y;

#if !defined(USE_ROCM)
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (out_dtype == at::kHalf) {
    gemm_w8a8_tc_kernel<__half><<<grid, 256, 0, stream>>>(
        reinterpret_cast<const int8_t*>(x.data_ptr()),
        reinterpret_cast<const int8_t*>(w.data_ptr()), xs.data_ptr<float>(),
        wsc.data_ptr<float>(),
        b ? reinterpret_cast<const __half*>(b->data_ptr()) : nullptr,
        reinterpret_cast<__half*>(y.data_ptr()), static_cast<int>(M),
        static_cast<int>(N), static_cast<int>(K));
  } else {
    gemm_w8a8_tc_kernel<__nv_bfloat16><<<grid, 256, 0, stream>>>(
        reinterpret_cast<const int8_t*>(x.data_ptr()),
        reinterpret_cast<const int8_t*>(w.data_ptr()), xs.data_ptr<float>(),
        wsc.data_ptr<float>(),
        b ? reinterpret_cast<const __nv_bfloat16*>(b->data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), static_cast<int>(M),
        static_cast<int>(N), static_cast<int>(K));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
  return y;
}

}  // namespace w8a8

TORCH_LIBRARY(rwkv7_w8a8, m) {
  m.def(
      "gemm_w8a8_tc(Tensor x, Tensor w, Tensor x_scale, Tensor w_scale, "
      "ScalarType out_dtype, Tensor? bias) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_w8a8, CUDA, m) {
  m.impl("gemm_w8a8_tc", &w8a8::gemm_w8a8_tc);
}
