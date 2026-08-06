// Copyright 2025-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
//
// AMD ROCm small-batch weight-only W8/W4 projections for RWKV-7.
//
// This is deliberately a small, portable HIP fast path.  It contains only the
// bandwidth-bound M<=8 kernels and does not depend on NVIDIA WMMA, cp.async, or
// architecture-specific matrix instructions.  gfx11 uses 32-lane wavefronts,
// matching the reduction layout below.  Other AMD targets fall back in Python.

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <torch/library.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace rwkv7_rocm_quant {

constexpr int GROUP = 64;
constexpr int THREADS = 128;
constexpr int OUT_TILE = 2;

__device__ __forceinline__ float wave32_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down(x, offset, 32);
  }
  return x;
}

template <typename Act>
__device__ __forceinline__ float2 load2_as_float2(const Act* ptr);

template <>
__device__ __forceinline__ float2 load2_as_float2<__half>(const __half* ptr) {
  return __half22float2(*reinterpret_cast<const __half2*>(ptr));
}

template <>
__device__ __forceinline__ float2 load2_as_float2<__hip_bfloat16>(
    const __hip_bfloat16* ptr) {
  return __bfloat1622float2(
      *reinterpret_cast<const __hip_bfloat162*>(ptr));
}

template <typename Act>
__device__ __forceinline__ Act from_float(float value);

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __hip_bfloat16 from_float<__hip_bfloat16>(
    float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ float dot4_w8(uint32_t p, float2 a0, float2 a1) {
  const int q0 = static_cast<int>(static_cast<int8_t>(p & 0xff));
  const int q1 = static_cast<int>(static_cast<int8_t>((p >> 8) & 0xff));
  const int q2 = static_cast<int>(static_cast<int8_t>((p >> 16) & 0xff));
  const int q3 = static_cast<int>(static_cast<int8_t>((p >> 24) & 0xff));
  return a0.x * static_cast<float>(q0) + a0.y * static_cast<float>(q1) +
         a1.x * static_cast<float>(q2) + a1.y * static_cast<float>(q3);
}

__device__ __forceinline__ int signed_nibble(uint32_t p, int shift) {
  int q = static_cast<int>((p >> shift) & 0xf);
  return q - ((q & 8) << 1);
}

__device__ __forceinline__ float dot8_w4(
    uint32_t p, float2 a0, float2 a1, float2 a2, float2 a3) {
  return a0.x * static_cast<float>(signed_nibble(p, 0)) +
         a0.y * static_cast<float>(signed_nibble(p, 4)) +
         a1.x * static_cast<float>(signed_nibble(p, 8)) +
         a1.y * static_cast<float>(signed_nibble(p, 12)) +
         a2.x * static_cast<float>(signed_nibble(p, 16)) +
         a2.y * static_cast<float>(signed_nibble(p, 20)) +
         a3.x * static_cast<float>(signed_nibble(p, 24)) +
         a3.y * static_cast<float>(signed_nibble(p, 28));
}

template <typename Act, int M>
__global__ __launch_bounds__(THREADS, 1) void w8_small_kernel(
    int K,
    int N,
    int NG,
    const Act* __restrict__ x,
    const uint32_t* __restrict__ qweight,
    const __half* __restrict__ scale,
    Act* __restrict__ y) {
  const int n0 = blockIdx.x * OUT_TILE;
  const int words_per_row = K >> 2;
  constexpr int words_per_group = GROUP / 4;
  float acc[M][OUT_TILE] = {};

  for (int t = threadIdx.x; t < words_per_row; t += THREADS) {
    const int k = t << 2;
    const int group = t / words_per_group;
    float2 activation[M][2];
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const Act* row = x + static_cast<int64_t>(m) * K + k;
      activation[m][0] = load2_as_float2(row);
      activation[m][1] = load2_as_float2(row + 2);
    }
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
      const uint32_t packed =
          qweight[static_cast<int64_t>(n0 + j) * words_per_row + t];
      const float s = __half2float(
          scale[static_cast<int64_t>(n0 + j) * NG + group]);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        acc[m][j] = fmaf(
            dot4_w8(packed, activation[m][0], activation[m][1]),
            s,
            acc[m][j]);
      }
    }
  }

  __shared__ float partial[THREADS / 32][M][OUT_TILE];
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
      const float v = wave32_sum(acc[m][j]);
      if (lane == 0) partial[wave][m][j] = v;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int j = 0; j < OUT_TILE; ++j) {
        float sum = 0.0f;
#pragma unroll
        for (int wave_id = 0; wave_id < THREADS / 32; ++wave_id) {
          sum += partial[wave_id][m][j];
        }
        y[static_cast<int64_t>(m) * N + n0 + j] = from_float<Act>(sum);
      }
    }
  }
}

template <typename Act, int M>
__global__ __launch_bounds__(THREADS, 1) void w4_small_kernel(
    int K,
    int N,
    int NG,
    const Act* __restrict__ x,
    const uint32_t* __restrict__ qweight,
    const __half* __restrict__ scale,
    Act* __restrict__ y) {
  const int n0 = blockIdx.x * OUT_TILE;
  const int words_per_row = K >> 3;
  constexpr int words_per_group = GROUP / 8;
  float acc[M][OUT_TILE] = {};

  for (int t = threadIdx.x; t < words_per_row; t += THREADS) {
    const int k = t << 3;
    const int group = t / words_per_group;
    float2 activation[M][4];
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const Act* row = x + static_cast<int64_t>(m) * K + k;
      activation[m][0] = load2_as_float2(row);
      activation[m][1] = load2_as_float2(row + 2);
      activation[m][2] = load2_as_float2(row + 4);
      activation[m][3] = load2_as_float2(row + 6);
    }
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
      const uint32_t packed =
          qweight[static_cast<int64_t>(n0 + j) * words_per_row + t];
      const float s = __half2float(
          scale[static_cast<int64_t>(n0 + j) * NG + group]);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        acc[m][j] = fmaf(
            dot8_w4(
                packed,
                activation[m][0],
                activation[m][1],
                activation[m][2],
                activation[m][3]),
            s,
            acc[m][j]);
      }
    }
  }

  __shared__ float partial[THREADS / 32][M][OUT_TILE];
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
      const float v = wave32_sum(acc[m][j]);
      if (lane == 0) partial[wave][m][j] = v;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int j = 0; j < OUT_TILE; ++j) {
        float sum = 0.0f;
#pragma unroll
        for (int wave_id = 0; wave_id < THREADS / 32; ++wave_id) {
          sum += partial[wave_id][m][j];
        }
        y[static_cast<int64_t>(m) * N + n0 + j] = from_float<Act>(sum);
      }
    }
  }
}

template <typename Launch>
at::Tensor launch_small(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scale,
    int64_t expected_qk,
    Launch launch) {
  TORCH_CHECK(x.is_cuda(), "x must be on an AMD GPU");
  TORCH_CHECK(
      x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
      "ROCm quant kernel requires fp16 or bf16 activation");
  TORCH_CHECK(x.dim() == 2, "x must have shape [M,K]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(qweight.is_contiguous() && scale.is_contiguous(), "quant tensors must be contiguous");
  TORCH_CHECK(qweight.device() == x.device() && scale.device() == x.device(), "all tensors must share a device");
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = qweight.size(0);
  TORCH_CHECK(M >= 1 && M <= 8, "ROCm quant kernel requires 1<=M<=8");
  TORCH_CHECK(K % GROUP == 0, "ROCm quant kernel requires K%64==0");
  TORCH_CHECK(N % OUT_TILE == 0, "ROCm quant kernel requires an even N");
  TORCH_CHECK(qweight.dim() == 2 && qweight.size(1) == expected_qk, "qweight shape mismatch");
  TORCH_CHECK(scale.scalar_type() == at::kHalf, "scale must be fp16");
  TORCH_CHECK(scale.dim() == 2 && scale.size(0) == N && scale.size(1) == K / GROUP, "scale shape mismatch");

  auto y = at::empty({M, N}, x.options());
  if (M == 0 || K == 0 || N == 0) return y;
  launch(static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), y);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return y;
}

template <typename Act>
void launch_w8_kernel(
    int M,
    int K,
    int N,
    int NG,
    int blocks,
    hipStream_t stream,
    const Act* x,
    const uint32_t* qweight,
    const __half* scale,
    Act* y) {
  switch (M) {
    case 1: hipLaunchKernelGGL((w8_small_kernel<Act, 1>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 2: hipLaunchKernelGGL((w8_small_kernel<Act, 2>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 3: hipLaunchKernelGGL((w8_small_kernel<Act, 3>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 4: hipLaunchKernelGGL((w8_small_kernel<Act, 4>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 5: hipLaunchKernelGGL((w8_small_kernel<Act, 5>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 6: hipLaunchKernelGGL((w8_small_kernel<Act, 6>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 7: hipLaunchKernelGGL((w8_small_kernel<Act, 7>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    default: hipLaunchKernelGGL((w8_small_kernel<Act, 8>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
  }
}

template <typename Act>
void launch_w4_kernel(
    int M,
    int K,
    int N,
    int NG,
    int blocks,
    hipStream_t stream,
    const Act* x,
    const uint32_t* qweight,
    const __half* scale,
    Act* y) {
  switch (M) {
    case 1: hipLaunchKernelGGL((w4_small_kernel<Act, 1>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 2: hipLaunchKernelGGL((w4_small_kernel<Act, 2>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 3: hipLaunchKernelGGL((w4_small_kernel<Act, 3>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 4: hipLaunchKernelGGL((w4_small_kernel<Act, 4>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 5: hipLaunchKernelGGL((w4_small_kernel<Act, 5>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 6: hipLaunchKernelGGL((w4_small_kernel<Act, 6>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    case 7: hipLaunchKernelGGL((w4_small_kernel<Act, 7>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
    default: hipLaunchKernelGGL((w4_small_kernel<Act, 8>), dim3(blocks), dim3(THREADS), 0, stream, K, N, NG, x, qweight, scale, y); break;
  }
}

at::Tensor linear_w8(
    const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scale) {
  TORCH_CHECK(qweight.scalar_type() == at::kChar, "W8 qweight must be int8");
  return launch_small(
      x, qweight, scale, x.size(1),
      [&](int M, int K, int N, at::Tensor& y) {
        const int NG = K / GROUP;
        const int blocks = N / OUT_TILE;
        auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
        const auto* qp = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int8_t>());
        const auto* sp = reinterpret_cast<const __half*>(scale.data_ptr<at::Half>());
        if (x.scalar_type() == at::kHalf) {
          const auto* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
          auto* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
          launch_w8_kernel(M, K, N, NG, blocks, stream, xp, qp, sp, yp);
        } else {
          const auto* xp = reinterpret_cast<const __hip_bfloat16*>(
              x.data_ptr<at::BFloat16>());
          auto* yp = reinterpret_cast<__hip_bfloat16*>(
              y.data_ptr<at::BFloat16>());
          launch_w8_kernel(M, K, N, NG, blocks, stream, xp, qp, sp, yp);
        }
      });
}

at::Tensor linear_w4(
    const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scale) {
  TORCH_CHECK(qweight.scalar_type() == at::kByte, "W4 qweight must be uint8");
  return launch_small(
      x, qweight, scale, x.size(1) / 2,
      [&](int M, int K, int N, at::Tensor& y) {
        const int NG = K / GROUP;
        const int blocks = N / OUT_TILE;
        auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
        const auto* qp = reinterpret_cast<const uint32_t*>(qweight.data_ptr<uint8_t>());
        const auto* sp = reinterpret_cast<const __half*>(scale.data_ptr<at::Half>());
        if (x.scalar_type() == at::kHalf) {
          const auto* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
          auto* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
          launch_w4_kernel(M, K, N, NG, blocks, stream, xp, qp, sp, yp);
        } else {
          const auto* xp = reinterpret_cast<const __hip_bfloat16*>(
              x.data_ptr<at::BFloat16>());
          auto* yp = reinterpret_cast<__hip_bfloat16*>(
              y.data_ptr<at::BFloat16>());
          launch_w4_kernel(M, K, N, NG, blocks, stream, xp, qp, sp, yp);
        }
      });
}

TORCH_LIBRARY(rwkv7_rocm_quant, m) {
  m.def("linear_w8(Tensor x, Tensor qweight, Tensor scale) -> Tensor");
  m.def("linear_w4(Tensor x, Tensor qweight, Tensor scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_rocm_quant, CUDA, m) {
  m.impl("linear_w8", &linear_w8);
  m.impl("linear_w4", &linear_w4);
}

}  // namespace rwkv7_rocm_quant
