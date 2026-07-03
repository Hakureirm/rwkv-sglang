// RWKV-7 x sglang fused fp16 decode GEMV (M6 / ADR-0004).
//
// Adapted from BlinkDL/Albatross `faster3a_2605/cuda/rwkv7_v3a_ops.cu`
// (Apache-2.0, (c) BlinkDL / Bo Peng). See ALBATROSS_LICENSE + NOTICE in this dir.
//
// MODIFICATIONS vs upstream (per Apache-2.0 §4(b)):
//   * Extracted ONLY the row-1 exact GEMV (`gemv_m1`) that the bsz1-decode path
//     uses for the r/k/v/o + ffn projections, into a standalone minimal-dependency
//     extension (no cublasLt / WMMA) so it JIT-builds fast and carries no unused
//     surface. (Upstream's fused LoRA GEMVs are NOT vendored — the LoRA chains run
//     on sglang's per-chain ReplicatedLinear by default, or on the fused `lora4_m1`
//     op (rwkv7_lora.cu) under RWKV_FUSED_LORA=1; see models/rwkv7.py.)
//   * fp32 accumulation throughout (upstream convention), IEEE arithmetic (the JIT
//     build does NOT pass --use_fast_math, so no FTZ / approx transcendentals) —
//     the greedy-EXACT + batch-invariance gates (bench/verify_m1d.py,
//     bench/verify_batch.py, RWKV_FAST_LINEAR=1) hold without fast-math.
//
// gemv_m1: M==1, fp16 IO, fp32 accumulate, static shapes, current stream, no host
// sync, caller-independent buffers -> cuda-graph capturable. K%4==0 and N even are
// required and are guarded by the Python caller (else it falls back to cuBLAS).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>   // TORCH_LIBRARY / TORCH_LIBRARY_IMPL
#include <cuda_fp16.h>
#include <vector>

using dtype = at::Half;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

// ---------------------------------------------------------------------------
// gemv_m1: y[1,N] = x[1,K] @ W[N,K]^T   (torch nn.Linear layout, fp32 accum).
// Adapted from albatross linear_orig_row1_exact4_f16_kernel. K%4==0, N%OutTile==0.
// ---------------------------------------------------------------------------
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void gemv_m1_kernel(
    int K, int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,   // [N, K]
    dtype* __restrict__ y) {
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
}

at::Tensor gemv_m1(at::Tensor x, at::Tensor weight) {
  const int64_t K = x.size(-1);
  const int64_t N = weight.size(0);
  TORCH_CHECK(x.numel() == K, "gemv_m1 requires M==1");
  TORCH_CHECK(weight.size(1) == K, "gemv_m1 weight [N,K] mismatch");
  TORCH_CHECK((K % 4) == 0, "gemv_m1 requires K%4==0");
  auto y = at::empty({1, N}, x.options());
  if (K == 0 || N == 0) return y;
  auto stream = at::cuda::getCurrentCUDAStream();
  if ((N % 2) == 0) {
    gemv_m1_kernel<128, 2><<<N / 2, 128, 0, stream>>>(
        K, N, x.data_ptr<dtype>(), weight.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    gemv_m1_kernel<128, 1><<<N, 128, 0, stream>>>(
        K, N, x.data_ptr<dtype>(), weight.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

TORCH_LIBRARY(rwkv7_fast, m) {
  m.def("gemv_m1(Tensor x, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_fast, CUDA, m) {
  m.impl("gemv_m1", &gemv_m1);
}
