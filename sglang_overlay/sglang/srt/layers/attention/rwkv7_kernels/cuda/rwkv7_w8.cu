// RWKV-7 x sglang hand-written WEIGHT-ONLY int8 decode kernels (companion to rwkv7_w4.cu).
//
// Why weight-only int8 (w8a16) in addition to the existing w8a8 path:
//   * sglang's w8a8 uses sgl-kernel's cutlass int8 GEMM, which only ships sm80-90
//     configs (fails on Turing sm75 AND Blackwell sm100/120 — measured). These
//     kernels JIT-build per-arch and run EVERYWHERE our int4 family runs.
//   * No activation quantization -> accuracy is essentially fp16 (per-channel-group
//     int8 weight RTN is near-lossless), unlike w8a8's small drift.
//   * Decode (small M) is weight-bandwidth-bound: int8 reads 1/2 the bytes of fp16
//     -> faster than fp16 at small M, while keeping better accuracy than int4.
//
// Same skeleton, quant structure and accumulation-order guarantees as rwkv7_w4.cu:
//   * group-wise symmetric int8, GROUP=64 along K: scale[n,g] = max|W|/127;
//     q = round(W/scale) clamped [-127,127]. qweight is int8[N,K] read as uint32[N,K/4]
//     (4 int8 per word, little-endian).
//   * fp32 accumulate, IEEE (no fast-math), cuda-graph safe.
//   * gemm_w8_small: every row's k-iteration/accumulation order is IDENTICAL to
//     gemv_w8_m1 -> per-row results are bit-identical to the M==1 kernel.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

namespace w8 {

using dtype = at::Half;
constexpr int GROUP = 64;                    // quant group along K (K%GROUP==0)
constexpr int WORDS_PER_GROUP = GROUP / 4;   // uint32 words per group (4 int8 each)

__device__ __forceinline__ float warp_sum_w8(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

// unpack 4 signed int8 from a word and fma against 4 activations
__device__ __forceinline__ float dot4_w8(uint32_t p, float2 a0, float2 a1) {
  const int q0 = (int)(int8_t)(p & 0xFF);
  const int q1 = (int)(int8_t)((p >> 8) & 0xFF);
  const int q2 = (int)(int8_t)((p >> 16) & 0xFF);
  const int q3 = (int)(int8_t)((p >> 24) & 0xFF);
  return a0.x * (float)q0 + a0.y * (float)q1 + a1.x * (float)q2 + a1.y * (float)q3;
}

// ---------------------------------------------------------------------------
// gemv_w8_m1: y[1,N] = sum_g scale[n,g] * (x[group g] . Qint[n, group g])
// ---------------------------------------------------------------------------
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void gemv_w8_m1_kernel(
    int K, int N, int NG,
    const __half* __restrict__ x,       // [K]
    const uint32_t* __restrict__ qw,    // [N, K/4]
    const __half* __restrict__ scale,   // [N, NG]
    __half* __restrict__ y) {           // [N]
  const int n0 = blockIdx.x * OutTile;
  const int KW = K >> 2;                // uint32 words per row (4 int8 each)
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) acc[j] = 0.0f;

  for (int t = threadIdx.x; t < KW; t += Threads) {
    const int k = t << 2;
    const int g = t / WORDS_PER_GROUP;
    const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint32_t p = qw[static_cast<int64_t>(n0 + j) * KW + t];
      const float s = __half2float(scale[static_cast<int64_t>(n0 + j) * NG + g]);
      acc[j] = fmaf(dot4_w8(p, a0, a1), s, acc[j]);
    }
  }

  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum_w8(acc[j]);
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

at::Tensor gemv_w8_m1(at::Tensor x, at::Tensor qweight, at::Tensor scale) {
  const int64_t K = x.numel();
  const int64_t N = qweight.size(0);
  const int64_t NG = K / GROUP;
  TORCH_CHECK((K % GROUP) == 0, "gemv_w8_m1 requires K%64==0");
  TORCH_CHECK(qweight.size(1) == K, "gemv_w8_m1 qweight [N,K] mismatch");
  TORCH_CHECK(scale.size(0) == N && scale.size(1) == NG, "gemv_w8_m1 scale [N,K/64] mismatch");
  auto y = at::empty({1, N}, x.options());
  if (K == 0 || N == 0) return y;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* qptr = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int8_t>());
  const auto* xptr = reinterpret_cast<const __half*>(x.data_ptr<dtype>());
  const auto* sptr = reinterpret_cast<const __half*>(scale.data_ptr<dtype>());
  auto* yptr = reinterpret_cast<__half*>(y.data_ptr<dtype>());
  if ((N % 2) == 0) {
    gemv_w8_m1_kernel<128, 2><<<N / 2, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr);
  } else {
    gemv_w8_m1_kernel<128, 1><<<N, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---------------------------------------------------------------------------
// gemm_w8_small: y[M,N] for 2<=M<=8 — one int8 weight-word read feeds all M rows;
// per-row accumulation order identical to gemv_w8_m1 (batch-invariant).
// ---------------------------------------------------------------------------
template <int Threads, int OutTile, int M>
__global__ __launch_bounds__(Threads, 1) void gemm_w8_small_kernel(
    int K, int N, int NG,
    const __half* __restrict__ x,       // [M, K]
    const uint32_t* __restrict__ qw,    // [N, K/4]
    const __half* __restrict__ scale,   // [N, NG]
    __half* __restrict__ y) {           // [M, N]
  const int n0 = blockIdx.x * OutTile;
  const int KW = K >> 2;
  float acc[M][OutTile];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int j = 0; j < OutTile; ++j) acc[m][j] = 0.0f;

  for (int t = threadIdx.x; t < KW; t += Threads) {
    const int k = t << 2;
    const int g = t / WORDS_PER_GROUP;
    float2 a[M][2];
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const __half* xm = x + static_cast<int64_t>(m) * K + k;
      a[m][0] = __half22float2(*reinterpret_cast<const __half2*>(xm));
      a[m][1] = __half22float2(*reinterpret_cast<const __half2*>(xm + 2));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint32_t p = qw[static_cast<int64_t>(n0 + j) * KW + t];
      const float s = __half2float(scale[static_cast<int64_t>(n0 + j) * NG + g]);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        acc[m][j] = fmaf(dot4_w8(p, a[m][0], a[m][1]), s, acc[m][j]);
      }
    }
  }

  __shared__ float partial[Threads / 32][M][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float v = warp_sum_w8(acc[m][j]);
      if (lane == 0) partial[warp][m][j] = v;
    }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < Threads / 32; ++w) sum += partial[w][m][j];
        y[static_cast<int64_t>(m) * N + n0 + j] = __float2half_rn(sum);
      }
  }
}

at::Tensor gemm_w8_small(at::Tensor x, at::Tensor qweight, at::Tensor scale) {
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = qweight.size(0);
  const int64_t NG = K / GROUP;
  TORCH_CHECK(M >= 2 && M <= 8, "gemm_w8_small requires 2<=M<=8");
  TORCH_CHECK((K % GROUP) == 0, "gemm_w8_small requires K%64==0");
  TORCH_CHECK((N % 2) == 0, "gemm_w8_small requires N even");
  TORCH_CHECK(qweight.size(1) == K, "gemm_w8_small qweight [N,K] mismatch");
  TORCH_CHECK(scale.size(0) == N && scale.size(1) == NG, "gemm_w8_small scale [N,K/64] mismatch");
  auto y = at::empty({M, N}, x.options());
  if (K == 0 || N == 0) return y;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* qptr = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int8_t>());
  const auto* xptr = reinterpret_cast<const __half*>(x.data_ptr<dtype>());
  const auto* sptr = reinterpret_cast<const __half*>(scale.data_ptr<dtype>());
  auto* yptr = reinterpret_cast<__half*>(y.data_ptr<dtype>());
  const int blocks = static_cast<int>(N / 2);
  switch (M) {
    case 2: gemm_w8_small_kernel<128, 2, 2><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    case 3: gemm_w8_small_kernel<128, 2, 3><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    case 4: gemm_w8_small_kernel<128, 2, 4><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    case 5: gemm_w8_small_kernel<128, 2, 5><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    case 6: gemm_w8_small_kernel<128, 2, 6><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    case 7: gemm_w8_small_kernel<128, 2, 7><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    default: gemm_w8_small_kernel<128, 2, 8><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---------------------------------------------------------------------------
// dequant_w8: int8 -> fp16 weight [N, K] for the M>8 (batched/prefill) path -> cuBLAS.
// ---------------------------------------------------------------------------
__global__ void dequant_w8_kernel(
    int K, int NG,
    const uint32_t* __restrict__ qw,   // [N, K/4]
    const __half* __restrict__ scale,  // [N, NG]
    __half* __restrict__ out) {        // [N, K]
  const int n = blockIdx.y;
  const int KW = K >> 2;
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < KW; t += gridDim.x * blockDim.x) {
    const uint32_t p = qw[static_cast<int64_t>(n) * KW + t];
    const float s = __half2float(scale[static_cast<int64_t>(n) * NG + (t / WORDS_PER_GROUP)]);
    __half* o = out + static_cast<int64_t>(n) * K + (t << 2);
    o[0] = __float2half_rn((float)(int)(int8_t)(p & 0xFF) * s);
    o[1] = __float2half_rn((float)(int)(int8_t)((p >> 8) & 0xFF) * s);
    o[2] = __float2half_rn((float)(int)(int8_t)((p >> 16) & 0xFF) * s);
    o[3] = __float2half_rn((float)(int)(int8_t)((p >> 24) & 0xFF) * s);
  }
}

at::Tensor dequant_w8(at::Tensor qweight, at::Tensor scale) {
  const int64_t N = qweight.size(0);
  const int64_t K = qweight.size(1);
  const int64_t NG = scale.size(1);
  auto out = at::empty({N, K}, scale.options());
  if (N == 0 || K == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* qptr = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int8_t>());
  const int threads = 256;
  const int blocks_x = static_cast<int>(((K >> 2) + threads - 1) / threads);
  dim3 grid(blocks_x, N);
  dequant_w8_kernel<<<grid, threads, 0, stream>>>(
      K, NG, qptr, reinterpret_cast<const __half*>(scale.data_ptr<dtype>()),
      reinterpret_cast<__half*>(out.data_ptr<dtype>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace w8

TORCH_LIBRARY(rwkv7_w8, m) {
  m.def("gemv_w8_m1(Tensor x, Tensor qweight, Tensor scale) -> Tensor");
  m.def("gemm_w8_small(Tensor x, Tensor qweight, Tensor scale) -> Tensor");
  m.def("dequant_w8(Tensor qweight, Tensor scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_w8, CUDA, m) {
  m.impl("gemv_w8_m1", &w8::gemv_w8_m1);
  m.impl("gemm_w8_small", &w8::gemm_w8_small);
  m.impl("dequant_w8", &w8::dequant_w8);
}
