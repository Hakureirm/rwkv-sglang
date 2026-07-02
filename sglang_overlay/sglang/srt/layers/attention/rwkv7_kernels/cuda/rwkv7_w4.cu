// RWKV-7 x sglang hand-written weight-only int4 decode GEMV (M7 / req#5, ADR-0004).
//
// Decode is weight-bandwidth-bound: the r/k/v/o + ffn projections are read once per
// token and dominate the byte traffic. Storing those weights as symmetric int4
// (per-output-channel scale) cuts that traffic ~4x vs fp16, so a bsz1 decode GEMV
// over int4 weights is *faster* than fp16 (not merely parity) AND cuts weight VRAM
// ~4x — the two things 4-bit must deliver (VRAM down, speed >= 16-bit).
//
// This is a hand-written int4 variant of `rwkv7_fast.cu::gemv_m1` (same block/warp
// reduction skeleton); it does NOT use bitsandbytes (whose nf4 GEMV dequant is
// slower than fp16 at M==1) or any FLA. IEEE arithmetic (no --use_fast_math),
// fp32 accumulate.
//
// Quantization (must match bench/verify_w4.py / quant_w4):
//   * group-wise symmetric int4, GROUP=64 along K (AWQ/GPTQ-style; far finer than
//     per-channel): scale[n,g] = max_{k in group g}|W[n,k]| / 7;
//     q[n,k] = round(W[n,k]/scale[n, k/GROUP]) clamped [-7,7]. Dequant = scale*q.
//   * packed 8 nibbles per uint32 (little-endian): bits[4i..4i+3] = q[k_base+i]&0xF
//     (2's-complement 4-bit). qweight is uint8[N, K/2], read here as uint32[N, K/8].
//   * GROUP=64 is a multiple of 8, so every uint32 (8 int4) lies entirely inside one
//     group -> apply that group's scale to the word's integer partial-sum in fp32.
//
// gemv_w4_m1: M==1, fp16 activation + fp16 output, int4 weight, fp32 accumulate,
// static shapes, current stream, no host sync -> cuda-graph capturable. K%GROUP==0
// and N even are required (guarded by the Python caller; else it falls back to cuBLAS).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_fp16.h>
#include <cstdint>

using dtype = at::Half;
constexpr int GROUP = 64;              // int4 quant group size along K (K%GROUP==0)
constexpr int WORDS_PER_GROUP = GROUP / 8;  // uint32 words per group (8 int4 each)
// GROUP=64 chosen empirically: RTN sym g64 max-scale gives the best end-to-end lambada
// of the calibration-free int4 schemes (g64 > g128; MSE-clip and asym both HURT task
// accuracy — weight-MSE-optimal != task-optimal; see docs/findings on w4).

__device__ __forceinline__ float warp_sum_w4(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

// y[1,N] = sum_g scale[n,g] * ( x[group g] · Qint[n, group g] )  — group-wise int4.
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void gemv_w4_m1_kernel(
    int K, int N, int NG,               // NG = K/GROUP groups per row
    const __half* __restrict__ x,       // [K]
    const uint32_t* __restrict__ qw,    // [N, K/8]  (8 int4 per uint32)
    const __half* __restrict__ scale,   // [N, NG]
    __half* __restrict__ y) {           // [N]
  const int n0 = blockIdx.x * OutTile;
  const int KW = K >> 3;                // uint32 words per row
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) acc[j] = 0.0f;

  for (int t = threadIdx.x; t < KW; t += Threads) {
    const int k = t << 3;              // base activation index for this word
    const int g = t / WORDS_PER_GROUP; // this whole word lies in group g
    const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    const float2 a2 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 4));
    const float2 a3 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 6));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint32_t p = qw[static_cast<int64_t>(n0 + j) * KW + t];
      // unpack 8 signed int4 (2's complement nibbles)
      int q0 = (int)((p >> 0) & 0xF);  q0 -= (q0 & 8) << 1;
      int q1 = (int)((p >> 4) & 0xF);  q1 -= (q1 & 8) << 1;
      int q2 = (int)((p >> 8) & 0xF);  q2 -= (q2 & 8) << 1;
      int q3 = (int)((p >> 12) & 0xF); q3 -= (q3 & 8) << 1;
      int q4 = (int)((p >> 16) & 0xF); q4 -= (q4 & 8) << 1;
      int q5 = (int)((p >> 20) & 0xF); q5 -= (q5 & 8) << 1;
      int q6 = (int)((p >> 24) & 0xF); q6 -= (q6 & 8) << 1;
      int q7 = (int)((p >> 28) & 0xF); q7 -= (q7 & 8) << 1;
      float part = a0.x * (float)q0 + a0.y * (float)q1
                 + a1.x * (float)q2 + a1.y * (float)q3
                 + a2.x * (float)q4 + a2.y * (float)q5
                 + a3.x * (float)q6 + a3.y * (float)q7;
      acc[j] = fmaf(part, __half2float(scale[static_cast<int64_t>(n0 + j) * NG + g]), acc[j]);
    }
  }

  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum_w4(acc[j]);
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

// x: [.,K] fp16 (M==1);  qweight: uint8 [N, K/2];  scale: fp16 [N, K/GROUP].  -> y:[1,N] fp16
at::Tensor gemv_w4_m1(at::Tensor x, at::Tensor qweight, at::Tensor scale) {
  const int64_t K = x.numel();
  const int64_t N = qweight.size(0);
  const int64_t NG = K / GROUP;
  TORCH_CHECK((K % GROUP) == 0, "gemv_w4_m1 requires K%64==0");
  TORCH_CHECK(qweight.size(1) == K / 2, "gemv_w4_m1 qweight [N,K/2] mismatch");
  TORCH_CHECK(scale.size(0) == N && scale.size(1) == NG, "gemv_w4_m1 scale [N,K/64] mismatch");
  auto y = at::empty({1, N}, x.options());
  if (K == 0 || N == 0) return y;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* qptr = reinterpret_cast<const uint32_t*>(qweight.data_ptr<uint8_t>());
  const auto* xptr = reinterpret_cast<const __half*>(x.data_ptr<dtype>());
  const auto* sptr = reinterpret_cast<const __half*>(scale.data_ptr<dtype>());
  auto* yptr = reinterpret_cast<__half*>(y.data_ptr<dtype>());
  if ((N % 2) == 0) {
    gemv_w4_m1_kernel<128, 2><<<N / 2, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr);
  } else {
    gemv_w4_m1_kernel<128, 1><<<N, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---------------------------------------------------------------------------
// dequant_w4: int4 (qweight,scale) -> fp16 weight [N, K]. Memory-bound; used by the
// M>1 (prefill / batched-decode) path which then calls cuBLAS. At M>1 the GEMM is
// compute-bound and the weight read is amortized across the batch, so w4 cannot beat
// fp16 there — the goal is only to MATCH it (req#5 "not slower than 16-bit") while
// keeping the int4 checkpoint. One fp16 word per thread-iteration; coalesced.
// ---------------------------------------------------------------------------
__global__ void dequant_w4_kernel(
    int K, int NG,
    const uint32_t* __restrict__ qw,   // [N, K/8]
    const __half* __restrict__ scale,  // [N, NG]
    __half* __restrict__ out) {        // [N, K]
  const int n = blockIdx.y;
  const int KW = K >> 3;
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < KW; t += gridDim.x * blockDim.x) {
    const uint32_t p = qw[static_cast<int64_t>(n) * KW + t];
    const float s = __half2float(scale[static_cast<int64_t>(n) * NG + (t / WORDS_PER_GROUP)]);
    __half* o = out + static_cast<int64_t>(n) * K + (t << 3);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int q = (int)((p >> (4 * i)) & 0xF);
      q -= (q & 8) << 1;
      o[i] = __float2half_rn((float)q * s);
    }
  }
}

at::Tensor dequant_w4(at::Tensor qweight, at::Tensor scale) {
  const int64_t N = qweight.size(0);
  const int64_t K = qweight.size(1) * 2;
  const int64_t NG = scale.size(1);
  auto out = at::empty({N, K}, scale.options());  // fp16
  if (N == 0 || K == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* qptr = reinterpret_cast<const uint32_t*>(qweight.data_ptr<uint8_t>());
  const int threads = 256;
  const int blocks_x = static_cast<int>(((K >> 3) + threads - 1) / threads);
  dim3 grid(blocks_x, N);
  dequant_w4_kernel<<<grid, threads, 0, stream>>>(
      K, NG, qptr, reinterpret_cast<const __half*>(scale.data_ptr<dtype>()),
      reinterpret_cast<__half*>(out.data_ptr<dtype>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY(rwkv7_w4, m) {
  m.def("gemv_w4_m1(Tensor x, Tensor qweight, Tensor scale) -> Tensor");
  m.def("dequant_w4(Tensor qweight, Tensor scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_w4, CUDA, m) {
  m.impl("gemv_w4_m1", &gemv_w4_m1);
  m.impl("dequant_w4", &dequant_w4);
}
