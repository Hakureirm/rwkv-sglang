// RWKV-7 x sglang hand-written weight-only int4 decode GEMV (M7 / req#5, ADR-0004).
//
// Decode is weight-bandwidth-bound: the r/k/v/o + ffn projections are read once per
// token and dominate the byte traffic. Storing those weights as symmetric int4
// (per-output-channel scale) cuts that traffic ~4x vs fp16, so a bsz1 decode GEMV
// over int4 weights is *faster* than fp16 (not merely parity) AND cuts weight VRAM
// ~4x — the two things 4-bit quantization must deliver (VRAM down, speed >= 16-bit).
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
// gemm_w4a8_tc: M>64, int8 activation (per-token quant) x int4 weight on the s8
// tensor-core pipeline from rwkv7_w8a8.cu — replaces the dequant->cuBLAS fallback
// whose effective weight traffic (~36 bits/element) was worse than plain fp16.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <algorithm>
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
// gemm_w4_small: y[M,N] for 2<=M<=8 (small batched decode). ONE int4 weight-word
// read feeds all M rows — the weight traffic (the bandwidth cost) is amortized
// across the batch instead of falling back to dequant->cuBLAS (which round-trips
// a full fp16 copy of the weights through HBM every call). Each row's k-iteration
// and accumulation order is IDENTICAL to gemv_w4_m1, so every row's result is
// bit-identical to the M==1 kernel -> batch-invariant by construction.
// Activations x[M,K] are tiny (<=64KB) and L2-resident across blocks.
// ---------------------------------------------------------------------------
template <int Threads, int OutTile, int M>
__global__ __launch_bounds__(Threads, 1) void gemm_w4_small_kernel(
    int K, int N, int NG,
    const __half* __restrict__ x,       // [M, K]
    const uint32_t* __restrict__ qw,    // [N, K/8]
    const __half* __restrict__ scale,   // [N, NG]
    __half* __restrict__ y) {           // [M, N]
  const int n0 = blockIdx.x * OutTile;
  const int KW = K >> 3;
  float acc[M][OutTile];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int j = 0; j < OutTile; ++j) acc[m][j] = 0.0f;

  for (int t = threadIdx.x; t < KW; t += Threads) {
    const int k = t << 3;
    const int g = t / WORDS_PER_GROUP;
    float2 a[M][4];
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const __half* xm = x + static_cast<int64_t>(m) * K + k;
      a[m][0] = __half22float2(*reinterpret_cast<const __half2*>(xm));
      a[m][1] = __half22float2(*reinterpret_cast<const __half2*>(xm + 2));
      a[m][2] = __half22float2(*reinterpret_cast<const __half2*>(xm + 4));
      a[m][3] = __half22float2(*reinterpret_cast<const __half2*>(xm + 6));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint32_t p = qw[static_cast<int64_t>(n0 + j) * KW + t];
      int q0 = (int)((p >> 0) & 0xF);  q0 -= (q0 & 8) << 1;
      int q1 = (int)((p >> 4) & 0xF);  q1 -= (q1 & 8) << 1;
      int q2 = (int)((p >> 8) & 0xF);  q2 -= (q2 & 8) << 1;
      int q3 = (int)((p >> 12) & 0xF); q3 -= (q3 & 8) << 1;
      int q4 = (int)((p >> 16) & 0xF); q4 -= (q4 & 8) << 1;
      int q5 = (int)((p >> 20) & 0xF); q5 -= (q5 & 8) << 1;
      int q6 = (int)((p >> 24) & 0xF); q6 -= (q6 & 8) << 1;
      int q7 = (int)((p >> 28) & 0xF); q7 -= (q7 & 8) << 1;
      const float s = __half2float(scale[static_cast<int64_t>(n0 + j) * NG + g]);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float part = a[m][0].x * (float)q0 + a[m][0].y * (float)q1
                   + a[m][1].x * (float)q2 + a[m][1].y * (float)q3
                   + a[m][2].x * (float)q4 + a[m][2].y * (float)q5
                   + a[m][3].x * (float)q6 + a[m][3].y * (float)q7;
        acc[m][j] = fmaf(part, s, acc[m][j]);
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
      const float v = warp_sum_w4(acc[m][j]);
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

// x: [M,K] fp16 (2<=M<=8, N even);  qweight: uint8 [N,K/2];  scale: fp16 [N,K/GROUP].
at::Tensor gemm_w4_small(at::Tensor x, at::Tensor qweight, at::Tensor scale) {
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = qweight.size(0);
  const int64_t NG = K / GROUP;
  TORCH_CHECK(M >= 2 && M <= 8, "gemm_w4_small requires 2<=M<=8");
  TORCH_CHECK((K % GROUP) == 0, "gemm_w4_small requires K%64==0");
  TORCH_CHECK((N % 2) == 0, "gemm_w4_small requires N even");
  TORCH_CHECK(qweight.size(1) == K / 2, "gemm_w4_small qweight [N,K/2] mismatch");
  TORCH_CHECK(scale.size(0) == N && scale.size(1) == NG, "gemm_w4_small scale [N,K/64] mismatch");
  auto y = at::empty({M, N}, x.options());
  if (K == 0 || N == 0) return y;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* qptr = reinterpret_cast<const uint32_t*>(qweight.data_ptr<uint8_t>());
  const auto* xptr = reinterpret_cast<const __half*>(x.data_ptr<dtype>());
  const auto* sptr = reinterpret_cast<const __half*>(scale.data_ptr<dtype>());
  auto* yptr = reinterpret_cast<__half*>(y.data_ptr<dtype>());
  const int blocks = static_cast<int>(N / 2);
  switch (M) {
    case 2: gemm_w4_small_kernel<128, 2, 2><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    case 3: gemm_w4_small_kernel<128, 2, 3><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    case 4: gemm_w4_small_kernel<128, 2, 4><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    case 5: gemm_w4_small_kernel<128, 2, 5><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    case 6: gemm_w4_small_kernel<128, 2, 6><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    case 7: gemm_w4_small_kernel<128, 2, 7><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
    case 8: gemm_w4_small_kernel<128, 2, 8><<<blocks, 128, 0, stream>>>(K, N, NG, xptr, qptr, sptr, yptr); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---------------------------------------------------------------------------
// gemm_w4_tc: y[M,N] for 8<M<=64 via TENSOR CORES (wmma m16n16k16, fp32 accum),
// dequantizing the int4 weight tile to fp16 in shared memory per K-step — no fp16
// weight copy ever touches HBM, so weight traffic is 1/4 of a cuBLAS fp16 GEMM.
// K_TILE == GROUP == 64, so each (n, k-tile) has EXACTLY one quant scale (clean).
// Numerics: fp16 inputs, fp32 accumulators; deterministic per-row reduction order
// (fixed k-loop + mma structure), independent of batch composition.
// Layout: y = X[M,K] @ W[N,K]^T. B(k,n) = W[n,k] -> the dequanted smem tile
// (row-major [N_TILE][K_TILE+pad]) is read as a col_major wmma fragment.
// ---------------------------------------------------------------------------
constexpr int TC_M = 16;    // rows per block tile (wmma m)
constexpr int TC_N = 64;    // output cols per block tile (4 warps x 16)
constexpr int TC_K = 64;    // k-step == GROUP
constexpr int TC_KPAD = 8;  // smem padding halfs to dodge bank conflicts

// MT = number of 16-row m-subtiles held in registers (block covers MT*16 rows).
// The weight tile is dequantized ONCE per block regardless of M — weight HBM traffic
// stays 1/4 of fp16 for the whole batch (this is where the int4 win comes from).
template <int MT, bool WritePartial>
__global__ __launch_bounds__(128, 1) void gemm_w4_tc_kernel(
    int M, int K, int N, int NG, int k_chunk,
    const __half* __restrict__ x,       // [M, K]
    const uint32_t* __restrict__ qw,    // [N, K/8]
    const __half* __restrict__ scale,   // [N, NG]
    __half* __restrict__ y,             // [M, N]        (WritePartial=false)
    float* __restrict__ ws) {           // [Z, M, N] f32 (WritePartial=true; split-K partials)
#if __CUDA_ARCH__ >= 800
  // sm80+ 2-stage cp.async pipeline: while tile t is dequanted + MMA'd, tile
  // t+1's activation rows and RAW int4 words stream into the other buffer —
  // the global-load latency that stalls the synchronous path at M=64 long-K
  // shapes hides behind compute. Same k order + wmma structure as the
  // synchronous path below (identical accumulation order; split-K unchanged).
  using namespace nvcuda;
  const int n0 = blockIdx.x * TC_N;
  const int kb = blockIdx.z * k_chunk;
  const int ke = (kb + k_chunk < K) ? kb + k_chunk : K;
  const int KW = K >> 3;
  const int lane = threadIdx.x;        // 0..127
  const int warp = lane >> 5;          // 0..3 -> n-subtile [n0+warp*16, +16)

  __shared__ __align__(16) __half smem_a[2][MT * TC_M][TC_K + TC_KPAD];
  __shared__ __align__(16) uint32_t smem_q[2][TC_N][8];   // raw words: 8/row/k-tile
  __shared__ __half smem_w[TC_N][TC_K + TC_KPAD];
  __shared__ float smem_c[TC_M][TC_N + TC_KPAD];

  // rows >= M never change: zero them once in both buffers.
#pragma unroll
  for (int r = 0; r < MT; ++r) {
    const int elt = (lane + r * 128) * 8;
    const int am = elt / TC_K;
    const int ak = elt % TC_K;
    if (am >= M) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        smem_a[0][am][ak + i] = __float2half(0.0f);
        smem_a[1][am][ak + i] = __float2half(0.0f);
      }
    }
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) wmma::fill_fragment(acc[mt], 0.0f);

  const int a_chunks = M * 8;  // live A rows, 8 x 16B chunks per row
  auto stage = [&](int k0s, int buf) {
    for (int t = lane; t < a_chunks; t += 128) {
      const int am = t >> 3;
      const int ak = (t & 7) * 8;
      __pipeline_memcpy_async(&smem_a[buf][am][ak],
                              x + static_cast<int64_t>(am) * K + k0s + ak, 16);
    }
    for (int t = lane; t < TC_N * 2; t += 128) {  // 8 words = 2 x 16B per row
      const int wn = t >> 1;
      const int wc = (t & 1) * 4;
      __pipeline_memcpy_async(&smem_q[buf][wn][wc],
                              qw + static_cast<int64_t>(n0 + wn) * KW + (k0s >> 3) + wc,
                              16);
    }
  };

  stage(kb, 0);
  __pipeline_commit();

  for (int k0 = kb; k0 < ke; k0 += TC_K) {
    const int cur = ((k0 - kb) >> 6) & 1;
    if (k0 + TC_K < ke) stage(k0 + TC_K, cur ^ 1);
    __pipeline_commit();
    __pipeline_wait_prior(1);  // current tile's copies complete; next stays in flight
    __syncthreads();
    {
      const int g = k0 / TC_K;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int t = lane + r * 128;
        const int wn = t >> 3;
        const int wk = t & 7;
        const uint32_t p = smem_q[cur][wn][wk];
        const float s = __half2float(scale[static_cast<int64_t>(n0 + wn) * NG + g]);
        __half* dst = &smem_w[wn][wk << 3];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          int q = (int)((p >> (4 * i)) & 0xF);
          q -= (q & 8) << 1;
          dst[i] = __float2half_rn((float)q * s);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TC_K; kk += 16) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
      wmma::load_matrix_sync(b_frag, &smem_w[warp * 16][kk], TC_K + TC_KPAD);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, &smem_a[cur][mt * TC_M][kk], TC_K + TC_KPAD);
        wmma::mma_sync(acc[mt], a_frag, b_frag, acc[mt]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    wmma::store_matrix_sync(&smem_c[0][warp * 16], acc[mt], TC_N + TC_KPAD, wmma::mem_row_major);
    __syncthreads();
    const int elt = lane * 8;
    const int cm = elt / TC_N;
    const int cn = elt % TC_N;
    const int gm = mt * TC_M + cm;
    if (gm < M) {
      if (WritePartial) {
        float* dst = ws + (static_cast<int64_t>(blockIdx.z) * M + gm) * N + n0 + cn;
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = smem_c[cm][cn + i];
      } else {
        __half* dst = y + static_cast<int64_t>(gm) * N + n0 + cn;
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = __float2half_rn(smem_c[cm][cn + i]);
      }
    }
    __syncthreads();
  }
#elif __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  const int n0 = blockIdx.x * TC_N;
  const int kb = blockIdx.z * k_chunk;                 // this split's K range
  const int ke = (kb + k_chunk < K) ? kb + k_chunk : K;
  const int KW = K >> 3;
  const int lane = threadIdx.x;        // 0..127
  const int warp = lane >> 5;          // 0..3 -> n-subtile [n0+warp*16, +16)

  __shared__ __half smem_a[MT * TC_M][TC_K + TC_KPAD];
  __shared__ __half smem_w[TC_N][TC_K + TC_KPAD];
  __shared__ float smem_c[TC_M][TC_N + TC_KPAD];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) wmma::fill_fragment(acc[mt], 0.0f);

  for (int k0 = kb; k0 < ke; k0 += TC_K) {
    // ---- stage A tile (MT*16 x 64), zero-padded for m >= M rows ----
    // 128 threads x 8 halfs x MT rounds = MT*16*64
#pragma unroll
    for (int r = 0; r < MT; ++r) {
      const int elt = (lane + r * 128) * 8;
      const int am = elt / TC_K;                   // 0..MT*16-1
      const int ak = elt % TC_K;
      if (am < M) {
        const __half* src = x + static_cast<int64_t>(am) * K + k0 + ak;
#pragma unroll
        for (int i = 0; i < 8; ++i) smem_a[am][ak + i] = src[i];
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) smem_a[am][ak + i] = __float2half(0.0f);
      }
    }
    // ---- stage + dequant W tile (64 x 64) ONCE: 64 rows x 8 words; 512 words ----
    {
      const int g = k0 / TC_K;                     // group index for this k-tile
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int t = lane + r * 128;              // 0..511
        const int wn = t >> 3;                     // 0..63
        const int wk = t & 7;                      // word within row
        const uint32_t p = qw[static_cast<int64_t>(n0 + wn) * KW + (k0 >> 3) + wk];
        const float s = __half2float(scale[static_cast<int64_t>(n0 + wn) * NG + g]);
        __half* dst = &smem_w[wn][wk << 3];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          int q = (int)((p >> (4 * i)) & 0xF);
          q -= (q & 8) << 1;
          dst[i] = __float2half_rn((float)q * s);
        }
      }
    }
    __syncthreads();
    // ---- tensor-core MACs: per kk, load b once, drive all MT m-subtiles ----
#pragma unroll
    for (int kk = 0; kk < TC_K; kk += 16) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
      // B(k,n) = smem_w[n][k]: col_major with ld = row stride of smem_w
      wmma::load_matrix_sync(b_frag, &smem_w[warp * 16][kk], TC_K + TC_KPAD);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, &smem_a[mt * TC_M][kk], TC_K + TC_KPAD);
        wmma::mma_sync(acc[mt], a_frag, b_frag, acc[mt]);
      }
    }
    __syncthreads();
  }

  // ---- epilogue: per m-subtile, acc -> smem (f32) -> global ----
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    wmma::store_matrix_sync(&smem_c[0][warp * 16], acc[mt], TC_N + TC_KPAD, wmma::mem_row_major);
    __syncthreads();
    const int elt = lane * 8;
    const int cm = elt / TC_N;
    const int cn = elt % TC_N;
    const int gm = mt * TC_M + cm;
    if (gm < M) {
      if (WritePartial) {
        float* dst = ws + (static_cast<int64_t>(blockIdx.z) * M + gm) * N + n0 + cn;
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = smem_c[cm][cn + i];
      } else {
        __half* dst = y + static_cast<int64_t>(gm) * N + n0 + cn;
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = __float2half_rn(smem_c[cm][cn + i]);
      }
    }
    __syncthreads();
  }
#endif
}

// deterministic split-K reduce: y[i] = sum_z ws[z][i] in fixed z order.
__global__ void splitk_reduce_kernel(int64_t MN, int Z,
                                     const float* __restrict__ ws,
                                     __half* __restrict__ y) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= MN) return;
  float acc = 0.0f;
  for (int z = 0; z < Z; ++z) acc += ws[static_cast<int64_t>(z) * MN + i];
  y[i] = __float2half_rn(acc);
}

// x: [M,K] fp16 (8<M<=64);  qweight: uint8 [N,K/2];  scale: fp16 [N,K/GROUP]. N%64==0, K%64==0.
at::Tensor gemm_w4_tc(at::Tensor x, at::Tensor qweight, at::Tensor scale) {
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = qweight.size(0);
  const int64_t NG = K / GROUP;
  TORCH_CHECK(M >= 1 && M <= 64, "gemm_w4_tc requires 1<=M<=64");
  TORCH_CHECK((K % TC_K) == 0, "gemm_w4_tc requires K%64==0");
  TORCH_CHECK((N % TC_N) == 0, "gemm_w4_tc requires N%64==0");
  TORCH_CHECK(qweight.size(1) == K / 2, "gemm_w4_tc qweight [N,K/2] mismatch");
  TORCH_CHECK(scale.size(0) == N && scale.size(1) == NG, "gemm_w4_tc scale [N,K/64] mismatch");
  auto y = at::empty({M, N}, x.options());
  if (K == 0 || N == 0) return y;
  auto stream = at::cuda::getCurrentCUDAStream();
  // split-K: one block covers all M (weight dequant once per block), so grid = N/64
  // blocks — too few to fill the GPU for small N. Split K until ~256 blocks are in
  // flight. Deterministic: each split writes an f32 partial; a second kernel reduces
  // in fixed z order (no atomics).
  const int64_t nb = N / TC_N;
  const int splits = static_cast<int>(std::min<int64_t>(
      std::min<int64_t>((K + TC_K - 1) / TC_K, 8),
      std::max<int64_t>(1, (256 + nb - 1) / nb)));
  const auto* xp = reinterpret_cast<const __half*>(x.data_ptr<dtype>());
  const auto* qp = reinterpret_cast<const uint32_t*>(qweight.data_ptr<uint8_t>());
  const auto* sp = reinterpret_cast<const __half*>(scale.data_ptr<dtype>());
  auto* yp = reinterpret_cast<__half*>(y.data_ptr<dtype>());
  const int mt = static_cast<int>((M + TC_M - 1) / TC_M);  // 1..4 m-subtiles, one block covers all M
  if (splits == 1) {
    dim3 grid(N / TC_N, 1, 1);
    const int kc = static_cast<int>(K);
    switch (mt) {
      case 1: gemm_w4_tc_kernel<1, false><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, yp, nullptr); break;
      case 2: gemm_w4_tc_kernel<2, false><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, yp, nullptr); break;
      case 3: gemm_w4_tc_kernel<3, false><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, yp, nullptr); break;
      default: gemm_w4_tc_kernel<4, false><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, yp, nullptr); break;
    }
  } else {
    // k_chunk: multiple of TC_K covering K in `splits` pieces
    int64_t k_chunk = ((K + splits - 1) / splits + TC_K - 1) / TC_K * TC_K;
    auto ws = at::empty({splits, M, N}, x.options().dtype(at::kFloat));
    float* wp = ws.data_ptr<float>();
    dim3 grid(N / TC_N, 1, splits);
    const int kc = static_cast<int>(k_chunk);
    switch (mt) {
      case 1: gemm_w4_tc_kernel<1, true><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, nullptr, wp); break;
      case 2: gemm_w4_tc_kernel<2, true><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, nullptr, wp); break;
      case 3: gemm_w4_tc_kernel<3, true><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, nullptr, wp); break;
      default: gemm_w4_tc_kernel<4, true><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, nullptr, wp); break;
    }
    const int64_t MN = M * N;
    const int threads = 256;
    const int64_t blocks = (MN + threads - 1) / threads;
    splitk_reduce_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
        MN, splits, ws.data_ptr<float>(), yp);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---------------------------------------------------------------------------
// gemm_w4a8_tc: y[M,N] for M>64 (high-concurrency decode / prefill) via int8
// TENSOR CORES — the w4a8 path that replaces the dequant->HBM->cuBLAS fallback,
// whose effective weight traffic (~36 bits/element: 4 read + 16 write + 16 read
// back) is worse than plain fp16's 16 and is why int4 lost to fp16 past M=64.
//
// Route (c): int4 weight tiles are loaded PACKED (half the global bytes of the
// w8a8 path), unpacked nibble->s8 during the smem staging, then the proven
// s8xs8 wmma machinery from rwkv7_w8a8.cu (m16n16k16, int32 accumulators) runs
// unchanged. Activations arrive per-token-quantized to s8 (same
// per_token_quant_int8 the w8a8 tier uses), so the MMAs issue at the s8
// tensor-core rate (2x fp16 on sm86/sm120) instead of dequanting to fp16.
//
// Numerics contract (bit-exactness-gated by bench/verify_w4a8.py):
//   * K_TILE == GROUP == 64: each staged k-tile is exactly one quant group. The
//     int32 accumulation within a group is order-exact (integers), and the
//     group partial folds into fp32 once per group, ascending g:
//       facc = __fadd_rn(facc, __fmul_rn((float)S_g, w_scale[n,g]))
//     Explicit mul+add (NOT fmaf): contraction-proof AND exactly reproducible
//     as torch fp32 tensor ops, so the gate's reference computes the identical
//     chain. |S_g| <= 128*7*64 = 57344 < 2^24, so (float)S_g is exact.
//   * epilogue: y = half_rn(__fmul_rn(facc, x_scale[m])). No atomics, no
//     split-K: a row's bits depend only on its content -> batch-invariant
//     across M and grid shape by construction (same class as w8a8).
//
// Block tile (MFRAG*32)(M) x 128(N) x 64(K), 256 threads = 8 warps (2x4); each
// warp owns an (MFRAG*16)x32 register tile (MFRAG=1 mirrors w8a8 V1, =2 its
// V2; auto-crossover measured on-card: MFRAG=2 from M>=192, earlier when the
// grid is wide enough — see the launcher). 2-stage cp.async double buffer for
// A (s8) and the RAW packed B words; the same 2 block barriers per k-step as
// w8a8 — the extra stages ride inside the warp-pair:
//   * the two warps sharing an n-column strip (wm=0/1 of one wn) each unpack
//     16 of the strip's 32 B-rows, then meet on a NAMED barrier (bar.sync
//     1+wn, 64 — the standard producer/consumer idiom) -> the unpack never
//     serializes the whole block, and s_b stays one 8 KB tile (32 KB total
//     smem -> 3 blocks/SM, matching w8a8 V2's occupancy).
//   * the per-group rescale stages each int32 fragment through a per-warp
//     16x16 smem tile (the V2 epilogue idiom). Its (r,c)<->lane map has
//     c = lane&15 CONSTANT per lane, so each lane needs exactly one weight
//     scale per n-fragment per group — served as 2 coalesced loads from the
//     TRANSPOSED scale operand [NG, N] (the caller passes scale.t().contig;
//     the [N, NG] checkpoint layout would make these 128 scattered sectors
//     per k-step, which measurably dominated the kernel).
// sm80+ only (cp.async + s8 wmma) — the Python caller gates on capability.
// Built WITHOUT --use_fast_math. cuda-graph safe (no allocs, current stream).
// ---------------------------------------------------------------------------
constexpr int A8_BN = 128;               // output cols per block
constexpr int A8_BK = 64;                // K staged per iteration == GROUP
constexpr int A8_WORDS = A8_BK / 8;      // packed uint32 words per row per k-tile

// Unpack 8 int4 nibbles (2's complement, little-endian along K) of p into two
// s8x4 words in K order. lo/hi split gives nibbles {0,2,4,6}/{1,3,5,7} in byte
// lanes; __byte_perm re-interleaves; (v^8)-8 per byte sign-extends 4->8 bit.
__device__ __forceinline__ void unpack8_s4_s8(uint32_t p, uint32_t& w0, uint32_t& w1) {
  const uint32_t lo = p & 0x0F0F0F0Fu;
  const uint32_t hi = (p >> 4) & 0x0F0F0F0Fu;
  const uint32_t b0123 = __byte_perm(lo, hi, 0x5140);  // bytes = nibbles 0,1,2,3
  const uint32_t b4567 = __byte_perm(lo, hi, 0x7362);  // bytes = nibbles 4,5,6,7
  w0 = __vsub4(b0123 ^ 0x08080808u, 0x08080808u);
  w1 = __vsub4(b4567 ^ 0x08080808u, 0x08080808u);
}

// Named barrier over one wn-pair (64 threads): the producer/consumer handoff
// between the pair's two unpack halves and their shared s_b strip. bar.sync
// performs the same smem ordering as __syncthreads, scoped to the named group.
__device__ __forceinline__ void bar_sync_pair(int barrier_id) {
  asm volatile("bar.sync %0, 64;" ::"r"(barrier_id));
}

template <int MFRAG>  // wmma m-fragments per warp: 1 -> block M=32, 2 -> block M=64
__global__ __launch_bounds__(256, MFRAG == 1 ? 3 : 2) void gemm_w4a8_tc_kernel(
    const int8_t* __restrict__ x,        // [M,K] s8 (per-token quantized)
    const uint32_t* __restrict__ qw,     // [N,K/8] packed int4
    const __half* __restrict__ scale_t,  // [K/64,N] TRANSPOSED per-(n,group) scale
    const float* __restrict__ xs,        // [M] per-token activation scale
    __half* __restrict__ y,              // [M,N]
    int M, int N, int K) {
#if (!defined(__CUDA_ARCH__)) || (__CUDA_ARCH__ >= 800)
  using namespace nvcuda;
  constexpr int BM = MFRAG * 32;
  const int n0 = blockIdx.x * A8_BN;
  const int m0 = blockIdx.y * BM;
  const int tid = threadIdx.x;   // 0..255
  const int warp = tid >> 5;     // 0..7
  const int lane = tid & 31;
  const int wm = warp >> 2;      // 0..1 -> rows [m0 + wm*MFRAG*16, +MFRAG*16)
  const int wn = warp & 3;       // 0..3 -> cols [n0 + wn*32, +32)
  const int cl = lane & 15;      // this lane's column within a 16-wide fragment
  const int KW = K >> 3;         // packed words per weight row

  __shared__ __align__(16) int8_t s_a[2][BM][A8_BK];
  __shared__ __align__(16) uint32_t s_bq[2][A8_BN][A8_WORDS];
  // One s8 tile; rows [wn*32,+32) are written by the wn-pair (16 rows each) and
  // read only by that pair — the handoff is the pair's named barrier, never a
  // block barrier (2 block barriers per k-step total, same as w8a8).
  __shared__ __align__(16) int8_t s_b[A8_BN][A8_BK];
  __shared__ __align__(16) int32_t s_epi[8][16][16];  // per-warp rescale staging

  const int a_rows_live = (M - m0 < BM) ? (M - m0) : BM;
  const int b_rows_live = (N - n0 < A8_BN) ? (N - n0) : A8_BN;

  // Rows beyond M/N never change across k-steps: zero them once in both buffers
  // (zero words unpack to s8 zeros -> exact zero MMA products; stores are guarded).
  for (int r = a_rows_live; r < BM; ++r)
    for (int c = tid; c < A8_BK; c += 256) { s_a[0][r][c] = 0; s_a[1][r][c] = 0; }
  for (int r = b_rows_live; r < A8_BN; ++r)
    for (int c = tid; c < A8_WORDS; c += 256) { s_bq[0][r][c] = 0; s_bq[1][r][c] = 0; }
  __syncthreads();

  // Stage k-tile [k0s, k0s+64) into buffer `buf`: 16B cp.async per op.
  // A: <=BM rows x 4 chunks; B packed: <=128 rows x 2 chunks (HALF the w8 bytes).
  auto stage = [&](int k0s, int buf) {
    const int a_ops = a_rows_live * 4;
    for (int t = tid; t < a_ops; t += 256) {
      const int r = t >> 2, c = (t & 3) * 16;
      __pipeline_memcpy_async(&s_a[buf][r][c],
                              x + static_cast<int64_t>(m0 + r) * K + k0s + c, 16);
    }
    const int b_ops = b_rows_live * 2;
    for (int t = tid; t < b_ops; t += 256) {
      const int r = t >> 1, c = (t & 1) * 4;  // word offset 0 / 4 within the tile row
      __pipeline_memcpy_async(&s_bq[buf][r][c],
                              qw + static_cast<int64_t>(n0 + r) * KW + (k0s >> 3) + c, 16);
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[MFRAG][2];
  float facc[MFRAG][2][8];
#pragma unroll
  for (int mf = 0; mf < MFRAG; ++mf)
#pragma unroll
    for (int nf = 0; nf < 2; ++nf) {
      wmma::fill_fragment(acc[mf][nf], 0);
#pragma unroll
      for (int e = 0; e < 8; ++e) facc[mf][nf][e] = 0.0f;
    }

  stage(0, 0);
  __pipeline_commit();

  const int nk = K / A8_BK;  // == NG: k-step i IS quant group i
  for (int i = 0; i < nk; ++i) {
    if (i + 1 < nk) {
      stage((i + 1) * A8_BK, (i + 1) & 1);
      __pipeline_commit();
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();  // buf i landed (and buf i+1's restage can't race iter i-1)
    const int buf = i & 1;
    // ---- pair-split unpack: this warp fills 16 of its strip's 32 B-rows ----
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int t = lane + e * 32;                    // 0..127 over the warp
      const int rn = wn * 32 + wm * 16 + (t >> 3);    // this warp's 16 rows
      const int wk = t & 7;
      uint32_t w0, w1;
      unpack8_s4_s8(s_bq[buf][rn][wk], w0, w1);
      uint32_t* dst = reinterpret_cast<uint32_t*>(&s_b[rn][wk << 3]);
      dst[0] = w0;
      dst[1] = w1;
    }
    // This lane's two group scales (column cl of each n-fragment): coalesced
    // 2B loads from the transposed [NG,N] layout, issued early so the L2 round
    // trip hides behind the unpack + mma work; consumed in the rescale below.
    const int gnc = n0 + wn * 32 + cl;
    const float sw0 =
        (gnc < N) ? __half2float(scale_t[static_cast<int64_t>(i) * N + gnc]) : 0.0f;
    const float sw1 = (gnc + 16 < N)
                          ? __half2float(scale_t[static_cast<int64_t>(i) * N + gnc + 16])
                          : 0.0f;
    bar_sync_pair(1 + wn);  // pair handoff: both 16-row halves of the strip ready
#pragma unroll
    for (int kk = 0; kk < A8_BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a[MFRAG];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b[2];
#pragma unroll
      for (int mf = 0; mf < MFRAG; ++mf)
        wmma::load_matrix_sync(a[mf], &s_a[buf][wm * (MFRAG * 16) + mf * 16][kk], A8_BK);
#pragma unroll
      for (int nf = 0; nf < 2; ++nf)
        // s_b holds B as [n][k] rows; a col_major KxN fragment with ldm=A8_BK
        // reads element (k, n_local) at n_local*A8_BK + k — exactly this layout.
        wmma::load_matrix_sync(b[nf], &s_b[wn * 32 + nf * 16][kk], A8_BK);
#pragma unroll
      for (int mf = 0; mf < MFRAG; ++mf)
#pragma unroll
        for (int nf = 0; nf < 2; ++nf)
          wmma::mma_sync(acc[mf][nf], a[mf], b[nf], acc[mf][nf]);
    }
    // ---- per-group rescale: int32 fragment -> per-warp smem -> fp32 chain ----
    // store_matrix_sync's row-major (r,c) map gives c = (lane + e*32) & 15 =
    // lane & 15 = cl (CONSTANT per lane) and r = (lane>>4) + 2e, so facc[e]
    // tracks one fixed output element across all groups and sw0/sw1 are the
    // only scales this lane ever needs. Dead columns fold scale 0 against
    // exact-zero sums. __syncwarp only — s_epi[warp] is warp-private.
#pragma unroll
    for (int mf = 0; mf < MFRAG; ++mf)
#pragma unroll
      for (int nf = 0; nf < 2; ++nf) {
        wmma::store_matrix_sync(&s_epi[warp][0][0], acc[mf][nf], 16,
                                wmma::mem_row_major);
        __syncwarp();
        const float sw = nf ? sw1 : sw0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int r = (lane >> 4) + 2 * e;
          facc[mf][nf][e] = __fadd_rn(
              facc[mf][nf][e],
              __fmul_rn(static_cast<float>(s_epi[warp][r][cl]), sw));
        }
        __syncwarp();
        wmma::fill_fragment(acc[mf][nf], 0);
      }
    __syncthreads();  // all warps done with s_a[buf]/s_bq[buf] before restage
  }

  // ---- epilogue: apply the per-token activation scale, store guarded ----
#pragma unroll
  for (int mf = 0; mf < MFRAG; ++mf)
#pragma unroll
    for (int nf = 0; nf < 2; ++nf)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = (lane >> 4) + 2 * e;
        const int gm = m0 + wm * (MFRAG * 16) + mf * 16 + r;
        const int gn = n0 + wn * 32 + nf * 16 + cl;
        if (gm < M && gn < N)
          y[static_cast<int64_t>(gm) * N + gn] =
              __float2half_rn(__fmul_rn(facc[mf][nf][e], xs[gm]));
      }
#endif  // __CUDA_ARCH__ >= 800
}

// x: s8 [M,K] (per-token quantized); qweight: uint8 [N,K/2]; scale_t: fp16
// [K/GROUP, N] — the TRANSPOSED (contiguous) view of the checkpoint's [N,K/64]
// scale, cached once per layer by the caller (coalesces the per-group scale
// reads); x_scale: fp32 [M]. K%64==0 (caller zero-pads — s8 zeros add exact
// zeros). Any M, N. algo: -1 auto, 0/1 force. Auto (3090 A/B, bench/ab_algo):
// the 64-row tile wins from M>=192 on every shape, and already from M=65 when
// N>=4096 keeps the half-sized grid full; the 32-row tile only holds the
// narrow-N low-M corner (its extra m-blocks fill the grid, and 32-row padding
// wastes less mma on M just past 64). Speed-only knob — identical bits.
at::Tensor gemm_w4a8_tc(at::Tensor x, at::Tensor qweight, at::Tensor scale_t,
                        at::Tensor x_scale, int64_t algo) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kChar && x.dim() == 2 &&
                  x.is_contiguous(),
              "gemm_w4a8_tc: x must be int8 [M,K] contiguous");
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = qweight.size(0);
  const int64_t NG = K / GROUP;
  TORCH_CHECK((K % GROUP) == 0, "gemm_w4a8_tc requires K%64==0");
  TORCH_CHECK(qweight.scalar_type() == at::kByte && qweight.is_contiguous() &&
                  qweight.size(1) == K / 2,
              "gemm_w4a8_tc qweight [N,K/2] mismatch");
  TORCH_CHECK(scale_t.scalar_type() == at::kHalf && scale_t.is_contiguous() &&
                  scale_t.size(0) == NG && scale_t.size(1) == N,
              "gemm_w4a8_tc scale_t [K/64,N] mismatch (pass scale.t().contiguous())");
  TORCH_CHECK(x_scale.scalar_type() == at::kFloat && x_scale.numel() == M,
              "gemm_w4a8_tc x_scale must be fp32 [M]");
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8,
              "gemm_w4a8_tc requires sm80+ (cp.async + s8 wmma)");
  auto xs = x_scale.contiguous();
  auto y = at::empty({M, N}, scale_t.options());
  if (M == 0 || N == 0) return y;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const int8_t*>(x.data_ptr());
  const auto* qp = reinterpret_cast<const uint32_t*>(qweight.data_ptr<uint8_t>());
  const auto* sp = reinterpret_cast<const __half*>(scale_t.data_ptr<dtype>());
  const float* xsp = xs.data_ptr<float>();
  auto* yp = reinterpret_cast<__half*>(y.data_ptr<dtype>());
  const int Mi = static_cast<int>(M), Ni = static_cast<int>(N), Ki = static_cast<int>(K);
  int use = static_cast<int>(algo);
  if (use < 0) use = (Mi >= 192 || Ni >= 4096) ? 1 : 0;
  if (use == 1) {
    dim3 grid((N + A8_BN - 1) / A8_BN, (M + 63) / 64);
    gemm_w4a8_tc_kernel<2><<<grid, 256, 0, stream>>>(xp, qp, sp, xsp, yp, Mi, Ni, Ki);
  } else {
    dim3 grid((N + A8_BN - 1) / A8_BN, (M + 31) / 32);
    gemm_w4a8_tc_kernel<1><<<grid, 256, 0, stream>>>(xp, qp, sp, xsp, yp, Mi, Ni, Ki);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---------------------------------------------------------------------------
// dequant_w4: int4 (qweight,scale) -> fp16 weight [N, K]. Memory-bound; used by the
// M>64 (prefill) path which then calls cuBLAS. At prefill the GEMM is compute-bound
// and the weight read is amortized across many tokens, so w4 only needs to MATCH
// fp16 there while keeping the int4 checkpoint. One fp16 word per thread-iteration.
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
  m.def("gemm_w4_small(Tensor x, Tensor qweight, Tensor scale) -> Tensor");
  m.def("gemm_w4_tc(Tensor x, Tensor qweight, Tensor scale) -> Tensor");
  m.def(
      "gemm_w4a8_tc(Tensor x, Tensor qweight, Tensor scale_t, Tensor x_scale, "
      "int algo) -> Tensor");
  m.def("dequant_w4(Tensor qweight, Tensor scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_w4, CUDA, m) {
  m.impl("gemv_w4_m1", &gemv_w4_m1);
  m.impl("gemm_w4_small", &gemm_w4_small);
  m.impl("gemm_w4_tc", &gemm_w4_tc);
  m.impl("gemm_w4a8_tc", &gemm_w4a8_tc);
  m.impl("dequant_w4", &dequant_w4);
}
