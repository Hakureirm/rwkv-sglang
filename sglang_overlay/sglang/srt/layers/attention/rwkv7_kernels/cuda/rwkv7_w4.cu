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
  m.def("dequant_w4(Tensor qweight, Tensor scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_w4, CUDA, m) {
  m.impl("gemv_w4_m1", &gemv_w4_m1);
  m.impl("gemm_w4_small", &gemm_w4_small);
  m.impl("gemm_w4_tc", &gemm_w4_tc);
  m.impl("dequant_w4", &dequant_w4);
}
