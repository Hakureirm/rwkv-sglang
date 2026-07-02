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
#include <cuda_pipeline.h>
#include <mma.h>
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
// gemm_w8_tc: y[M,N] for 8<M<=64 via TENSOR CORES (wmma m16n16k16, fp32 accum),
// dequantizing the int8 weight tile to fp16 in shared memory per K-step — no fp16
// weight copy ever touches HBM, so weight traffic is 1/2 of a cuBLAS fp16 GEMM.
// Same structure as rwkv7_w4.cu's gemm_w4_tc (K_TILE == GROUP == 64 -> exactly one
// scale per (n, k-tile); one block covers all M rows so the weight tile is
// dequantized once per block; deterministic split-K: f32 partials + fixed-order
// reduce, no atomics). Only the unpack differs: 4 int8 per uint32 (16 words per
// row per k-tile) instead of 8 int4.
// ---------------------------------------------------------------------------
constexpr int TC_M = 16;    // rows per block tile (wmma m)
constexpr int TC_N = 64;    // output cols per block tile (4 warps x 16)
constexpr int TC_K = 64;    // k-step == GROUP
constexpr int TC_KPAD = 8;  // smem padding halfs to dodge bank conflicts

template <int MT, bool WritePartial>
__global__ __launch_bounds__(128, 1) void gemm_w8_tc_kernel(
    int M, int K, int N, int NG, int k_chunk,
    const __half* __restrict__ x,       // [M, K]
    const uint32_t* __restrict__ qw,    // [N, K/4]
    const __half* __restrict__ scale,   // [N, NG]
    __half* __restrict__ y,             // [M, N]        (WritePartial=false)
    float* __restrict__ ws) {           // [Z, M, N] f32 (WritePartial=true; split-K partials)
#if __CUDA_ARCH__ >= 800
  // sm80+ 2-stage cp.async pipeline (the w8 sibling of gemm_w4_tc's): tile t+1's
  // activation rows and RAW int8 words (16 uint32/row/k-tile) stream in while
  // tile t is dequanted + MMA'd. Same k order + wmma structure as the
  // synchronous path (identical accumulation order; split-K unchanged).
  using namespace nvcuda;
  const int n0 = blockIdx.x * TC_N;
  const int kb = blockIdx.z * k_chunk;
  const int ke = (kb + k_chunk < K) ? kb + k_chunk : K;
  const int KW = K >> 2;
  const int lane = threadIdx.x;        // 0..127
  const int warp = lane >> 5;          // 0..3 -> n-subtile [n0+warp*16, +16)

  __shared__ __align__(16) __half smem_a[2][MT * TC_M][TC_K + TC_KPAD];
  __shared__ __align__(16) uint32_t smem_q[2][TC_N][16];  // raw words: 16/row/k-tile
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
    for (int t = lane; t < TC_N * 4; t += 128) {  // 16 words = 4 x 16B per row
      const int wn = t >> 2;
      const int wc = (t & 3) * 4;
      __pipeline_memcpy_async(&smem_q[buf][wn][wc],
                              qw + static_cast<int64_t>(n0 + wn) * KW + (k0s >> 2) + wc,
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
      for (int r = 0; r < 8; ++r) {
        const int t = lane + r * 128;
        const int wn = t >> 4;
        const int wk = t & 15;
        const uint32_t p = smem_q[cur][wn][wk];
        const float s = __half2float(scale[static_cast<int64_t>(n0 + wn) * NG + g]);
        __half* dst = &smem_w[wn][wk << 2];
        dst[0] = __float2half_rn((float)(int)(int8_t)(p & 0xFF) * s);
        dst[1] = __float2half_rn((float)(int)(int8_t)((p >> 8) & 0xFF) * s);
        dst[2] = __float2half_rn((float)(int)(int8_t)((p >> 16) & 0xFF) * s);
        dst[3] = __float2half_rn((float)(int)(int8_t)((p >> 24) & 0xFF) * s);
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
  const int KW = K >> 2;
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
    // ---- stage + dequant W tile (64 x 64) ONCE: 64 rows x 16 words = 1024 words ----
    {
      const int g = k0 / TC_K;                     // group index for this k-tile
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int t = lane + r * 128;              // 0..1023
        const int wn = t >> 4;                     // 0..63
        const int wk = t & 15;                     // word within row's k-tile
        const uint32_t p = qw[static_cast<int64_t>(n0 + wn) * KW + (k0 >> 2) + wk];
        const float s = __half2float(scale[static_cast<int64_t>(n0 + wn) * NG + g]);
        __half* dst = &smem_w[wn][wk << 2];
        dst[0] = __float2half_rn((float)(int)(int8_t)(p & 0xFF) * s);
        dst[1] = __float2half_rn((float)(int)(int8_t)((p >> 8) & 0xFF) * s);
        dst[2] = __float2half_rn((float)(int)(int8_t)((p >> 16) & 0xFF) * s);
        dst[3] = __float2half_rn((float)(int)(int8_t)((p >> 24) & 0xFF) * s);
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
__global__ void splitk_reduce_w8_kernel(int64_t MN, int Z,
                                        const float* __restrict__ ws,
                                        __half* __restrict__ y) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= MN) return;
  float acc = 0.0f;
  for (int z = 0; z < Z; ++z) acc += ws[static_cast<int64_t>(z) * MN + i];
  y[i] = __float2half_rn(acc);
}

// x: [M,K] fp16 (1<=M<=64);  qweight: int8 [N,K];  scale: fp16 [N,K/GROUP]. N%64==0, K%64==0.
at::Tensor gemm_w8_tc(at::Tensor x, at::Tensor qweight, at::Tensor scale) {
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = qweight.size(0);
  const int64_t NG = K / GROUP;
  TORCH_CHECK(M >= 1 && M <= 64, "gemm_w8_tc requires 1<=M<=64");
  TORCH_CHECK((K % TC_K) == 0, "gemm_w8_tc requires K%64==0");
  TORCH_CHECK((N % TC_N) == 0, "gemm_w8_tc requires N%64==0");
  TORCH_CHECK(qweight.size(1) == K, "gemm_w8_tc qweight [N,K] mismatch");
  TORCH_CHECK(scale.size(0) == N && scale.size(1) == NG, "gemm_w8_tc scale [N,K/64] mismatch");
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
  const auto* qp = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int8_t>());
  const auto* sp = reinterpret_cast<const __half*>(scale.data_ptr<dtype>());
  auto* yp = reinterpret_cast<__half*>(y.data_ptr<dtype>());
  const int mt = static_cast<int>((M + TC_M - 1) / TC_M);  // 1..4 m-subtiles, one block covers all M
  if (splits == 1) {
    dim3 grid(N / TC_N, 1, 1);
    const int kc = static_cast<int>(K);
    switch (mt) {
      case 1: gemm_w8_tc_kernel<1, false><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, yp, nullptr); break;
      case 2: gemm_w8_tc_kernel<2, false><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, yp, nullptr); break;
      case 3: gemm_w8_tc_kernel<3, false><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, yp, nullptr); break;
      default: gemm_w8_tc_kernel<4, false><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, yp, nullptr); break;
    }
  } else {
    // k_chunk: multiple of TC_K covering K in `splits` pieces
    int64_t k_chunk = ((K + splits - 1) / splits + TC_K - 1) / TC_K * TC_K;
    auto ws = at::empty({splits, M, N}, x.options().dtype(at::kFloat));
    float* wp = ws.data_ptr<float>();
    dim3 grid(N / TC_N, 1, splits);
    const int kc = static_cast<int>(k_chunk);
    switch (mt) {
      case 1: gemm_w8_tc_kernel<1, true><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, nullptr, wp); break;
      case 2: gemm_w8_tc_kernel<2, true><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, nullptr, wp); break;
      case 3: gemm_w8_tc_kernel<3, true><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, nullptr, wp); break;
      default: gemm_w8_tc_kernel<4, true><<<grid, 128, 0, stream>>>(M, K, N, NG, kc, xp, qp, sp, nullptr, wp); break;
    }
    const int64_t MN = M * N;
    const int threads = 256;
    const int64_t blocks = (MN + threads - 1) / threads;
    splitk_reduce_w8_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
        MN, splits, ws.data_ptr<float>(), yp);
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
  m.def("gemm_w8_tc(Tensor x, Tensor qweight, Tensor scale) -> Tensor");
  m.def("dequant_w8(Tensor qweight, Tensor scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_w8, CUDA, m) {
  m.impl("gemv_w8_m1", &w8::gemv_w8_m1);
  m.impl("gemm_w8_small", &w8::gemm_w8_small);
  m.impl("gemm_w8_tc", &w8::gemm_w8_tc);
  m.impl("dequant_w8", &w8::dequant_w8);
}
