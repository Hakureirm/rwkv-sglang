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

#include "rwkv7_pdl.cuh"  // PDL chain (task #50 sm120 step); no-op unarmed

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
  // PDL: x comes from the stream predecessor; wait for its stores (no-op when
  // launched plain / below sm_90 — see rwkv7_pdl.cuh).
  rwkv7_pdl_wait();
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
  rwkv7_pdl_launch_dependents();  // let the next PDL stage schedule early
}

at::Tensor gemv_m1(at::Tensor x, at::Tensor weight) {
  const int64_t K = x.size(-1);
  const int64_t N = weight.size(0);
  TORCH_CHECK(x.numel() == K, "gemv_m1 requires M==1");
  TORCH_CHECK(weight.size(1) == K, "gemv_m1 weight [N,K] mismatch");
  TORCH_CHECK((K % 4) == 0, "gemv_m1 requires K%4==0");
  auto y = at::empty({1, N}, x.options());
  if (N == 0) return y;
  if (K == 0) return y.zero_();  // empty reduction = 0, not uninitialized memory
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool pdl = rwkv7_pdl_enabled("fast");
  if ((N % 2) == 0) {
    rwkv7_launch_maybe_pdl(pdl, gemv_m1_kernel<128, 2>,
        dim3(static_cast<unsigned>(N / 2)), dim3(128), 0, stream.stream(),
        static_cast<int>(K), static_cast<int>(N), x.data_ptr<dtype>(),
        weight.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    rwkv7_launch_maybe_pdl(pdl, gemv_m1_kernel<128, 1>,
        dim3(static_cast<unsigned>(N)), dim3(128), 0, stream.stream(),
        static_cast<int>(K), static_cast<int>(N), x.data_ptr<dtype>(),
        weight.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---------------------------------------------------------------------------
// gemv_m1_cfg: same kernel, but (Threads, OutTile) chosen by the Python
// arch-aware autotuner (fast_linear.py::_select_config) instead of the fixed
// <128,2>/<128,1> in gemv_m1. Candidate grid: Threads in {64,128,256},
// OutTile in {1,2,4}. Occupancy of these is compile-time (regs/smem), so the
// autotuner keys purely on (sm_arch, N, K). Requires N % OutTile == 0, K%4==0
// (guaranteed by the caller). cuda-graph safe (static shapes, no host sync).
// ---------------------------------------------------------------------------
#define RWKV7_GEMV_LAUNCH(T, OT)                                             \
  rwkv7_launch_maybe_pdl(pdl, gemv_m1_kernel<T, OT>,                         \
      dim3(static_cast<unsigned>(N / (OT))), dim3(T), 0, stream.stream(),    \
      static_cast<int>(K), static_cast<int>(N), x.data_ptr<dtype>(),        \
      weight.data_ptr<dtype>(), y.data_ptr<dtype>())

at::Tensor gemv_m1_cfg(at::Tensor x, at::Tensor weight, int64_t threads,
                       int64_t out_tile) {
  const int64_t K = x.size(-1);
  const int64_t N = weight.size(0);
  TORCH_CHECK(x.numel() == K, "gemv_m1_cfg requires M==1");
  TORCH_CHECK(weight.size(1) == K, "gemv_m1_cfg weight [N,K] mismatch");
  TORCH_CHECK((K % 4) == 0, "gemv_m1_cfg requires K%4==0");
  TORCH_CHECK((N % out_tile) == 0, "gemv_m1_cfg requires N % out_tile == 0");
  auto y = at::empty({1, N}, x.options());
  if (N == 0) return y;
  if (K == 0) return y.zero_();  // empty reduction = 0, not uninitialized memory
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool pdl = rwkv7_pdl_enabled("fast");
  // key = threads*100 + out_tile: out_tile < 100, so no (threads,out_tile)
  // aliasing (threads*10+out_tile collided, e.g. (63,11) -> (64,1)).
  switch (threads * 100 + out_tile) {
    case 64 * 100 + 1:  RWKV7_GEMV_LAUNCH(64, 1);  break;
    case 64 * 100 + 2:  RWKV7_GEMV_LAUNCH(64, 2);  break;
    case 64 * 100 + 4:  RWKV7_GEMV_LAUNCH(64, 4);  break;
    case 128 * 100 + 1: RWKV7_GEMV_LAUNCH(128, 1); break;
    case 128 * 100 + 2: RWKV7_GEMV_LAUNCH(128, 2); break;
    case 128 * 100 + 4: RWKV7_GEMV_LAUNCH(128, 4); break;
    case 256 * 100 + 1: RWKV7_GEMV_LAUNCH(256, 1); break;
    case 256 * 100 + 2: RWKV7_GEMV_LAUNCH(256, 2); break;
    case 256 * 100 + 4: RWKV7_GEMV_LAUNCH(256, 4); break;
    default: TORCH_CHECK(false, "gemv_m1_cfg unsupported (threads,out_tile)=(",
                         threads, ",", out_tile, "); use {64,128,256}x{1,2,4}");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
#undef RWKV7_GEMV_LAUNCH

// ---------------------------------------------------------------------------
// gemv_m1_sqrelu_cfg: EPILOGUE-FUSED GEMV. Identical fp32-accumulate GEMV as
// gemv_m1_kernel, but the store folds the FFN channel-mix activation
//   act = relu(k)^2      (== torch.relu(k) ** 2, the ffn.key epilogue)
// into the same kernel, so the intermediate k[1,N] never round-trips to HBM and
// the standalone relu + pow launches (2 tiny elementwise kernels/layer) vanish.
// This is the "epilogue-fuse INTO the GEMV" lever from F0051 §5.1 — Albatross's
// technique — applied to the one pure-elementwise op that directly follows a GEMV
// output in the model (models/rwkv7.py: FFN `act = torch.relu(key(xk)) ** 2`).
//
// BYTE-EXACTNESS (gated by bench/test_sqrelu_gate.py BEFORE the model enables it):
// the accumulation is the SAME code as gemv_m1_kernel, so `sum` is bit-identical
// to the plain path with the SAME (Threads, OutTile). We then reproduce torch's
// EXACT two-step rounding: round sum -> fp16 `k` (what the plain GEMV stores and
// torch.relu reads), take relu on that fp16 value, and square in fp32 opmath —
// because aten's pow(x, 2) special-case computes `b*b` with b = (float)base, then
// rounds once to fp16. So act == fp16( f * f ), f = (float)relu(fp16(sum)). No
// transcendental, no fast-math (the JIT build omits --use_fast_math), so this is
// bit-exact, not merely close. relu uses `f > 0 ? f : 0`; the square erases the
// sign-of-zero, so this matches torch for every finite input (the gate sweeps
// saturating + knife-edge finite values). N%OutTile==0, K%4==0 (caller-guarded).
// cuda-graph safe (static shapes, current stream, no host sync).
// ---------------------------------------------------------------------------
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void gemv_m1_sqrelu_kernel(
    int K, int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,   // [N, K]
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  rwkv7_pdl_wait();  // x from the ffn shift_lerp1 predecessor (no-op unarmed)
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
      // sqrelu epilogue, bit-exact with torch.relu(k)**2 where k=fp16(sum):
      //   k  = round(sum -> fp16)         (== the plain gemv store & torch's relu input)
      //   f  = (float)k                   (fp16->fp32 is lossless)
      //   r  = relu(f) = f > 0 ? f : 0    (sign-of-zero erased by the square)
      //   act= round(r*r -> fp16)         (aten pow(.,2): b*b in fp32 opmath, one round)
      const float f = __half2float(__float2half_rn(sum));
      const float r = f > 0.0f ? f : 0.0f;
      y[n0 + j] = __float2half_rn(r * r);
    }
  }
  rwkv7_pdl_launch_dependents();  // sparse_cmix / ffn.value schedules early
}

#define RWKV7_SQRELU_LAUNCH(T, OT)                                            \
  rwkv7_launch_maybe_pdl(pdl, gemv_m1_sqrelu_kernel<T, OT>,                   \
      dim3(static_cast<unsigned>(N / (OT))), dim3(T), 0, stream.stream(),     \
      static_cast<int>(K), static_cast<int>(N), x.data_ptr<dtype>(),          \
      weight.data_ptr<dtype>(), y.data_ptr<dtype>())

at::Tensor gemv_m1_sqrelu_cfg(at::Tensor x, at::Tensor weight, int64_t threads,
                              int64_t out_tile) {
  const int64_t K = x.size(-1);
  const int64_t N = weight.size(0);
  TORCH_CHECK(x.numel() == K, "gemv_m1_sqrelu_cfg requires M==1");
  TORCH_CHECK(weight.size(1) == K, "gemv_m1_sqrelu_cfg weight [N,K] mismatch");
  TORCH_CHECK((K % 4) == 0, "gemv_m1_sqrelu_cfg requires K%4==0");
  TORCH_CHECK((N % out_tile) == 0, "gemv_m1_sqrelu_cfg requires N % out_tile == 0");
  auto y = at::empty({1, N}, x.options());
  if (N == 0) return y;
  if (K == 0) return y.zero_();  // empty reduction -> relu(0)^2 == 0
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool pdl = rwkv7_pdl_enabled("fast");
  switch (threads * 100 + out_tile) {
    case 64 * 100 + 1:  RWKV7_SQRELU_LAUNCH(64, 1);  break;
    case 64 * 100 + 2:  RWKV7_SQRELU_LAUNCH(64, 2);  break;
    case 64 * 100 + 4:  RWKV7_SQRELU_LAUNCH(64, 4);  break;
    case 128 * 100 + 1: RWKV7_SQRELU_LAUNCH(128, 1); break;
    case 128 * 100 + 2: RWKV7_SQRELU_LAUNCH(128, 2); break;
    case 128 * 100 + 4: RWKV7_SQRELU_LAUNCH(128, 4); break;
    case 256 * 100 + 1: RWKV7_SQRELU_LAUNCH(256, 1); break;
    case 256 * 100 + 2: RWKV7_SQRELU_LAUNCH(256, 2); break;
    case 256 * 100 + 4: RWKV7_SQRELU_LAUNCH(256, 4); break;
    default: TORCH_CHECK(false, "gemv_m1_sqrelu_cfg unsupported (threads,out_tile)=(",
                         threads, ",", out_tile, "); use {64,128,256}x{1,2,4}");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
#undef RWKV7_SQRELU_LAUNCH

// ---------------------------------------------------------------------------
// gemv_mb_cfg: batch-invariant M-row GEMV. Each row m of y[M,N] is computed by
// the EXACT same per-output fp32 reduction as gemv_m1_kernel (one row per
// blockIdx.y), so with the SAME (threads, out_tile) the decode path picks,
// y[m] is BIT-IDENTICAL to gemv_m1(x[m]). Purpose: the chain-spec verify runs
// the target over K positions in one launch while staying bit-exact against the
// M=1 baseline decode — closing the F0031 gate flip (M=K cuBLAS GEMM reduction
// order) without giving up the one-forward-per-round structure. cuda-graph safe.
// ---------------------------------------------------------------------------
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void gemv_mb_kernel(
    int K, int N, int M,
    const dtype* __restrict__ x,        // [M, K]
    const dtype* __restrict__ weight,   // [N, K]
    dtype* __restrict__ y) {            // [M, N]
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const dtype* xm = x + static_cast<int64_t>(m) * K;
  dtype* ym = y + static_cast<int64_t>(m) * N;
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) acc[j] = 0.0f;
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(xm + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(xm + k + 2));
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
      ym[n0 + j] = __float2half_rn(sum);
    }
  }
}

#define RWKV7_GEMB_LAUNCH(T, OT)                                              \
  gemv_mb_kernel<T, OT><<<dim3(static_cast<int>(N) / (OT), static_cast<int>(M)), \
                          (T), 0, stream>>>(                                  \
      static_cast<int>(K), static_cast<int>(N), static_cast<int>(M),         \
      x.data_ptr<dtype>(), weight.data_ptr<dtype>(), y.data_ptr<dtype>())

at::Tensor gemv_mb_cfg(at::Tensor x, at::Tensor weight, int64_t threads,
                       int64_t out_tile) {
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = weight.size(0);
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "gemv_mb_cfg x must be [M,K] contiguous");
  TORCH_CHECK(weight.size(1) == K, "gemv_mb_cfg weight [N,K] mismatch");
  TORCH_CHECK((K % 4) == 0, "gemv_mb_cfg requires K%4==0");
  TORCH_CHECK((N % out_tile) == 0, "gemv_mb_cfg requires N % out_tile == 0");
  auto y = at::empty({M, N}, x.options());
  if (N == 0 || M == 0) return y;
  if (K == 0) return y.zero_();
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (threads * 100 + out_tile) {
    case 64 * 100 + 1:  RWKV7_GEMB_LAUNCH(64, 1);  break;
    case 64 * 100 + 2:  RWKV7_GEMB_LAUNCH(64, 2);  break;
    case 64 * 100 + 4:  RWKV7_GEMB_LAUNCH(64, 4);  break;
    case 128 * 100 + 1: RWKV7_GEMB_LAUNCH(128, 1); break;
    case 128 * 100 + 2: RWKV7_GEMB_LAUNCH(128, 2); break;
    case 128 * 100 + 4: RWKV7_GEMB_LAUNCH(128, 4); break;
    case 256 * 100 + 1: RWKV7_GEMB_LAUNCH(256, 1); break;
    case 256 * 100 + 2: RWKV7_GEMB_LAUNCH(256, 2); break;
    case 256 * 100 + 4: RWKV7_GEMB_LAUNCH(256, 4); break;
    default: TORCH_CHECK(false, "gemv_mb_cfg unsupported (threads,out_tile)=(",
                         threads, ",", out_tile, "); use {64,128,256}x{1,2,4}");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
#undef RWKV7_GEMB_LAUNCH

TORCH_LIBRARY(rwkv7_fast, m) {
  m.def("gemv_m1(Tensor x, Tensor weight) -> Tensor");
  m.def("gemv_m1_cfg(Tensor x, Tensor weight, int threads, int out_tile) -> Tensor");
  m.def("gemv_m1_sqrelu_cfg(Tensor x, Tensor weight, int threads, int out_tile) -> Tensor");
  m.def("gemv_mb_cfg(Tensor x, Tensor weight, int threads, int out_tile) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_fast, CUDA, m) {
  m.impl("gemv_m1", &gemv_m1);
  m.impl("gemv_m1_cfg", &gemv_m1_cfg);
  m.impl("gemv_m1_sqrelu_cfg", &gemv_m1_sqrelu_cfg);
  m.impl("gemv_mb_cfg", &gemv_mb_cfg);
}
