// RWKV-7 x sglang fused 4-chain LoRA for bsz1 fp16 decode (M9).
//
// Replaces the per-layer cluster of ~12 tiny launches (4 x [down-GEMV + act +
// up-GEMV(+bias)]) with TWO kernels behind one custom op:
//
//   stage1: one block per row of the row-stacked down matrix d_cat[R_total, H]:
//           h[r] = act( dot_fp32(d_cat[r,:], xs[chain_of(r),:]) )
//   stage2: one warp per output element (c, n):
//           y[c,n] = bias_cat[c,n] + dot_fp32(u_cat[n, roff:roff+rank], h[roff:...])
//
// PACKED LAYOUTS (chosen for coalescing; the Python packer builds them):
//   xs       fp16 [C, H]        lerped chain inputs (xw, xa, xg[, xv])
//   d_cat    fp16 [R_total, H]  down weights row-stacked (nn.Linear [rank,H] rows)
//   u_cat    fp16 [H, R_total]  up weights column-stacked == torch.cat(up.weight, dim=1);
//                               stage2's warp reads u_cat[n, roff+lane...] -> the rank dim
//                               is innermost/contiguous, so lanes issue coalesced
//                               (__half2 when 4B-aligned) loads.
//   bias_cat fp16 [C, H]        zeros where a chain has no bias (g)
//   meta     int32 [C, 3]       (rank_offset, rank, act_code); act 0=id 1=tanh 2=sigmoid
//
// GREEDY-EXACTNESS: fp32 accumulation, IEEE (no --use_fast_math), deterministic
// reduction order. The torch reference chain rounds each stage to fp16
// (F.linear -> fp16, act -> fp16, F.linear+bias -> fp16); we reproduce those
// intermediate fp16 roundings exactly — the scratch h holds fp32 values that are
// EXACTLY fp16-rounded post-activation outputs — so the only residual difference
// vs torch is dot-product reduction order (~1 fp16 ULP, same class as gemv_m1,
// which already holds the greedy gate). Activations use libdevice tanhf /
// 1/(1+expf(-x)), the same opmath-float functions torch's fp16 tanh/sigmoid use.
//
// cuda-graph safe: static shapes, current stream, no host sync (meta is read on
// device; grid dims come from tensor SIZES only).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>   // TORCH_LIBRARY / TORCH_LIBRARY_IMPL
#include <cuda_fp16.h>

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
// stage1: h[r] = act(dot(d_cat[r,:], xs[chain_of(r),:])), fp32 accumulate.
// One block (Threads) per down-row; same 4-halves-per-thread pattern as gemv_m1.
// H%4==0 required (guarded host-side).
// ---------------------------------------------------------------------------
template <int Threads>
__global__ __launch_bounds__(Threads, 1) void lora_stage1_kernel(
    int H, int C,
    const dtype* __restrict__ xs,     // [C, H]
    const dtype* __restrict__ d_cat,  // [R_total, H]
    const int* __restrict__ meta,     // [C, 3]
    float* __restrict__ h) {          // [R_total] scratch
  const int r = blockIdx.x;
  int chain = 0, act = 0;
  for (int c = 0; c < C; ++c) {  // C <= 4: trivial linear scan
    const int roff = meta[c * 3];
    const int rank = meta[c * 3 + 1];
    if (r >= roff && r < roff + rank) {
      chain = c;
      act = meta[c * 3 + 2];
    }
  }
  const dtype* x = xs + static_cast<int64_t>(chain) * H;
  const dtype* w = d_cat + static_cast<int64_t>(r) * H;
  rwkv7_pdl_wait();  // xs from shift_lerp6 predecessor (no-op unarmed)
  // F0065: LATENCY-bound at decode (grid = Rtot ~ few hundred blocks, 8 serial
  // load rounds/thread; measured 6.4us vs ~1.8us byte floor = ~28% of
  // achievable — the opposite regime from F0064's BW-wall GEMVs). So the
  // F0064 V1+V2 load treatment IS the medicine here: one 64-bit int2 load per
  // 4-half chunk + K-unroll x2 with hoisted loads (2x bytes in flight). The
  // per-acc FMA sequence is unchanged (chunk k fully, then chunk k+kstride
  // fully — exactly what the serial loop produced) => byte-identical output;
  // the existing byte gates apply as-is.
  float acc = 0.0f;
  const int kstride = Threads << 2;
  int k = threadIdx.x << 2;
  for (; k + kstride < H; k += kstride << 1) {
    const int2 xp0 = *reinterpret_cast<const int2*>(x + k);
    const int2 wp0 = *reinterpret_cast<const int2*>(w + k);
    const int2 xp1 = *reinterpret_cast<const int2*>(x + k + kstride);
    const int2 wp1 = *reinterpret_cast<const int2*>(w + k + kstride);
    const float2 xa0 = __half22float2(*reinterpret_cast<const __half2*>(&xp0.x));
    const float2 xb0 = __half22float2(*reinterpret_cast<const __half2*>(&xp0.y));
    const float2 wa0 = __half22float2(*reinterpret_cast<const __half2*>(&wp0.x));
    const float2 wb0 = __half22float2(*reinterpret_cast<const __half2*>(&wp0.y));
    const float2 xa1 = __half22float2(*reinterpret_cast<const __half2*>(&xp1.x));
    const float2 xb1 = __half22float2(*reinterpret_cast<const __half2*>(&xp1.y));
    const float2 wa1 = __half22float2(*reinterpret_cast<const __half2*>(&wp1.x));
    const float2 wb1 = __half22float2(*reinterpret_cast<const __half2*>(&wp1.y));
    acc = fmaf(xa0.x, wa0.x, acc);
    acc = fmaf(xa0.y, wa0.y, acc);
    acc = fmaf(xb0.x, wb0.x, acc);
    acc = fmaf(xb0.y, wb0.y, acc);
    acc = fmaf(xa1.x, wa1.x, acc);
    acc = fmaf(xa1.y, wa1.y, acc);
    acc = fmaf(xb1.x, wb1.x, acc);
    acc = fmaf(xb1.y, wb1.y, acc);
  }
  for (; k < H; k += kstride) {  // tail: 0 or 1 remaining chunk
    const int2 xp = *reinterpret_cast<const int2*>(x + k);
    const int2 wp = *reinterpret_cast<const int2*>(w + k);
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(&xp.x));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(&xp.y));
    const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(&wp.x));
    const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(&wp.y));
    acc = fmaf(x0.x, w0.x, acc);
    acc = fmaf(x0.y, w0.y, acc);
    acc = fmaf(x1.x, w1.x, acc);
    acc = fmaf(x1.y, w1.y, acc);
  }
  __shared__ float partial[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const float v = warp_sum(acc);
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < Threads / 32; ++i) sum += partial[i];
    // Reproduce the torch chain's fp16 intermediates exactly: the down-GEMM
    // output is rounded to fp16 BEFORE the activation, and the activation
    // output is stored as fp16 (kept here as its exact fp32 image).
    float t = __half2float(__float2half_rn(sum));
    if (act == 1) t = tanhf(t);
    else if (act == 2) t = 1.0f / (1.0f + expf(-t));
    h[r] = __half2float(__float2half_rn(t));
  }
  rwkv7_pdl_launch_dependents();  // stage2 schedules early (PDL, no-op unarmed)
}

// ---------------------------------------------------------------------------
// stage2: y[c,n] = bias_cat[c,n] + dot(u_cat[n, roff:roff+rank], h[roff:...]).
// One warp per output element; h (R_total fp32, <=2KB) staged in smem; lanes
// read the contiguous rank segment of u_cat's row n (coalesced, __half2 when
// the segment is 4B-aligned). Bias added AFTER the accumulated sum (cuBLAS
// epilogue order). Deterministic per-warp shuffle reduction.
// ---------------------------------------------------------------------------
template <int Warps>
__global__ __launch_bounds__(Warps * 32, 1) void lora_stage2_kernel(
    int H, int C, int Rtot,
    const dtype* __restrict__ u_cat,     // [H, R_total]
    const dtype* __restrict__ bias_cat,  // [C, H]
    const int* __restrict__ meta,        // [C, 3]
    const float* __restrict__ h,         // [R_total]
    dtype* __restrict__ y) {             // [C, H]
  extern __shared__ float hs[];  // R_total floats
  rwkv7_pdl_wait();  // h is stage1's output; wait before staging (no-op unarmed)
  for (int i = threadIdx.x; i < Rtot; i += Warps * 32) hs[i] = h[i];
  __syncthreads();
  const int gw = blockIdx.x * Warps + (threadIdx.x >> 5);
  if (gw >= C * H) return;
  const int c = gw / H;
  const int n = gw - c * H;
  const int roff = meta[c * 3];
  const int rank = meta[c * 3 + 1];
  const dtype* u = u_cat + static_cast<int64_t>(n) * Rtot + roff;
  const int lane = threadIdx.x & 31;
  float acc = 0.0f;
  if (((Rtot | roff | rank) & 1) == 0) {
    // 4B-aligned segment (Rtot, roff, rank all even): vectorized __half2 loads.
    for (int r = lane << 1; r < rank; r += 64) {
      const float2 uv = __half22float2(*reinterpret_cast<const __half2*>(u + r));
      acc = fmaf(uv.x, hs[roff + r], acc);
      acc = fmaf(uv.y, hs[roff + r + 1], acc);
    }
  } else {
    for (int r = lane; r < rank; r += 32) {
      acc = fmaf(__half2float(u[r]), hs[roff + r], acc);
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    y[gw] = __float2half_rn(acc + __half2float(bias_cat[gw]));
  }
  rwkv7_pdl_launch_dependents();  // next stage schedules early (no-op unarmed)
}

at::Tensor lora4_m1(at::Tensor xs, at::Tensor d_cat, at::Tensor u_cat,
                    at::Tensor bias_cat, at::Tensor meta) {
  const int64_t C = xs.size(0);
  const int64_t H = xs.size(1);
  const int64_t Rtot = d_cat.size(0);
  TORCH_CHECK(xs.is_cuda() && xs.scalar_type() == at::kHalf, "lora4_m1: xs must be CUDA fp16");
  TORCH_CHECK(d_cat.scalar_type() == at::kHalf && u_cat.scalar_type() == at::kHalf &&
              bias_cat.scalar_type() == at::kHalf, "lora4_m1: weights/bias must be fp16");
  TORCH_CHECK(meta.scalar_type() == at::kInt && meta.dim() == 2 &&
              meta.size(0) == C && meta.size(1) == 3 && meta.is_cuda(),
              "lora4_m1: meta must be CUDA int32 [C,3]");
  TORCH_CHECK(d_cat.size(1) == H, "lora4_m1: d_cat [R_total,H] mismatch");
  TORCH_CHECK(u_cat.size(0) == H && u_cat.size(1) == Rtot, "lora4_m1: u_cat [H,R_total] mismatch");
  TORCH_CHECK(bias_cat.size(0) == C && bias_cat.size(1) == H, "lora4_m1: bias_cat [C,H] mismatch");
  TORCH_CHECK(xs.is_contiguous() && d_cat.is_contiguous() && u_cat.is_contiguous() &&
              bias_cat.is_contiguous() && meta.is_contiguous(), "lora4_m1: inputs must be contiguous");
  TORCH_CHECK((H % 4) == 0, "lora4_m1 requires H%4==0");
  TORCH_CHECK(C >= 1 && C <= 8, "lora4_m1: 1<=C<=8");
  TORCH_CHECK(Rtot >= 1, "lora4_m1: empty rank total");
  auto h = at::empty({Rtot}, xs.options().dtype(at::kFloat));
  auto y = at::empty({C, H}, xs.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool pdl = rwkv7_pdl_enabled("lora");
  constexpr int kThreads = 128;
  rwkv7_launch_maybe_pdl(pdl, lora_stage1_kernel<kThreads>,
      dim3(static_cast<unsigned>(Rtot)), dim3(kThreads), 0, stream.stream(),
      static_cast<int>(H), static_cast<int>(C),
      xs.data_ptr<dtype>(), d_cat.data_ptr<dtype>(),
      static_cast<const int*>(meta.data_ptr<int>()), h.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  constexpr int kWarps = 8;
  const int64_t total = C * H;
  const int64_t blocks = (total + kWarps - 1) / kWarps;
  const size_t smem = static_cast<size_t>(Rtot) * sizeof(float);
  rwkv7_launch_maybe_pdl(pdl, lora_stage2_kernel<kWarps>,
      dim3(static_cast<unsigned>(blocks)), dim3(kWarps * 32), smem,
      stream.stream(),
      static_cast<int>(H), static_cast<int>(C), static_cast<int>(Rtot),
      u_cat.data_ptr<dtype>(), bias_cat.data_ptr<dtype>(),
      static_cast<const int*>(meta.data_ptr<int>()),
      static_cast<const float*>(h.data_ptr<float>()), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ===========================================================================
// Batched-M variant (F0023 #3 / ADR-0005 R3): xs[M,C,H] -> y[M,C,H], so the
// 2-launch fusion covers M>1 decode instead of falling back to per-chain
// ReplicatedLinear (the one place LoRA ran against albatross, which fuses M<=8).
// Per-(m) reduction order is byte-identical to lora4_m1 (same fp16 intermediate
// roundings), so lora4_mn at M==1 must equal lora4_m1 token-for-token.
// h scratch is [M, R_total]; stage2 reads h from global (no smem staging, since
// warps in a block may span different m) — correctness first; smem-per-m tiling
// is a later optimization. cuda-graph safe (grid from sizes only, no host sync).
// ===========================================================================
template <int Threads>
__global__ __launch_bounds__(Threads, 1) void lora_stage1_mn_kernel(
    int H, int C, int Rtot,
    const dtype* __restrict__ xs,     // [M, C, H]
    const dtype* __restrict__ d_cat,  // [R_total, H]
    const int* __restrict__ meta,     // [C, 3]
    float* __restrict__ h) {          // [M, R_total]
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  int chain = 0, act = 0;
  for (int c = 0; c < C; ++c) {
    const int roff = meta[c * 3];
    const int rank = meta[c * 3 + 1];
    if (r >= roff && r < roff + rank) { chain = c; act = meta[c * 3 + 2]; }
  }
  const dtype* x = xs + (static_cast<int64_t>(m) * C + chain) * H;
  const dtype* w = d_cat + static_cast<int64_t>(r) * H;
  float acc = 0.0f;
  for (int k = threadIdx.x << 2; k < H; k += Threads << 2) {
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(w + k));
    const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(w + k + 2));
    acc = fmaf(x0.x, w0.x, acc);
    acc = fmaf(x0.y, w0.y, acc);
    acc = fmaf(x1.x, w1.x, acc);
    acc = fmaf(x1.y, w1.y, acc);
  }
  __shared__ float partial[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const float v = warp_sum(acc);
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < Threads / 32; ++i) sum += partial[i];
    float t = __half2float(__float2half_rn(sum));
    if (act == 1) t = tanhf(t);
    else if (act == 2) t = 1.0f / (1.0f + expf(-t));
    h[static_cast<int64_t>(m) * Rtot + r] = __half2float(__float2half_rn(t));
  }
}

template <int Warps>
__global__ __launch_bounds__(Warps * 32, 1) void lora_stage2_mn_kernel(
    int H, int C, int Rtot, int M,
    const dtype* __restrict__ u_cat,     // [H, R_total]
    const dtype* __restrict__ bias_cat,  // [C, H]
    const int* __restrict__ meta,        // [C, 3]
    const float* __restrict__ h,         // [M, R_total]
    dtype* __restrict__ y) {             // [M, C, H]
  const int64_t gw = static_cast<int64_t>(blockIdx.x) * Warps + (threadIdx.x >> 5);
  const int64_t CH = static_cast<int64_t>(C) * H;
  if (gw >= static_cast<int64_t>(M) * CH) return;
  const int m = static_cast<int>(gw / CH);
  const int rem = static_cast<int>(gw - static_cast<int64_t>(m) * CH);
  const int c = rem / H;
  const int n = rem - c * H;
  const int roff = meta[c * 3];
  const int rank = meta[c * 3 + 1];
  const dtype* u = u_cat + static_cast<int64_t>(n) * Rtot + roff;
  const float* hm = h + static_cast<int64_t>(m) * Rtot + roff;
  const int lane = threadIdx.x & 31;
  float acc = 0.0f;
  if (((Rtot | roff | rank) & 1) == 0) {
    for (int r = lane << 1; r < rank; r += 64) {
      const float2 uv = __half22float2(*reinterpret_cast<const __half2*>(u + r));
      acc = fmaf(uv.x, hm[r], acc);
      acc = fmaf(uv.y, hm[r + 1], acc);
    }
  } else {
    for (int r = lane; r < rank; r += 32) acc = fmaf(__half2float(u[r]), hm[r], acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) y[gw] = __float2half_rn(acc + __half2float(bias_cat[c * H + n]));
}

at::Tensor lora4_mn(at::Tensor xs, at::Tensor d_cat, at::Tensor u_cat,
                    at::Tensor bias_cat, at::Tensor meta) {
  // xs [M, C, H] -> y [M, C, H]
  TORCH_CHECK(xs.dim() == 3, "lora4_mn: xs must be [M,C,H]");
  const int64_t M = xs.size(0);
  const int64_t C = xs.size(1);
  const int64_t H = xs.size(2);
  const int64_t Rtot = d_cat.size(0);
  TORCH_CHECK(xs.is_cuda() && xs.scalar_type() == at::kHalf, "lora4_mn: xs CUDA fp16");
  TORCH_CHECK(d_cat.scalar_type() == at::kHalf && u_cat.scalar_type() == at::kHalf &&
              bias_cat.scalar_type() == at::kHalf, "lora4_mn: weights/bias fp16");
  TORCH_CHECK(meta.scalar_type() == at::kInt && meta.dim() == 2 && meta.size(0) == C &&
              meta.size(1) == 3 && meta.is_cuda(), "lora4_mn: meta int32 [C,3] CUDA");
  TORCH_CHECK(d_cat.size(1) == H, "lora4_mn: d_cat [R,H] mismatch");
  TORCH_CHECK(u_cat.size(0) == H && u_cat.size(1) == Rtot, "lora4_mn: u_cat [H,R] mismatch");
  TORCH_CHECK(bias_cat.size(0) == C && bias_cat.size(1) == H, "lora4_mn: bias_cat [C,H] mismatch");
  TORCH_CHECK(xs.is_contiguous() && d_cat.is_contiguous() && u_cat.is_contiguous() &&
              bias_cat.is_contiguous() && meta.is_contiguous(), "lora4_mn: inputs contiguous");
  TORCH_CHECK((H % 4) == 0, "lora4_mn requires H%4==0");
  TORCH_CHECK(C >= 1 && C <= 8, "lora4_mn: 1<=C<=8");
  TORCH_CHECK(M >= 1 && Rtot >= 1, "lora4_mn: empty M/rank");
  TORCH_CHECK(M <= 65535, "lora4_mn: M exceeds gridDim.y limit (and the fused path is M-gated to tiny M anyway)");
  auto h = at::empty({M, Rtot}, xs.options().dtype(at::kFloat));
  auto y = at::empty({M, C, H}, xs.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 128;
  dim3 g1(static_cast<unsigned>(Rtot), static_cast<unsigned>(M));
  lora_stage1_mn_kernel<kThreads><<<g1, kThreads, 0, stream>>>(
      static_cast<int>(H), static_cast<int>(C), static_cast<int>(Rtot),
      xs.data_ptr<dtype>(), d_cat.data_ptr<dtype>(), meta.data_ptr<int>(), h.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  constexpr int kWarps = 8;
  const int64_t total = M * C * H;
  const int64_t blocks = (total + kWarps - 1) / kWarps;
  lora_stage2_mn_kernel<kWarps><<<blocks, kWarps * 32, 0, stream>>>(
      static_cast<int>(H), static_cast<int>(C), static_cast<int>(Rtot), static_cast<int>(M),
      u_cat.data_ptr<dtype>(), bias_cat.data_ptr<dtype>(), meta.data_ptr<int>(),
      h.data_ptr<float>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

TORCH_LIBRARY(rwkv7_lora, m) {
  m.def("lora4_m1(Tensor xs, Tensor d_cat, Tensor u_cat, Tensor bias_cat, Tensor meta) -> Tensor");
  m.def("lora4_mn(Tensor xs, Tensor d_cat, Tensor u_cat, Tensor bias_cat, Tensor meta) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_lora, CUDA, m) {
  m.impl("lora4_m1", &lora4_m1);
  m.impl("lora4_mn", &lora4_mn);
}
