// RWKV-7 x sglang fused layer-boundary glue for decode (ADR-0005 R2).
//
// Fuses the paged token-shift (gather prev conv + scatter current) with the
// lerp, keeping the shifted intermediate ON-CHIP (registers) instead of writing
// it to HBM between a token_shift kernel and a separate lerp kernel. Under
// cuda-graph the launch overhead is already captured; the win here is the HBM
// round-trip of `shifted` (and dropping token_shift's full `.clone()` copy) —
// the memory-bound bsz1-decode cost that keeps us at 0.73x vs albatross's
// whole-layer mega-fusion (F0007 / F0023 §4).
//
// TWO ops (attn entry = 6-way lerp; ffn entry = 1-way lerp), decode path only
// (one token per request, cache_indices[t] = the request's conv slot). normed
// is fp16 and the conv state is fp32 (both guarded host-side; other configs
// keep the torch path).
//
// GREEDY-EXACTNESS: replicates fused_lerp6's exact fp16 rounding
// (fused.py:_lerp6_kernel): d = round_fp16(shifted - x); per output
// prod = round_fp16(mix*d); o = round_fp16(x + prod). token_shift semantics
// (rwkv7_backend.py:134-136): prev is read as round_fp16(conv_fp32) (exactly
// prev.to(x.dtype)) BEFORE the scatter, and conv <- float(normed_fp16) (exact
// upcast). So this is byte-identical to token_shift + fused_lerp6
// (see bench/test_glue.py).
//
// PAD SLOTS: under padded cuda-graph decode replay sglang fills the tail of
// mamba_cache_indices with PAD_SLOT_ID = -1 (see upstream mamba backends /
// causal_conv1d). Pad rows must not touch the conv pool (ci = -1 would index
// one H-row BEFORE the pool for layer 0, or the previous layer's row `size` -
// an allocatable live slot - for later layers). We guard ci against [0, S)
// and write zeros to the (discarded) pad output rows, mirroring
// wkv_recurrent's s_mask. Valid indices are distinct per running request, so
// the conv scatter has no cross-token race; pad rows write nothing.
//
// cuda-graph safe: static shapes, current stream, no host sync, no allocation
// beyond the output.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_fp16.h>

using dtype = at::Half;

// The conv token-shift state is fp32 (rwkv7_backend.py). token_shift stores
// conv = normed.to(fp32) (exact upcast) and returns prev.to(fp16) (rounds), which
// fused_lerp6 then loads as fp16 -> f32. We replicate both: sh = round_fp16(conv);
// conv <- float(normed_fp16).

// attn entry: out[6,T,H] = lerp6(normed, prev);  conv[ci] <- normed.
// Grid is (T, ceil(H/Threads)): every element is touched by exactly ONE
// thread and the math is pure elementwise, so H-tiling changes nothing about
// the bits (re-gated by bench/test_glue.py). The original 1-block-per-token
// launch was sized for the bsz1 decode path this kernel was built for; at
// bs320 serving it left the SMs under-filled (320 blocks) and measured
// SLOWER than the unfused gather/index_put cluster it replaces (W1' leg
// attribution) - the tiled grid scales with T*H.
template <int Threads>
__global__ __launch_bounds__(Threads, 2) void shift_lerp6_kernel(
    int T, int H, int S,
    const dtype* __restrict__ normed,         // [T, H] fp16
    const dtype* __restrict__ mix6,           // [6, H] fp16 (xr,xk,xw,xa,xg,xv)
    const int* __restrict__ cache_indices,    // [T]
    float* __restrict__ conv,                 // [S, H, 1] fp32, row stride H
    dtype* __restrict__ out) {                // [6, T, H] fp16
  const int t = blockIdx.x;
  const int ci = cache_indices[t];
  const int64_t obase = static_cast<int64_t>(t) * H;
  const int k = blockIdx.y * Threads + threadIdx.x;
  if (k >= H) return;
  if (ci < 0 || ci >= S) {  // PAD_SLOT_ID (-1) padded replay: no conv access, zero the discarded row
#pragma unroll
    for (int j = 0; j < 6; ++j)
      out[static_cast<int64_t>(j) * T * H + obase + k] = __float2half_rn(0.f);
    return;
  }
  const dtype* xr = normed + static_cast<int64_t>(t) * H;
  float* cr = conv + static_cast<int64_t>(ci) * H;
  {
    const float x = static_cast<float>(xr[k]);
    const float sh = __half2float(__float2half_rn(cr[k]));  // prev.to(fp16) BEFORE scatter
    cr[k] = x;                                              // conv <- normed.to(fp32)
    const float d = __half2float(__float2half_rn(sh - x));  // round_fp16(shifted - x)
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      const float m = static_cast<float>(mix6[static_cast<int64_t>(j) * H + k]);
      const float prod = __half2float(__float2half_rn(m * d));
      out[static_cast<int64_t>(j) * T * H + obase + k] = __float2half_rn(x + prod);
    }
  }
}

// ffn entry: xk[T,H] = lerp1(normed, prev, x_k);  conv[ci] <- normed.
template <int Threads>
__global__ __launch_bounds__(Threads, 2) void shift_lerp1_kernel(
    int T, int H, int S,
    const dtype* __restrict__ normed,         // [T, H] fp16
    const dtype* __restrict__ x_k,            // [H] fp16
    const int* __restrict__ cache_indices,    // [T]
    float* __restrict__ conv,                 // [S, H, 1] fp32
    dtype* __restrict__ out) {                // [T, H] fp16
  const int t = blockIdx.x;
  const int ci = cache_indices[t];
  dtype* orow = out + static_cast<int64_t>(t) * H;
  const int k = blockIdx.y * Threads + threadIdx.x;  // (T, ceil(H/Threads)) grid, see lerp6
  if (k >= H) return;
  if (ci < 0 || ci >= S) {  // PAD_SLOT_ID (-1) padded replay: no conv access, zero the discarded row
    orow[k] = __float2half_rn(0.f);
    return;
  }
  const dtype* xr = normed + static_cast<int64_t>(t) * H;
  float* cr = conv + static_cast<int64_t>(ci) * H;
  {
    const float x = static_cast<float>(xr[k]);
    const float sh = __half2float(__float2half_rn(cr[k]));
    cr[k] = x;
    const float d = __half2float(__float2half_rn(sh - x));
    const float m = static_cast<float>(x_k[k]);
    const float prod = __half2float(__float2half_rn(m * d));
    orow[k] = __float2half_rn(x + prod);
  }
}

at::Tensor shift_lerp6(at::Tensor normed, at::Tensor mix6,
                       at::Tensor cache_indices, at::Tensor conv) {
  const int64_t T = normed.size(0);
  const int64_t H = normed.size(1);
  TORCH_CHECK(normed.is_cuda() && normed.scalar_type() == at::kHalf, "shift_lerp6: normed CUDA fp16");
  TORCH_CHECK(mix6.scalar_type() == at::kHalf && mix6.size(0) == 6 && mix6.size(1) == H,
              "shift_lerp6: mix6 fp16 [6,H]");
  TORCH_CHECK(conv.scalar_type() == at::kFloat && conv.is_contiguous(), "shift_lerp6: conv fp32 contiguous");
  TORCH_CHECK(conv.dim() >= 2 && conv.size(1) == H, "shift_lerp6: conv [S+1,H,1] hidden must == normed H (kernel indexes row stride H)");
  TORCH_CHECK(cache_indices.scalar_type() == at::kInt && cache_indices.size(0) == T,
              "shift_lerp6: cache_indices int32 [T]");
  TORCH_CHECK(normed.is_contiguous() && mix6.is_contiguous() && cache_indices.is_contiguous(),
              "shift_lerp6: inputs contiguous");
  auto out = at::empty({6, T, H}, normed.options());
  if (T == 0 || H == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  const dim3 grid6(static_cast<unsigned>(T),
                   static_cast<unsigned>((H + kThreads - 1) / kThreads));
  shift_lerp6_kernel<kThreads><<<grid6, kThreads, 0, stream>>>(
      static_cast<int>(T), static_cast<int>(H), static_cast<int>(conv.size(0)),
      normed.data_ptr<dtype>(),
      mix6.data_ptr<dtype>(), cache_indices.data_ptr<int>(),
      conv.data_ptr<float>(), out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor shift_lerp1(at::Tensor normed, at::Tensor x_k,
                       at::Tensor cache_indices, at::Tensor conv) {
  const int64_t T = normed.size(0);
  const int64_t H = normed.size(1);
  TORCH_CHECK(normed.is_cuda() && normed.scalar_type() == at::kHalf, "shift_lerp1: normed CUDA fp16");
  TORCH_CHECK(x_k.scalar_type() == at::kHalf && x_k.numel() == H, "shift_lerp1: x_k fp16 [H]");
  TORCH_CHECK(conv.scalar_type() == at::kFloat && conv.is_contiguous(), "shift_lerp1: conv fp32 contiguous");
  TORCH_CHECK(conv.dim() >= 2 && conv.size(1) == H, "shift_lerp1: conv [S+1,H,1] hidden must == normed H (kernel indexes row stride H)");
  TORCH_CHECK(cache_indices.scalar_type() == at::kInt && cache_indices.size(0) == T,
              "shift_lerp1: cache_indices int32 [T]");
  TORCH_CHECK(normed.is_contiguous() && x_k.is_contiguous() && cache_indices.is_contiguous(),
              "shift_lerp1: inputs contiguous");
  auto out = at::empty({T, H}, normed.options());
  if (T == 0 || H == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  const dim3 grid1(static_cast<unsigned>(T),
                   static_cast<unsigned>((H + kThreads - 1) / kThreads));
  shift_lerp1_kernel<kThreads><<<grid1, kThreads, 0, stream>>>(
      static_cast<int>(T), static_cast<int>(H), static_cast<int>(conv.size(0)),
      normed.data_ptr<dtype>(),
      x_k.data_ptr<dtype>(), cache_indices.data_ptr<int>(),
      conv.data_ptr<float>(), out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY(rwkv7_glue, m) {
  // conv is mutated in-place (the token-shift scatter) - declare it (a!) so
  // functionalization/compile passes (e.g. piecewise cuda graph) see the write.
  m.def("shift_lerp6(Tensor normed, Tensor mix6, Tensor cache_indices, Tensor(a!) conv) -> Tensor");
  m.def("shift_lerp1(Tensor normed, Tensor x_k, Tensor cache_indices, Tensor(a!) conv) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_glue, CUDA, m) {
  m.impl("shift_lerp6", &shift_lerp6);
  m.impl("shift_lerp1", &shift_lerp1);
}
