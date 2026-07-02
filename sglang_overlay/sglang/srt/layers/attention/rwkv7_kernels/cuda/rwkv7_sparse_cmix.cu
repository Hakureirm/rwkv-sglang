// RWKV-7 x sglang — sparse channel-mix (FFN) value-projection (M6 phase-2 / ADR-0004).
//
// Adapted from BlinkDL/Albatross `faster3b_2606/cuda/rwkv7_mega_ops_260602.cu`
// (`cmix_sparse_down_relu_one_vtile_hfma2_split2_kernel`, Apache-2.0, (c) BlinkDL /
// Bo Peng). See ALBATROSS_LICENSE + NOTICE in this dir.
//
// The channel-mix value projection is  out[H] = W[H,inter] @ (relu(k)^2)[inter].
// On real prompts relu(k)^2 is 86-90% EXACT ZERO (measured), so ~9/10 of the value
// weight never needs to be read. We keep albatross's per-tile ballot/popc compaction of
// the nonzero rows, but MODIFY it (per Apache-2.0 §4(b)):
//   * fp32 per-tile register accumulation + an fp32 output buffer (albatross uses half2
//     hfma2) — the cuBLAS rounding class;
//   * the kernel takes the RAW key preactivation `k` and applies relu()^2 itself
//     (fusing the elementwise square into the load).
// Accuracy: skipping a zero row is bit-exact (0*w = 0). The cross-inter-tile combine is a
// float atomicAdd, so each output's last-ULP rounding is ORDER-NONDETERMINISTIC (~1 ULP,
// same class as a cuBLAS split-K) — not a run-to-run bit guarantee. Empirically passes the
// greedy-EXACT + verify_batch gates (0.1B/1.5B/7.2B, cuda-graph). Static grid
// (inter/FFN_TILE, H/C_TILE), all sparsity handled in-block -> cuda-graph safe.
//
// The value weight must be pre-tiled to [inter/FFN_TILE, H/C_TILE, FFN_TILE, C_TILE]
// (done in Python: value.weight.t().reshape(...).permute(0,2,1,3)); requires
// inter%FFN_TILE==0 and H%C_TILE==0 (checked by the caller, else dense fallback).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_fp16.h>

namespace {
constexpr int FFN_TILE = 128;   // inter rows per block-tile
constexpr int C_TILE = 256;     // output (H) cols per block-tile
constexpr int THREADS = C_TILE / 2;  // 128 threads, each owns 2 output cols (half2 read)
constexpr int NWARP = FFN_TILE / 32;
}  // namespace

// grid = (inter/FFN_TILE, H/C_TILE); block = THREADS.
__global__ __launch_bounds__(THREADS, 4) void sparse_cmix_f32acc_kernel(
    int C,                                   // = H
    const half* __restrict__ preact,         // raw key preactivation k [inter]
    const half* __restrict__ wt,             // value weight, tiled [.. FFN_TILE, C_TILE]
    float* __restrict__ out_f32) {           // [H], pre-zeroed
  __shared__ half vec[FFN_TILE];
  __shared__ int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[NWARP];
  __shared__ int warp_prefix[NWARP];

  const int fb = blockIdx.x;
  const int cb = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int start_f = fb * FFN_TILE;

  // relu(k)^2 for this tile's FFN_TILE rows (THREADS == FFN_TILE, one row per thread)
  const float v = fmaxf(__half2float(preact[start_f + tid]), 0.0f);
  const half r2 = __float2half_rn(v * v);
  vec[tid] = r2;

  // ballot/popc/warp-prefix compaction of nonzero rows into nnz_ids
  const bool nz = bool(__half_as_ushort(r2) << 1);  // nonzero iff any bit below sign set
  const unsigned mask = __ballot_sync(0xffffffffu, nz);
  const int local_pos = __popc(mask & ((1u << lane) - 1u));
  if (lane == 0) warp_counts[warp] = __popc(mask);
  __syncthreads();
  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < NWARP; ++w) { warp_prefix[w] = s; s += warp_counts[w]; }
    nnz_count = s;
  }
  __syncthreads();
  if (nz) nnz_ids[warp_prefix[warp] + local_pos] = tid;
  __syncthreads();

  // accumulate this tile's contribution to 2 output cols in fp32
  float acc0 = 0.0f, acc1 = 0.0f;
  const int c_blocks = C / C_TILE;
  const int c0 = cb * C_TILE + tid * 2;
  const int64_t tile_base = static_cast<int64_t>(fb * c_blocks + cb) * FFN_TILE * C_TILE;
  for (int i = 0; i < nnz_count; ++i) {
    const int fl = nnz_ids[i];
    const half* wp = wt + tile_base + static_cast<int64_t>(fl) * C_TILE + tid * 2;
    const float a = __half2float(vec[fl]);
    acc0 = fmaf(a, __half2float(wp[0]), acc0);
    acc1 = fmaf(a, __half2float(wp[1]), acc1);
  }
  // combine partials across the inter-tiles (fb) in fp32
  atomicAdd(&out_f32[c0], acc0);
  atomicAdd(&out_f32[c0 + 1], acc1);
}

// out[1,H] = tiled_value_weight @ relu(preact)^2 , fp32-accumulate, fp16 output.
at::Tensor sparse_cmix(at::Tensor preact, at::Tensor wt, int64_t H) {
  const int64_t inter = preact.numel();
  TORCH_CHECK(preact.scalar_type() == at::kHalf, "sparse_cmix: preact must be fp16");
  TORCH_CHECK(wt.scalar_type() == at::kHalf, "sparse_cmix: weight must be fp16");
  TORCH_CHECK(wt.numel() == inter * H, "sparse_cmix: tiled weight size mismatch");
  TORCH_CHECK((inter % FFN_TILE) == 0 && (H % C_TILE) == 0, "sparse_cmix: shape not conforming");
  auto pc = preact.contiguous();
  auto wc = wt.contiguous();
  auto out32 = at::zeros({H}, preact.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(inter / FFN_TILE), static_cast<unsigned>(H / C_TILE));
  // torch only instantiates data_ptr<> for its scalar types (at::Half), not __half.
  sparse_cmix_f32acc_kernel<<<grid, THREADS, 0, stream>>>(
      static_cast<int>(H),
      reinterpret_cast<const half*>(pc.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(wc.data_ptr<at::Half>()),
      out32.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out32.to(at::kHalf).view({1, H});
}

TORCH_LIBRARY(rwkv7_sparse_cmix, m) {
  m.def("sparse_cmix(Tensor preact, Tensor wt, int H) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_sparse_cmix, CUDA, m) {
  m.impl("sparse_cmix", &sparse_cmix);
}
