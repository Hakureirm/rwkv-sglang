// RWKV-7 x sglang fused norm-boundary kernels (W1' / ADR-0005 follow-on).
//
// Two ops, both pure same-math fusions of the layer's norm boundaries (the
// large-batch decode profile vs vllm-rwkv PR#8 found ~15 stock torch kernels
// of per-layer "glue" vs their 6 hand-fused ops; GEMM and WKV are excluded):
//
//   add_ln:      x_new = x + delta (fp16 residual add), y = LayerNorm(x_new).
//                Replaces the (add kernel + vectorized LN kernel) pair at the
//                pre-attn / pre-ffn / final-norm boundaries; x_new is written
//                once and the LN row stats are computed from registers.
//
//   gn_gatecorr: y = (GroupNorm(o) + (r*k*r_k).sum(-1,keepdim)*v) * g.
//                Replaces torch GroupNorm's RowwiseMoments + 1d-forward pair
//                PLUS the Triton _gate_corr kernel (their tmix_lnx_rkvres_xg
//                analog), dropping the o_norm HBM round-trip.
//
// GREEDY-EXACTNESS (the same bar every other fused kernel in this tree meets):
// both ops replicate their reference BIT-FOR-BIT, which for norms means
// replicating the REDUCTION ALGORITHM, not just the math:
//
//   * add_ln transcribes torch's vectorized_layer_norm_kernel (pytorch v2.11
//     aten/src/ATen/native/cuda/layer_norm_kernel.cu): one block of
//     (warp_size x 4) threads per row, aligned_vector<half,4> loads, the
//     WelfordDataLN online sum with reciprocal-multiply (delta * (1/new_count)),
//     the intra-warp shfl_down(16..1) cuWelfordCombine tree, the smem
//     inter-warp tree, sigma2/N at the end, rsqrtf, and the fp32 affine apply
//     with one final round to fp16. The residual add is done in fp32 and
//     rounded to fp16 BEFORE the stats (bit-identical to torch's a+b add
//     kernel: float add of two halves is exact, one rn round), so the stats
//     see exactly the tensor torch's LN would have seen.
//   * gn_gatecorr transcribes torch's RowwiseMomentsCUDAKernel for the 1d
//     GroupNorm case (group_norm_kernel.cu; D=head_dim < 512 -> one 32-thread
//     warp per (token, head), WelfordOps with true division + int64 n, the
//     WarpReduce shfl_down(16..1) combine with zero-count guards), including
//     two torch quirks that matter for bits: eps is rounded through fp16
//     (GroupNormKernelImpl casts the double eps to scalar_t=half), and
//     mean/rstd pass through the fp16 mean/rstd TENSORS between the moments
//     and apply kernels (we round them through fp16 in registers). The
//     gate-correction epilogue then replicates the deployed Triton
//     _gate_corr kernel's rounding sequence (fused.py): every binary op
//     computes in fp32 and rounds to fp16 before the next, and the 64-wide
//     (r*k*r_k) reduction uses the exact summation tree the Triton kernel
//     lowers to (butterfly-xor within each 32-lane half, halves added; probed
//     bit-for-bit against the live kernel before enabling - see
//     bench/test_ln_fused.py).
//
// FMA NOTE: the LN/GN APPLY expressions are written exactly like the aten
// sources and compiled with default -O3 (fmad on), reproducing the
// contraction pattern of the torch build; the fp16-rounded epilogue chains
// cannot contract across the explicit __float2half_rn round-trips. The gates
// in bench/test_ln_fused.py compare against the live torch/Triton ops on
// randomized + adversarial inputs and must show ZERO differing bytes before
// either op is enabled (RWKV_FUSED_ADDLN / RWKV_FUSED_GNGC, default OFF).
//
// cuda-graph safe: static shapes, current stream, no host sync, allocations
// only for the outputs. Decode + extend both eligible (shape-agnostic in T).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_fp16.h>

using dtype = at::Half;

namespace {

constexpr int kVec = 4;  // torch ln vec_size (dtype-independent, see aten)

struct alignas(sizeof(__half) * kVec) half4 {
  __half val[kVec];
};

// ---- torch vectorized-LN Welford (WelfordDataLN, float count) ----
struct WelfordLN {
  float mean;
  float sigma2;
  float count;
};

__device__ __forceinline__ WelfordLN welford_ln_onlinesum(const float val,
                                                          const WelfordLN& c) {
  float delta = val - c.mean;
  float new_count = c.count + 1.f;
  float new_mean = c.mean + delta * (1.f / new_count);  // aten: reciprocal mult
  return {new_mean, c.sigma2 + delta * (val - new_mean), new_count};
}

__device__ __forceinline__ WelfordLN welford_ln_combine(const WelfordLN dataB,
                                                        const WelfordLN dataA) {
  // transcribed verbatim from aten cuWelfordCombine (arg order dataB, dataA!)
  float delta = dataB.mean - dataA.mean;
  float count = dataA.count + dataB.count;
  float mean, sigma2;
  if (count > 0.f) {
    float coef = 1.f / count;
    float nA = dataA.count * coef;
    float nB = dataB.count * coef;
    mean = nA * dataA.mean + nB * dataB.mean;
    sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
  } else {
    mean = 0.f;
    sigma2 = 0.f;
  }
  return {mean, sigma2, count};
}

// compute_stats over the fp16 row held in registers (vals[n_own] = this
// thread's vec chunks, chunk i covers row vecs thrx, thrx+numx, ...), matching
// aten's per-thread order + warp/block trees. blockDim must be (32, NW).
template <int MaxVecPerThread>
__device__ WelfordLN ln_compute_stats(const half4* vals, int n_own, int N,
                                      float* buf) {
  WelfordLN wd{0.f, 0.f, 0.f};
  for (int i = 0; i < n_own; ++i) {
#pragma unroll
    for (int ii = 0; ii < kVec; ii++) {
      wd = welford_ln_onlinesum(__half2float(vals[i].val[ii]), wd);
    }
  }
  // intra-warp reduction (shfl_down 16..1), combine(current, shuffled)
  for (int offset = 16; offset > 0; offset >>= 1) {
    WelfordLN wdB{__shfl_down_sync(0xffffffff, wd.mean, offset),
                  __shfl_down_sync(0xffffffff, wd.sigma2, offset),
                  __shfl_down_sync(0xffffffff, wd.count, offset)};
    wd = welford_ln_combine(wd, wdB);
  }
  // inter-warp tree via smem (transcribed; blockDim.y power of two)
  if (blockDim.y > 1) {
    float* meansigmabuf = buf;
    float* countbuf = buf + blockDim.y;
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
      if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
        const int wrt_y = threadIdx.y - offset;
        meansigmabuf[2 * wrt_y] = wd.mean;
        meansigmabuf[2 * wrt_y + 1] = wd.sigma2;
        countbuf[wrt_y] = wd.count;
      }
      __syncthreads();
      if (threadIdx.x == 0 && threadIdx.y < offset) {
        WelfordLN wdB{meansigmabuf[2 * threadIdx.y],
                      meansigmabuf[2 * threadIdx.y + 1], countbuf[threadIdx.y]};
        wd = welford_ln_combine(wd, wdB);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      meansigmabuf[0] = wd.mean;
      meansigmabuf[1] = wd.sigma2 / float(N);
    }
    __syncthreads();
    return WelfordLN{meansigmabuf[0], meansigmabuf[1], 0.f};
  } else {
    return WelfordLN{__shfl_sync(0xffffffff, wd.mean, 0),
                     __shfl_sync(0xffffffff, wd.sigma2, 0) / float(N), 0.f};
  }
}

// x_new = round_fp16(x + delta); y = LayerNorm(x_new) — torch bit order.
// One block per row, blockDim (32, 4); N % 4 == 0; N <= MaxVecPerThread*128*4.
template <int MaxVecPerThread>
__global__ void add_ln_kernel(int N, float eps,
                              const dtype* __restrict__ x,
                              const dtype* __restrict__ delta,
                              const dtype* __restrict__ gamma,
                              const dtype* __restrict__ beta,
                              dtype* __restrict__ x_new,
                              dtype* __restrict__ y) {
  extern __shared__ float s_data[];
  const int64_t i1 = blockIdx.x;
  const half4* xv = reinterpret_cast<const half4*>(x + i1 * N);
  const half4* dv = reinterpret_cast<const half4*>(delta + i1 * N);
  const half4* gv = reinterpret_cast<const half4*>(gamma);
  const half4* bv = reinterpret_cast<const half4*>(beta);
  half4* xnv = reinterpret_cast<half4*>(x_new + i1 * N);
  half4* yv = reinterpret_cast<half4*>(y + i1 * N);

  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  const int n_vec = N / kVec;

  // residual add: float add of two halves is EXACT; one rn round to fp16 ==
  // torch's elementwise add kernel. Keep this thread's x_new chunks in
  // registers (written once; stats + apply reuse the identical bits).
  half4 own[MaxVecPerThread];
  int n_own = 0;
  for (int i = thrx; i < n_vec; i += numx) {
    half4 a = xv[i];
    half4 b = dv[i];
    half4 s;
#pragma unroll
    for (int ii = 0; ii < kVec; ii++) {
      s.val[ii] =
          __float2half_rn(__half2float(a.val[ii]) + __half2float(b.val[ii]));
    }
    xnv[i] = s;
    own[n_own++] = s;
  }

  WelfordLN wd = ln_compute_stats<MaxVecPerThread>(own, n_own, N, s_data);
  float rstd_val = rsqrtf(wd.sigma2 + eps);

  // affine apply, aten expression (fp32 compute, one implicit round at store)
  int slot = 0;
  for (int i = thrx; i < n_vec; i += numx, ++slot) {
    half4 data = own[slot];
    half4 g4 = gv[i];
    half4 b4 = bv[i];
    half4 out;
#pragma unroll
    for (int ii = 0; ii < kVec; ii++) {
      out.val[ii] = __float2half_rn(
          __half2float(g4.val[ii]) *
              (rstd_val * (__half2float(data.val[ii]) - wd.mean)) +
          __half2float(b4.val[ii]));
    }
    yv[i] = out;
  }
}

// ---- torch GroupNorm RowwiseMoments Welford (WelfordOps: true division) ----
struct WelfordGN {
  float mean;
  float m2;
  long long n;
  float nf;
};

__device__ __forceinline__ WelfordGN welford_gn_reduce(WelfordGN acc, float data) {
  long long new_n = acc.n + 1;
  float new_nf = static_cast<float>(new_n);
  float delta = data - acc.mean;
  float new_mean = acc.mean + delta / new_nf;  // true fdiv (aten WelfordOps)
  float new_delta = data - new_mean;
  return {new_mean, acc.m2 + delta * new_delta, new_n, new_nf};
}

__device__ __forceinline__ WelfordGN welford_gn_combine(WelfordGN a, WelfordGN b) {
  if (a.nf == 0.f) return b;
  if (b.nf == 0.f) return a;
  float delta = b.mean - a.mean;
  float new_count = a.nf + b.nf;
  float nb_over_n = b.nf / new_count;  // true fdiv
  return {a.mean + delta * nb_over_n,
          a.m2 + b.m2 + delta * delta * a.nf * nb_over_n, -1, new_count};
}

// The deployed Triton _gate_corr kernel's 64-wide fp32 sum lowers to a
// butterfly-xor tree within each 32-lane half of the axis, the two half-sums
// added low+high. Probed bit-for-bit against the live kernel (BK=64,
// num_warps=4, triton 3.6) by bench/test_ln_fused.py BEFORE enabling; if that
// probe ever fails on a new stack the op stays disabled (gate, not hope).
// Here thread t of the 32-thread warp holds a=p2[t] (low half) and b=p2[t+32]
// (high half).
__device__ __forceinline__ float gate_corr_sum64(float a, float b) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    a += __shfl_xor_sync(0xffffffff, a, offset);
    b += __shfl_xor_sync(0xffffffff, b, offset);
  }
  return a + b;  // low half + high half
}

// y = (GroupNorm(o) + (r*k*r_k).sum(head)*v) * g, head_dim = D <= 512.
// One 32-thread block per (token, head); mirrors torch's 1d GroupNorm
// RowwiseMoments launch (num_threads = warp_size when D < 512).
__global__ void gn_gatecorr_kernel(int D, float eps /* fp16-rounded by host */,
                                   int NH,
                                   const dtype* __restrict__ o,
                                   const dtype* __restrict__ r,
                                   const dtype* __restrict__ k,
                                   const dtype* __restrict__ rk,  // [NH*D]
                                   const dtype* __restrict__ v,
                                   const dtype* __restrict__ g,
                                   const dtype* __restrict__ gamma,  // [NH*D]
                                   const dtype* __restrict__ beta,   // [NH*D]
                                   dtype* __restrict__ out) {
  const int64_t tg = blockIdx.x;   // t * NH + h
  const int h = static_cast<int>(tg % NH);
  const int64_t rowbase = tg * D;  // == (t*NH + h) * D = t*H + h*D
  const int64_t pbase = static_cast<int64_t>(h) * D;

  // --- RowwiseMoments over o[rowbase : rowbase+D] (torch order: j += 32) ---
  WelfordGN val{0.f, 0.f, 0, 0.f};
  for (int j = threadIdx.x; j < D; j += 32) {
    val = welford_gn_reduce(val, __half2float(o[rowbase + j]));
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    WelfordGN vb{__shfl_down_sync(0xffffffff, val.mean, offset),
                 __shfl_down_sync(0xffffffff, val.m2, offset),
                 __shfl_down_sync(0xffffffff, val.n, offset),
                 __shfl_down_sync(0xffffffff, val.nf, offset)};
    val = welford_gn_combine(val, vb);
  }
  // project on lane 0, then broadcast the HALF-ROUNDED mean/rstd (torch stores
  // them into fp16 tensors between its two kernels; correction=0 -> /nf).
  float mean_f, rstd_f;
  if (threadIdx.x == 0) {
    float var = val.m2 / val.nf;
    mean_f = __half2float(__float2half_rn(val.mean));
    rstd_f = __half2float(__float2half_rn(rsqrtf(var + eps)));
  }
  mean_f = __shfl_sync(0xffffffff, mean_f, 0);
  rstd_f = __shfl_sync(0xffffffff, rstd_f, 0);

  // --- apply + gate-corr epilogue for this thread's channels j, j+32 ---
  // GroupNorm1dForward: (x - mean) * rstd * gamma + beta, fp32, one round.
  // Then the Triton _gate_corr rounding chain (each op rounds to fp16).
  float p2f[2];
  __half onormh[2];
  int jj[2] = {static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.x) + 32};
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const int j = jj[c];
    if (j < D) {
      const int64_t idx = rowbase + j;
      onormh[c] = __float2half_rn(
          (__half2float(o[idx]) - mean_f) * rstd_f *
              __half2float(gamma[pbase + j]) +
          __half2float(beta[pbase + j]));
      float rf = __half2float(r[idx]);
      float kf = __half2float(k[idx]);
      float rkf = __half2float(rk[pbase + j]);
      float p1 = __half2float(__float2half_rn(rf * kf));
      p2f[c] = __half2float(__float2half_rn(p1 * rkf));
    } else {
      onormh[c] = __float2half_rn(0.f);
      p2f[c] = 0.f;
    }
  }
  float s = gate_corr_sum64(p2f[0], p2f[1]);
  float s16 = __half2float(__float2half_rn(s));  // tl.sum(...).to(DT)
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const int j = jj[c];
    if (j < D) {
      const int64_t idx = rowbase + j;
      float gc = __half2float(__float2half_rn(s16 * __half2float(v[idx])));
      float oo = __half2float(__float2half_rn(__half2float(onormh[c]) + gc));
      out[idx] = __float2half_rn(oo * __half2float(g[idx]));
    }
  }
}

// y = relu(x)^2 in ONE kernel (their `relu_square` analog). The reference is
// torch.relu(k) ** 2 on fp16: relu is exact (max(x,0), no rounding); pow2
// computes float(r)*float(r) and rounds once. Both reproduced verbatim; pure
// elementwise, so fusing the pair is bit-exact by construction (gated anyway
// by bench/test_ln_fused.py). Fires on the M>1 dense ffn path - the M==1
// path already has gemv_m1_sqrelu / sparse_cmix.
__global__ void relu_sq_kernel(int64_t n, const dtype* __restrict__ x,
                               dtype* __restrict__ y) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = __half2float(x[i]);
    float r = v > 0.f ? v : 0.f;
    y[i] = __float2half_rn(r * r);
  }
}

}  // namespace

at::Tensor relu_sq(at::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous(),
              "relu_sq: x CUDA fp16 contiguous");
  auto y = at::empty_like(x);
  const int64_t n = x.numel();
  if (n == 0) return y;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  relu_sq_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
      n, x.data_ptr<dtype>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::tuple<at::Tensor, at::Tensor> add_ln(at::Tensor x, at::Tensor delta,
                                          at::Tensor gamma, at::Tensor beta,
                                          double eps) {
  const int64_t T = x.size(0);
  const int64_t N = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "add_ln: x CUDA fp16");
  TORCH_CHECK(delta.scalar_type() == at::kHalf && delta.sizes() == x.sizes(),
              "add_ln: delta fp16 same shape");
  TORCH_CHECK(gamma.scalar_type() == at::kHalf && gamma.numel() == N &&
                  beta.scalar_type() == at::kHalf && beta.numel() == N,
              "add_ln: affine fp16 [N]");
  TORCH_CHECK(x.is_contiguous() && delta.is_contiguous() &&
                  gamma.is_contiguous() && beta.is_contiguous(),
              "add_ln: inputs contiguous");
  TORCH_CHECK(N % kVec == 0, "add_ln: N % 4 == 0 (torch vectorized-LN tier)");
  constexpr int kMaxVecPerThread = 16;  // N <= 16*128*4 = 8192
  TORCH_CHECK(N <= kMaxVecPerThread * 128 * kVec, "add_ln: N too large");
  auto x_new = at::empty_like(x);
  auto y = at::empty_like(x);
  if (T == 0) return {x_new, y};
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 threads(32, 4, 1);
  const int nshared = threads.y > 1 ? threads.y * 3 / 2 * sizeof(float) : 0;
  add_ln_kernel<kMaxVecPerThread>
      <<<static_cast<unsigned>(T), threads, nshared, stream>>>(
          static_cast<int>(N), static_cast<float>(eps), x.data_ptr<dtype>(),
          delta.data_ptr<dtype>(), gamma.data_ptr<dtype>(),
          beta.data_ptr<dtype>(), x_new.data_ptr<dtype>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_new, y};
}

at::Tensor gn_gatecorr(at::Tensor o, at::Tensor r, at::Tensor k, at::Tensor rk,
                       at::Tensor v, at::Tensor g, at::Tensor gamma,
                       at::Tensor beta, double eps, int64_t nh) {
  const int64_t T = o.size(0);
  const int64_t H = o.size(1);
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == at::kHalf, "gn_gatecorr: o CUDA fp16");
  TORCH_CHECK(nh > 0 && H % nh == 0, "gn_gatecorr: H % nh == 0");
  const int64_t D = H / nh;
  TORCH_CHECK(D <= 64, "gn_gatecorr: head_dim <= 64 (sum tree is 2x32 lanes)");
  for (const auto& t : {r, k, v, g}) {
    TORCH_CHECK(t.scalar_type() == at::kHalf && t.numel() == T * H &&
                    t.is_contiguous(),
                "gn_gatecorr: r/k/v/g fp16 [T,H] contiguous");
  }
  TORCH_CHECK(rk.scalar_type() == at::kHalf && rk.numel() == H && rk.is_contiguous(),
              "gn_gatecorr: r_k fp16 [H]");
  TORCH_CHECK(gamma.scalar_type() == at::kHalf && gamma.numel() == H &&
                  beta.scalar_type() == at::kHalf && beta.numel() == H &&
                  gamma.is_contiguous() && beta.is_contiguous(),
              "gn_gatecorr: affine fp16 [H]");
  TORCH_CHECK(o.is_contiguous(), "gn_gatecorr: o contiguous");
  auto out = at::empty_like(o);
  if (T == 0) return out;
  // torch rounds the GroupNorm eps through the input dtype (fp16) first.
  const float eps_h = __half2float(__float2half_rn(static_cast<float>(eps)));
  auto stream = at::cuda::getCurrentCUDAStream();
  gn_gatecorr_kernel<<<static_cast<unsigned>(T * nh), 32, 0, stream>>>(
      static_cast<int>(D), eps_h, static_cast<int>(nh), o.data_ptr<dtype>(),
      r.data_ptr<dtype>(), k.data_ptr<dtype>(), rk.data_ptr<dtype>(),
      v.data_ptr<dtype>(), g.data_ptr<dtype>(), gamma.data_ptr<dtype>(),
      beta.data_ptr<dtype>(), out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY(rwkv7_ln, m) {
  m.def("add_ln(Tensor x, Tensor delta, Tensor gamma, Tensor beta, float eps) -> (Tensor, Tensor)");
  m.def(
      "gn_gatecorr(Tensor o, Tensor r, Tensor k, Tensor rk, Tensor v, Tensor g, "
      "Tensor gamma, Tensor beta, float eps, int nh) -> Tensor");
  m.def("relu_sq(Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_ln, CUDA, m) {
  m.impl("add_ln", &add_ln);
  m.impl("gn_gatecorr", &gn_gatecorr);
  m.impl("relu_sq", &relu_sq);
}
