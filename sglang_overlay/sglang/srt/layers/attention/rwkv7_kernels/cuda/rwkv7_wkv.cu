// RWKV-7 WKV decode-step recurrence, hand-CUDA (task #54, W1'' kernel line).
//
// Replaces the Triton `_wkv_recurrent_kernel` on the serving-hot path only:
// batched decode (T==1) with the in-place indexed state pool (state_pool +
// cache_indices), both state storage dtypes (fp32 pool = bitwise-oracle tier,
// fp16 pool = RWKV_STATE_FP16 throughput tier). Everything else (varlen
// recurrent prefill, non-indexed h0/ht API) stays on the Triton kernel.
//
// Math (identical to wkv_recurrent.py, ground truth bench/oracle_numpy.py):
//   decay[k] = exp(w[k])                      (w is log-decay)
//   sa[v]    = sum_k (-kk[k]) * S[k,v]        (pre-update S)
//   S[k,v]   = decay[k]*S[k,v] + (kk[k]*a[k])*sa[v] + k[k]*v[v]
//   o[v]     = sum_k S[k,v] * (scale*r[k])
//
// BIT-EXACTNESS CONTRACT (gate: bench/test_wkv_cuda.py, zero differing bytes
// vs the Triton kernel per state dtype, o AND pool): all arithmetic is fp32
// in-register with fp16 touched only at the HBM boundary, and every rounding
// site replicates the Triton kernel's compiled arithmetic exactly:
//   * loads:  cvt.f32.f16 (exact); the carried state additionally passes
//     through Triton's `zeros + load` normalization (+0.0f add, which maps
//     -0.0 -> +0.0) -> __fadd_rn(x, 0.0f).
//   * decay:  Triton tl.exp lowers to ex2.approx.f32(w * 0x3FB8AA3B); we pin
//     the same instruction by inline PTX (NOT expf/exp2f, which are the
//     accurate-path functions and differ in ULPs).
//   * update: m = mul(b[k], sa); m = fma(S[k], decay[k], m);
//             S'[k] = fma(k[k], v[v], m)   (exact fusion pattern LLVM picked
//             for the Triton source `decay*S + b*sa + k*v`).
//   * reductions: fp addition is commutative but not associative, so the
//     exact association TREE of the Triton kernel's tl.sum (per state dtype:
//     the compiled thread layouts differ) is reproduced serially per column.
//     fp32 pool: 16 leaves, leaf(g) over k in {g, g+16, g+32, g+48} as
//       fma(x0, fma(x2, fma(x3? ...)))  -- precisely:
//       t = mul(m[g+16], s[g+16]); t = fma(m[g], s[g], t);
//       t = fma(m[g+32], s[g+32], t); t = fma(m[g+48], s[g+48], t)
//       warp(w) = (P[4w]+P[4w+2]) + (P[4w+1]+P[4w+3]); total = (W0+W2)+(W1+W3)
//     fp16 pool: 32 leaves, leaf(g) = fma(m[g], s[g], mul(m[g+32], s[g+32]));
//       warp(w) = ((P[8w]+P[8w+4])+(P[8w+2]+P[8w+6]))
//               + ((P[8w+1]+P[8w+5])+(P[8w+3]+P[8w+7]));
//       total   = (W0+W2)+(W1+W3)
//   * stores: cvt.rn.f16.f32 for the fp16 pool and the o output (__float2half_rn).
// Association trees extracted from the Triton kernel's own PTX for the pinned
// decode configs (BV=32, num_warps=4; sm120); numerics do NOT depend on our
// own launch geometry, so this kernel is batch-invariant by construction
// (one block per (request, head), no cross-request reduction).
//
// PAD SLOTS (cuda-graph padded replay, cache_indices[i] = -1 or >= pool size):
// state reads as zero, state write is skipped, and the o row is still written
// from the S=0 computation - byte-identical to the Triton kernel's s_mask
// semantics (its pad rows compute o from S=0 and store; callers discard them).
//
// PERFORMANCE DESIGN (F0058): at bs>=256 both the Triton kernel and this one
// sit at the same standalone in-place r+w wall (~1.52 TB/s on the 5090), but
// in serving the kernel is followed by compute-bound GEMMs, so state WRITES
// can drain from L2 during the next kernels - the kernel's own wall time is
// then read-shaped, and what matters is issuing reads/writes as few, wide,
// fully-coalesced bursts: state loads ride cp.async in 16B units (evict_first
// policy), stores leave in one back-loaded 32B-unit burst (evict_last) after
// the math; gate vectors are read once per (n,h) and staged as precomputed
// f32 (the 2-program Triton tiling read them twice), and the two reductions
// run serially per column with zero cross-warp smem round-trips (the Triton
// kernel pays 4 bar.syncs per launch for them). One (n,h) per 64-thread
// block, thread t owns state column t; smem column reads/writes are
// bank-conflict free (2 half lanes share one 32-bit word; fp32 maps 1:1).
// Wins: bs<=128 device-time (up to 1.7x paired) + eager launch-to-launch
// (3-6x, thin C++ op); bs>=320 is parity-plus (+1.0-1.5%) - both kernels are
// at the drain-regime floor there, and the serving-visible delta is ~+0.3%.
//
// No borrowed code (ADR-0004): authored against our own Triton kernel's
// semantics; smem staging / vectorized-burst layouts are standard CUDA idioms.
// cuda-graph safe: static shapes, current stream, no host sync, no allocation
// beyond the output tensor. Built WITHOUT fast-math (IEEE, no FTZ).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_fp16.h>

#include "rwkv7_pdl.cuh"  // PDL chain (task #50 sm120 step); no-op unarmed

namespace {

constexpr int D = 64;              // head_dim == K == V
constexpr int kThreads = D;        // one thread per state column

// Triton tl.exp: mul.f32 by 0f3FB8AA3B (fp32 log2 e) then ex2.approx.f32.
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float log2e() { return __int_as_float(0x3FB8AA3B); }

// L2 eviction-priority hints for the state stream (perf only - same bytes,
// same values). The carried state is read once (streaming) and its updated
// lines are what the NEXT kernels drain from L2: reads take evict_first so
// the single-use inbound lines don't displace the outbound dirty lines, and
// stores take evict_last so the dirty lines survive in L2 long enough to be
// written back under the following compute-bound kernels instead of inside
// this kernel's own wall time (the serving-regime lever; see header).
//
// The load side rides cp.async (16B .cg with an L2::cache_hint policy): the
// payload never passes through registers - that both frees ~16 registers
// (register count is this kernel's occupancy limiter) and lets the state
// copy fly while the gate-vector ALU prologue runs.
__device__ __forceinline__ void cp_async_16_evict_first(void* smem_dst,
                                                        const void* gmem_src) {
  const unsigned saddr =
      static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
               : "=l"(pol));
  asm volatile(
      "cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(
          saddr),
      "l"(gmem_src), "l"(pol));
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
}
// ptxas (CUDA 13, sm120) only accepts L2:: eviction hints on 32-byte vector
// accesses, so the store side uses .v4.b64 (ulonglong4) units.
__device__ __forceinline__ void st_evict_last(ulonglong4* p, ulonglong4 x) {
  asm volatile("st.global.L2::evict_last.v4.b64 [%0], {%1,%2,%3,%4};" ::"l"(p),
               "l"(x.x), "l"(x.y), "l"(x.z), "l"(x.w));
}

template <typename StateT>
__device__ __forceinline__ float state_to_f32(StateT x);
template <>
__device__ __forceinline__ float state_to_f32<__half>(__half x) {
  return __half2float(x);
}
template <>
__device__ __forceinline__ float state_to_f32<float>(float x) { return x; }

template <typename StateT>
__device__ __forceinline__ StateT f32_to_state(float x);
template <>
__device__ __forceinline__ __half f32_to_state<__half>(float x) {
  return __float2half_rn(x);
}
template <>
__device__ __forceinline__ float f32_to_state<float>(float x) { return x; }

// Load-side view of the carried state: exact upcast + Triton's `zeros + load`
// +0.0f normalization (-0.0 -> +0.0). __fadd_rn keeps the add under -O3.
template <typename StateT>
__device__ __forceinline__ float s_old(const StateT* sh, int kk_, int t) {
  return __fadd_rn(state_to_f32<StateT>(sh[kk_ * D + t]), 0.0f);
}

template <typename StateT>
__global__ __launch_bounds__(kThreads) void wkv_decode_kernel(
    const __half* __restrict__ r,   // [B, 1, H, D] fp16
    const __half* __restrict__ w,   // log-decay
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ kk,  // L2-normalized by the caller
    const __half* __restrict__ a,
    __half* __restrict__ o,         // [B, 1, H, D]
    StateT* __restrict__ pool,      // [n_slots, H, D, D] contiguous
    const int* __restrict__ ci,     // [B] cache_indices
    int h_shift, int n_slots, float scale) {
  // H is a power of two (32 or 64 for the RWKV-7 line): shift/mask instead of
  // an integer division pipeline (saves registers; regs are the occupancy cap).
  const int nh = blockIdx.x;
  const int n = nh >> h_shift;
  const int h = nh & ((1 << h_shift) - 1);
  const int t = threadIdx.x;  // column owner AND staging lane

  extern __shared__ __align__(32) unsigned char smraw[];
  StateT* sh_s = reinterpret_cast<StateT*>(smraw);  // [D*D] state staging
  float* sv = reinterpret_cast<float*>(smraw + D * D * sizeof(StateT));
  float* rs = sv;            // scale*r
  float* dec = sv + D;       // exp(w)
  float* bb = sv + 2 * D;    // kk*a
  float* nkk = sv + 3 * D;   // 0 - kk
  float* kf = sv + 4 * D;    // k
  // v needs no smem: thread t only ever reads its own column's v[t].

  const int cidx = ci[n];
  const bool live = (cidx >= 0) && (cidx < n_slots);

  // ---- issue the state copy FIRST (critical path: ci -> addresses -> DMA),
  // then do the gate-vector loads + ALU while the async copy flies ----
  constexpr int kCp = 16 / sizeof(StateT);           // elems per cp.async unit
  constexpr int kLd = (D * D) / (kThreads * kCp);    // units per thread
  if (live) {
    const int4* g16 = reinterpret_cast<const int4*>(
        pool + ((static_cast<int64_t>(cidx) << h_shift) + h) * D * D);
    int4* s16 = reinterpret_cast<int4*>(sh_s);
#pragma unroll
    for (int j = 0; j < kLd; ++j)
      cp_async_16_evict_first(s16 + j * kThreads + t, g16 + j * kThreads + t);
  } else {  // pad slot: S = 0, same bits as the Triton masked load
    ulonglong4* s32 = reinterpret_cast<ulonglong4*>(sh_s);
#pragma unroll
    for (int j = 0; j < kLd / 2; ++j)
      s32[j * kThreads + t] = ulonglong4{0ull, 0ull, 0ull, 0ull};
  }

  // ---- stage gate vectors (one element per thread), precomputed to f32 ----
  // PDL: the state cp.async above touches only this kernel's own pool (safe
  // pre-wait -> it overlaps the producer's tail); the gate vectors r/w/k/v/kk/a
  // ARE the predecessors' outputs, so wait here. No-op unarmed / sm<90.
  rwkv7_pdl_wait();
  const int64_t vecbase = static_cast<int64_t>(nh) * D;
  const float vv = __half2float(v[vecbase + t]);
  {
    const float rf = __half2float(r[vecbase + t]);
    const float wf = __half2float(w[vecbase + t]);
    const float kkf = __half2float(kk[vecbase + t]);
    const float af = __half2float(a[vecbase + t]);
    rs[t] = __fmul_rn(rf, scale);
    dec[t] = ex2_approx(__fmul_rn(wf, log2e()));
    bb[t] = __fmul_rn(kkf, af);
    nkk[t] = __fsub_rn(0.0f, kkf);
    kf[t] = __half2float(k[vecbase + t]);
  }
  cp_async_wait_all();
  __syncthreads();

  // ---- pass 1: sa[t] = sum_k nkk[k] * S_old[k][t], Triton association tree ----
  // Leaves land in a P[] array first (all smem loads issued up front = deep
  // ILP for latency hiding; a combine-order rewrite with ~8 live registers
  // measured 5% SLOWER - the serial chains starved the memory pipeline).
  float sa;
  if constexpr (sizeof(StateT) == 4) {
    // fp32 pool: 16 serial-4 leaves over {g, g+16, g+32, g+48}
    float P[16];
#pragma unroll
    for (int g = 0; g < 16; ++g) {
      float x = __fmul_rn(nkk[g + 16], s_old(sh_s, g + 16, t));
      x = __fmaf_rn(nkk[g], s_old(sh_s, g, t), x);
      x = __fmaf_rn(nkk[g + 32], s_old(sh_s, g + 32, t), x);
      P[g] = __fmaf_rn(nkk[g + 48], s_old(sh_s, g + 48, t), x);
    }
    const float W0 = (P[0] + P[2]) + (P[1] + P[3]);
    const float W1 = (P[4] + P[6]) + (P[5] + P[7]);
    const float W2 = (P[8] + P[10]) + (P[9] + P[11]);
    const float W3 = (P[12] + P[14]) + (P[13] + P[15]);
    sa = (W0 + W2) + (W1 + W3);
  } else {
    // fp16 pool: 32 serial-2 leaves over {g, g+32}
    float P[32];
#pragma unroll
    for (int g = 0; g < 32; ++g) {
      P[g] = __fmaf_rn(nkk[g], s_old(sh_s, g, t),
                       __fmul_rn(nkk[g + 32], s_old(sh_s, g + 32, t)));
    }
    float W[4];
#pragma unroll
    for (int wp = 0; wp < 4; ++wp) {
      const int b = wp * 8;
      W[wp] = ((P[b + 0] + P[b + 4]) + (P[b + 2] + P[b + 6])) +
              ((P[b + 1] + P[b + 5]) + (P[b + 3] + P[b + 7]));
    }
    sa = (W[0] + W[2]) + (W[1] + W[3]);
  }

  // ---- pass 2: state update (in smem, in place) + o reduction, same tree ----
  auto upd = [&](int kk_) -> float {
    float m = __fmul_rn(bb[kk_], sa);
    m = __fmaf_rn(s_old(sh_s, kk_, t), dec[kk_], m);
    const float s_new = __fmaf_rn(kf[kk_], vv, m);
    sh_s[kk_ * D + t] = f32_to_state<StateT>(s_new);
    return s_new;
  };
  float o_acc;
  if constexpr (sizeof(StateT) == 4) {
    float P[16];
#pragma unroll
    for (int g = 0; g < 16; ++g) {
      float x = __fmul_rn(rs[g + 16], upd(g + 16));
      x = __fmaf_rn(rs[g], upd(g), x);
      x = __fmaf_rn(rs[g + 32], upd(g + 32), x);
      P[g] = __fmaf_rn(rs[g + 48], upd(g + 48), x);
    }
    const float W0 = (P[0] + P[2]) + (P[1] + P[3]);
    const float W1 = (P[4] + P[6]) + (P[5] + P[7]);
    const float W2 = (P[8] + P[10]) + (P[9] + P[11]);
    const float W3 = (P[12] + P[14]) + (P[13] + P[15]);
    o_acc = (W0 + W2) + (W1 + W3);
  } else {
    float P[32];
#pragma unroll
    for (int g = 0; g < 32; ++g) {
      P[g] = __fmaf_rn(rs[g], upd(g), __fmul_rn(rs[g + 32], upd(g + 32)));
    }
    float W[4];
#pragma unroll
    for (int wp = 0; wp < 4; ++wp) {
      const int b = wp * 8;
      W[wp] = ((P[b + 0] + P[b + 4]) + (P[b + 2] + P[b + 6])) +
              ((P[b + 1] + P[b + 5]) + (P[b + 3] + P[b + 7]));
    }
    o_acc = (W[0] + W[2]) + (W[1] + W[3]);
  }
  o[vecbase + t] = __float2half_rn(o_acc);
  // PDL: o (the only tensor the next stage consumes) is stored; let the
  // dependent schedule now — its own wait still spans our state store below
  // (wait releases only at full grid completion). No-op unarmed / sm<90.
  rwkv7_pdl_launch_dependents();

  // ---- back-loaded bulk state store (skipped for pad slots) ----
  __syncthreads();
  if (live) {
    constexpr int kSt = (D * D) / (kThreads * (32 / sizeof(StateT)));
    ulonglong4* g32 = reinterpret_cast<ulonglong4*>(
        pool + ((static_cast<int64_t>(cidx) << h_shift) + h) * D * D);
    const ulonglong4* s32 = reinterpret_cast<const ulonglong4*>(sh_s);
#pragma unroll
    for (int j = 0; j < kSt; ++j)
      st_evict_last(g32 + j * kThreads + t, s32[j * kThreads + t]);
  }
}

at::Tensor wkv_decode(at::Tensor r, at::Tensor w, at::Tensor k, at::Tensor v,
                      at::Tensor kk, at::Tensor a, at::Tensor pool,
                      at::Tensor ci, double scale) {
  TORCH_CHECK(r.is_cuda() && r.dim() == 4 && r.size(1) == 1 && r.size(3) == D,
              "wkv_decode: r must be CUDA [B,1,H,64]");
  const int64_t B = r.size(0);
  const int64_t H = r.size(2);
  for (const auto& x : {r, w, k, v, kk, a}) {
    TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous() &&
                    x.sizes() == r.sizes(),
                "wkv_decode: r/w/k/v/kk/a must be contiguous fp16 [B,1,H,64]");
  }
  TORCH_CHECK(pool.is_cuda() && pool.is_contiguous() && pool.dim() == 4 &&
                  pool.size(1) == H && pool.size(2) == D && pool.size(3) == D,
              "wkv_decode: pool must be contiguous [S,H,64,64]");
  TORCH_CHECK(pool.scalar_type() == at::kHalf || pool.scalar_type() == at::kFloat,
              "wkv_decode: pool must be fp16 or fp32");
  TORCH_CHECK(ci.is_cuda() && ci.scalar_type() == at::kInt && ci.is_contiguous() &&
                  ci.numel() == B,
              "wkv_decode: cache_indices must be contiguous int32 [B]");
  TORCH_CHECK((H & (H - 1)) == 0 && H > 0, "wkv_decode: H must be a power of two");
  int h_shift = 0;
  while ((1 << h_shift) < H) ++h_shift;
  auto o = at::empty_like(v);
  if (B == 0) return o;
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(B * H));
  const int n_slots = static_cast<int>(pool.size(0));
  const float fscale = static_cast<float>(scale);
  const bool pdl = rwkv7_pdl_enabled("wkv");
  if (pool.scalar_type() == at::kHalf) {
    const size_t smem = D * D * sizeof(__half) + 5 * D * sizeof(float);
    rwkv7_launch_maybe_pdl(pdl, wkv_decode_kernel<__half>,
        grid, dim3(kThreads), smem, stream.stream(),
        reinterpret_cast<const __half*>(r.data_ptr()),
        reinterpret_cast<const __half*>(w.data_ptr()),
        reinterpret_cast<const __half*>(k.data_ptr()),
        reinterpret_cast<const __half*>(v.data_ptr()),
        reinterpret_cast<const __half*>(kk.data_ptr()),
        reinterpret_cast<const __half*>(a.data_ptr()),
        reinterpret_cast<__half*>(o.data_ptr()),
        reinterpret_cast<__half*>(pool.data_ptr()),
        static_cast<const int*>(ci.data_ptr<int>()),
        h_shift, n_slots, fscale);
  } else {
    const size_t smem = D * D * sizeof(float) + 5 * D * sizeof(float);
    rwkv7_launch_maybe_pdl(pdl, wkv_decode_kernel<float>,
        grid, dim3(kThreads), smem, stream.stream(),
        reinterpret_cast<const __half*>(r.data_ptr()),
        reinterpret_cast<const __half*>(w.data_ptr()),
        reinterpret_cast<const __half*>(k.data_ptr()),
        reinterpret_cast<const __half*>(v.data_ptr()),
        reinterpret_cast<const __half*>(kk.data_ptr()),
        reinterpret_cast<const __half*>(a.data_ptr()),
        reinterpret_cast<__half*>(o.data_ptr()),
        pool.data_ptr<float>(),
        static_cast<const int*>(ci.data_ptr<int>()),
        h_shift, n_slots, fscale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

}  // namespace

TORCH_LIBRARY(rwkv7_wkv, m) {
  // pool is mutated in place (the carried state) - declare (a!) so
  // functionalization/compile passes see the write (mirrors rwkv7_glue).
  m.def(
      "wkv_decode(Tensor r, Tensor w, Tensor k, Tensor v, Tensor kk, Tensor a, "
      "Tensor(a!) pool, Tensor ci, float scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(rwkv7_wkv, CUDA, m) { m.impl("wkv_decode", &wkv_decode); }
