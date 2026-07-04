// Cross-arch occupancy probe for F0023 §5 (launch-tuning axis).
//
// Validates, empirically and WITHOUT a profiler (no ncu / no perf-counter perms),
// the finding's central cross-arch claim: albatross's 64-thread linear configs are
// capped at ~67% occupancy on sm_86/sm_89 (16 resident-block limit) but reach higher
// on sm_90 (Hopper, 32 blocks/SM); and it measures our gemv_m1 candidate configs'
// real per-arch occupancy to seed the arch-aware autotune table (roadmap #6).
//
// Occupancy depends only on (registers/thread, static smem/block, blockSize) vs the
// arch's per-SM limits -- all compile-time -- so no input tensors are needed. We use
// cudaOccupancyMaxActiveBlocksPerMultiprocessor (a runtime API, zero special perms).
//
// The three kernel bodies below are copied VERBATIM from source so ptxas produces the
// real register counts:
//   * gemv_m1_kernel                     <- ours, rwkv7_fast.cu:43-85
//   * linear_orig_row2_exact_f16_kernel  <- albatross rwkv7_v3a_ops.cu:619-674 (rows==2)
//   * linear_orig_rows_f16_kernel        <- albatross rwkv7_v3a_ops.cu:426-509 (rows_cfg body)
//
// Build (per arch, on the target GPU):
//   nvcc -O3 -arch=sm_XX -o occ occupancy_probe.cu && ./occ
// Emits one JSON object per line: a "device" header then one per (kernel,config).

#include <cuda_fp16.h>
#include <cstdio>

using dtype = __half;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    x += __shfl_down_sync(0xffffffffu, x, offset);
  return x;
}

// --- OURS: gemv_m1_kernel (rwkv7_fast.cu:43-85), verbatim ------------------- //
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void gemv_m1_kernel(
    int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
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
}

// --- ALBATROSS: linear_orig_row2_exact_f16_kernel (rwkv7_v3a_ops.cu:619), verbatim -- //
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row2_exact_f16_kernel(
    int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
  for (int k2 = threadIdx.x; k2 < (K >> 1); k2 += Threads) {
    const int k = k2 << 1;
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_orig + static_cast<int64_t>(n0 + j) * K + k));
      acc0[j] = fmaf(x0.x, wv.x, acc0[j]);
      acc0[j] = fmaf(x0.y, wv.y, acc0[j]);
      acc1[j] = fmaf(x1.x, wv.x, acc1[j]);
      acc1[j] = fmaf(x1.y, wv.y, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) { partial[warp][0][j] = v0; partial[warp][1][j] = v1; }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) { sum0 += partial[w][0][j]; sum1 += partial[w][1][j]; }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}

// --- ALBATROSS: linear_orig_rows_f16_kernel (rwkv7_v3a_ops.cu:426), verbatim ---- //
template <int Threads, int RowTile, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_rows_f16_kernel(
    int M, int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m0 = blockIdx.y * RowTile;
  float acc[RowTile][OutTile];
#pragma unroll
  for (int r = 0; r < RowTile; ++r)
#pragma unroll
    for (int j = 0; j < OutTile; ++j) acc[r][j] = 0.0f;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    float2 wv[OutTile];
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      wv[j] = (n < N)
          ? __half22float2(*reinterpret_cast<const __half2*>(weight_orig + static_cast<int64_t>(n) * K + k))
          : make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + static_cast<int64_t>(m) * K + k));
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          acc[r][j] = fmaf(xv.x, wv[j].x, acc[r][j]);
          acc[r][j] = fmaf(xv.y, wv[j].y, acc[r][j]);
        }
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float wv = __half2float(*reinterpret_cast<const __half*>(weight_orig + static_cast<int64_t>(n) * K + K - 1));
#pragma unroll
        for (int r = 0; r < RowTile; ++r) {
          const int m = m0 + r;
          if (m < M) {
            const float xv = __half2float(*reinterpret_cast<const __half*>(x + static_cast<int64_t>(m) * K + K - 1));
            acc[r][j] = fmaf(xv, wv, acc[r][j]);
          }
        }
      }
    }
  }
  __shared__ float partial[Threads / 32][RowTile][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < RowTile; ++r)
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float v = warp_sum(acc[r][j]);
      if (lane == 0) partial[warp][r][j] = v;
    }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < RowTile; ++r)
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < Threads / 32; ++w) sum += partial[w][r][j];
        const int m = m0 + r, n = n0 + j;
        if (m < M && n < N) y[static_cast<int64_t>(m) * N + n] = __float2half_rn(sum);
      }
  }
}

// --------------------------------------------------------------------------- //
static int g_maxWarps, g_maxBlocks, g_numSM;

template <typename K>
static void probe(const char* owner, const char* name, int threads, K kernel) {
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, (const void*)kernel);
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (const void*)kernel, threads, 0);
  const int warps = blocks * threads / 32;
  const double occ = 100.0 * warps / g_maxWarps;
  // limiter attribution
  const char* lim = "warps/threads";
  if (blocks == g_maxBlocks) lim = "block-count-cap";
  printf("{\"owner\":\"%s\",\"kernel\":\"%s\",\"threads\":%d,\"regs\":%d,"
         "\"smem_static_B\":%d,\"max_active_blocks_per_sm\":%d,\"warps_per_sm\":%d,"
         "\"max_warps_per_sm\":%d,\"occupancy_pct\":%.1f,\"limiter\":\"%s\"}\n",
         owner, name, threads, fa.numRegs, (int)fa.sharedSizeBytes, blocks, warps,
         g_maxWarps, occ, lim);
}

int main() {
  int dev = 0; cudaGetDevice(&dev);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
  g_numSM = p.multiProcessorCount;
  g_maxWarps = p.maxThreadsPerMultiProcessor / 32;
  cudaDeviceGetAttribute(&g_maxBlocks, cudaDevAttrMaxBlocksPerMultiprocessor, dev);
  printf("{\"device\":\"%s\",\"cc\":\"%d.%d\",\"sm_arch\":%d,\"num_sm\":%d,"
         "\"max_threads_per_sm\":%d,\"max_warps_per_sm\":%d,\"max_blocks_per_sm\":%d,"
         "\"regs_per_sm\":%d,\"smem_per_sm_B\":%d,\"smem_per_block_optin_B\":%d}\n",
         p.name, p.major, p.minor, p.major * 10 + p.minor, g_numSM,
         p.maxThreadsPerMultiProcessor, g_maxWarps, g_maxBlocks,
         p.regsPerMultiprocessor, (int)p.sharedMemPerMultiprocessor,
         (int)p.sharedMemPerBlockOptin);

  // OURS: gemv_m1 candidate autotune space {64,128,256} x {1,2,4}
  probe("ours", "gemv_m1<64,1>", 64, gemv_m1_kernel<64, 1>);
  probe("ours", "gemv_m1<64,2>", 64, gemv_m1_kernel<64, 2>);
  probe("ours", "gemv_m1<64,4>", 64, gemv_m1_kernel<64, 4>);
  probe("ours", "gemv_m1<128,1>", 128, gemv_m1_kernel<128, 1>);
  probe("ours", "gemv_m1<128,2>", 128, gemv_m1_kernel<128, 2>);  // current default (N even)
  probe("ours", "gemv_m1<128,4>", 128, gemv_m1_kernel<128, 4>);
  probe("ours", "gemv_m1<256,1>", 256, gemv_m1_kernel<256, 1>);
  probe("ours", "gemv_m1<256,2>", 256, gemv_m1_kernel<256, 2>);
  probe("ours", "gemv_m1<256,4>", 256, gemv_m1_kernel<256, 4>);

  // ALBATROSS: the 64-thread configs the finding flags as occupancy-capped
  probe("albatross", "row2_exact<64,2>", 64, linear_orig_row2_exact_f16_kernel<64, 2>);
  probe("albatross", "row2_exact<128,2>", 128, linear_orig_row2_exact_f16_kernel<128, 2>);
  probe("albatross", "row2_exact<256,1>", 256, linear_orig_row2_exact_f16_kernel<256, 1>);
  probe("albatross", "rows_cfg<64,3,4>", 64, linear_orig_rows_f16_kernel<64, 3, 4>);  // worst offender
  probe("albatross", "rows_cfg<64,2,4>", 64, linear_orig_rows_f16_kernel<64, 2, 4>);
  probe("albatross", "rows_f16<128,3,4>", 128, linear_orig_rows_f16_kernel<128, 3, 4>);
  probe("albatross", "rows_f16<128,4,2>", 128, linear_orig_rows_f16_kernel<128, 4, 2>);
  probe("albatross", "rows_f16<128,3,2>", 128, linear_orig_rows_f16_kernel<128, 3, 2>);
  return 0;
}
