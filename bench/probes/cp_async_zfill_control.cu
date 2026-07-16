// Positive/negative sanitizer control for the "predicated-off cp.async still
// zero-fills its full destination" race class (the class behind Albatross's
// 2026-07-14 `bvec` hotfix; construction below is ours — it models the CLASS,
// not their code). AUDIT PROBE ONLY: never linked into any extension.
//
// PTX semantics under test: `cp.async.ca.shared.global [dst], [src], cp-size,
// src-size` ALWAYS writes cp-size bytes to shared memory; src-size < cp-size
// zero-fills the tail. So "predicating" a copy by setting src-size = 0 does
// not skip it — it turns it into an in-flight async ZERO-FILL of dst. If a
// predicated-off thread computes the same dst as another thread's real copy,
// the zero-fill races the data (WAW; consumer then reads whichever landed
// last).
//
//   buggy: 64 threads. Warp 0 cp.async's real data into s[lane]; warp 1
//          issues the src-size-0 form AT THE SAME ADDRESS s[lane].
//   fixed: warp 1's zero-fill is routed to a scratch buffer (the fix class:
//          give predicated-off lanes a dummy destination).
//
// Expected: compute-sanitizer --tool racecheck flags `buggy` (proves the tool
// detects this hazard class on this arch/driver) and reports nothing on
// `fixed`. The binary also functionally counts wiped (zero) outputs — the
// race manifesting as data corruption, timing permitting.
//
// Build (sm86 box):  nvcc -arch=sm_86 -O3 -o cp_async_zfill_control \
//                        cp_async_zfill_control.cu
// Run:               ./cp_async_zfill_control buggy|fixed [iters]
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)

__device__ __forceinline__ void cp4(void* smem, const void* gmem, bool pred) {
  const int bytes = pred ? 4 : 0;  // src-size operand; cp-size stays 4
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(addr),
               "l"(gmem), "r"(bytes));
}

template <bool Fixed>
__global__ void control_kernel(const float* __restrict__ src,
                               float* __restrict__ out) {
  __shared__ float s[32];
  __shared__ float s_dummy[32];
  const int i = threadIdx.x;   // 0..63 (two warps)
  const int lane = i & 31;
  // buggy: every thread targets s[lane]; warp 1 is "predicated off" via
  // src-size 0 but still zero-fills s[lane] — racing warp 0's real copy.
  // fixed: warp 1's zero-fill lands in s_dummy instead.
  float* dst = (!Fixed || i < 32) ? &s[lane] : &s_dummy[lane];
  cp4(dst, src + lane, i < 32);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  out[static_cast<size_t>(blockIdx.x) * 64 + i] = s[lane];
}

int main(int argc, char** argv) {
  const bool fixed = (argc > 1 && std::strcmp(argv[1], "fixed") == 0);
  const int iters = (argc > 2) ? std::atoi(argv[2]) : 200;
  const int blocks = 1024;

  float* src;
  float* out;
  cudaMalloc(&src, 32 * sizeof(float));
  cudaMalloc(&out, static_cast<size_t>(blocks) * 64 * sizeof(float));
  float ones[32];
  for (int i = 0; i < 32; ++i) ones[i] = 1.0f;
  cudaMemcpy(src, ones, sizeof(ones), cudaMemcpyHostToDevice);

  long long wiped = 0, total = 0;
  float* host = new float[static_cast<size_t>(blocks) * 64];
  for (int it = 0; it < iters; ++it) {
    if (fixed)
      control_kernel<true><<<blocks, 64>>>(src, out);
    else
      control_kernel<false><<<blocks, 64>>>(src, out);
    cudaMemcpy(host, out, static_cast<size_t>(blocks) * 64 * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (long long j = 0; j < static_cast<long long>(blocks) * 64; ++j) {
      total += 1;
      if (host[j] == 0.0f) wiped += 1;
    }
  }
  cudaError_t err = cudaGetLastError();
  printf("[%s] launches=%d blocks=%d  wiped=%lld / %lld outputs  (cuda: %s)\n",
         fixed ? "fixed" : "buggy", iters, blocks, wiped, total,
         cudaGetErrorString(err));
  delete[] host;
  cudaFree(src);
  cudaFree(out);
  return 0;
}

#else
int main() {
  std::printf("cp.async control requires sm80+\n");
  return 0;
}
#endif
