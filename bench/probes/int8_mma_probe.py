"""sm120 int8-MMA feasibility probe: s8xs8->s32 wmma vs f16 wmma raw throughput."""
import torch, time
from torch.utils.cpp_extension import load_inline

src = r"""
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

__global__ void f16_mma_loop(const __half* A, const __half* B, float* C, int iters) {
  wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16,16,16, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  wmma::load_matrix_sync(a, A, 16);
  wmma::load_matrix_sync(b, B, 16);
  for (int i = 0; i < iters; ++i) wmma::mma_sync(acc, a, b, acc);
  if (threadIdx.x == 0 && blockIdx.x == 0)
    wmma::store_matrix_sync(C, acc, 16, wmma::mem_row_major);
}

__global__ void s8_mma_loop(const signed char* A, const signed char* B, int* C, int iters) {
  wmma::fragment<wmma::matrix_a, 16,16,16, signed char, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16,16,16, signed char, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16,16,16, int> acc;
  wmma::fill_fragment(acc, 0);
  wmma::load_matrix_sync(a, A, 16);
  wmma::load_matrix_sync(b, B, 16);
  for (int i = 0; i < iters; ++i) wmma::mma_sync(acc, a, b, acc);
  if (threadIdx.x == 0 && blockIdx.x == 0)
    wmma::store_matrix_sync(C, acc, 16, wmma::mem_row_major);
}

void run_f16(torch::Tensor A, torch::Tensor B, torch::Tensor C, int blocks, int iters) {
  f16_mma_loop<<<blocks, 32>>>((const __half*)A.data_ptr(), (const __half*)B.data_ptr(),
                               C.data_ptr<float>(), iters);
}
void run_s8(torch::Tensor A, torch::Tensor B, torch::Tensor C, int blocks, int iters) {
  s8_mma_loop<<<blocks, 32>>>((const signed char*)A.data_ptr(), (const signed char*)B.data_ptr(),
                              C.data_ptr<int>(), iters);
}
"""
mod = load_inline("int8probe", cpp_sources="void run_f16(torch::Tensor,torch::Tensor,torch::Tensor,int,int); void run_s8(torch::Tensor,torch::Tensor,torch::Tensor,int,int);",
                  cuda_sources=src, functions=["run_f16","run_s8"], verbose=False,
                  extra_cuda_cflags=["-O3"])

dev = "cuda"
prop = torch.cuda.get_device_properties(0)
print(f"GPU: {torch.cuda.get_device_name(0)} sm{prop.major}{prop.minor} SMs={prop.multi_processor_count}")
Ah = torch.randn(16,16, dtype=torch.half, device=dev); Bh = torch.randn(16,16, dtype=torch.half, device=dev)
Ch = torch.zeros(16,16, dtype=torch.float, device=dev)
Ai = torch.randint(-128,127,(16,16), dtype=torch.int8, device=dev); Bi = torch.randint(-128,127,(16,16), dtype=torch.int8, device=dev)
Ci = torch.zeros(16,16, dtype=torch.int, device=dev)
blocks = prop.multi_processor_count * 8
iters = 200000
flop_per_mma = 2*16*16*16
def bench(fn, *args):
    fn(*args, blocks, 1000); torch.cuda.synchronize()  # warmup+JIT
    t0=time.perf_counter(); fn(*args, blocks, iters); torch.cuda.synchronize()
    dt=time.perf_counter()-t0
    return blocks * iters * flop_per_mma / dt / 1e12
tf16 = bench(mod.run_f16, Ah, Bh, Ch)
ts8  = bench(mod.run_s8, Ai, Bi, Ci)
print(f"fp16 wmma: {tf16:8.1f} TFLOPS")
print(f"s8   wmma: {ts8:8.1f} TOPS")
print(f"int8/fp16 ratio: {ts8/tf16:.4f}")
print("S8_WMMA_COMPILES_AND_RUNS_ON_THIS_ARCH" if ts8 > 0 else "FAIL")
