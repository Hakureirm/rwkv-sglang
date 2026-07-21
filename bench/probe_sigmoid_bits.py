#!/usr/bin/env python3
"""F0066c debug probe: which exp does triton's _lora_gates actually use?

Reproduces the failing gate case, finds the differing elements, and for each
computes the a-chain (rnd(1/(1+exp(-x)))) four ways on the exact fp16 input:
  T  = the deployed triton kernel's output (production bits)
  E  = CUDA expf   (IEEE libdevice __nv_expf — what lora4_m1_gated uses today)
  F  = CUDA __expf (fast ex2.approx path)
  S  = torch.sigmoid on the fp16 tensor (ATen reference)
Prints the raw uint16 bits of all four so the divergent implementation is
identified by evidence, not guess. Also runs a FULL-RANGE sweep (all 65536
fp16 bit patterns) counting T-vs-E / T-vs-F / T-vs-S mismatches — the
distribution-free answer.
"""
import torch
from torch.utils.cpp_extension import load_inline

from sglang.srt.layers.attention.rwkv7_kernels import lora_fused
from sglang.srt.layers.attention.rwkv7_kernels.fused import fused_lora_gates

assert lora_fused.available()

cpp_src = "std::vector<torch::Tensor> sig2(torch::Tensor x);"
cu_src = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
__global__ void sig2_kernel(int n, const at::Half* x, at::Half* ye, at::Half* yf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __half2float(*reinterpret_cast<const __half*>(x + i));
  ye[i] = at::Half(__float2half_rn(1.f / (1.f + expf(-v))));
  yf[i] = at::Half(__float2half_rn(1.f / (1.f + __expf(-v))));
}
std::vector<torch::Tensor> sig2(torch::Tensor x) {
  auto ye = torch::empty_like(x);
  auto yf = torch::empty_like(x);
  int n = x.numel();
  sig2_kernel<<<(n + 255) / 256, 256>>>(
      n, x.data_ptr<at::Half>(), ye.data_ptr<at::Half>(), yf.data_ptr<at::Half>());
  return {ye, yf};
}
"""
mod = load_inline(name="sig2probe", cpp_sources=cpp_src, cuda_sources=cu_src,
                  functions=["sig2"], extra_cuda_cflags=["-O3"], verbose=False)


def bits(t):
    return [hex(v) for v in t.view(torch.uint16).cpu().tolist()]


# --- Part 1: the exact failing case (C=3 ranks=[96,96,128] seed=1, a-row) ---
H = 4096
g = torch.Generator(device="cuda").manual_seed(1)
ranks = [96, 96, 128]
C = 3
Rtot = sum(ranks)
xs = (torch.randn((C, H), generator=g, device="cuda") * 0.6).half()
d_cat = (torch.randn((Rtot, H), generator=g, device="cuda") * 0.05).half()
u_cat = (torch.randn((H, Rtot), generator=g, device="cuda") * 0.05).half()
bias = (torch.randn((C, H), generator=g, device="cuda") * 0.3).half()
meta_rows, off = [], 0
for i, r in enumerate(ranks):
    meta_rows.append([off, r, 1 if i == 0 else 0])
    off += r
meta = torch.tensor(meta_rows, dtype=torch.int32, device="cuda")
v = (torch.randn((1, H), generator=g, device="cuda") * 0.6).half()
vf = (torch.randn((1, H), generator=g, device="cuda") * 0.6).half()

lo = lora_fused.lora4_m1(xs, d_cat, u_cat, bias, meta)
w_t, a_t, _ = fused_lora_gates(lo, v, vf, False)          # triton bits
yb = lora_fused.lora4_m1_gated(xs, d_cat, u_cat, bias, meta,
                               v.reshape(-1).contiguous(),
                               vf.reshape(-1).contiguous(), 0.6065306597126334)
a_mine = yb[1:2]
diff_idx = (a_t != a_mine).reshape(-1).nonzero().reshape(-1)
print(f"failing-case a-row differing elements: {diff_idx.numel()}")
for i in diff_idx[:8].cpu().tolist():
    x_in = lo[1:2, i:i + 1].reshape(1)
    ye, yf = mod.sig2(x_in.contiguous())
    s = torch.sigmoid(x_in)
    print(f"  idx={i} lo_bits={bits(x_in)[0]}  triton={bits(a_t.reshape(-1)[i:i+1])[0]} "
          f"expf={bits(ye)[0]} __expf={bits(yf)[0]} torch={bits(s)[0]} "
          f"mine={bits(a_mine.reshape(-1)[i:i+1])[0]}")

# --- Part 2: full fp16 sweep — distribution-free census ---
allbits = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.float16).cuda()
finite = torch.isfinite(allbits)
x_all = torch.where(finite, allbits, torch.zeros_like(allbits)).half().contiguous()
lo_sw = torch.zeros((4, 65536), device="cuda", dtype=torch.float16)
lo_sw[1] = x_all  # a-row input; other rows zero
w_sw, a_sw, _ = fused_lora_gates(lo_sw, torch.zeros(1, 65536, device="cuda").half(),
                                 torch.zeros(1, 65536, device="cuda").half(), False)
ye, yf = mod.sig2(x_all)
s = torch.sigmoid(x_all)
m = finite
print("full-sweep a-chain mismatch counts (finite inputs only):")
print("  triton vs expf   :", (a_sw.reshape(-1)[m] != ye[m]).sum().item())
print("  triton vs __expf :", (a_sw.reshape(-1)[m] != yf[m]).sum().item())
print("  triton vs torch  :", (a_sw.reshape(-1)[m] != s[m]).sum().item())
print("  expf   vs torch  :", (ye[m] != s[m]).sum().item())
