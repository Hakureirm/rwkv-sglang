"""Correctness gate for lora4_mn (batched-M fused LoRA, ADR-0005 R3).

The strongest self-consistent gate: lora4_mn(xs[M,C,H])[m] must be byte-identical
(torch.equal) to lora4_m1(xs[m]) for every m — since the batched kernel uses the
exact same per-(m) reduction/rounding as the proven M==1 kernel. If that holds for
all m over random inputs, the M-dim indexing is correct and token-exactness is
inherited from lora4_m1 (already greedy-gated).

Run on box: python bench/test_lora_mn.py
"""
import sys
from pathlib import Path
import torch

OV = Path(__file__).resolve().parents[1] / "sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels"
sys.path.insert(0, str(OV))

from torch.utils.cpp_extension import load  # noqa: E402
load(name="rwkv7_lora", sources=[str(OV / "cuda" / "rwkv7_lora.cu")], is_python_module=False,
     verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-Xptxas", "-O3"])

dev = torch.device("cuda")
torch.manual_seed(0)
H = 2048
# RWKV-7 4-chain ranks (decay w=96, a=96, v=64, gate g=256); acts: tanh/sigmoid/id mix
RANKS = [96, 96, 64, 256]
ACTS = [1, 2, 0, 2]  # 0=id 1=tanh 2=sigmoid (codes just need to be exercised)
C = len(RANKS)
Rtot = sum(RANKS)
meta = torch.zeros(C, 3, dtype=torch.int32)
off = 0
for c in range(C):
    meta[c, 0] = off; meta[c, 1] = RANKS[c]; meta[c, 2] = ACTS[c]; off += RANKS[c]
meta = meta.to(dev)
d_cat = torch.randn(Rtot, H, dtype=torch.float16, device=dev) * 0.1
u_cat = torch.randn(H, Rtot, dtype=torch.float16, device=dev) * 0.1
bias_cat = torch.randn(C, H, dtype=torch.float16, device=dev) * 0.1

all_ok = True
for M in [1, 2, 4, 8, 16, 32]:
    xs = torch.randn(M, C, H, dtype=torch.float16, device=dev) * 0.5
    ymn = torch.ops.rwkv7_lora.lora4_mn(xs, d_cat, u_cat, bias_cat, meta)  # [M,C,H]
    ok_all_m = True
    max_m_diff = 0
    for m in range(M):
        y1 = torch.ops.rwkv7_lora.lora4_m1(xs[m].contiguous(), d_cat, u_cat, bias_cat, meta)  # [C,H]
        if not torch.equal(ymn[m], y1):
            ok_all_m = False
            max_m_diff = max(max_m_diff, (ymn[m].float() - y1.float()).abs().max().item())
    tag = "EXACT" if ok_all_m else f"MISMATCH(max={max_m_diff:.2e})"
    print(f"M={M:3d}: lora4_mn[m] == lora4_m1(xs[m]) for all m -> {tag}")
    all_ok = all_ok and ok_all_m

print("\nRESULT:", "ALL EXACT — R3 correct (batched-M == M==1 per token)" if all_ok else "FAILED")
sys.exit(0 if all_ok else 1)
