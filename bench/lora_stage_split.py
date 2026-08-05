"""Where does lora4_mn's time go, stage1 vs stage2, and how does it compare to cuBLAS?

The batch gate sits at 8 because the fused kernel loses above it. Before rewriting
anything, split its cost: the two stages have completely different shapes (stage1
is a per-(m,rank) dot over H, stage2 is a warp per output element reducing over
rank), so the wrong one can absorb a rewrite and give nothing back.
"""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.layers.attention.rwkv7_kernels import lora_fused

H = 2048
RANKS = [64, 64, 128, 32]          # w, a, g, v -- 1.5B
ACTS = [1, 1, 1, 1]                # tanh
dev = "cuda"


def build_pack(C):
    ranks = RANKS[:C]
    rtot = sum(ranks)
    d_cat = torch.randn(rtot, H, device=dev, dtype=torch.float16) * 0.05
    u_cat = torch.randn(H, rtot, device=dev, dtype=torch.float16) * 0.05
    bias = torch.randn(C, H, device=dev, dtype=torch.float16) * 0.05
    meta = torch.zeros(C, 3, device=dev, dtype=torch.int32)
    off = 0
    for c, r in enumerate(ranks):
        meta[c, 0] = off
        meta[c, 1] = r
        meta[c, 2] = ACTS[c]
        off += r
    return d_cat.contiguous(), u_cat.contiguous(), bias.contiguous(), meta.contiguous()


def cublas_chain(xs, downs, ups, biases):
    out = []
    for c in range(xs.shape[1]):
        t = torch.tanh(xs[:, c] @ downs[c].t())
        out.append(t @ ups[c].t() + biases[c])
    return torch.stack(out, dim=1)


def main():
    assert lora_fused.available(), "lora extension did not build"
    C = 4
    pack = build_pack(C)
    ranks = RANKS[:C]
    downs = [torch.randn(r, H, device=dev, dtype=torch.float16) * 0.05 for r in ranks]
    ups = [torch.randn(H, r, device=dev, dtype=torch.float16) * 0.05 for r in ranks]
    biases = [torch.randn(H, device=dev, dtype=torch.float16) * 0.05 for _ in ranks]

    print(f"{'M':>4} {'stage1 us':>10} {'stage2 us':>10} {'fused us':>10} {'cuBLAS us':>10} {'fused/cuBLAS':>13}")
    for M in (1, 2, 4, 8, 12, 16, 24, 32):
        xs = torch.randn(M, C, H, device=dev, dtype=torch.float16).contiguous()
        for _ in range(30):
            lora_fused.lora4_mn(xs, *pack)
            cublas_chain(xs, downs, ups, biases)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(200):
                lora_fused.lora4_mn(xs, *pack)
            torch.cuda.synchronize()
        s1 = s2 = 0.0
        for e in prof.key_averages():
            if "stage1_mn" in e.key:
                s1 = e.self_device_time_total / 200
            elif "stage2_mn" in e.key:
                s2 = e.self_device_time_total / 200

        with profile(activities=[ProfilerActivity.CUDA]) as prof2:
            for _ in range(200):
                cublas_chain(xs, downs, ups, biases)
            torch.cuda.synchronize()
        cub = sum(e.self_device_time_total for e in prof2.key_averages()
                  if e.device_type == torch.autograd.DeviceType.CUDA) / 200

        fused = s1 + s2
        print(f"{M:>4} {s1:>10.2f} {s2:>10.2f} {fused:>10.2f} {cub:>10.2f} {fused / cub:>13.3f}")


main()
