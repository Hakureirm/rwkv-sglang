#!/usr/bin/env python3
"""Standalone numerics + speed test for the hand-written int4 decode GEMV (rwkv7_w4.cu).

De-risks the kernel in isolation before any model wiring:
  * numerics: kernel output vs the dequantized-fp16 reference (isolates KERNEL error)
              and vs the full-fp16 matmul (the QUANTIZATION error we pay for 4-bit).
  * speed:    kernel vs torch fp16 matmul (cuBLAS) at M==1, over the real projection
              shapes — decode is weight-bandwidth-bound, so int4 should BEAT fp16.

  source ~/rwkv_env.sh && CUDA_VISIBLE_DEVICES=0 ~/envs/rwkv-sgl/bin/python bench/verify_w4.py
"""
import time
from pathlib import Path

import torch


def build():
    from torch.utils.cpp_extension import load
    here = Path(__file__).resolve().parent
    candidates = [
        here / "rwkv7_w4.cu",  # flat (rsync'd next to this script)
        here.parent / "sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda/rwkv7_w4.cu",
    ]
    src = next((c for c in candidates if c.exists()), None)
    if src is None:
        raise FileNotFoundError(f"rwkv7_w4.cu not found in {[str(c) for c in candidates]}")
    load(name="rwkv7_w4", sources=[str(src)],
         is_python_module=False, verbose=False,
         extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-Xptxas", "-O3"])


GROUP = 64


def quantize_w4(W: torch.Tensor):
    """Group-wise (G=128) symmetric int4. Returns (qweight uint8[N,K/2], scale fp16[N,K/G], q_int).

    Must match rwkv7_w4.cu: scale[n,g]=max_{group}|W|/7; q=round(W/scale[group]) clamped
    [-7,7]; packed 2 nibbles/byte along K (byte[c] = q[2c]&0xF | (q[2c+1]&0xF)<<4)."""
    N, K = W.shape
    assert K % GROUP == 0, f"K={K} not divisible by GROUP={GROUP}"
    NG = K // GROUP
    Wf = W.float()
    Wg = Wf.view(N, NG, GROUP)
    scale = Wg.abs().amax(dim=2) / 7.0            # [N, NG]
    scale = torch.clamp(scale, min=1e-8)
    q = torch.round(Wg / scale[:, :, None]).clamp_(-7, 7).to(torch.int32).view(N, K)  # [N,K]
    nib = (q & 0xF).to(torch.uint8)               # 2's-complement nibble
    low = nib[:, 0::2]
    high = nib[:, 1::2]
    qweight = (low | (high << 4)).contiguous()    # [N, K/2] uint8
    return qweight, scale.to(torch.float16), q


def dequant_ref(q_int: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    N, K = q_int.shape
    NG = scale.shape[1]
    return (q_int.view(N, NG, GROUP).float() * scale.float()[:, :, None]).view(N, K)


def main():
    build()
    dev = "cuda"
    torch.manual_seed(0)
    # (K, N) projection shapes across the model sizes (r/k/v/o square + a wide ffn).
    shapes = [(768, 768), (2048, 2048), (4096, 4096), (4096, 14336), (14336, 4096)]
    print(f"{'K':>6} {'N':>6} | {'kernel vs dequant (rel)':>24} | "
          f"{'quant err vs fp16 (rel)':>24} | {'fp16 us':>8} {'w4 us':>8} {'speedup':>8}")
    print("-" * 104)
    all_ok = True
    for K, N in shapes:
        W = (torch.randn(N, K, device=dev) * 0.02).to(torch.float16)
        x = (torch.randn(K, device=dev) * 1.0).to(torch.float16)
        qweight, scale, q_int = quantize_w4(W)
        qweight = qweight.to(dev); scale = scale.to(dev); q_int = q_int.to(dev)

        y_w4 = torch.ops.rwkv7_w4.gemv_w4_m1(x.view(1, -1), qweight, scale).view(-1).float()
        y_dq = (x.view(1, -1).float() @ dequant_ref(q_int, scale).t()).view(-1)   # kernel target
        y_fp16 = (x.view(1, -1).float() @ W.float().t()).view(-1)                 # ideal

        rel_kernel = ((y_w4 - y_dq).norm() / (y_dq.norm() + 1e-9)).item()
        rel_quant = ((y_dq - y_fp16).norm() / (y_fp16.norm() + 1e-9)).item()
        ok = rel_kernel < 2e-3
        all_ok &= ok

        # ---- speed (M==1) ----
        Wt = W.t().contiguous()  # torch matmul wants [K,N] for x[1,K]@[K,N]
        for _ in range(20):
            _ = x.view(1, -1) @ Wt
            _ = torch.ops.rwkv7_w4.gemv_w4_m1(x.view(1, -1), qweight, scale)
        torch.cuda.synchronize()
        it = 200
        t0 = time.perf_counter()
        for _ in range(it):
            _ = x.view(1, -1) @ Wt
        torch.cuda.synchronize()
        fp16_us = (time.perf_counter() - t0) / it * 1e6
        t0 = time.perf_counter()
        for _ in range(it):
            _ = torch.ops.rwkv7_w4.gemv_w4_m1(x.view(1, -1), qweight, scale)
        torch.cuda.synchronize()
        w4_us = (time.perf_counter() - t0) / it * 1e6

        flag = "OK " if ok else "BAD"
        print(f"{K:>6} {N:>6} | {rel_kernel:>19.2e} {flag} | {rel_quant:>24.2e} | "
              f"{fp16_us:>8.1f} {w4_us:>8.1f} {fp16_us / w4_us:>7.2f}x")
    print("-" * 104)
    print("ALL KERNEL-NUMERICS OK" if all_ok else "SOME KERNEL NUMERICS FAILED")


if __name__ == "__main__":
    main()
