# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Standalone gate + micro-bench for the fused fp16 decode GEMV (rwkv7_fast.cu).

Validates the CUDA extension in isolation:
  1. build the JIT extension (CUDA_HOME=/usr/local/cuda-12.9, sm_86),
  2. numerical correctness of gemv_m1 vs an fp32 torch reference (the fp32-accumulate
     kernel should be AT LEAST as close to fp32 as torch's fp16 matmul),
  3. micro-benchmark gemv_m1 vs torch F.linear at M=1 for real RWKV-7 shapes. NOTE
     this is an EAGER micro-bench: the 1.1-1.6x it shows is mostly launch/dispatch
     overhead that cuda-graph amortizes — the honest end-to-end (cuda-graph ON) gain
     is +5-9% bsz1 at 1.5B/7.2B (see docs/findings/0015 + bench/results/fast_linear/).

Usage (box, sglang venv, after `source ~/rwkv_env.sh`):
    ~/envs/rwkv-sgl/bin/python bench/verify_fast_linear.py
"""

import os
import sys

import torch
import torch.nn.functional as F

# Import the JIT loader from the deployed overlay.
from sglang.srt.layers.attention.rwkv7_kernels import fast_linear as fl

DEV = "cuda"
DT = torch.float16


def _err(a, b):
    a = a.float()
    b = b.float()
    denom = b.abs().max().clamp_min(1e-6)
    return (a - b).abs().max().item(), ((a - b).abs().max() / denom).item()


def check_gemv_m1():
    print("\n== gemv_m1 correctness (y[1,N] = x[1,K] @ W[N,K]^T) ==")
    ok = True
    for (K, N) in [(768, 768), (2048, 2048), (4096, 4096), (4096, 16384), (16384, 4096)]:
        torch.manual_seed(0)
        x = torch.randn(1, K, device=DEV, dtype=DT) * 0.1
        W = torch.randn(N, K, device=DEV, dtype=DT) * 0.05
        y = fl.gemv_m1(x, W)
        ref32 = F.linear(x.float(), W.float())          # fp32 truth
        ref16 = F.linear(x, W)                           # torch fp16 path
        ek_abs, ek_rel = _err(y, ref32)                  # kernel vs fp32
        et_abs, et_rel = _err(ref16, ref32)              # torch-fp16 vs fp32
        verdict = "OK" if ek_rel <= max(et_rel * 1.5, 2e-3) else "BAD"
        if verdict != "OK":
            ok = False
        print(f"  K={K:5d} N={N:5d}  kernel_vs_fp32 rel={ek_rel:.2e}  "
              f"torch16_vs_fp32 rel={et_rel:.2e}  [{verdict}]")
    return ok


def bench_gemv():
    print("\n== micro-bench: gemv_m1 vs torch F.linear at M=1 (decode bsz1 ceiling) ==")
    iters = 2000
    print(f"  {'shape (K x N)':>16} | {'kernel us':>10} | {'cublas us':>10} | speedup")
    for (K, N, tag) in [
        (768, 768, "0.1B r/k/v/o"),
        (2048, 2048, "1.5B r/k/v/o"),
        (4096, 4096, "7.2B r/k/v/o"),
        (4096, 16384, "7.2B ffn key"),
        (16384, 4096, "7.2B ffn value"),
    ]:
        torch.manual_seed(0)
        x = torch.randn(1, K, device=DEV, dtype=DT) * 0.1
        W = torch.randn(N, K, device=DEV, dtype=DT) * 0.05
        # warmup
        for _ in range(50):
            fl.gemv_m1(x, W)
            F.linear(x, W)
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fl.gemv_m1(x, W)
        e.record(); torch.cuda.synchronize()
        t_k = s.elapsed_time(e) / iters * 1000  # us
        s.record()
        for _ in range(iters):
            F.linear(x, W)
        e.record(); torch.cuda.synchronize()
        t_c = s.elapsed_time(e) / iters * 1000
        print(f"  {tag:>16} | {t_k:10.2f} | {t_c:10.2f} | {t_c/t_k:5.2f}x  ({K}x{N})")


def main():
    if not torch.cuda.is_available():
        print("no CUDA"); sys.exit(1)
    print(f"torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}")
    print("building rwkv7_fast extension (JIT)...")
    if not fl.available():
        print("BUILD FAILED"); sys.exit(2)
    print("build OK")
    a = check_gemv_m1()
    bench_gemv()
    print(f"\nRESULT: gemv={'PASS' if a else 'FAIL'}")
    sys.exit(0 if a else 3)


if __name__ == "__main__":
    main()
