# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Standalone gate + micro-bench for the sparse channel-mix value projection.

Validates rwkv7_sparse_cmix.cu in isolation before model wiring:
  1. build the JIT extension,
  2. correctness of the fp32-accum sparse kernel vs an fp32 dense reference (must be at
     least as close to fp32 as torch's own fp16 matmul — i.e. same rounding class),
  3. micro-bench sparse vs dense F.linear at realistic sparsity (50% and ~90%, matching
     the measured 86-90% real-prompt sqrelu sparsity) for real RWKV-7 FFN shapes.

Usage (box): ~/envs/rwkv-sgl/bin/python bench/verify_sparse_cmix.py
"""

import sys

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.rwkv7_kernels import sparse_cmix as sc

DEV = "cuda"
DT = torch.float16
SHAPES = [(768, 3072), (2048, 8192), (4096, 16384)]  # (H, inter) for 0.1B/1.5B/7.2B


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def check():
    print("\n== sparse_cmix correctness (out[H] = Wt @ relu(k)^2) ==")
    ok = True
    for (H, inter) in SHAPES:
        torch.manual_seed(0)
        W = (torch.randn(H, inter, device=DEV, dtype=DT) * 0.05)
        k = (torch.randn(inter, device=DEV, dtype=DT) * 0.3)  # ~50% negative
        act16 = torch.relu(k) ** 2
        ref32 = F.linear(act16.float(), W.float())            # fp32 dense truth
        dense16 = F.linear(act16, W).float()                  # torch fp16 path
        tiled = sc.tile_value_weight(W)
        out = sc.sparse_cmix(k, tiled, H).float().view(-1)
        es, ed = _rel(out, ref32), _rel(dense16, ref32)
        zero = (act16 == 0).float().mean().item()
        v = "OK" if es <= max(ed * 1.5, 2e-3) else "BAD"
        ok = ok and v == "OK"
        print(f"  H={H:5d} inter={inter:6d} sparsity={zero:.2f}  "
              f"sparse_vs_fp32={es:.2e}  dense16_vs_fp32={ed:.2e}  [{v}]")
    return ok


def bench():
    print("\n== micro-bench: sparse_cmix vs dense F.linear (value proj, bsz1) ==")
    iters = 2000
    print(f"  {'H x inter':>14} | {'sparsity':>8} | {'sparse us':>9} | {'dense us':>9} | speedup")
    for (H, inter) in SHAPES:
        W = (torch.randn(H, inter, device=DEV, dtype=DT) * 0.05)
        tiled = sc.tile_value_weight(W)
        for shift in (0.0, 0.9):  # ~50% and ~90% negative -> zero
            torch.manual_seed(1)
            k = (torch.randn(inter, device=DEV, dtype=DT) * 0.5 - shift)
            act = torch.relu(k) ** 2
            zero = (act == 0).float().mean().item()
            for _ in range(50):
                sc.sparse_cmix(k, tiled, H)
                F.linear(act, W)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters):
                sc.sparse_cmix(k, tiled, H)
            e.record(); torch.cuda.synchronize()
            t_s = s.elapsed_time(e) / iters * 1000
            s.record()
            for _ in range(iters):
                F.linear(act, W)
            e.record(); torch.cuda.synchronize()
            t_d = s.elapsed_time(e) / iters * 1000
            print(f"  {H:5d}x{inter:<7d} | {zero:8.2f} | {t_s:9.2f} | {t_d:9.2f} | {t_d/t_s:5.2f}x")


def main():
    if not torch.cuda.is_available():
        print("no CUDA"); sys.exit(1)
    print(f"torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}")
    if not sc.available():
        print("BUILD FAILED"); sys.exit(2)
    print("build OK")
    a = check()
    bench()
    print(f"\nRESULT: sparse_cmix={'PASS' if a else 'FAIL'}")
    sys.exit(0 if a else 3)


if __name__ == "__main__":
    main()
