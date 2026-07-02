#!/usr/bin/env python3
"""Standalone numerics + speed test for the hand-written weight-only int8 kernels
(rwkv7_w8.cu): gemv_w8_m1 + gemm_w8_small + gemm_w8_tc. Same de-risk pattern as verify_w4.py.

  source ~/rwkv_env.sh && CUDA_VISIBLE_DEVICES=0 python bench/verify_w8.py
"""
import time
from pathlib import Path

import torch

GROUP = 64


def build():
    from torch.utils.cpp_extension import load
    here = Path(__file__).resolve().parent
    candidates = [
        here / "rwkv7_w8.cu",
        here.parent / "sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda/rwkv7_w8.cu",
    ]
    src = next((c for c in candidates if c.exists()), None)
    if src is None:
        raise FileNotFoundError("rwkv7_w8.cu not found")
    load(name="rwkv7_w8", sources=[str(src)], is_python_module=False, verbose=False,
         extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-Xptxas", "-O3"])


def quantize_w8(W: torch.Tensor):
    """Group-wise (G=64) symmetric int8. Returns (qweight int8[N,K], scale fp16[N,K/G], q_int)."""
    N, K = W.shape
    assert K % GROUP == 0
    NG = K // GROUP
    Wg = W.float().view(N, NG, GROUP)
    scale = (Wg.abs().amax(dim=2) / 127.0).clamp(min=1e-8)
    q = torch.round(Wg / scale[:, :, None]).clamp_(-127, 127).to(torch.int8).view(N, K)
    return q.contiguous(), scale.to(torch.float16), q.to(torch.int32)


def dequant_ref(q_int, scale):
    N, K = q_int.shape
    NG = scale.shape[1]
    return (q_int.view(N, NG, GROUP).float() * scale.float()[:, :, None]).view(N, K)


def main():
    build()
    dev = "cuda"
    torch.manual_seed(0)
    print(f"{'M':>3} {'K':>6} {'N':>6} | {'vs dequant (rel)':>17} | {'quant err vs fp16':>18} | "
          f"{'fp16 us':>8} {'w8 us':>7} {'vs fp16':>8} | {'rows==M1':>9}")
    print("-" * 108)
    ok_all = True
    for M in (1, 2, 4, 8, 16, 32, 64):
        for K, N in [(2048, 2048), (4096, 4096), (4096, 14336)]:
            if M > 8 and N % 64 != 0:
                continue
            W = (torch.randn(N, K, device=dev) * 0.02).to(torch.float16)
            X = (torch.randn(M, K, device=dev)).to(torch.float16)
            qw, sc, qi = quantize_w8(W)
            qw = qw.to(dev); sc = sc.to(dev); qi = qi.to(dev)

            if M == 1:
                y = torch.ops.rwkv7_w8.gemv_w8_m1(X, qw, sc).float()
                bitexact = "-"
            elif M <= 8:
                y = torch.ops.rwkv7_w8.gemm_w8_small(X, qw, sc).float()
                y_rows = torch.cat([torch.ops.rwkv7_w8.gemv_w8_m1(X[m:m+1], qw, sc)
                                    for m in range(M)], dim=0).float()
                bitexact = "BIT-EXACT" if torch.equal(y.half(), y_rows.half()) else "MISMATCH"
                ok_all &= (bitexact == "BIT-EXACT")
            else:
                # tensor-core path: fp16 wmma + fp32 accum — deterministic but not
                # bit-identical to the scalar GEMV (different reduction structure);
                # gate on rel-err vs the dequant reference instead.
                y = torch.ops.rwkv7_w8.gemm_w8_tc(X, qw, sc).float()
                bitexact = "tc"
            y_ref = X.float() @ dequant_ref(qi, sc).t()
            rel = ((y - y_ref).norm() / (y_ref.norm() + 1e-9)).item()
            y_fp = X.float() @ W.float().t()
            qerr = ((y_ref - y_fp).norm() / (y_fp.norm() + 1e-9)).item()
            ok = rel < 2e-3
            ok_all &= ok

            Wt = W.t().contiguous()
            if M == 1:
                op = lambda: torch.ops.rwkv7_w8.gemv_w8_m1(X, qw, sc)
            elif M <= 8:
                op = lambda: torch.ops.rwkv7_w8.gemm_w8_small(X, qw, sc)
            else:
                op = lambda: torch.ops.rwkv7_w8.gemm_w8_tc(X, qw, sc)
            for _ in range(20):
                _ = X @ Wt; _ = op()
            torch.cuda.synchronize()
            it = 200
            t0 = time.perf_counter()
            for _ in range(it): _ = X @ Wt
            torch.cuda.synchronize(); fp16_us = (time.perf_counter() - t0) / it * 1e6
            t0 = time.perf_counter()
            for _ in range(it): _ = op()
            torch.cuda.synchronize(); w8_us = (time.perf_counter() - t0) / it * 1e6

            flag = "OK " if ok else "BAD"
            print(f"{M:>3} {K:>6} {N:>6} | {rel:>13.2e} {flag} | {qerr:>18.2e} | "
                  f"{fp16_us:>8.1f} {w8_us:>7.1f} {fp16_us / w8_us:>7.2f}x | {bitexact:>9}")
    print("-" * 108)
    print("W8 ALL OK" if ok_all else "W8 FAILED")


if __name__ == "__main__":
    main()
