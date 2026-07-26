# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Standalone per-kernel bench for the add_ln boundary (F0068 / task #57).

WHY: the F0066-E0 per-kernel table (7.2B, 5090) puts add_ln at 255.09 us/step
over 64.56 calls = 3.95 us/call — and its launch is grid=dim3(T), so at decode
bsz1 that is ONE block on the whole GPU (512 threads at the deployed WIDE
config). That single item is 3.5% of the 7202.5 us/step BUSY, more than the
whole shift_lerp6+shift_lerp1 pair (103.9 us/step) the banked inline-lerp
design targets. Before writing any fusion we measure what the 3.95 us is
actually SPENT on — memory, or the block-wide Welford reduction tree.

This loads rwkv7_ln.cu directly (no sglang stack needed) and times add_ln
under a CUDA graph (the deployed condition: launch overhead already captured,
same-stream kernels serialize, so N replays == N x kernel time).

Configs are selected by env (read once, statically, inside the kernel launcher)
so each config needs its own process — driven by bench_addln_sweep.sh.

  RWKV_ADDLN_WIDE=0  -> (32,4)  MaxVecPerThread=16   torch-parity partition
  RWKV_ADDLN_WIDE=1  -> (32,16) MaxVecPerThread=2    F0065 WIDE (deployed)

Usage:
  python3 bench_addln_configs.py --hidden 4096 --iters 500 --reps 30
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch


def _load_ext(cuda_dir: Path):
    from torch.utils.cpp_extension import load

    load(
        name="rwkv7_ln",
        sources=[str(cuda_dir / "rwkv7_ln.cu")],
        is_python_module=False,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
    )


def time_graph(fn, iters: int, reps: int):
    """Median us/call over `reps` replays of a graph holding `iters` calls."""
    # warmup on a side stream (required before capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(20):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)  # ms -> us/call
    samples.sort()
    return {
        "us_per_call_p50": statistics.median(samples),
        "us_per_call_min": samples[0],
        "us_per_call_p90": samples[int(0.9 * (len(samples) - 1))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=4096, help="N (7.2B=4096, 1.5B=2048)")
    ap.add_argument("--tokens", type=int, default=1, help="T (decode bsz1 = 1)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--cuda-dir", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cuda_dir = Path(args.cuda_dir) if args.cuda_dir else Path(__file__).resolve().parent.parent / \
        "sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda"
    _load_ext(cuda_dir)

    T, N = args.tokens, args.hidden
    dev = "cuda"
    torch.manual_seed(0)
    x = torch.randn(T, N, device=dev, dtype=torch.float16)
    delta = torch.randn(T, N, device=dev, dtype=torch.float16)
    gamma = torch.randn(N, device=dev, dtype=torch.float16)
    beta = torch.randn(N, device=dev, dtype=torch.float16)
    eps = 1e-5

    def call():
        torch.ops.rwkv7_ln.add_ln(x, delta, gamma, beta, eps)

    res = time_graph(call, args.iters, args.reps)

    cap = torch.cuda.get_device_capability()
    rec = {
        "kernel": "add_ln",
        "T": T,
        "N": N,
        "wide": os.environ.get("RWKV_ADDLN_WIDE", "0"),
        "fasttree": os.environ.get("RWKV_ADDLN_FASTTREE", "0"),
        "device": torch.cuda.get_device_name(),
        "sm": f"{cap[0]}{cap[1]}",
        "iters": args.iters,
        "reps": args.reps,
        **res,
    }
    # bytes touched per call: read x,delta,gamma,beta + write x_new,y (fp16)
    rec["bytes_per_call"] = int(2 * (4 * T * N + 2 * T * N)) if T else 0
    rec["eff_gbps"] = rec["bytes_per_call"] / (rec["us_per_call_p50"] * 1e-6) / 1e9
    print(json.dumps(rec))
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    sys.exit(main())
