# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Per-kernel FLOOR probe for the boundary cluster (F0068 / task #57).

add_ln at decode bsz1 fits `t = f + k*N` with f ~= 2.7 us on the 3090 — 61% of
the 4.41 us/call. That fixed term is either (a) the graph-node dispatch floor
(nothing intra-kernel can remove it: the only fix is FEWER launches, i.e.
fusion) or (b) the block-wide Welford reduction tree (fixable by a better
reduction / more blocks). The two imply opposite designs, so measure it.

Probes, all timed the same way as bench_addln_configs.py (CUDA graph, N
sequential same-stream replays -> N x kernel time):

  null_1b    relu_sq on 4 elements   -> 1 block,  ~0 work   = dispatch floor
  null_16b   relu_sq on 4096 elems   -> 16 blocks, ~0 work  = floor at the
                                        shift_lerp6 grid shape
  addln      add_ln at the current config (env-selected)

Usage: python3 bench_kernel_floor.py --cuda-dir <dir>
"""
import argparse
import json
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
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    samples.sort()
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--cuda-dir", type=str, required=True)
    args = ap.parse_args()

    _load_ext(Path(args.cuda_dir))
    dev = "cuda"
    N = args.hidden
    torch.manual_seed(0)

    tiny = torch.randn(4, device=dev, dtype=torch.float16)
    row = torch.randn(N, device=dev, dtype=torch.float16)
    x = torch.randn(1, N, device=dev, dtype=torch.float16)
    delta = torch.randn(1, N, device=dev, dtype=torch.float16)
    gamma = torch.randn(N, device=dev, dtype=torch.float16)
    beta = torch.randn(N, device=dev, dtype=torch.float16)

    out = {
        "device": torch.cuda.get_device_name(),
        "N": N,
        # relu_sq grid = ceil(n/256) blocks; 4 elems -> 1 block (dispatch floor)
        "null_1block_us": time_graph(
            lambda: torch.ops.rwkv7_ln.relu_sq(tiny), args.iters, args.reps),
        # N=4096 -> 16 blocks, same grid shape as shift_lerp6 at H=4096
        "null_16block_us": time_graph(
            lambda: torch.ops.rwkv7_ln.relu_sq(row), args.iters, args.reps),
        "addln_us": time_graph(
            lambda: torch.ops.rwkv7_ln.add_ln(x, delta, gamma, beta, 1e-5),
            args.iters, args.reps),
    }
    out["addln_minus_floor_us"] = out["addln_us"] - out["null_1block_us"]
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
