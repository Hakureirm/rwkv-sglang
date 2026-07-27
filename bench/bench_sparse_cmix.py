# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Floor decomposition for sparse_cmix (F0068 cut 3 / task #57).

sparse_cmix is the LARGEST addressable non-bandwidth-wall item in the 7.2B bsz1
step: 418.17 us/step over 32.28 calls = 12.95 us/call (F0066-E0 table), i.e.
25.9% of the 1617 us/step pool that is not already at a DRAM wall. Unlike
add_ln it is NOT a single-block latency problem — its grid is
(inter/128, H/256) = (128, 16) = 2048 blocks. So the question is different:
how far is it from its own byte floor, and what holds it there?

Two candidate bounds, and they imply opposite fixes:
  (a) the sparse weight stream — nnz_count x C_TILE x 2 B per block;
  (b) the cross-tile combine — every one of the inter/FFN_TILE tiles atomicAdds
      into the same [H] fp32 buffer, so each output element takes 128 atomic f32
      adds per call (7.2B: 524k atomics, ~2 MB of traffic into 16 KB).

This measures (b) directly by A/B-ing the production kernel against the
_probe_cmix_noatomic probe (plain store instead of atomicAdd — WRONG RESULTS by
construction, probe only), at the real 7.2B geometry across the measured
sparsity band (the kernel header records relu(k)^2 as 86-90% exact zero).

Usage:
  python3 bench_sparse_cmix.py --cuda-dir <rwkv7_kernels/cuda> [--hidden 4096]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

FFN_TILE, C_TILE = 128, 256


def _load_ext(cuda_dir: Path):
    from torch.utils.cpp_extension import load

    load(
        name="rwkv7_sparse_cmix",
        sources=[str(cuda_dir / "rwkv7_sparse_cmix.cu")],
        is_python_module=False,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
    )


def time_graph(fn, iters: int, reps: int):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(10):
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
    return statistics.median(sorted(samples))


def make_preact(inter, density, gen):
    """Raw key preactivation whose relu()^2 has exactly `density` nonzeros.

    The kernel computes relu(k)^2 itself and treats <=0 as the zero it skips, so
    sparsity is controlled by the SIGN of k: negatives become exact zeros.
    """
    k = torch.empty(inter, device="cuda", dtype=torch.float16)
    k.normal_(generator=gen)
    n_nz = int(round(inter * density))
    perm = torch.randperm(inter, generator=gen, device="cuda")
    k[perm[:n_nz]] = k[perm[:n_nz]].abs().clamp(min=0.05)   # strictly positive
    k[perm[n_nz:]] = -k[perm[n_nz:]].abs().clamp(min=0.05)  # strictly negative
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-dir", required=True)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--inter", type=int, default=16384)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    _load_ext(Path(args.cuda_dir))
    H, inter = args.hidden, args.inter
    gen = torch.Generator(device="cuda").manual_seed(20260727)
    wt = torch.randn(inter * H, device="cuda", dtype=torch.float16) * 0.02

    peak = {"NVIDIA GeForce RTX 3090": 936.2}.get(torch.cuda.get_device_name())
    rows = []
    for density in (0.10, 0.12, 0.14, 0.25, 0.50, 1.00):
        pre = make_preact(inter, density, gen)
        t_atomic = time_graph(
            lambda: torch.ops.rwkv7_sparse_cmix._probe_cmix_noatomic(pre, wt, H, False),
            args.iters, args.reps)
        t_plain = time_graph(
            lambda: torch.ops.rwkv7_sparse_cmix._probe_cmix_noatomic(pre, wt, H, True),
            args.iters, args.reps)
        t_full = time_graph(
            lambda: torch.ops.rwkv7_sparse_cmix.sparse_cmix(pre, wt, H),
            args.iters, args.reps)
        # weight bytes actually streamed: nnz rows x C_TILE x 2 B x (H/C_TILE) tiles
        wbytes = density * inter * H * 2
        rec = {
            "density": density,
            "us_atomic": round(t_atomic, 3),
            "us_plain_store_PROBE": round(t_plain, 3),
            "atomic_cost_us": round(t_atomic - t_plain, 3),
            "atomic_share": round((t_atomic - t_plain) / t_atomic, 3),
            "us_full_with_finalize": round(t_full, 3),
            "weight_MB": round(wbytes / 1e6, 2),
            "achieved_GBps": round(wbytes / (t_atomic * 1e-6) / 1e9, 1),
        }
        if peak:
            rec["pct_of_peak_BW"] = round(100.0 * rec["achieved_GBps"] / peak, 1)
        rows.append(rec)
        print(json.dumps(rec), flush=True)

    out = {"device": torch.cuda.get_device_name(), "H": H, "inter": inter,
           "grid_blocks": (inter // FFN_TILE) * (H // C_TILE), "rows": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
