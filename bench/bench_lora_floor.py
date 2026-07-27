# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Floor decomposition for the LoRA pair (F0068 cut 3 / task #57).

lora_stage1 + lora_stage2 = 206.03 + 257.40 = 463.4 us/step in the 7.2B bsz1
step (F0066-E0 table) = 28.7% of the 1617 us/step pool that is not already at a
DRAM wall — the largest remaining unexamined block after add_ln (bracketed,
F0068 §2-4) and sparse_cmix (measured at 81-87% of its own achievable rate).

Real 7.2B geometry, from the checkpoint config (not assumed):
  hidden_size 4096, layers 32, decay/a/v/gate low-rank dims 128/128/96/480
  => R_total = 128+128+480+96 = 832 for layers > 0 (736 at layer 0)
  d_cat [R_total, H] fp16 = 6.82 MB ; u_cat [H, R_total] fp16 = 6.82 MB

stage1 launches ONE block per down-row (R_total blocks of 128 threads), so the
xs row is re-read by every block; stage2 launches one warp per output element.
Both read ~6.8 MB of DRAM per call, which sets a hard byte floor — but "floor"
must mean ACHIEVABLE bandwidth, not peak: the sparse_cmix probe measured this
kernel class topping out at 91.3% of peak on a dense stream, so peak-relative
numbers overstate the headroom (the mistake this harness exists to avoid).

Reports us/call, achieved GB/s, and the share of the graph-node dispatch floor,
so a fusion/rewrite proposal can be priced before it is built.

Usage: python3 bench_lora_floor.py --cuda-dir <rwkv7_kernels/cuda>
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ACT_IDENTITY, ACT_TANH, ACT_SIGMOID = 0, 1, 2
PEAK_GBPS = {"NVIDIA GeForce RTX 3090": 936.2, "NVIDIA GeForce RTX 5090": 1691.7}


def _load_ext(cuda_dir: Path):
    from torch.utils.cpp_extension import load

    for name in ("rwkv7_lora", "rwkv7_ln"):
        load(name=name, sources=[str(cuda_dir / f"{name}.cu")],
             is_python_module=False, verbose=False,
             extra_cflags=["-O3"], extra_cuda_cflags=["-O3"])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-dir", required=True)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--ranks", type=int, nargs="+", default=[128, 128, 480, 96],
                    help="decay, a, gate, v low-rank dims (checkpoint order w,a,g,v)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    _load_ext(Path(args.cuda_dir))
    H = args.hidden
    ranks = args.ranks
    C = len(ranks)
    R_total = sum(ranks)
    dev, dt = "cuda", torch.float16
    gen = torch.Generator(device=dev).manual_seed(20260727)

    xs = (torch.randn(C, H, generator=gen, device=dev, dtype=torch.float32) * 0.5).to(dt)
    d_cat = (torch.randn(R_total, H, generator=gen, device=dev, dtype=torch.float32) * 0.02).to(dt)
    u_cat = (torch.randn(H, R_total, generator=gen, device=dev, dtype=torch.float32) * 0.02).to(dt)
    bias_cat = torch.zeros(C, H, device=dev, dtype=dt)
    meta = torch.zeros(C, 3, device=dev, dtype=torch.int32)
    acts = [ACT_TANH, ACT_SIGMOID, ACT_IDENTITY, ACT_SIGMOID][:C]
    off = 0
    for c, r in enumerate(ranks):
        meta[c, 0], meta[c, 1], meta[c, 2] = off, r, acts[c]
        off += r
    meta = meta.contiguous()

    # dispatch floor in the same harness (relu_sq on 4 elements = ~no work)
    tiny = torch.randn(4, device=dev, dtype=dt)
    floor = time_graph(lambda: torch.ops.rwkv7_ln.relu_sq(tiny), args.iters, args.reps)

    t_pair = time_graph(
        lambda: torch.ops.rwkv7_lora.lora4_m1(xs, d_cat, u_cat, bias_cat, meta),
        args.iters, args.reps)

    dev_name = torch.cuda.get_device_name()
    peak = PEAK_GBPS.get(dev_name)
    d_MB, u_MB = d_cat.numel() * 2 / 1e6, u_cat.numel() * 2 / 1e6
    # lora4_m1 = stage1 + stage2 in one op (two launches) -> 2 dispatch floors
    work = t_pair - 2 * floor
    rec = {
        "device": dev_name,
        "H": H, "ranks": ranks, "R_total": R_total,
        "stage1_blocks": R_total,
        "dispatch_floor_us": round(floor, 3),
        "lora4_m1_us_per_call": round(t_pair, 3),
        "dispatch_share": round(2 * floor / t_pair, 3),
        "work_us": round(work, 3),
        "dram_MB": round(d_MB + u_MB, 2),
        "achieved_GBps_on_work": round((d_MB + u_MB) * 1e6 / (work * 1e-6) / 1e9, 1),
    }
    if peak:
        rec["pct_of_peak_on_work"] = round(
            100.0 * rec["achieved_GBps_on_work"] / peak, 1)
        # the achievable ceiling this kernel class actually reaches (sparse_cmix
        # dense probe): 91.3% of peak, not 100%
        rec["pct_of_achievable_91p3"] = round(
            100.0 * rec["achieved_GBps_on_work"] / (peak * 0.913), 1)
    print(json.dumps(rec, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    sys.exit(main())
