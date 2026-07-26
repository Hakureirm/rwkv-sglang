# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Byte gate for the F0068 FastTree inter-warp Welford reduction (task #57).

CLAIM UNDER TEST: RWKV_ADDLN_FASTTREE=1 replays the aten inter-warp tree's
combine schedule inside warp 0 (lane y holds warp y's partial; offset =
NW/2..1) instead of 4 rounds of smem ping-pong, so it must be BIT-IDENTICAL —
same operand pairs, same order, no reassociation. Anything less than
torch.equal on BOTH outputs (x_new and y) means the schedule was not actually
reproduced and the change carries hidden numerics, so it must not ship.

The config is latched by a process-static getenv, so each arm runs in its own
subprocess and this driver diffs the dumps.

Covers both launcher paths (WIDE (32,16) and the parity (32,4) tier), both
deployed hidden sizes (1.5B 2048 / 7.2B 4096) plus an N that forces the parity
path, T=1 (decode) and T>1, and adversarial rows (huge/tiny/mixed magnitudes
that stress the Welford count/mean recurrence).

Usage:
  python3 test_addln_fasttree.py --cuda-dir <rwkv7_kernels/cuda>
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def _load_ext(cuda_dir: str):
    from torch.utils.cpp_extension import load

    load(
        name="rwkv7_ln",
        sources=[str(Path(cuda_dir) / "rwkv7_ln.cu")],
        is_python_module=False,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
    )


# (name, builder) — builder(T, N) -> (x, delta, gamma, beta)
def _cases():
    def normal(T, N, g):
        return (torch.randn(T, N, generator=g, device="cuda", dtype=torch.float16),
                torch.randn(T, N, generator=g, device="cuda", dtype=torch.float16))

    def large(T, N, g):
        return ((torch.randn(T, N, generator=g, device="cuda", dtype=torch.float16) * 60.0),
                (torch.randn(T, N, generator=g, device="cuda", dtype=torch.float16) * 60.0))

    def tiny(T, N, g):
        return ((torch.randn(T, N, generator=g, device="cuda", dtype=torch.float16) * 1e-3),
                (torch.randn(T, N, generator=g, device="cuda", dtype=torch.float16) * 1e-3))

    def mixed(T, N, g):
        # half the row huge, half denormal-ish -> stresses the running mean
        x = torch.randn(T, N, generator=g, device="cuda", dtype=torch.float16)
        x[..., : N // 2] *= 200.0
        x[..., N // 2:] *= 1e-4
        d = torch.randn(T, N, generator=g, device="cuda", dtype=torch.float16)
        return x, d

    return {"normal": normal, "large": large, "tiny": tiny, "mixed": mixed}


def run_arm(cuda_dir: str, out_path: str):
    _load_ext(cuda_dir)
    g = torch.Generator(device="cuda")
    res = {}
    for (T, N), (cname, build) in itertools.product(
            [(1, 2048), (1, 4096), (4, 4096), (1, 8192)], _cases().items()):
        g.manual_seed(1234 + N + T)
        x, delta = build(T, N, g)
        gamma = torch.randn(N, generator=g, device="cuda", dtype=torch.float16)
        beta = torch.randn(N, generator=g, device="cuda", dtype=torch.float16)
        x_new, y = torch.ops.rwkv7_ln.add_ln(x, delta, gamma, beta, 1e-5)
        key = f"{T}x{N}:{cname}"
        res[key + ":x_new"] = x_new.cpu()
        res[key + ":y"] = y.cpu()
    torch.save(res, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-dir", required=True)
    ap.add_argument("--dump", default=None, help="internal: run one arm")
    args = ap.parse_args()

    if args.dump:
        run_arm(args.cuda_dir, args.dump)
        return 0

    tmp = tempfile.mkdtemp(prefix="addln_fasttree_")
    arms = {}
    fails = []
    for wide in ("0", "1", "2"):
        for fast in ("0", "1"):
            path = os.path.join(tmp, f"w{wide}_f{fast}.pt")
            env = dict(os.environ, RWKV_ADDLN_WIDE=wide, RWKV_ADDLN_FASTTREE=fast)
            r = subprocess.run(
                [sys.executable, __file__, "--cuda-dir", args.cuda_dir, "--dump", path],
                env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[FAIL] arm wide={wide} fast={fast} crashed:\n{r.stdout}\n{r.stderr}")
                return 1
            arms[(wide, fast)] = torch.load(path)

    total = 0
    for wide in ("0", "1", "2"):
        ref, got = arms[(wide, "0")], arms[(wide, "1")]
        assert ref.keys() == got.keys()
        for k in sorted(ref.keys()):
            total += 1
            if not torch.equal(ref[k], got[k]):
                diff = (ref[k].float() - got[k].float()).abs()
                nbad = int((ref[k] != got[k]).sum())
                fails.append(f"wide={wide} {k}: {nbad} differing elems, "
                             f"max|d|={diff.max().item():.6g}")

    tier = "(32,4) parity + (32,16) WIDE"
    if fails:
        print(f"OVERALL: FAIL — FastTree is NOT bit-identical ({len(fails)}/{total})")
        for f in fails[:20]:
            print("  " + f)
        return 1
    print(json.dumps({
        "gate": "addln_fasttree",
        "verdict": "PASS",
        "comparisons": total,
        "tiers": tier,
        "note": "torch.equal on x_new and y, FastTree=1 vs FastTree=0, both launcher tiers",
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
