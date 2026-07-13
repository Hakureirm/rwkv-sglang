#!/usr/bin/env python3
"""Byte-exactness gate for the W1' fused norm-boundary kernels (rwkv7_ln.cu).

Two ops, each compared BYTE-FOR-BYTE against the live reference stack it
replaces (not a numpy model of it - the actual torch / Triton kernels the
deployed fp16 path runs):

  add_ln      vs  (x + delta) followed by torch nn.LayerNorm    [both outputs]
  gn_gatecorr vs  torch nn.GroupNorm followed by fused.fused_gate_corr

plus a dedicated summation-TREE probe for gn_gatecorr that isolates the
Triton _gate_corr 64-wide fp32 reduction (gamma=0/beta=0 -> GroupNorm output
is 0; k=r_k=v=g=1 -> kernel output == fp16(tree_sum(r_row)) broadcast), so a
tree mismatch is reported as such instead of a generic byte diff.

Shapes cover the real model geometry (0.1B/1.5B/7.2B: H=768/2048/4096,
head_dim 64) x decode/extend batch sizes, uniform + heavy-tailed + subnormal
inputs. PASS requires ZERO differing bytes everywhere. Exit 0 iff all pass.

  python bench/test_ln_fused.py [--quick]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sglang.srt.layers.attention.rwkv7_kernels import ln_fused  # noqa: E402
from sglang.srt.layers.attention.rwkv7_kernels.fused import (  # noqa: E402
    fused_gate_corr,
)

DEV = "cuda"


def _mk(shape, gen, kind):
    if kind == "uniform":
        t = (torch.rand(shape, generator=gen, device=DEV) * 4 - 2)
    elif kind == "heavy":
        t = torch.randn(shape, generator=gen, device=DEV) * (
            10 ** (torch.rand(shape, generator=gen, device=DEV) * 4 - 2)
        )
    else:  # tiny: subnormal-adjacent magnitudes
        t = torch.randn(shape, generator=gen, device=DEV) * 6e-5
    return t.to(torch.float16)


def _diff_bytes(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum().item())


def test_add_ln(quick=False):
    gen = torch.Generator(device=DEV).manual_seed(20260713)
    fails = 0
    Ts = [1, 7, 320] if quick else [1, 2, 7, 64, 320, 1024, 4096]
    for H in (768, 2048, 4096):
        ln = torch.nn.LayerNorm(H, eps=1e-5).to(DEV, torch.float16)
        with torch.no_grad():
            ln.weight.copy_(_mk((H,), gen, "uniform").float() + 1.0)
            ln.bias.copy_(_mk((H,), gen, "uniform").float() * 0.5)
        for T in Ts:
            for kind in ("uniform", "heavy", "tiny"):
                x = _mk((T, H), gen, kind)
                d = _mk((T, H), gen, kind)
                xr = x + d
                yr = ln(xr)
                xf, yf = ln_fused.add_ln(x, d, ln)
                dx, dy = _diff_bytes(xr, xf), _diff_bytes(yr, yf)
                status = "OK " if dx == 0 and dy == 0 else "FAIL"
                if dx or dy:
                    fails += 1
                print(f"[add_ln] H={H:5d} T={T:5d} {kind:8s} "
                      f"x_new diff={dx} y diff={dy} {status}")
    return fails


def test_gn_gatecorr(quick=False):
    gen = torch.Generator(device=DEV).manual_seed(20260714)
    fails = 0
    Ts = [1, 7, 320] if quick else [1, 2, 7, 64, 320, 1024, 4096]
    for H, NH in ((768, 12), (2048, 32), (4096, 64)):
        HD = H // NH
        gn = torch.nn.GroupNorm(NH, H, eps=HD * 1e-5).to(DEV, torch.float16)
        with torch.no_grad():
            gn.weight.copy_(_mk((H,), gen, "uniform").float() + 1.0)
            gn.bias.copy_(_mk((H,), gen, "uniform").float() * 0.5)
        rk = _mk((NH, HD), gen, "uniform")
        for T in Ts:
            for kind in ("uniform", "heavy", "tiny"):
                o = _mk((T, H), gen, kind)
                r = _mk((T, NH, HD), gen, kind)
                k = _mk((T, NH, HD), gen, kind)
                v = _mk((T, NH, HD), gen, kind)
                g = _mk((T, H), gen, kind)
                o_norm = gn(o)
                ref = fused_gate_corr(o_norm, r, k, rk, v, g, NH)
                got = ln_fused.gn_gatecorr(o, r, k, rk, v, g, gn, NH)
                db = _diff_bytes(ref, got)
                status = "OK " if db == 0 else "FAIL"
                if db:
                    fails += 1
                print(f"[gn_gatecorr] H={H:5d} T={T:5d} {kind:8s} diff={db} {status}")
    return fails


def probe_sum_tree(trials=6, T=4096):
    """Isolate the 64-wide reduction: out == fp16(tree_sum(r_head_row))."""
    gen = torch.Generator(device=DEV).manual_seed(20260715)
    H, NH, HD = 4096, 64, 64
    gn = torch.nn.GroupNorm(NH, H, eps=HD * 1e-5).to(DEV, torch.float16)
    with torch.no_grad():
        gn.weight.zero_()
        gn.bias.zero_()
    ones = torch.ones(T, NH, HD, device=DEV, dtype=torch.float16)
    rk = torch.ones(NH, HD, device=DEV, dtype=torch.float16)
    g = torch.ones(T, H, device=DEV, dtype=torch.float16)
    mism = 0
    total = 0
    for _ in range(trials):
        o = _mk((T, H), gen, "uniform")
        r = _mk((T, NH, HD), gen, "heavy")
        ref = fused_gate_corr(gn(o), r, ones, rk, ones, g, NH)
        got = ln_fused.gn_gatecorr(o, r, ones, rk, ones, g, gn, NH)
        mism += _diff_bytes(ref, got)
        total += ref.numel()
    print(f"[sum-tree probe] rows={trials * T * NH} bytes_compared={total} "
          f"mismatches={mism} {'OK' if mism == 0 else 'FAIL'}")
    return 0 if mism == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    assert ln_fused.available(), "rwkv7_ln JIT build failed"
    fails = probe_sum_tree(trials=2 if args.quick else 6)
    fails += test_add_ln(args.quick)
    fails += test_gn_gatecorr(args.quick)
    print("OVERALL:", "PASS" if fails == 0 else f"FAIL ({fails} cases)")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
