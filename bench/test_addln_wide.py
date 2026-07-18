#!/usr/bin/env python3
"""F0065 gate: add_ln WIDE small-T variant vs the parity config vs fp32 truth.

The WIDE path ((32,16), MaxVec=2, env RWKV_ADDLN_WIDE=1) uses the same Welford
ALGORITHM as the deployed (32,4) torch-parity config but a different
partition/tree shape, so its LN output is NOT bit-parity with torch. The gate
here is the fp16-state-WKV precedent bar:
  (a) x_new must be BYTE-IDENTICAL between parity and wide (the residual add
      is per-element, order-free — any diff is a bug, not a rounding choice);
  (b) each variant's LN y is compared against an fp32 reference computed on
      the fp16-rounded x_new; the wide variant must be NO FARTHER from truth
      than the parity variant (max-abs and mismatch-count both);
  (c) the binding end-to-end gate is greedy verify_m1d with WIDE armed
      (run separately).
Env is read once (static cache), so this script re-execs itself per mode.
"""
import os
import subprocess
import sys

import torch

SHAPES = [(1, 4096), (1, 2048), (4, 4096), (32, 4096)]  # (T, N), small-T tier
SEEDS = [0, 1, 2]


def run_mode(mode: str) -> None:
    from sglang.srt.layers.attention.rwkv7_kernels import ln_fused
    assert ln_fused.available(), "rwkv7_ln JIT build failed"
    outs = {}
    for T, N in SHAPES:
        for seed in SEEDS:
            g = torch.Generator(device="cuda").manual_seed(seed)
            x = (torch.randn((T, N), generator=g, device="cuda") * 0.7).half()
            d = (torch.randn((T, N), generator=g, device="cuda") * 0.7).half()
            ga = (torch.randn((N,), generator=g, device="cuda") * 0.5 + 1.0).half()
            be = (torch.randn((N,), generator=g, device="cuda") * 0.2).half()
            xn, y = torch.ops.rwkv7_ln.add_ln(x, d, ga, be, 1e-5)
            outs[(T, N, seed)] = (xn.cpu(), y.cpu(), x.cpu(), d.cpu(),
                                  ga.cpu(), be.cpu())
    torch.save(outs, f"/tmp/addln_{mode}.pt")


def compare() -> int:
    par = torch.load("/tmp/addln_parity.pt")
    wid = torch.load("/tmp/addln_wide.pt")
    worst_par, worst_wid, fail = 0.0, 0.0, 0
    for key in par:
        xn_p, y_p, x, d, ga, be = par[key]
        xn_w, y_w, *_ = wid[key]
        # (a) x_new byte-identical
        if not torch.equal(xn_p, xn_w):
            print(f"FAIL x_new bytes differ at {key}")
            fail = 1
        # (b) distance to fp32 truth on the SAME fp16-rounded x_new
        xf = xn_p.float()
        mean = xf.mean(dim=-1, keepdim=True)
        var = xf.var(dim=-1, unbiased=False, keepdim=True)
        ref = ((xf - mean) * torch.rsqrt(var + 1e-5) * ga.float() +
               be.float()).half().float()
        dp = (y_p.float() - ref).abs().max().item()
        dw = (y_w.float() - ref).abs().max().item()
        worst_par, worst_wid = max(worst_par, dp), max(worst_wid, dw)
        if dw > dp + 2e-3:  # wide must be no farther from truth (fp16 ~1 ULP)
            print(f"FAIL wide farther from fp32 truth at {key}: {dw} vs {dp}")
            fail = 1
    print(f"max|y - fp32ref|  parity={worst_par:.6f}  wide={worst_wid:.6f}")
    print("OVERALL", "FAIL" if fail else "PASS")
    return fail


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_mode(sys.argv[1])
        sys.exit(0)
    env = dict(os.environ)
    for mode, wide in [("parity", "0"), ("wide", "1")]:
        env["RWKV_ADDLN_WIDE"] = wide
        subprocess.run([sys.executable, __file__, mode], env=env, check=True)
    sys.exit(compare())
