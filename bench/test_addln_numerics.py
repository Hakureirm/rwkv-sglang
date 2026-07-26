# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Numerics gate for the add_ln tier ladder + FastTree (F0068 / task #57).

Self-contained: loads rwkv7_ln.cu directly, so it gates the kernel WITHOUT
touching the container's deployed sglang tree (bench/test_ln_fused.py imports
the installed package and would need the overlay copied in first).

Two bars, matching what each change is allowed to claim:

  TIER 0 (torch-parity (32,4) partition) — the transcription contract:
      BYTE-EXACT vs (x + delta) then torch nn.LayerNorm. Must hold at
      FastTree=0 AND FastTree=1 (FastTree must not disturb the parity tier).

  TIER 1/2 (F0065 WIDE (32,16), F0068 WIDER (32,32)) — a DIFFERENT Welford
      partition, so byte-parity with torch is not the bar and never was. The
      F0065 bar is used instead: y must sit NO FARTHER from the fp32 reference
      truth than the parity tier's own y does (max + mean abs error), i.e. the
      rewrite may not degrade accuracy relative to the shipped baseline.

Usage: python3 test_addln_numerics.py --cuda-dir <rwkv7_kernels/cuda>
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

CASES = [(1, 2048), (1, 4096), (4, 4096), (1, 8192)]
KINDS = ("uniform", "heavy", "tiny")
EPS = 1e-5


def _mk(shape, gen, kind):
    if kind == "uniform":
        t = torch.rand(shape, generator=gen, device="cuda") * 4 - 2
    elif kind == "heavy":
        t = torch.randn(shape, generator=gen, device="cuda") * (
            10 ** (torch.rand(shape, generator=gen, device="cuda") * 4 - 2))
    else:  # subnormal-adjacent
        t = torch.randn(shape, generator=gen, device="cuda") * 6e-5
    return t.to(torch.float16)


def _inputs(T, N, kind):
    gen = torch.Generator(device="cuda").manual_seed(20260727 + N * 7 + T)
    x = _mk((T, N), gen, kind)
    delta = _mk((T, N), gen, kind)
    gamma = _mk((N,), gen, "uniform")
    beta = _mk((N,), gen, "uniform")
    return x, delta, gamma, beta


def run_arm(cuda_dir: str, out_path: str):
    from torch.utils.cpp_extension import load
    load(name="rwkv7_ln", sources=[str(Path(cuda_dir) / "rwkv7_ln.cu")],
         is_python_module=False, verbose=False,
         extra_cflags=["-O3"], extra_cuda_cflags=["-O3"])
    res = {}
    for (T, N), kind in itertools.product(CASES, KINDS):
        x, delta, gamma, beta = _inputs(T, N, kind)
        x_new, y = torch.ops.rwkv7_ln.add_ln(x, delta, gamma, beta, EPS)
        res[f"{T}x{N}:{kind}:x_new"] = x_new.cpu()
        res[f"{T}x{N}:{kind}:y"] = y.cpu()
    torch.save(res, out_path)


def references():
    """torch fp16 reference (the byte contract) + fp32 truth (the accuracy bar)."""
    ref, truth = {}, {}
    for (T, N), kind in itertools.product(CASES, KINDS):
        x, delta, gamma, beta = _inputs(T, N, kind)
        xn = (x + delta)  # fp16 add, exactly what the kernel's first phase does
        ln = torch.nn.LayerNorm(N, eps=EPS, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            ln.weight.copy_(gamma)
            ln.bias.copy_(beta)
            y = ln(xn)
        ref[f"{T}x{N}:{kind}:x_new"] = xn.cpu()
        ref[f"{T}x{N}:{kind}:y"] = y.cpu()
        # fp32 truth from the SAME fp16 x_new (isolates the LN reduction, not the add)
        xf = xn.float()
        mu = xf.mean(-1, keepdim=True)
        var = xf.var(-1, unbiased=False, keepdim=True)
        truth[f"{T}x{N}:{kind}:y"] = (
            (xf - mu) * torch.rsqrt(var + EPS) * gamma.float() + beta.float()).cpu()
    return ref, truth


def _bytes_diff(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-dir", required=True)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    if args.dump:
        run_arm(args.cuda_dir, args.dump)
        return 0

    tmp = tempfile.mkdtemp(prefix="addln_numerics_")
    arms = {}
    for wide, fast in itertools.product(("0", "1", "2"), ("0", "1")):
        path = os.path.join(tmp, f"w{wide}_f{fast}.pt")
        env = dict(os.environ, RWKV_ADDLN_WIDE=wide, RWKV_ADDLN_FASTTREE=fast)
        r = subprocess.run([sys.executable, __file__, "--cuda-dir", args.cuda_dir,
                            "--dump", path], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[FAIL] arm wide={wide} fast={fast}:\n{r.stdout}\n{r.stderr}")
            return 1
        arms[(wide, fast)] = torch.load(path)

    ref, truth = references()
    report, failed = [], False

    # --- bar 1: tier 0 must be byte-exact vs torch, at both FastTree settings
    for fast in ("0", "1"):
        nbad = sum(_bytes_diff(arms[("0", fast)][k], ref[k]) for k in ref)
        ok = nbad == 0
        failed |= not ok
        report.append({"bar": "tier0_byte_vs_torch", "fasttree": fast,
                       "differing_bytes": nbad, "verdict": "PASS" if ok else "FAIL"})

    # --- bar 2: tiers 1/2 no farther from fp32 truth than tier 0.
    # Cases whose fp16 output is byte-identical to parity are reported as such
    # (that is the STRONGEST outcome, not a skip): the partition change did not
    # move a single bit there, so there is nothing to compare. The remaining
    # cases must not be farther from truth. Bars: max error strictly not worse;
    # mean error with a relative tolerance, because the mean of |err| is itself
    # an fp32/fp64 accumulation whose last bits are order-dependent noise (the
    # observed deltas are ~1e-11 on errors of ~1e-3 — 8 orders down).
    MEAN_RTOL, MEAN_ATOL = 1e-6, 1e-12
    for wide in ("1", "2"):
        for fast in ("0", "1"):
            worst_max, worst_mean = 0.0, 0.0
            checked, identical, offenders = 0, 0, []
            for k in truth:
                if torch.equal(arms[(wide, fast)][k], arms[("0", fast)][k]):
                    identical += 1
                    continue
                base = arms[("0", fast)][k].float()
                got = arms[(wide, fast)][k].float()
                t = truth[k]
                e_base_max = (base - t).abs().max().item()
                e_got_max = (got - t).abs().max().item()
                e_base_mean = (base - t).abs().mean().item()
                e_got_mean = (got - t).abs().mean().item()
                dmax = e_got_max - e_base_max
                dmean = e_got_mean - e_base_mean
                worst_max, worst_mean = max(worst_max, dmax), max(worst_mean, dmean)
                if dmax > 0.0 or dmean > e_base_mean * MEAN_RTOL + MEAN_ATOL:
                    offenders.append({"case": k, "d_max": dmax, "d_mean": dmean,
                                      "parity_max": e_base_max})
                checked += 1
            ok = not offenders
            failed |= not ok
            report.append({"bar": "tier_equidistant_from_fp32_truth",
                           "tier": wide, "fasttree": fast,
                           "cases_byte_identical_to_parity": identical,
                           "cases_compared_numerically": checked,
                           "worst_max_err_delta_vs_parity": worst_max,
                           "worst_mean_err_delta_vs_parity": worst_mean,
                           "offenders": offenders,
                           "verdict": "PASS" if ok else "FAIL"})

    print(json.dumps({"gate": "addln_numerics",
                      "overall": "FAIL" if failed else "PASS",
                      "results": report}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
