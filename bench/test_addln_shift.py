#!/usr/bin/env python3
"""F0066 gate: fused add_ln_shift{6,1} == (add_ln WIDE -> shift_lerp{6,1}).

The fused kernel claims BYTE-EXACT composition: its add/stats/apply phases are
add_ln's WIDE config verbatim and its shift/lerp phase is shift_lerp*'s exact
rounding chain. So the gate is torch.equal on ALL THREE observable effects —
x_new, the lerp output, and the conv state after — vs running the two deployed
ops in sequence. Pads (ci = -1 / ci >= S) included: composition writes zeros to
the out rows and leaves conv untouched; the fused op must match.

RWKV_ADDLN_WIDE=1 is forced before the first kernel call (the C++ side caches
the env once) so the composition's add_ln takes the WIDE path the fused kernel
replicates.
"""
import os

os.environ["RWKV_ADDLN_WIDE"] = "1"  # must precede the first add_ln call

import torch  # noqa: E402

from sglang.srt.layers.attention.rwkv7_kernels import glue, ln_fused  # noqa: E402

assert ln_fused.available(), "rwkv7_ln JIT build failed"
assert glue.available(), "rwkv7_glue JIT build failed"

SHAPES = [(1, 4096), (1, 2048), (4, 4096), (32, 4096)]  # (T, N)
SEEDS = [0, 1, 2]
S_POOL = 48  # conv slots


def mk_ci(T, S, seed):
    """Distinct valid slots + a pad(-1) pattern (and an out-of-range ci on one)."""
    g = torch.Generator().manual_seed(seed + 991)
    perm = torch.randperm(S, generator=g)[:T].to(torch.int32)
    if T >= 2:
        perm[T - 1] = -1  # PAD_SLOT_ID
    if T >= 4:
        perm[T - 2] = S + 3  # out-of-range guard case
    return perm.cuda()


def run_case(T, N, seed, six: bool):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn((T, N), generator=g, device="cuda") * 0.7).half()
    d = (torch.randn((T, N), generator=g, device="cuda") * 0.7).half()
    ga = (torch.randn((N,), generator=g, device="cuda") * 0.5 + 1.0).half()
    be = (torch.randn((N,), generator=g, device="cuda") * 0.2).half()
    J = 6 if six else 1
    mixes = (torch.randn((J, N), generator=g, device="cuda") * 0.4).half()
    conv0 = (torch.randn((S_POOL, N, 1), generator=g, device="cuda") * 0.6).float()
    ci = mk_ci(T, S_POOL, seed)
    eps = 1e-5

    # Path A: composition (deployed ops, WIDE add_ln)
    conv_a = conv0.clone()
    xn_a, y = torch.ops.rwkv7_ln.add_ln(x, d, ga, be, eps)
    if six:
        out_a = torch.ops.rwkv7_glue.shift_lerp6(y, mixes, ci, conv_a)
    else:
        out_a = torch.ops.rwkv7_glue.shift_lerp1(y, mixes.view(-1), ci, conv_a)

    # Path B: fused
    conv_b = conv0.clone()
    if six:
        xn_b, out_b = torch.ops.rwkv7_ln.add_ln_shift6(x, d, ga, be, eps,
                                                       mixes, ci, conv_b)
    else:
        xn_b, out_b = torch.ops.rwkv7_ln.add_ln_shift1(x, d, ga, be, eps,
                                                       mixes.view(-1), ci, conv_b)

    ok = (torch.equal(xn_a, xn_b) and torch.equal(out_a, out_b)
          and torch.equal(conv_a, conv_b))
    tag = f"J={J} T={T} N={N} seed={seed}"
    if not ok:
        print(f"FAIL {tag}: x_new={torch.equal(xn_a, xn_b)} "
              f"out={torch.equal(out_a, out_b)} conv={torch.equal(conv_a, conv_b)}")
        if not torch.equal(out_a, out_b):
            diff = (out_a.float() - out_b.float()).abs()
            print(f"     out max|diff|={diff.max().item()} at "
                  f"{(diff > 0).sum().item()} positions")
    return ok


def main() -> int:
    fails = 0
    for six in (True, False):
        for T, N in SHAPES:
            for seed in SEEDS:
                if not run_case(T, N, seed, six):
                    fails += 1
    print(f"OVERALL {'FAIL' if fails else 'PASS'} "
          f"({len(SHAPES) * len(SEEDS) * 2 - fails}/{len(SHAPES) * len(SEEDS) * 2})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
