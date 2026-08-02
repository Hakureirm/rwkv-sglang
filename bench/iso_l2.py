#!/usr/bin/env python3
"""Tune two lattices to the SAME total weight error with opposite origin resolution.

Every arm so far moved total error and origin spacing together, because the error-optimal
allocation of levels is itself "fine where the mass is" -- so the two accounts of why fp4
beats int4 have never been pulled apart. Holding error fixed and moving only the gap at
zero is the arm that does pull them apart:

  same rel. error  ->  the error account predicts the two land on top of each other
  5x apart at zero ->  the origin account predicts the fine one wins by several points

Both sit well above int4's error, so both are expected to score below it. The comparison
that matters is between them, not against int4.
"""
import glob, os, sys
import torch
from safetensors.torch import load_file

TARGET_SUFFIXES = ("r_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                   ".key.weight", ".value.weight")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "/data/rwkv-sglang/models/rwkv7-1.5b-fla"
TARGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1.33


def load_mats():
    mats = []
    for sf in sorted(glob.glob(os.path.join(MODEL, "*.safetensors"))):
        sd = load_file(sf)
        for name, W in sd.items():
            if W.ndim == 2 and name.endswith(TARGET_SUFFIXES) and W.shape[1] % 64 == 0:
                N, K = W.shape
                mats.append(W.float().view(N, K // 64, 64))
    return mats


def rel_err(mats, pos):
    lat = torch.tensor(sorted([-p for p in pos] + [0.0] + list(pos)), dtype=torch.float32)
    err = norm = 0.0
    for Wg in mats:
        s = Wg.abs().amax(2).clamp(min=1e-8)[:, :, None]
        idx = ((Wg / s).unsqueeze(-1) - lat).abs().argmin(-1)
        err += (lat[idx] * s - Wg).pow(2).sum().item()
        norm += Wg.pow(2).sum().item()
    return (err / norm) ** 0.5


def fine(g):
    """Origin pinned fine at 0.05; `g` starves the tail until the error target is met.

    Refining the origin only ever lowers error, so a fine-origin lattice cannot be pushed
    to 1.33x from the origin end -- the extra error has to be bought somewhere else, and
    the tail is the only region left.
    """
    return [0.05, 0.10, 0.10 + g, 0.10 + 2 * g, 0.10 + 3 * g, 0.10 + 4 * g, 1.0]


def coarse(t):
    """t pushes the innermost level away from zero; upper levels stay dense."""
    return [t, t + 0.15, t + 0.28, t + 0.40, t + 0.52, (t + 0.52 + 1.0) / 2, 1.0]


def solve(mats, family, lo, hi, base, target, name):
    for _ in range(18):
        mid = (lo + hi) / 2
        r = rel_err(mats, family(mid)) / base
        if r < target:
            lo = mid
        else:
            hi = mid
    pos = family((lo + hi) / 2)
    r = rel_err(mats, pos) / base
    print(f"{name:8} gap@0={pos[0]:.4f}  rel.err={r:.4f}x  levels={[round(p, 4) for p in pos]}")
    return pos, r


def main():
    mats = load_mats()
    base = rel_err(mats, [1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7, 1.0])
    print(f"{len(mats)} matrices, int4 baseline rel.err {base:.6f}\n")
    solve(mats, fine, 0.20, 0.005, base, TARGET, "fine")
    solve(mats, coarse, 0.02, 0.45, base, TARGET, "coarse")


if __name__ == "__main__":
    main()
