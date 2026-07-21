#!/usr/bin/env python3
"""F0066c gate: lora4_m1_gated == lora4_m1 -> the TORCH gate chain, byte-exact.

REFERENCE CHOICE (probe-adjudicated, bench/results/f0066c/probe_sigmoid_bits.log):
the gate anchors to the TORCH chain (-sigmoid(lo0)*inv_sqrt_e, sigmoid(lo1),
raw lo2, v + (vf-v)*sigmoid(lo3)) — the project's original semantic baseline —
NOT to the deployed triton _lora_gates_kernel. A full 65536-pattern fp16
census showed the CUDA expf chain is bit-identical to torch.sigmoid on EVERY
finite pattern (0/65536 mismatches), while triton's tl.exp deviates from
torch/expf/__expf by 1 ULP on exactly 2/65536 rare patterns (its own gate
never sampled them). Emulating triton's anomalous bits would enshrine the
deviation; the gated kernel matches the reference everywhere instead. The
triton delta is still printed as an INFORMATIONAL census (expected tiny, not
a failure); greedy e2e remains the binding production gate.
"""
import torch

from sglang.srt.layers.attention.rwkv7_kernels import lora_fused
from sglang.srt.layers.attention.rwkv7_kernels.fused import fused_lora_gates

assert lora_fused.available(), "rwkv7_lora JIT build failed"

_INV_SQRT_E = 0.6065306597126334
H = 4096
CASES = [
    # (C, ranks) — pack order (w, a, g[, v]); ranks mirror 7.2B-class splits
    (4, [96, 96, 128, 64]),
    (3, [96, 96, 128]),
    (4, [64, 64, 96, 32]),
]
SEEDS = [0, 1, 2]


def run_case(C, ranks, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    Rtot = sum(ranks)
    xs = (torch.randn((C, H), generator=g, device="cuda") * 0.6).half()
    d_cat = (torch.randn((Rtot, H), generator=g, device="cuda") * 0.05).half()
    u_cat = (torch.randn((H, Rtot), generator=g, device="cuda") * 0.05).half()
    bias = (torch.randn((C, H), generator=g, device="cuda") * 0.3).half()
    meta_rows, off = [], 0
    for i, r in enumerate(ranks):
        act = 1 if i == 0 else 0  # w chain uses tanh on h (act=1), others none
        meta_rows.append([off, r, act])
        off += r
    meta = torch.tensor(meta_rows, dtype=torch.int32, device="cuda")
    v = (torch.randn((1, H), generator=g, device="cuda") * 0.6).half()
    vf = (torch.randn((1, H), generator=g, device="cuda") * 0.6).half()
    has_v = C == 4

    # Path A: lora4_m1 + the TORCH gate chain (the semantic reference — see
    # module docstring; replicates the model's non-fused fallback exactly)
    lo = lora_fused.lora4_m1(xs, d_cat, u_cat, bias, meta)
    w_a = -torch.sigmoid(lo[0:1]) * _INV_SQRT_E
    a_a = torch.sigmoid(lo[1:2])
    g_a = lo[2:3]
    v_a = v + (vf - v) * torch.sigmoid(lo[3:4]) if has_v else v

    # Informational census vs the deployed triton kernel (expected: tiny 1-ULP
    # deltas on the 2/65536 anomalous patterns, when sampled)
    w_t, a_t, v_t = fused_lora_gates(lo, v, vf, has_v)
    tri_delta = int((w_t != w_a).sum() + (a_t != a_a).sum()
                    + ((v_t != v_a).sum() if has_v else 0))

    # Path B: gated
    yb = lora_fused.lora4_m1_gated(
        xs, d_cat, u_cat, bias, meta,
        v.reshape(-1).contiguous(), vf.reshape(-1).contiguous(), _INV_SQRT_E)
    w_b, a_b, g_b = yb[0:1], yb[1:2], yb[2:3]

    ok = (torch.equal(w_a, w_b) and torch.equal(a_a, a_b)
          and torch.equal(g_a, g_b))
    if has_v:
        ok = ok and torch.equal(v_a, yb[3:4])
    if tri_delta:
        print(f"  (info) triton-vs-torch delta elements C={C} seed={seed}: {tri_delta}")
    if not ok:
        print(f"FAIL C={C} ranks={ranks} seed={seed}: "
              f"w={torch.equal(w_a, w_b)} a={torch.equal(a_a, a_b)} "
              f"g={torch.equal(g_a, g_b)}"
              + (f" v={torch.equal(v_a, yb[3:4])}" if has_v else ""))
    return ok


def main() -> int:
    fails = sum(not run_case(C, r, s) for C, r in CASES for s in SEEDS)
    total = len(CASES) * len(SEEDS)
    print(f"OVERALL {'FAIL' if fails else 'PASS'} ({total - fails}/{total})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
