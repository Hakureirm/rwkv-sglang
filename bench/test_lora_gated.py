#!/usr/bin/env python3
"""F0066c gate: lora4_m1_gated == lora4_m1 -> fused_lora_gates, byte-for-byte.

The gated kernel claims: rows (w_log, a, g_raw[, v_new]) identical to running
lora4_m1 then the triton _lora_gates_kernel (g is the raw lo row in both).
Gate = torch.equal per row, C in {3, 4} (layer0 / layer>0), realistic rank
splits, several seeds.
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

    # Path A: composition (deployed ops)
    lo = lora_fused.lora4_m1(xs, d_cat, u_cat, bias, meta)
    w_a, a_a, v_a = fused_lora_gates(lo, v, vf, has_v)
    g_a = lo[2:3]

    # Path B: gated
    yb = lora_fused.lora4_m1_gated(
        xs, d_cat, u_cat, bias, meta,
        v.reshape(-1).contiguous(), vf.reshape(-1).contiguous(), _INV_SQRT_E)
    w_b, a_b, g_b = yb[0:1], yb[1:2], yb[2:3]

    ok = (torch.equal(w_a, w_b) and torch.equal(a_a, a_b)
          and torch.equal(g_a, g_b))
    if has_v:
        ok = ok and torch.equal(v_a, yb[3:4])
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
