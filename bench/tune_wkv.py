# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Tune the WKV decode kernel (BV / num_warps) for BATCHED decode (the bsz32 bottleneck).

Profiling showed the WKV recurrence is the only decode component that scales with batch
(7.2B: 11.5us@bsz1 -> 248us@bsz32/layer, ~27% of state bandwidth). The launcher pins
BV=32,num_warps=4 for bit-reproducibility; this sweeps (BV,num_warps) at bsz32, timing each
and checking it is BIT-IDENTICAL to the pinned config (so greedy-EXACT + verify_batch stay
valid). Reports the fastest bit-identical config.

Usage (box): ~/envs/rwkv-sgl/bin/python bench/tune_wkv.py
"""
import sys, itertools
import torch
from sglang.srt.layers.attention.rwkv7_kernels.wkv_recurrent import wkv_recurrent

DEV = "cuda"


def bench(fn, iters=200):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def main():
    torch.manual_seed(0)
    # 7.2B decode: bsz=32 sequences, T=1, H=64 heads, head_dim=64
    B, T, H, D = 32, 1, 64, 64
    dt = torch.bfloat16
    mk = lambda V=D: torch.randn(B, T, H, V, device=DEV, dtype=dt)
    r, w, k, v, a = mk(), (-torch.rand(B, T, H, D, device=DEV, dtype=dt)), mk(), mk(), torch.sigmoid(mk())
    kk = mk(); kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    s0 = torch.randn(B, H, D, D, device=DEV, dtype=torch.float32)

    def run(bv, nw):
        return wkv_recurrent(r, w, k, v.clone(), kk, a, scale=1.0, initial_state=s0,
                             output_final_state=True, _bv=bv, _nw=nw)

    o_ref, s_ref = run(32, 4)  # the pinned config
    t_ref = bench(lambda: run(32, 4))
    print(f"7.2B bsz32 WKV decode — pinned BV=32,nw=4: {t_ref:.2f} us  (baseline)")
    print(f"{'BV':>3} {'nw':>3} {'us':>8} {'vs_ref':>7} {'bit-identical':>13}")
    best = (t_ref, 32, 4)
    for bv, nw in itertools.product([16, 32, 64], [1, 2, 4, 8]):
        try:
            o, st = run(bv, nw)
            t = bench(lambda: run(bv, nw))
        except Exception as ex:
            print(f"{bv:>3} {nw:>3}  {'ERR':>7}  {str(ex)[:40]}"); continue
        bit = bool(torch.equal(o, o_ref) and torch.equal(st, s_ref))
        maxerr = (o.float() - o_ref.float()).abs().max().item()
        print(f"{bv:>3} {nw:>3} {t:>8.2f} {t_ref/t:>6.2f}x {str(bit):>13}  (maxerr {maxerr:.1e})")
        if bit and t < best[0]:
            best = (t, bv, nw)
    print(f"\nFASTEST bit-identical: BV={best[1]} nw={best[2]} @ {best[0]:.2f} us "
          f"({t_ref/best[0]:.2f}x vs pinned)")


if __name__ == "__main__":
    main()
