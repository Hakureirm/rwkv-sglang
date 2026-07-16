#!/usr/bin/env python3
"""Stage-A megakernel increment gate + microbench (task #50 / F0060).

gemv_rkv_m1 (rwkv7_mega.cu) packs the r/k/v decode GEMVs into ONE launch.
Two checks:
  1. BIT-EXACT gate: grouped output == stack of 3 gemv_m1 (torch.equal, zero
     differing bytes), across shapes + input families + both parities.
  2. Microbench: eager (launch-visible) and graphed (pure GPU-busy) device time,
     grouped vs 3 separate launches — graphed is the production-relevant delta.

Run in the rwkvmain container:
  python bench/test_mega_rkv.py
"""
import torch

from sglang.srt.layers.attention.rwkv7_kernels import fast_linear, mega

assert fast_linear.available(), "rwkv7_fast failed to build"
assert mega.available(), "rwkv7_mega failed to build"

dev = torch.device("cuda")
arch = torch.cuda.get_device_capability()
print(f"# device={torch.cuda.get_device_name(0)} cap={arch}")


def oracle(xr, xk, xv, wr, wk, wv, t, ot):
    r = torch.ops.rwkv7_fast.gemv_m1_cfg(xr.view(1, -1), wr, t, ot)
    k = torch.ops.rwkv7_fast.gemv_m1_cfg(xk.view(1, -1), wk, t, ot)
    v = torch.ops.rwkv7_fast.gemv_m1_cfg(xv.view(1, -1), wv, t, ot)
    return torch.cat([r, k, v], dim=0)  # [3, N]


def mk(N, K, scale, gen):
    def one(shape):
        if gen == "uniform":
            return (torch.randn(shape, device=dev, dtype=torch.float16) * scale)
        if gen == "heavy":  # heavy-tailed
            return (torch.randn(shape, device=dev, dtype=torch.float16) ** 3 * scale)
        raise ValueError(gen)
    xr, xk, xv = one((1, K)), one((1, K)), one((1, K))
    wr, wk, wv = one((N, K)), one((N, K)), one((N, K))
    return xr, xk, xv, wr, wk, wv


# ---------------------------------------------------------------- gate
print("\n## BIT-EXACT GATE (torch.equal vs 3x gemv_m1)")
SHAPES = [
    ("1.5B r/k/v", 2048, 2048),
    ("7.2B r/k/v", 4096, 4096),
    ("0.1B r/k/v", 768, 768),
    ("odd-N", 6, 2048),        # OutTile=1 path
    ("small-K", 2048, 8),
]
all_ok = True
for name, N, K in SHAPES:
    t, ot = mega.rkv_config(N, K)
    for gen in ("uniform", "heavy"):
        for scale in (0.5, 2.0, 8.0):
            xr, xk, xv, wr, wk, wv = mk(N, K, scale, gen)
            got = torch.ops.rwkv7_mega.gemv_rkv_m1(
                xr.view(-1), xk.view(-1), xv.view(-1), wr, wk, wv, t, ot)
            exp = oracle(xr, xk, xv, wr, wk, wv, t, ot)
            ok = torch.equal(got, exp)
            all_ok &= ok
            if not ok:
                nd = (got != exp).sum().item()
                print(f"  FAIL {name:12s} cfg=({t},{ot}) {gen}/{scale}: "
                      f"{nd} differing / {got.numel()}")
    print(f"  {'PASS' if all_ok else 'FAIL'} {name:12s} N={N} K={K} cfg=({t},{ot})")
print(f"\nGATE (rkv): {'PASS (zero differing bytes)' if all_ok else 'FAIL'}")


# ---- Stage-A2 gate: o_proj as a role (G=1) + whole-block r/k/v/o (G=4) ------
# o_proj is another M==1 [N,K]·[1,K]^T GEMV with (N,K)=(H,H) like r/k/v, so it
# takes the SAME _select_config; gemv_o_m1[0] must be byte-identical to
# gemv_m1(xo, wo), and gemv_rkvo_m1's 4 rows to [r,k,v,o] stacked.
print("\n## STAGE-A2 GATE (o_proj role G=1 + whole-block r/k/v/o G=4)")
o_ok = rkvo_ok = True
for name, N, K in SHAPES:
    t, ot = mega.rkv_config(N, K)
    for gen in ("uniform", "heavy"):
        for scale in (0.5, 2.0, 8.0):
            xr, xk, xv, wr, wk, wv = mk(N, K, scale, gen)
            xo, wo = mk(N, K, scale, gen)[0], mk(N, K, scale, gen)[3]
            # G=1: o_proj alone
            go = torch.ops.rwkv7_mega.gemv_o_m1(xo.view(-1), wo, t, ot)
            eo = torch.ops.rwkv7_fast.gemv_m1_cfg(xo.view(1, -1), wo, t, ot)
            o_ok &= torch.equal(go, eo)
            # G=4: r/k/v/o in one launch
            g4 = torch.ops.rwkv7_mega.gemv_rkvo_m1(
                xr.view(-1), xk.view(-1), xv.view(-1), xo.view(-1),
                wr, wk, wv, wo, t, ot)
            e4 = torch.cat([oracle(xr, xk, xv, wr, wk, wv, t, ot), eo], dim=0)
            rkvo_ok &= torch.equal(g4, e4)
    print(f"  {'PASS' if o_ok and rkvo_ok else 'FAIL'} {name:12s} "
          f"N={N} K={K} cfg=({t},{ot})  o(G1)={o_ok} rkvo(G4)={rkvo_ok}")
print(f"\nGATE (o G=1): {'PASS (zero differing bytes)' if o_ok else 'FAIL'}")
print(f"GATE (rkvo G=4): {'PASS (zero differing bytes)' if rkvo_ok else 'FAIL'}")
all_ok = all_ok and o_ok and rkvo_ok


# ---------------------------------------------------------------- microbench
def bench_eager(fn, n=200, warmup=50, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        s.record()
        for _ in range(n):
            fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / n)
    return best * 1000  # us/call


def bench_graph(fn, n=200, warmup=50, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    torch.cuda.synchronize()
    best = float("inf")
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / n)
    return best * 1000  # us/call


print("\n## MICROBENCH (us per r/k/v block; graphed = production-relevant)")
print(f"{'shape':14s} {'cfg':8s} {'sep_eager':>10s} {'grp_eager':>10s} "
      f"{'sep_grph':>10s} {'grp_grph':>10s} {'grph_delta':>11s}")
for name, N, K in [("1.5B r/k/v", 2048, 2048), ("7.2B r/k/v", 4096, 4096)]:
    t, ot = mega.rkv_config(N, K)
    xr, xk, xv, wr, wk, wv = mk(N, K, 1.0, "uniform")
    xrv, xkv, xvv = xr.view(-1), xk.view(-1), xv.view(-1)

    def sep():
        torch.ops.rwkv7_fast.gemv_m1_cfg(xr, wr, t, ot)
        torch.ops.rwkv7_fast.gemv_m1_cfg(xk, wk, t, ot)
        torch.ops.rwkv7_fast.gemv_m1_cfg(xv, wv, t, ot)

    def grp():
        torch.ops.rwkv7_mega.gemv_rkv_m1(xrv, xkv, xvv, wr, wk, wv, t, ot)

    se, ge = bench_eager(sep), bench_eager(grp)
    sg, gg = bench_graph(sep), bench_graph(grp)
    print(f"{name:14s} ({t},{ot})  {se:10.2f} {ge:10.2f} {sg:10.2f} {gg:10.2f} "
          f"{sg - gg:+10.2f}")
