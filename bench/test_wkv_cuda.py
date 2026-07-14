# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bit-exactness + batch-invariance gate for the hand-CUDA WKV decode kernel
(task #54, rwkv7_kernels/cuda/rwkv7_wkv.cu) vs the Triton `_wkv_recurrent_kernel`.

Contract under test (kernel header, wkv_cuda.py):
  * ZERO differing bytes on the o output AND the state pool, per state dtype
    (fp32 pool = bitwise-oracle tier; fp16 pool = RWKV_STATE_FP16 tier), for
    the serving decode shape (T==1, in-place indexed pool), pad rows included.
  * Batch-invariance: any row of a bs=N launch is bit-identical to the same
    request launched at bs=1 (per-row reduction independence).
  * Multi-step: 64 chained decode steps stay byte-identical (carried-state
    round-trip drift would compound; none is allowed).

The Triton reference runs its PINNED decode config (BV=32, num_warps=4) - the
association tree the CUDA kernel transcribes. RWKV_WKV_FP16_CFG must be unset
(it re-tiles the Triton fp16-state path only; the CUDA kernel always carries
the pinned tree). Run on the box:
  ~/envs/rwkv-sgl/bin/python bench/test_wkv_cuda.py
"""
import os
import sys

# The CUDA path must not be routed around, and the Triton reference must be the
# pinned config: neutralize the two envs BEFORE importing the launcher.
os.environ.pop("RWKV_WKV_FP16_CFG", None)
os.environ["RWKV_WKV_CUDA"] = "0"  # gate calls each side explicitly

import torch

from sglang.srt.layers.attention.rwkv7_kernels import wkv_cuda
from sglang.srt.layers.attention.rwkv7_kernels.wkv_recurrent import wkv_recurrent

DEV = "cuda"
FAILS = []


def bits_equal(x, y):
    if x.dtype == torch.float16:
        return bool((x.view(torch.int16) == y.view(torch.int16)).all())
    return bool((x.view(torch.int32) == y.view(torch.int32)).all())


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


def make_inputs(B, H, kind, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    D = 64
    mk = lambda s=1.0: torch.randn(B, 1, H, D, device=DEV, dtype=torch.float16, generator=g) * s
    if kind == "uniform":
        r, k, v, kkr, a = mk(), mk(), mk(), mk(), torch.sigmoid(mk())
        w = -torch.rand(B, 1, H, D, device=DEV, dtype=torch.float16, generator=g) * 0.6 - 0.05
    elif kind == "heavy":
        r, k, v = mk(8.0), mk(8.0), mk(8.0)
        kkr, a = mk(4.0), torch.sigmoid(mk(4.0))
        w = -torch.rand(B, 1, H, D, device=DEV, dtype=torch.float16, generator=g) * 12.0
    elif kind == "edge":
        # exact zeros in kk (sign-of-zero paths), w == 0 (decay == 1), tiny values
        r, k, v = mk(), mk(1e-4), mk(1e-4)
        kkr = mk()
        kkr[:, :, :, ::3] = 0.0
        a = torch.sigmoid(mk())
        w = torch.zeros(B, 1, H, D, device=DEV, dtype=torch.float16)
    else:  # subnormal-heavy states come from the pool init; inputs mild
        r, k, v, kkr, a = mk(0.5), mk(0.5), mk(0.5), mk(0.5), torch.sigmoid(mk())
        w = -torch.rand(B, 1, H, D, device=DEV, dtype=torch.float16, generator=g) * 0.3
    kk = kkr / kkr.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).to(torch.float16)
    if kind == "edge":
        kk = torch.nan_to_num(kk, nan=0.0)
        kk[:, :, :, ::3] = 0.0
    return r, w, k, v, kk, a


def make_pool(slots, H, sdt, kind, seed):
    g = torch.Generator(device=DEV).manual_seed(seed + 777)
    pool = torch.randn(slots, H, 64, 64, device=DEV, dtype=sdt, generator=g) * 0.5
    if kind == "subnormal":
        tiny = torch.randn(slots, H, 64, 64, device=DEV, dtype=torch.float32, generator=g) * 1e-7
        pool.copy_(tiny.to(sdt))
    elif kind == "heavy":
        pool.mul_(64.0)
    return pool


def run_pair(B, H, sdt, kind, seed, ci_kind="shuffle", steps=1):
    slots = max(64, B + 3)
    pool_t = make_pool(slots, H, sdt, kind, seed)
    pool_c = pool_t.clone()
    if ci_kind == "shuffle":
        ci = torch.randperm(slots - 1, device=DEV)[:B].int() + 1
    elif ci_kind == "pads":
        ci = torch.randperm(slots - 1, device=DEV)[:B].int() + 1
        ci[0] = -1                      # PAD_SLOT_ID
        if B > 2:
            ci[B // 2] = slots          # >= pool size sentinel
        ci[B - 1] = -1
    else:
        ci = torch.arange(1, B + 1, device=DEV, dtype=torch.int32)
    ok_o, ok_s = True, True
    for s in range(steps):
        r, w, k, v, kk, a = make_inputs(B, H, kind, seed + s)
        o_t, _ = wkv_recurrent(r, w, k, v, kk, a, scale=1.0,
                               state_pool=pool_t, cache_indices=ci)
        o_c = wkv_cuda.wkv_decode(r, w, k, v, kk, a, pool_c, ci, 1.0)
        ok_o = ok_o and bits_equal(o_t, o_c)
        ok_s = ok_s and bits_equal(pool_t, pool_c)
        if not (ok_o and ok_s):
            do = (o_t.float() - o_c.float()).abs()
            ds = (pool_t.float() - pool_c.float()).abs()
            print(f"    step {s}: o maxdiff {do.max().item():.3e} "
                  f"({(o_t.view(torch.int16) != o_c.view(torch.int16)).sum().item()} bytes) "
                  f"pool maxdiff {ds.max().item():.3e}")
            break
    return ok_o and ok_s


def main():
    if not wkv_cuda.available():
        print("FAIL: rwkv7_wkv extension did not build")
        sys.exit(1)
    torch.manual_seed(0)

    for sdt, tag in ((torch.float32, "fp32-state"), (torch.float16, "fp16-state")):
        print(f"[{tag}] bit-exactness vs Triton (o + pool bytes)")
        for B, H in ((1, 64), (2, 64), (3, 32), (24, 64), (320, 64), (512, 64), (64, 32)):
            for kind in ("uniform", "heavy", "edge", "subnormal"):
                ok = run_pair(B, H, sdt, kind, seed=B * 131 + len(kind))
                check(f"{tag} B={B} H={H} {kind}", ok)
        print(f"[{tag}] pad slots (-1 and >=size sentinels)")
        for B in (2, 24, 320):
            ok = run_pair(B, 64, sdt, "uniform", seed=9000 + B, ci_kind="pads")
            check(f"{tag} B={B} pads", ok)
        print(f"[{tag}] 64-step chained recurrence (carried-state drift)")
        ok = run_pair(24, 64, sdt, "uniform", seed=4242, steps=64)
        check(f"{tag} 64-step chain", ok)

        # batch-invariance probe: row j of a bs=320 launch == the same request
        # launched alone (same slot, same inputs), torch.equal on o and slot.
        print(f"[{tag}] batch-invariance probe (row of bs=320 == bs=1)")
        B, H, slots = 320, 64, 345
        pool_a = make_pool(slots, H, sdt, "uniform", 31337)
        pool_b = pool_a.clone()
        ci = torch.randperm(slots - 1, device=DEV)[:B].int() + 1
        r, w, k, v, kk, a = make_inputs(B, H, "uniform", 31337)
        o_all = wkv_cuda.wkv_decode(r, w, k, v, kk, a, pool_a, ci, 1.0)
        ok = True
        for j in (0, 7, 160, 319):
            o_one = wkv_cuda.wkv_decode(
                r[j : j + 1].contiguous(), w[j : j + 1].contiguous(),
                k[j : j + 1].contiguous(), v[j : j + 1].contiguous(),
                kk[j : j + 1].contiguous(), a[j : j + 1].contiguous(),
                pool_b, ci[j : j + 1].contiguous(), 1.0)
            ok = ok and torch.equal(o_all[j : j + 1], o_one)
            ok = ok and torch.equal(pool_a[ci[j].item()], pool_b[ci[j].item()])
        check(f"{tag} batch-invariance", ok)

    if FAILS:
        print(f"\nOVERALL FAIL ({len(FAILS)}): {FAILS[:8]}")
        sys.exit(1)
    print("\nOVERALL PASS - hand-CUDA WKV is byte-identical to the Triton kernel "
          "(both state dtypes, pads included) and batch-invariant.")


if __name__ == "__main__":
    main()
