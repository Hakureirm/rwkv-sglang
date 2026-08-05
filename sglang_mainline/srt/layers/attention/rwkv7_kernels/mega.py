# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""JIT loader + adapter for the RWKV-7 megakernel-line fusions (rwkv7_mega.cu).

Stage-A (task #50, ADR-0008 / F0060): a role-generic grouped decode GEMV
(gemv_grouped_m1_kernel, blockIdx.y = role) onto which the time-mix block's
projection stages layer:
  gemv_rkv_m1  (G=3)  r/k/v stage        — Stage-A1 (F0060 §5)
  gemv_o_m1    (G=1)  output projection  — Stage-A2 (F0060 §7.5 "add o_proj")
  gemv_rkvo_m1 (G=4)  whole-block r/k/v/o stage the sm120 megakernel chains
Each output row is byte-identical to fast_linear.gemv_m1 because it reuses that
kernel's exact fp32 reduction and the SAME (threads, out_tile) the deployed
decode path picks for (N, K) — o_proj shares (N,K)=(H,H) with r/k/v so it takes
the identical config — so every role composes under the same greedy-EXACT gate.
Env-gated RWKV_MEGA (default OFF), fp16 M==1 only; anything else keeps the
per-projection path. gemv_rkvo_m1 is the bit-exact-gated PREFAB for the 5090
whole-block grid — on the 3090 o_proj is post-WKV so it still launches on its
own (no PDL persistent grid to share the r/k/v launch), see F0060 §7.5.

The PDL griddepcontrol overlap in the .cu is sm_90+ only and currently inert
(needs the launch-attribute + downstream-wait wiring, the documented sm120 step);
on sm_86 this gates structure + correctness, exactly like the WKV CUDA kernel.
"""

import os
from pathlib import Path

import torch

# Reuse fast_linear's arch-aware (threads, out_tile) selection so the grouped
# kernel's per-proj reduction matches the deployed gemv_m1 bit-for-bit.
from sglang.srt.layers.attention.rwkv7_kernels import fast_linear

_EXT_LOADED = False
_LOAD_FAILED = False

MEGA = os.environ.get("RWKV_MEGA", "0") == "1"


def _ensure_loaded() -> bool:
    """JIT-build + register torch.ops.rwkv7_mega on first use. Idempotent."""
    global _EXT_LOADED, _LOAD_FAILED
    if _EXT_LOADED:
        return True
    if _LOAD_FAILED:
        return False
    try:
        from torch.utils.cpp_extension import load

        cuda_dir = Path(__file__).parent / "cuda"
        load(
            name="rwkv7_mega",
            sources=[str(cuda_dir / "rwkv7_mega.cu")],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        )
        _register_fakes()
        _EXT_LOADED = True
        return True
    except Exception as e:  # pragma: no cover - build env dependent
        print(f"[rwkv7_mega] JIT load failed, falling back to per-proj GEMV: {e}")
        _LOAD_FAILED = True
        return False


def _register_fakes():
    try:
        @torch.library.register_fake("rwkv7_mega::gemv_rkv_m1")
        def _f(xr, xk, xv, wr, wk, wv, threads, out_tile):
            return xr.new_empty((3, wr.shape[0]))

        @torch.library.register_fake("rwkv7_mega::gemv_o_m1")
        def _fo(xo, wo, threads, out_tile):
            return xo.new_empty((1, wo.shape[0]))

        @torch.library.register_fake("rwkv7_mega::gemv_rkvo_m1")
        def _frkvo(xr, xk, xv, xo, wr, wk, wv, wo, threads, out_tile):
            return xr.new_empty((4, wr.shape[0]))
    except Exception:
        pass


def available() -> bool:
    return _ensure_loaded()


def gemv_rkv_m1(xr, xk, xv, wr, wk, wv):
    """y[3,N] = stack(xr@wr^T, xk@wk^T, xv@wv^T) for M==1, ONE launch.

    Bit-identical to stacking three fast_linear.gemv_m1 calls (same reduction,
    same arch-aware config). The three activations pass as separate pointers (no
    stack/gather launch); each row of the [3,N] output is byte-identical to
    gemv_m1(x_p, w_p). Caller guarantees fp16, K%4==0, r/k/v share (N, K)."""
    N, K = wr.size(0), wr.size(1)
    t, ot = fast_linear._select_config(N, K)
    return torch.ops.rwkv7_mega.gemv_rkv_m1(
        xr.contiguous().view(-1), xk.contiguous().view(-1),
        xv.contiguous().view(-1), wr, wk, wv, t, ot)


def gemv_o_m1(xo, wo):
    """y[1,N] = xo@wo^T for M==1, ONE launch (G=1 slice of the grouped kernel).

    Stage-A2 (F0060 §7.5): o_proj as a role of the megakernel's r/k/v/o stage.
    Byte-identical to fast_linear.gemv_m1(xo, wo) — same reduction, same config
    (o_proj is (N,K)=(H,H), so _select_config picks exactly the r/k/v config)."""
    N, K = wo.size(0), wo.size(1)
    t, ot = fast_linear._select_config(N, K)
    return torch.ops.rwkv7_mega.gemv_o_m1(xo.contiguous().view(-1), wo, t, ot)


def gemv_rkvo_m1(xr, xk, xv, xo, wr, wk, wv, wo):
    """y[4,N] = stack(r,k,v,o) for M==1, ONE launch (G=4 whole-block stage).

    The bit-exact-gated PREFAB for the sm120 megakernel's whole-block r/k/v/o
    projection stage (F0060 §7.5): the persistent grid runs r/k/v (roles 0..2),
    the WKV recurrence, then o_proj (role 3) with PDL between. Each row is
    byte-identical to the corresponding fast_linear.gemv_m1. On the 3090 this is
    the correctness/structure gate only (o depends on the WKV output, so it
    cannot share the r/k/v launch without the 5090's persistent-grid PDL)."""
    N, K = wr.size(0), wr.size(1)
    t, ot = fast_linear._select_config(N, K)
    return torch.ops.rwkv7_mega.gemv_rkvo_m1(
        xr.contiguous().view(-1), xk.contiguous().view(-1),
        xv.contiguous().view(-1), xo.contiguous().view(-1),
        wr, wk, wv, wo, t, ot)


def rkv_config(N, K):
    """The (threads, out_tile) the grouped kernel will use — exposed for gates."""
    return fast_linear._select_config(N, K)
