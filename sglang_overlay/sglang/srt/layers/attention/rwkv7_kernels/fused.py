# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Fused elementwise + low-rank kernels for the RWKV-7 decode/extend hot path (M5).

Profiling (bench/results/profile.md) found ~20% of the 1.5B bsz1 decode step is
elementwise "glue" spread across ~40 tiny CUDA kernels, plus 8 pathological tiny
LoRA GEMVs (15.3%). These triton kernels collapse that glue into a handful of
launches, lifting decode bandwidth utilization.

BIT-EXACTNESS (the hard gate). The deployed reference computes everything in plain
torch where every binary op on a low-precision tensor (bf16/fp16) rounds its result
back to that dtype (`at::opmath_type<bf16> == float`, so compute-in-fp32 →
round-to-bf16). We reproduce that EXACT rounding sequence: each sub-expression is
evaluated in fp32 then immediately `.to(DT)` (round to the storage dtype) before
being consumed by the next op. The single trick that makes this work for both bf16
AND fp32 with one kernel: ``x.to(DT).to(tl.float32)`` where ``DT == float32`` is the
identity (no precision change), while ``DT == bfloat16`` rounds.

CRITICAL: every launch passes ``enable_fp_fusion=False`` (-> ptxas ``--fmad=false``).
Without it triton/LLVM contracts ``x + m*d`` into one FMA and folds away the
intermediate ``.to(DT)`` round, making the result ~1 ULP MORE accurate than torch --
bit-DIFFERENT, which can flip a knife-edge bf16 argmax. With it the kernels are
bit-identical to the torch reference (verified max_abs_diff == 0.0 at fp32/bf16/fp16).
The hd-axis reductions (L2-norm, gate-correction sum) accumulate in fp32 exactly like
torch's reductions; the final round-to-DT absorbs any reduction-order ULP. So the SAME
kernels are bit-identical at fp32 (gate: verify_m1d --dtype float32) and bf16 (gate:
verify_m1d --cuda-graph --dtype bfloat16).

All kernels are cuda-graph safe: static shapes, no host syncs, output into
caller-allocated buffers.
"""

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------
# Kernel A: token-shift lerp (6x). xr/xw/xk/xv/xa/xg = x + x_*·(shifted - x).
# Replaces ~12 tiny torch kernels (1 sub + 6 mul + 6 add) with one launch.
# ----------------------------------------------------------------------------
@triton.jit
def _lerp6_kernel(
    x_ptr, sh_ptr, mix_ptr, out_ptr,
    T, H,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    offs = hb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H
    DT = out_ptr.dtype.element_ty

    row = t * H + offs
    x = tl.load(x_ptr + row, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(sh_ptr + row, mask=mask, other=0.0).to(tl.float32)
    # d = shifted - x  (one rounded torch op)
    d = (sh - x).to(DT).to(tl.float32)

    for i in tl.static_range(6):
        m = tl.load(mix_ptr + i * H + offs, mask=mask, other=0.0).to(tl.float32)
        # x_*·d  (rounded)  then  x + (...)  (rounded)
        prod = (m * d).to(DT).to(tl.float32)
        o = (x + prod).to(DT)
        tl.store(out_ptr + i * (T * H) + row, o, mask=mask)


def fused_lerp6(x, shifted, mix6):
    """x,shifted: [T,H]; mix6: [6,H] (stacked xr,xk,xw,xa,xg,xv mix vectors).

    Returns [6,T,H] (same dtype as x): xr,xk,xw,xa,xg,xv in that order (matches
    Rwkv7Attention._mix6_buf and the forward unpack).
    """
    T, H = x.shape
    # flat pointer math below assumes contiguous rows (no-op when already so)
    x, shifted, mix6 = x.contiguous(), shifted.contiguous(), mix6.contiguous()
    out = torch.empty(6, T, H, dtype=x.dtype, device=x.device)
    BLOCK = 1024
    grid = (T, triton.cdiv(H, BLOCK))
    _lerp6_kernel[grid](x, shifted, mix6, out, T, H, BLOCK=BLOCK, enable_fp_fusion=False)
    return out


# ----------------------------------------------------------------------------
# Kernel B: kk = L2norm(k·k_k) over head_dim, and k <- k + k·(a-1)·k_a.
# One launch over (T, n_head) replaces ~7 tiny torch kernels + a reduction.
# ----------------------------------------------------------------------------
@triton.jit
def _kk_kmix_kernel(
    k_ptr, a_ptr, kk_param_ptr, ka_param_ptr,
    kk_out_ptr, knew_out_ptr,
    T, H, NH,
    BK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    j = tl.arange(0, BK)
    HD = H // NH
    mask = j < HD
    DT = kk_out_ptr.dtype.element_ty

    base = t * H + h * HD + j
    pbase = h * HD + j
    k = tl.load(k_ptr + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + base, mask=mask, other=0.0).to(tl.float32)
    kkp = tl.load(kk_param_ptr + pbase, mask=mask, other=0.0).to(tl.float32)
    kap = tl.load(ka_param_ptr + pbase, mask=mask, other=0.0).to(tl.float32)

    # kk = k * k_k   (rounded)
    kk = (k * kkp).to(DT)
    kk_f = kk.to(tl.float32)
    # k_new = k + k*(a-1.0)*k_a   (each sub-op rounded, matches torch eval order)
    am = (a - 1.0).to(DT).to(tl.float32)
    t1 = (k * am).to(DT).to(tl.float32)
    t2 = (t1 * kap).to(DT).to(tl.float32)
    knew = (k + t2).to(DT)

    # L2 normalize kk over head_dim. torch: kk.norm(dim=-1) -> DT, clamp_min(1e-12),
    # then kk / norm (rounded). Accumulate sum-of-squares in fp32.
    ss = tl.sum(tl.where(mask, kk_f * kk_f, 0.0), axis=0)
    norm = tl.sqrt(ss)
    norm = norm.to(DT).to(tl.float32)          # torch norm returns storage dtype
    clamp = tl.full((), 1e-12, tl.float32).to(DT).to(tl.float32)
    norm = tl.maximum(norm, clamp)
    kk_n = (kk_f / norm).to(DT)

    tl.store(kk_out_ptr + base, kk_n, mask=mask)
    tl.store(knew_out_ptr + base, knew, mask=mask)


def fused_kk_kmix(k, a, kk_param, ka_param, num_heads):
    """k,a: [T,H]; kk_param,ka_param: [H]; returns (kk_norm [T,nh,hd], k_new [T,H]).

    kk_norm is the L2-normalized (k·k_k); k_new is the a-gated k. Bit-identical to
    the deployed torch sequence.
    """
    k, a = k.contiguous(), a.contiguous()
    kk_param, ka_param = kk_param.contiguous(), ka_param.contiguous()
    T, H = k.shape
    HD = H // num_heads
    BK = triton.next_power_of_2(HD)
    kk_out = torch.empty(T, H, dtype=k.dtype, device=k.device)
    knew_out = torch.empty(T, H, dtype=k.dtype, device=k.device)
    grid = (T, num_heads)
    _kk_kmix_kernel[grid](
        k, a, kk_param.reshape(-1), ka_param.reshape(-1),
        kk_out, knew_out, T, H, num_heads, BK=BK, enable_fp_fusion=False,
    )
    return kk_out.view(T, num_heads, HD), knew_out


# ----------------------------------------------------------------------------
# Kernel C: gate-correction + residual add + gate multiply.
#   o = o_norm + ((r*k*r_k).sum(-1,keepdim) * v);  o = o * g
# One launch over (T, n_head) replaces 3 muls + sum + add + reshape + mul.
# (g_norm/GroupNorm stays a torch op upstream — see model.)
# ----------------------------------------------------------------------------
@triton.jit
def _gate_corr_kernel(
    onorm_ptr, r_ptr, k_ptr, rk_ptr, v_ptr, g_ptr,
    out_ptr,
    T, H, NH,
    BK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    j = tl.arange(0, BK)
    HD = H // NH
    mask = j < HD
    DT = out_ptr.dtype.element_ty

    base = t * H + h * HD + j
    pbase = h * HD + j
    r = tl.load(r_ptr + base, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + base, mask=mask, other=0.0).to(tl.float32)
    rk = tl.load(rk_ptr + pbase, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + base, mask=mask, other=0.0).to(tl.float32)
    onorm = tl.load(onorm_ptr + base, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + base, mask=mask, other=0.0).to(tl.float32)

    # (r*k*r_k)  -> each mul rounded
    p1 = (r * k).to(DT).to(tl.float32)
    p2 = (p1 * rk).to(DT).to(tl.float32)
    s = tl.sum(tl.where(mask, p2, 0.0), axis=0)
    s = s.to(DT).to(tl.float32)            # .sum() returns storage dtype
    gc = (s * v).to(DT).to(tl.float32)     # broadcast scalar over head_dim
    o = (onorm + gc).to(DT).to(tl.float32)
    o = (o * g).to(DT)
    tl.store(out_ptr + base, o, mask=mask)


def fused_gate_corr(o_norm, r, k, r_k, v, g, num_heads):
    """o_norm,g: [T,H]; r,k,v: [T,nh,hd] (or [T,H]); r_k: [nh,hd]. Returns [T,H]."""
    T, H = o_norm.shape
    HD = H // num_heads
    BK = triton.next_power_of_2(HD)
    # flat pointer math assumes contiguous rows; .reshape() may return a strided
    # view, and a strided o_norm/g would be silently mis-read (the F0028 g bug) —
    # normalize here once (no-op when already contiguous).
    o_norm, g = o_norm.contiguous(), g.contiguous()
    out = torch.empty(T, H, dtype=o_norm.dtype, device=o_norm.device)
    grid = (T, num_heads)
    _gate_corr_kernel[grid](
        o_norm, r.reshape(T, H).contiguous(), k.reshape(T, H).contiguous(),
        r_k.reshape(-1).contiguous(),
        v.reshape(T, H).contiguous(), g, out, T, H, num_heads, BK=BK, enable_fp_fusion=False,
    )
    return out


# ----------------------------------------------------------------------------
# Kernel D: LoRA-output gate activations (bsz1 fp16 decode).
#   w_log = -sigmoid(lo[0]) * INV_SQRT_E
#   a     =  sigmoid(lo[1])
#   v     =  v + (v_first - v) * sigmoid(lo[3])       (only when HAS_V, i.e. layer>0)
# (g = lo[2] is an identity slice the caller keeps — not touched here.)
# One launch replaces the ~5-7 tiny torch elementwise kernels (3 sigmoids + neg/mul
# + sub/mul/add) that the profiler (H100, F0051) showed as the single largest
# un-fused launch-count cluster in the per-layer decode step.
#
# BIT-EXACTNESS: reproduce torch's exact fp16 rounding (opmath_type<half>==float, so
# each op computes in fp32 then rounds back to fp16 before the next). The one novelty
# vs the other fused kernels is the transcendental: torch's fp16 sigmoid is
#   (half)(1.f / (1.f + expf(-(float)x)))
# so we compute sigmoid the SAME way — 1/(1+exp(-x)) in fp32 with tl.exp (libdevice
# __nv_expf, == CUDA std::exp(float)) — then round to DT. enable_fp_fusion=False keeps
# ptxas from contracting the round-away FMAs (same trick as the other kernels). The
# byte-exactness of this (incl. the transcendental) is gated by bench/test_lora_gates.py
# BEFORE the path is enabled — if tl.exp ever diverges from expf by a ULP that survives
# the fp16 round, the gate fails and the caller keeps the torch path.
# ----------------------------------------------------------------------------
@triton.jit
def _lora_gates_kernel(
    lo_ptr, v_ptr, vfirst_ptr,
    wlog_ptr, a_ptr, vout_ptr,
    inv_sqrt_e,
    H,
    HAS_V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    hb = tl.program_id(0)
    offs = hb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H
    DT = wlog_ptr.dtype.element_ty

    # sigmoid(x) computed as torch does it: 1/(1+exp(-x)) in fp32, rounded to DT.
    lo0 = tl.load(lo_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s0 = (1.0 / (1.0 + tl.exp(-lo0))).to(DT).to(tl.float32)   # round to storage dtype
    # -sigmoid(lo0) is exact in fp16; the *INV_SQRT_E multiply is the one rounded op.
    wlog = ((-s0) * inv_sqrt_e).to(DT)
    tl.store(wlog_ptr + offs, wlog, mask=mask)

    lo1 = tl.load(lo_ptr + H + offs, mask=mask, other=0.0).to(tl.float32)
    a = (1.0 / (1.0 + tl.exp(-lo1))).to(DT)
    tl.store(a_ptr + offs, a, mask=mask)

    if HAS_V:
        lo3 = tl.load(lo_ptr + 3 * H + offs, mask=mask, other=0.0).to(tl.float32)
        s3 = (1.0 / (1.0 + tl.exp(-lo3))).to(DT).to(tl.float32)
        v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        vf = tl.load(vfirst_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        diff = (vf - v).to(DT).to(tl.float32)
        prod = (diff * s3).to(DT).to(tl.float32)
        vnew = (v + prod).to(DT)
        tl.store(vout_ptr + offs, vnew, mask=mask)


_INV_SQRT_E = 0.6065306597126334  # e^-0.5, == models/rwkv7.py:_INV_SQRT_E


def fused_lora_gates(lo, v, v_first, has_v):
    """lo: [C,H] fp16 (rows w,a,g[,v] in lora4_m1 order); v,v_first: [1,H] (or [H]) fp16.

    Returns (w_log[1,H], a[1,H], v_out) where v_out is the updated v[1,H] when has_v
    (layer>0) else the input v unchanged. Bit-identical to the torch gate math
    (bench/test_lora_gates.py). g = lo[2:3] stays a caller-side slice (no kernel).
    """
    C, H = lo.shape
    lo = lo.contiguous()
    w_log = torch.empty(1, H, dtype=lo.dtype, device=lo.device)
    a = torch.empty(1, H, dtype=lo.dtype, device=lo.device)
    if has_v:
        vflat = v.reshape(-1).contiguous()
        vf = v_first.reshape(-1).contiguous()
        v_out = torch.empty(1, H, dtype=lo.dtype, device=lo.device)
    else:
        # dummy pointers (kernel skips the v branch); keep v unchanged for the caller.
        vflat = vf = lo  # unused when HAS_V=False
        v_out = v
    BLOCK = 1024
    grid = (triton.cdiv(H, BLOCK),)
    _lora_gates_kernel[grid](
        lo, vflat, vf, w_log, a, (v_out if has_v else w_log),
        _INV_SQRT_E, H, HAS_V=has_v, BLOCK=BLOCK, enable_fp_fusion=False,
    )
    return w_log, a, v_out


