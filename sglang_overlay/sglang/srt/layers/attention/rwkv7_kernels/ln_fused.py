# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""JIT loader + wrappers for the fused norm-boundary kernels (rwkv7_ln.cu, W1').

Two pure same-math fusions of per-layer glue the large-batch decode profile
found running as stock torch kernels (vs vllm-rwkv PR#8's hand-fused analogs):

  add_ln(x, delta, ln)        -> (x_new, y): x_new = x + delta (fp16 residual
      add), y = LayerNorm(x_new). Bit-identical to torch's add + nn.LayerNorm
      (the vectorized LN algorithm is transcribed; see rwkv7_ln.cu).
  gn_gatecorr(o, r, k, r_k, v, g, gn, nh) -> out: GroupNorm + the Triton
      _gate_corr epilogue in ONE kernel (torch RowwiseMoments transcription +
      the _gate_corr rounding chain + its probed summation tree).

Both default OFF (RWKV_FUSED_ADDLN / RWKV_FUSED_GNGC in models/rwkv7.py) until
bench/test_ln_fused.py shows ZERO differing bytes vs the live reference ops on
the target stack, plus greedy 24/24 EXACT end-to-end. Pad rows need no special
casing here: these ops are pure functions of their inputs (no state pool
access); pad rows compute garbage that the caller discards, same as the
reference ops. cuda-graph safe (static shapes, no host sync); fakes registered
for piecewise capture.
"""
from pathlib import Path

import torch

_EXT_LOADED = False
_LOAD_FAILED = False


def _ensure_loaded():
    global _EXT_LOADED, _LOAD_FAILED
    if _EXT_LOADED:
        return True
    if _LOAD_FAILED:
        return False
    try:
        from torch.utils.cpp_extension import load

        cuda_dir = Path(__file__).parent / "cuda"
        load(
            name="rwkv7_ln",
            sources=[str(cuda_dir / "rwkv7_ln.cu")],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            # default fmad (ON), matching the torch build's contraction of the
            # LN/GN apply expressions - part of the bit-exactness contract.
            extra_cuda_cflags=["-O3"],
        )
        _register_fakes()
        _EXT_LOADED = True
        return True
    except Exception as e:  # pragma: no cover - build env dependent
        print(f"[rwkv7_ln] JIT load failed, falling back to torch norms: {e}")
        _LOAD_FAILED = True
        return False


def _register_fakes():
    try:
        @torch.library.register_fake("rwkv7_ln::add_ln")
        def _fa(x, delta, gamma, beta, eps):
            return torch.empty_like(x), torch.empty_like(x)

        @torch.library.register_fake("rwkv7_ln::gn_gatecorr")
        def _fg(o, r, k, rk, v, g, gamma, beta, eps, nh):
            return torch.empty_like(o)

        @torch.library.register_fake("rwkv7_ln::relu_sq")
        def _fr(x):
            return torch.empty_like(x)

        @torch.library.register_fake("rwkv7_ln::vres_gates")
        def _fv(wl, al, vl, v, vf, inv_sqrt_e):
            return torch.empty_like(wl), torch.empty_like(wl), torch.empty_like(v)
    except Exception:
        pass  # older torch without register_fake -> caller disables piecewise capture


def available() -> bool:
    return _ensure_loaded()


def add_ln(x, delta, ln: torch.nn.LayerNorm):
    """x_new = x + delta; y = ln(x_new). Returns (x_new, y).

    Caller guards eligibility (fp16, contiguous, N % 4 == 0, N <= 8192,
    affine LayerNorm) - mirrors models/rwkv7.py _addln_eligible."""
    return torch.ops.rwkv7_ln.add_ln(x, delta, ln.weight, ln.bias, ln.eps)


def relu_sq(x):
    """relu(x)**2 in one kernel (bit-identical to torch relu + pow on fp16)."""
    return torch.ops.rwkv7_ln.relu_sq(x)


def vres_gates(wl, al, vl, v, v_first, inv_sqrt_e):
    """Batched LoRA-gate activations: returns (w_log, a, v_new).

    w_log = -sigmoid(wl) * inv_sqrt_e; a = sigmoid(al);
    v_new = v + (v_first - v) * sigmoid(vl) (when vl is not None, layer>0)
    - bit-identical to the torch op chain (bench/test_ln_fused.py)."""
    return torch.ops.rwkv7_ln.vres_gates(wl, al, vl, v, v_first, inv_sqrt_e)


def gn_gatecorr(o, r, k, r_k, v, g, gn: torch.nn.GroupNorm, nh: int):
    """(gn(o) + (r*k*r_k).sum(head)*v) * g, one kernel. Returns [T, H]."""
    T, H = o.shape
    return torch.ops.rwkv7_ln.gn_gatecorr(
        o,
        r.reshape(T, H).contiguous(),
        k.reshape(T, H).contiguous(),
        r_k.reshape(-1).contiguous(),
        v.reshape(T, H).contiguous(),
        g.contiguous(),
        gn.weight,
        gn.bias,
        gn.eps,
        nh,
    )
