# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""JIT loader + wrapper for the hand-CUDA WKV decode kernel (rwkv7_wkv.cu, task #54).

Serving-hot path only: batched decode (T==1) through the in-place indexed state
pool, fp16 activations, fp16 OR fp32 state storage (fp32 in-register accumulation
either way). Gated by RWKV_WKV_CUDA (default off, read in wkv_recurrent.py).
Bit-exact vs the Triton `_wkv_recurrent_kernel` per state dtype - zero differing
bytes on o and the pool, including pad rows (bench/test_wkv_cuda.py). The varlen
recurrent-prefill and non-indexed h0/ht paths stay on the Triton kernel.

Built WITHOUT fast-math (IEEE, no FTZ) so the bit-exactness contract holds.
pool is declared mutable ((a!)) in the op schema; cuda-graph safe; fake
registered for piecewise capture.
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
            name="rwkv7_wkv",
            sources=[str(cuda_dir / "rwkv7_wkv.cu")],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        )
        _register_fakes()
        _EXT_LOADED = True
        return True
    except Exception as e:  # pragma: no cover - build env dependent
        print(f"[rwkv7_wkv] JIT load failed, falling back to the Triton kernel: {e}")
        _LOAD_FAILED = True
        return False


def _register_fakes():
    try:
        @torch.library.register_fake("rwkv7_wkv::wkv_decode")
        def _fwkv(r, w, k, v, kk, a, pool, ci, scale):
            return v.new_empty(v.shape)
    except Exception:
        pass  # older torch without register_fake -> caller disables piecewise cuda graph


def available() -> bool:
    return _ensure_loaded()


def wkv_decode(r, w, k, v, kk, a, pool, ci, scale):
    """o[B,1,H,64] = WKV decode step; pool[ci] updated in place. See rwkv7_wkv.cu."""
    return torch.ops.rwkv7_wkv.wkv_decode(r, w, k, v, kk, a, pool, ci, scale)
