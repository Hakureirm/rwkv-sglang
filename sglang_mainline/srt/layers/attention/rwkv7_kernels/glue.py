# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""JIT loader + wrappers for the fused layer-boundary glue (rwkv7_glue.cu, R2).

Fuses the paged token-shift (gather prev conv + scatter current, dropping the
`.clone()`) with the lerp, keeping the shifted intermediate on-chip. Decode path,
fp16 normed + fp32 conv state (guarded by the caller:
rwkv7_backend.try_fused_shift_lerp*). Gated by RWKV_FUSED_GLUE (default off).
Byte-exact vs token_shift + fused_lerp6/lerp1 (bench/test_glue.py). Pad slots
(PAD_SLOT_ID = -1 from padded cuda-graph replay) are guarded in-kernel: no conv
access, zeroed output rows. conv is declared mutable ((a!)) in the op schema so
functionalization sees the in-place scatter. cuda-graph safe; fakes registered
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
            name="rwkv7_glue",
            sources=[str(cuda_dir / "rwkv7_glue.cu")],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        )
        _register_fakes()
        _EXT_LOADED = True
        return True
    except Exception as e:  # pragma: no cover - build env dependent
        print(f"[rwkv7_glue] JIT load failed, falling back to token_shift+lerp: {e}")
        _LOAD_FAILED = True
        return False


def _register_fakes():
    try:
        @torch.library.register_fake("rwkv7_glue::shift_lerp6")
        def _f6(normed, mix6, cache_indices, conv):
            return normed.new_empty((6, normed.shape[0], normed.shape[1]))

        @torch.library.register_fake("rwkv7_glue::shift_lerp1")
        def _f1(normed, x_k, cache_indices, conv):
            return normed.new_empty((normed.shape[0], normed.shape[1]))
    except Exception:
        pass  # older torch without register_fake -> caller disables piecewise cuda graph


def available() -> bool:
    return _ensure_loaded()


def shift_lerp6(normed, mix6, cache_indices, conv):
    """lp6[6,T,H] = lerp6(normed, prev); conv[cache_indices] <- normed (in-place)."""
    return torch.ops.rwkv7_glue.shift_lerp6(normed, mix6, cache_indices, conv)


def shift_lerp1(normed, x_k, cache_indices, conv):
    """xk[T,H] = lerp1(normed, prev, x_k); conv[cache_indices] <- normed (in-place)."""
    return torch.ops.rwkv7_glue.shift_lerp1(normed, x_k, cache_indices, conv)
