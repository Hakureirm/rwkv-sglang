# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""JIT loader + adapters for the sparse channel-mix value projection (rwkv7_sparse_cmix.cu).

The RWKV-7 FFN is  out = value(relu(key(xk))^2).  The value-projection input relu(k)^2 is
86-90% exact-zero on real prompts (measured — bench/results/sparse_ffn/sparsity.log), so
~9/10 of the value weight never needs to be read. This runs a hand-written fp32-accumulate
sparse kernel (adapted from BlinkDL/Albatross, Apache-2.0; see cuda/NOTICE) that skips the
zero-activation weight rows — a TRUE bandwidth saving past the dense ceiling, cuda-graph safe
(static grid).

Accuracy: skipping exact-zero activations is bit-exact (0*w=0), and per-tile accumulation is
fp32 (cuBLAS rounding class). The cross-inter-tile combine uses a float `atomicAdd`, so the
last-ULP rounding of each output is order-nondeterministic (~1 ULP, same class as a cuBLAS
split-K) — NOT a run-to-run bit guarantee. Empirically it passes the project's gates
(verify_m1d greedy-EXACT + verify_batch batch-invariant, 0.1B/1.5B/7.2B, cuda-graph). Opt-in
(RWKV_SPARSE_FFN), default off. See docs/design/m6-sparse-ffn.md.

fp16-only, bsz1 (M==1) decode path; anything else uses the dense torch/ReplicatedLinear
path. Enabled via the model (RWKV_SPARSE_FFN env, default off). See docs/design/m6-sparse-ffn.md.
"""

from pathlib import Path

import torch

_TILE_F = 128   # must match FFN_TILE in the .cu
_TILE_C = 256   # must match C_TILE in the .cu
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
            name="rwkv7_sparse_cmix",
            sources=[str(cuda_dir / "rwkv7_sparse_cmix.cu")],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        )
        _EXT_LOADED = True
        return True
    except Exception as e:  # pragma: no cover - build env dependent
        print(f"[rwkv7_sparse_cmix] JIT load failed, falling back to dense: {e}")
        _LOAD_FAILED = True
        return False


def available() -> bool:
    return _ensure_loaded()


def conforms(value_weight: torch.Tensor) -> bool:
    """value_weight is [H, inter] (torch nn.Linear layout)."""
    H, inter = value_weight.shape
    return (inter % _TILE_F) == 0 and (H % _TILE_C) == 0


def tile_value_weight(value_weight: torch.Tensor) -> torch.Tensor:
    """[H, inter] -> tiled [inter/TF, H/TC, TF, TC] flattened, so skipping a zero
    activation row skips a contiguous coalesced weight block. Do this once at load."""
    H, inter = value_weight.shape
    wt = value_weight.t().contiguous()  # [inter, H]
    wt = wt.view(inter // _TILE_F, _TILE_F, H // _TILE_C, _TILE_C)
    wt = wt.permute(0, 2, 1, 3).contiguous().view(inter, H)
    return wt


def sparse_cmix(preact_k: torch.Tensor, tiled_weight: torch.Tensor, H: int) -> torch.Tensor:
    """out[1,H] = tiled_value_weight @ relu(preact_k)^2 (fp32 accum, fp16 out).

    preact_k = the RAW key projection output [.., inter] fp16 (the kernel applies relu()^2).
    tiled_weight = tile_value_weight(value.weight), fp16. Caller guarantees M==1 + fp16 +
    conforming shapes (else dense fallback)."""
    return torch.ops.rwkv7_sparse_cmix.sparse_cmix(
        preact_k.contiguous().view(-1), tiled_weight, H
    )
