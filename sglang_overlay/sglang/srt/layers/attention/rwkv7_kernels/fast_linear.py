# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""JIT loader + adapter for the fused fp16 decode GEMV (rwkv7_fast.cu).

Replaces the M==1 (bsz1 decode) r/k/v/o + ffn projection GEMVs with an fp32-accumulate
fused CUDA GEMV adapted from BlinkDL/Albatross (Apache-2.0; see cuda/NOTICE). fp16-only
(the op reads/writes ``at::Half``): our precision-matched comparison target is
ours-fp16 vs albatross-fp16, and on Ampere fp16==bf16 speed. bf16 / fp32 / quantized /
any M>1 keep the existing torch (ReplicatedLinear/cuBLAS) path. Our WKV recurrent state
stays fp32 (untouched); the LoRA down/up projections use sglang's ReplicatedLinear.

Built WITHOUT --use_fast_math (IEEE arithmetic, no FTZ) so the greedy-EXACT + batch-
invariance gates hold. cuda-graph safe: static shapes, current stream, no host sync.
Enabled via the model (RWKV_FAST_LINEAR env, default off; see docs/findings/0015).
"""

from pathlib import Path

import torch

_EXT_LOADED = False
_LOAD_FAILED = False


def _ensure_loaded():
    """JIT-build + register torch.ops.rwkv7_fast.gemv_m1 on first use. Idempotent."""
    global _EXT_LOADED, _LOAD_FAILED
    if _EXT_LOADED:
        return True
    if _LOAD_FAILED:
        return False
    try:
        from torch.utils.cpp_extension import load

        cuda_dir = Path(__file__).parent / "cuda"
        load(
            name="rwkv7_fast",
            sources=[str(cuda_dir / "rwkv7_fast.cu")],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        )
        _EXT_LOADED = True
        return True
    except Exception as e:  # pragma: no cover - build env dependent
        print(f"[rwkv7_fast] JIT load failed, falling back to torch: {e}")
        _LOAD_FAILED = True
        return False


def available() -> bool:
    return _ensure_loaded()


def gemv_m1(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """y[1,N] = x[1,K] @ weight[N,K]^T. weight = torch nn.Linear .weight (fp16).

    Caller (models/rwkv7.py::_proj_gemv) guarantees M==1, fp16, contiguous, K%4==0
    before dispatching here; anything else takes the torch path."""
    return torch.ops.rwkv7_fast.gemv_m1(x.contiguous().view(1, -1), weight)
