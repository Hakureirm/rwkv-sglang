# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""JIT loader + adapter for the hand-written weight-only int4 decode GEMV (rwkv7_w4.cu).

Weight-only group-wise (GROUP=64) symmetric int4 for the big r/k/v/o + ffn key/value
projections. Small-batch decode (M<=8) is weight-bandwidth-bound, so reading int4 weights
(~1/4 the bytes of fp16) makes decode *faster* than fp16 while cutting weight VRAM ~4x —
the two things 4-bit quantization must deliver (VRAM down, speed >= 16-bit). Not
bitsandbytes (its nf4 GEMV is slower than fp16 at M==1) and no FLA. Built WITHOUT
--use_fast_math (IEEE). cuda-graph safe.

Numerics validated bit-identically vs the dequant reference in bench/verify_w4.py
(rel err ~2e-4, i.e. same ULP as torch's own fp16 matmul).
"""
from pathlib import Path

import torch

GROUP = 64

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
            name="rwkv7_w4",
            sources=[str(cuda_dir / "rwkv7_w4.cu")],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        )
        _register_fakes()
        _EXT_LOADED = True
        return True
    except Exception as e:  # pragma: no cover - build env dependent
        print(f"[rwkv7_w4] JIT load failed, falling back to torch dequant: {e}")
        _LOAD_FAILED = True
        return False


def _register_fakes():
    """FakeTensor (meta) impls so torch.dynamo / piecewise-cuda-graph can trace the
    custom ops without a graph break (otherwise the default sglang server needs
    --disable-piecewise-cuda-graph). Idempotent; both ops are static-shape/graph-safe."""
    try:
        @torch.library.register_fake("rwkv7_w4::gemv_w4_m1")
        def _gemv_w4_m1_fake(x, qweight, scale):
            return x.new_empty((1, qweight.shape[0]))

        @torch.library.register_fake("rwkv7_w4::gemm_w4_small")
        def _gemm_w4_small_fake(x, qweight, scale):
            return x.new_empty((x.shape[0], qweight.shape[0]))

        @torch.library.register_fake("rwkv7_w4::dequant_w4")
        def _dequant_w4_fake(qweight, scale):
            return scale.new_empty((qweight.shape[0], qweight.shape[1] * 2))
    except Exception:
        pass  # older torch without register_fake -> caller must disable piecewise cuda graph


def available() -> bool:
    return _ensure_loaded()


def gemv_w4_m1(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """y[1,N] = dequant(qweight,scale) applied to x[1,K]. M==1, fp16. Caller guards
    fp16 + M==1 + K%64==0 before dispatching here."""
    return torch.ops.rwkv7_w4.gemv_w4_m1(x.contiguous().view(1, -1), qweight, scale)


def gemm_w4_small(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """y[M,N] for 2<=M<=8 (small batched decode): one int4 weight-word read feeds all M
    rows; each row is bit-identical to the M==1 kernel (same accumulation order) ->
    batch-invariant. Caller guards fp16 + 2<=M<=8 + K%64==0 + N even."""
    return torch.ops.rwkv7_w4.gemm_w4_small(x.contiguous(), qweight, scale)


def dequant(qweight: torch.Tensor, scale: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """Unpack group-wise symmetric int4 -> fp16 weight [N, K] for the M>1 (prefill/
    batched) path -> cuBLAS. Uses the CUDA dequant kernel when built (memory-bound,
    fast); else a torch reference. Matches rwkv7_w4.cu / bench/quant_w4.py exactly."""
    if _ensure_loaded():
        return torch.ops.rwkv7_w4.dequant_w4(qweight, scale)
    N = qweight.shape[0]
    K = qweight.shape[1] * 2
    NG = K // group
    lo = (qweight & 0xF).to(torch.int16)
    hi = (qweight >> 4).to(torch.int16)
    lo -= (lo & 8) << 1  # sign-extend 4-bit
    hi -= (hi & 8) << 1
    q = torch.empty(N, K, dtype=torch.int16, device=qweight.device)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    w = q.view(N, NG, group).to(scale.dtype) * scale[:, :, None]
    return w.view(N, K)
