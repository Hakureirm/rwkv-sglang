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
import os
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

        @torch.library.register_fake("rwkv7_w4::gemm_w4_tc")
        def _gemm_w4_tc_fake(x, qweight, scale):
            return x.new_empty((x.shape[0], qweight.shape[0]))

        @torch.library.register_fake("rwkv7_w4::gemm_w4a8_tc")
        def _gemm_w4a8_tc_fake(x, qweight, scale, x_scale, algo):
            return scale.new_empty((x.shape[0], qweight.shape[0]))

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


def gemm_w4_tc(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """y[M,N] for 8<M<=64 (batched decode) via tensor cores (wmma, fp32 accum): the int4
    weight tile is dequantized to fp16 in shared memory per K-step, so weight HBM traffic
    is 1/4 of a cuBLAS fp16 GEMM. Deterministic per-row reduction order (fixed k-loop),
    independent of batch composition. Caller guards fp16 + M<=64 + K%64==0 + N%64==0."""
    return torch.ops.rwkv7_w4.gemm_w4_tc(x.contiguous(), qweight, scale)


# algo for the w4a8 large-M kernel: -1 auto (64-row tile from M>=192, or already
# from M=65 when N>=4096 — the measured 3090 crossover, see the .cu launcher);
# 0/1 force. Speed-only knob — both tiles are bit-identical (same per-group fp32
# chain per element; gated by bench/verify_w4a8.py).
_W4A8_ALGO = int(os.environ.get("RWKV_W4A8_ALGO", "-1"))

_TC_S8_SUPPORTED = None


def tc_s8_supported() -> bool:
    """gemm_w4a8_tc needs sm80+ (cp.async + s8 wmma fragments; the device code is
    empty below that). Distinct from tc_supported() (sm70+, fp16 wmma)."""
    global _TC_S8_SUPPORTED
    if _TC_S8_SUPPORTED is None:
        try:
            _TC_S8_SUPPORTED = torch.cuda.get_device_capability()[0] >= 8
        except Exception:
            _TC_S8_SUPPORTED = False
    return _TC_S8_SUPPORTED


def gemm_w4a8_tc(x: torch.Tensor, qweight: torch.Tensor, scale_t: torch.Tensor,
                 algo: int = None) -> torch.Tensor:
    """y[M,N] for M>64 (high-concurrency decode / prefill) via w4a8: per-token int8
    activation quant (sglang's per_token_quant_int8, the same op the w8a8 tier uses)
    + int4-weight s8xs8 tensor-core GEMM (rwkv7_w4.cu::gemm_w4a8_tc). Weight HBM
    traffic is 1/4 of fp16 and the MMAs run at the s8 rate — this replaces the
    dequant->cuBLAS fallback whose ~36 bits/element weight traffic lost to fp16
    (the M=64 concurrency cliff). Semantics are w4a8 (quantized activations), NOT
    the w4a16 of the M<=64 kernels — accuracy is certified separately (Stage 3).

    `scale_t` is the TRANSPOSED scale, fp16 [K/64, N] contiguous — i.e.
    `.scale.t().contiguous()`, computed once and cached per layer by the caller
    (W4Linear stashes it as `_scale_t`): the kernel's per-group scale reads are
    coalesced in this layout and scattered in the checkpoint's [N, K/64].
    Caller guards fp16 + M>64 + K%64==0 + tc_s8_supported()."""
    from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8

    x_q, x_scale = per_token_quant_int8(x.contiguous())
    return torch.ops.rwkv7_w4.gemm_w4a8_tc(
        x_q, qweight, scale_t, x_scale.reshape(-1),
        _W4A8_ALGO if algo is None else algo,
    )


# ---- weight-only int8 family (rwkv7_w8.cu) — same skeleton, near-lossless accuracy,
# ---- runs on every arch (no cutlass; JIT per-arch). qweight int8[N,K] + scale[N,K/64].
_W8_LOADED = False
_W8_FAILED = False


def _ensure_w8_loaded():
    global _W8_LOADED, _W8_FAILED
    if _W8_LOADED:
        return True
    if _W8_FAILED:
        return False
    try:
        from torch.utils.cpp_extension import load

        cuda_dir = Path(__file__).parent / "cuda"
        load(
            name="rwkv7_w8",
            sources=[str(cuda_dir / "rwkv7_w8.cu")],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        )
        try:
            @torch.library.register_fake("rwkv7_w8::gemv_w8_m1")
            def _gemv_w8_m1_fake(x, qweight, scale):
                return x.new_empty((1, qweight.shape[0]))

            @torch.library.register_fake("rwkv7_w8::gemm_w8_small")
            def _gemm_w8_small_fake(x, qweight, scale):
                return x.new_empty((x.shape[0], qweight.shape[0]))

            @torch.library.register_fake("rwkv7_w8::gemm_w8_tc")
            def _gemm_w8_tc_fake(x, qweight, scale):
                return x.new_empty((x.shape[0], qweight.shape[0]))

            @torch.library.register_fake("rwkv7_w8::gemm_w8_tc_large")
            def _gemm_w8_tc_large_fake(x, qweight, scale):
                return x.new_empty((x.shape[0], qweight.shape[0]))

            @torch.library.register_fake("rwkv7_w8::dequant_w8")
            def _dequant_w8_fake(qweight, scale):
                return scale.new_empty((qweight.shape[0], qweight.shape[1]))
        except Exception:
            pass
        _W8_LOADED = True
        return True
    except Exception as e:  # pragma: no cover
        print(f"[rwkv7_w8] JIT load failed, falling back to torch dequant: {e}")
        _W8_FAILED = True
        return False


_TC_SUPPORTED = None


def tc_supported() -> bool:
    """Tensor-core (wmma) kernels need sm70+; the gemm_*_tc device code is empty
    below that (ARCH guard), so Pascal and older must never be routed to it —
    they fall back to dequant→cuBLAS (the scalar gemv/small kernels are plain
    FMA + warp shuffles and run fine from sm60)."""
    global _TC_SUPPORTED
    if _TC_SUPPORTED is None:
        try:
            _TC_SUPPORTED = torch.cuda.get_device_capability()[0] >= 7
        except Exception:
            _TC_SUPPORTED = False
    return _TC_SUPPORTED


def w8_available() -> bool:
    return _ensure_w8_loaded()


def gemv_w8_m1(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.ops.rwkv7_w8.gemv_w8_m1(x.contiguous().view(1, -1), qweight, scale)


def gemm_w8_small(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.ops.rwkv7_w8.gemm_w8_small(x.contiguous(), qweight, scale)


def gemm_w8_tc(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.ops.rwkv7_w8.gemm_w8_tc(x.contiguous(), qweight, scale)


def gemm_w8_tc_large(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """y[M,N] for the high-concurrency regime (64<M<=~256) via tensor cores with a 2-D
    (M x N) block grid: 64x64x64 tile, 256 threads / 8 warps (4(M)x2(N)), weight-stationary
    — the int8 weight tile is dequanted to fp16 in shared memory ONCE per K-step and reused
    across all 64 M-rows, so weight HBM traffic stays 1/2 of a cuBLAS fp16 GEMM even at large
    M (the amortization the M<=64 gemm_w8_tc cannot reach). 2-stage cp.async on sm80+, sync
    fallback on sm70-75, no split-K. Deterministic ascending-K fp32 accum -> same numerical
    class as gemm_w8_tc (~2.9e-4 rel vs dequant ref). Caller guards fp16 + N%64==0 + K%64==0
    + tc_supported()."""
    return torch.ops.rwkv7_w8.gemm_w8_tc_large(x.contiguous(), qweight, scale)


def dequant_w8(qweight: torch.Tensor, scale: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    if _ensure_w8_loaded():
        return torch.ops.rwkv7_w8.dequant_w8(qweight, scale)
    N, K = qweight.shape
    NG = K // group
    w = qweight.view(N, NG, group).to(scale.dtype) * scale[:, :, None]
    return w.view(N, K)


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
