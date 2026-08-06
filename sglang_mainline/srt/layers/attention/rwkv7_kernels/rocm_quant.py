# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""AMD ROCm weight-only W8/W4 kernels for RWKV-7 decode and prefill.

The fast FP16/BF16 path is a small, standalone HIP extension containing no
NVIDIA WMMA/cp.async dependencies.  A portable Triton implementation remains
the build-failure fallback and handles the measured prefill window.  Both paths
preserve the CUDA checkpoint layouts and fuse group-wise dequantization with
the linear reduction, so supported calls never materialize a full fp16/bf16
weight.  Unmeasured or non-winning W8 shapes retain the correctness-first torch
dequant + GEMM fallback in the model layer.
"""

import os
from pathlib import Path

import torch
import triton
import triton.language as tl


GROUP = 64
MAX_BATCH = 8
MAX_FUSED_PREFILL = 256
_PREFILL_ENABLED = os.environ.get("RWKV_ROCM_QUANT_PREFILL", "1") == "1"

_HIP_EXT_LOADED = False
_HIP_EXT_FAILED = False


def _register_fakes() -> None:
    try:
        @torch.library.register_fake("rwkv7_rocm_quant::linear_w8")
        def _linear_w8_fake(x, qweight, scale):
            return x.new_empty((x.shape[0], qweight.shape[0]))

        @torch.library.register_fake("rwkv7_rocm_quant::linear_w4")
        def _linear_w4_fake(x, qweight, scale):
            return x.new_empty((x.shape[0], qweight.shape[0]))

    except Exception:
        # Older torch versions may not expose register_fake.  The kernels still
        # work in eager mode; callers can disable piecewise graph capture there.
        pass


def _ensure_hip_extension() -> bool:
    global _HIP_EXT_LOADED, _HIP_EXT_FAILED
    if _HIP_EXT_LOADED:
        return True
    if _HIP_EXT_FAILED or torch.version.hip is None:
        return False
    try:
        from torch.utils.cpp_extension import load

        source = Path(__file__).parent / "hip" / "rwkv7_quant_hip.cu"
        load(
            name="rwkv7_rocm_quant",
            sources=[str(source)],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
        )
        _register_fakes()
        _HIP_EXT_LOADED = True
        return True
    except Exception as exc:  # pragma: no cover - build environment dependent
        print(
            "[rwkv7_rocm_quant] HIP JIT load failed; using Triton fallback: "
            f"{exc}"
        )
        _HIP_EXT_FAILED = True
        return False


def available() -> bool:
    return torch.version.hip is not None and torch.cuda.is_available()


@triton.jit
def _w8_linear_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    X_IS_BF16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    groups = K // GROUP_SIZE
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        q = tl.load(
            q_ptr + offs_k[:, None] + offs_n[None, :] * K,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.float32)
        scale = tl.load(
            scale_ptr
            + (offs_k // GROUP_SIZE)[:, None]
            + offs_n[None, :] * groups,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        weight = q * scale
        if X_IS_BF16:
            x = x.to(tl.bfloat16)
            weight = weight.to(tl.bfloat16)
        else:
            x = x.to(tl.float16)
            weight = weight.to(tl.float16)
        acc = tl.dot(x, weight, acc)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _w4_linear_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    X_IS_BF16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    packed_k = K // 2
    groups = K // GROUP_SIZE
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        packed = tl.load(
            q_ptr + (offs_k // 2)[:, None] + offs_n[None, :] * packed_k,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.int32)
        nibble = tl.where(
            (offs_k[:, None] & 1) == 0,
            packed & 0xF,
            (packed >> 4) & 0xF,
        )
        signed_q = tl.where(nibble < 8, nibble, nibble - 16).to(tl.float32)
        scale = tl.load(
            scale_ptr
            + (offs_k // GROUP_SIZE)[:, None]
            + offs_n[None, :] * groups,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        weight = signed_q * scale
        if X_IS_BF16:
            x = x.to(tl.bfloat16)
            weight = weight.to(tl.bfloat16)
        else:
            x = x.to(tl.float16)
            weight = weight.to(tl.float16)
        acc = tl.dot(x, weight, acc)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _validate(x, qweight, scale, *, max_m):
    if not available():
        raise RuntimeError("ROCm Triton quantized linear requested without HIP")
    if x.ndim != 2 or not 1 <= x.shape[0] <= max_m:
        raise ValueError(
            f"ROCm quant linear expects [M,K], 1<=M<={max_m}; got {x.shape}"
        )
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"ROCm quant linear expects fp16/bf16 activation; got {x.dtype}")
    if x.shape[1] % GROUP != 0:
        raise ValueError(f"K must be divisible by {GROUP}; got {x.shape[1]}")
    if qweight.ndim != 2 or scale.ndim != 2:
        raise ValueError("qweight and scale must be matrices")
    if qweight.device != x.device or scale.device != x.device:
        raise ValueError("activation and quant tensors must share a device")
    if scale.dtype != torch.float16:
        raise TypeError(f"ROCm quant scale must be fp16; got {scale.dtype}")
    if qweight.shape[0] != scale.shape[0]:
        raise ValueError("qweight and scale output dimensions differ")
    if scale.shape[1] != x.shape[1] // GROUP:
        raise ValueError("scale group dimension does not match K/64")
    expected_qk = x.shape[1] if qweight.dtype == torch.int8 else x.shape[1] // 2
    if qweight.dtype not in (torch.int8, torch.uint8):
        raise TypeError(f"ROCm qweight must be int8/uint8; got {qweight.dtype}")
    if qweight.shape[1] != expected_qk:
        raise ValueError("qweight packed dimension does not match K")


def _launch(kernel, x, qweight, scale, *, large=False):
    _validate(
        x,
        qweight,
        scale,
        max_m=MAX_FUSED_PREFILL if large else MAX_BATCH,
    )
    x = x.contiguous()
    qweight = qweight.contiguous()
    scale = scale.contiguous()
    M, K = x.shape
    N = qweight.shape[0]
    # The decode fallback pads to one 16-row matrix tile.  The prefill path uses
    # a fixed 64-row tile at every M so an unchunked call and the same rows split
    # into scheduler chunks execute the same reduction layout.  This fixed tile
    # was faster than dequant->rocBLAS throughout the W4 M=9..256 matrix on
    # gfx1100, and is used for W8 only behind the measured shape gate below.
    block_m = 64 if large else 16
    block_n = 32
    block_k = 64
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    kernel[grid](
        x,
        qweight,
        scale,
        out,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE=GROUP,
        X_IS_BF16=x.dtype == torch.bfloat16,
        num_warps=8 if kernel is _w4_linear_kernel or not large else 4,
        enable_fp_fusion=False,
    )
    return out


def _base_supported(x: torch.Tensor, qweight: torch.Tensor) -> bool:
    return (
        available()
        and x.ndim == 2
        and x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.shape[1] % GROUP == 0
        and qweight.ndim == 2
    )


def w4_supported(x: torch.Tensor, qweight: torch.Tensor) -> bool:
    """Whether the fused ROCm W4 path covers this decode/prefill shape."""
    return (
        _base_supported(x, qweight)
        and qweight.dtype == torch.uint8
        and qweight.shape[1] * 2 == x.shape[1]
        and 1 <= x.shape[0] <= MAX_FUSED_PREFILL
        and (x.shape[0] <= MAX_BATCH or _PREFILL_ENABLED)
    )


def w8_supported(x: torch.Tensor, qweight: torch.Tensor) -> bool:
    """Conservative gfx1100 W8 dispatch gate built from real RWKV projections.

    M<=8 always uses the HIP decode kernel.  The fixed-tile Triton path is used
    only where it beat dequant->rocBLAS in the all-size sweep.  Other shapes
    deliberately fall back in ``W8Linear.forward`` rather than taking a merely
    correct but slower fused path.
    """
    if not _base_supported(x, qweight):
        return False
    m, k = x.shape
    n = qweight.shape[0]
    if qweight.dtype != torch.int8 or qweight.shape[1] != k:
        return False
    if 1 <= m <= MAX_BATCH:
        return True
    if not _PREFILL_ENABLED:
        return False
    if not MAX_BATCH < m <= MAX_FUSED_PREFILL:
        return False
    if m <= 128:
        # Narrow-output small-model FFN-value projections were the only tested
        # M<=128 family where the fixed 64-row W8 tile did not beat fallback.
        return not (k > n and k < 8192)
    # At the stable 256-row prefill tile, retain only robust measured wins.
    # FP16 rocBLAS has a better dequant+GEMM crossover than BF16 for the 1.5B
    # and 2.9B forward FFN shapes, so those remain on fallback in FP16.
    forward_min_k = 4096 if x.dtype == torch.float16 else 2048
    return (
        (n == 4 * k and k >= forward_min_k)
        or (n == k and k >= 4096)
        or (k == 4 * n and k >= 16384)
    )


def linear_w8(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor):
    """Fused W8A16 group-64 linear for a gated ROCm decode/prefill shape."""
    if not w8_supported(x, qweight):
        raise ValueError(f"ROCm fused W8 shape is outside the measured gate: {x.shape}")
    if x.shape[0] <= MAX_BATCH and (
        x.dtype in (torch.float16, torch.bfloat16)
        and qweight.shape[0] % 2 == 0
        and _ensure_hip_extension()
    ):
        return torch.ops.rwkv7_rocm_quant.linear_w8(
            x.contiguous(), qweight.contiguous(), scale.contiguous()
        )
    return _launch(
        _w8_linear_kernel,
        x,
        qweight,
        scale,
        large=x.shape[0] > MAX_BATCH,
    )


def linear_w4(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor):
    """Fused W4A16 group-64 linear for ROCm decode and M<=256 prefill."""
    if not w4_supported(x, qweight):
        raise ValueError(f"ROCm fused W4 shape is outside the measured gate: {x.shape}")
    if x.shape[0] <= MAX_BATCH and (
        x.dtype in (torch.float16, torch.bfloat16)
        and qweight.shape[0] % 2 == 0
        and _ensure_hip_extension()
    ):
        return torch.ops.rwkv7_rocm_quant.linear_w4(
            x.contiguous(), qweight.contiguous(), scale.contiguous()
        )
    return _launch(
        _w4_linear_kernel,
        x,
        qweight,
        scale,
        large=x.shape[0] > MAX_BATCH,
    )
