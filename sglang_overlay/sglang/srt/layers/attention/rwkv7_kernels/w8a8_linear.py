# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""JIT loader + sm120 dispatch for the w8a8 int8×int8 tensor-core GEMM (rwkv7_w8a8.cu).

sglang's `--quantization w8a8_int8` linear method calls sgl_kernel's cutlass
`int8_scaled_mm`, which ships for sm80–90 only — on sm120 (RTX 5090 / Blackwell
consumer) it raises NotImplementedError, so the whole w8a8 tier is unavailable there.
This module provides a same-contract replacement (per-token × per-channel scales,
int32 accumulation over full K, single fp32 rescale epilogue) built from standard
sm80+ s8 wmma fragments, and a scoped patch that routes W8A8Int8LinearMethod.apply
through it exactly where the cutlass op is missing.

Dispatch policy:
  * sm120 (or any arch where sgl_kernel raises): our kernel.
  * sm80–90: untouched — cutlass keeps the path it already owns.
  * kill switch: RWKV_W8A8_TC=0 disables the patch entirely.

Determinism note: the int32 accumulation is order-exact, so outputs are bit-identical
across batch sizes for the same row — strictly stronger than the fp16 cuBLAS fallback.
"""
import os
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
            name="rwkv7_w8a8",
            sources=[str(cuda_dir / "rwkv7_w8a8.cu")],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        )
        _register_fakes()
        _EXT_LOADED = True
        return True
    except Exception as e:  # pragma: no cover - build env dependent
        print(f"[rwkv7_w8a8] JIT load failed, w8a8 stays on upstream paths: {e}")
        _LOAD_FAILED = True
        return False


def _register_fakes():
    """FakeTensor impl so piecewise-cuda-graph traces the op without a break."""
    try:

        @torch.library.register_fake("rwkv7_w8a8::gemm_w8a8_tc")
        def _gemm_w8a8_tc_fake(x, w, x_scale, w_scale, out_dtype, bias, algo):
            return x.new_empty((x.shape[0], w.shape[1]), dtype=out_dtype)

    except Exception:
        pass  # older torch without register_fake -> disable piecewise cuda graph


# algo: -1 = auto (per-M: V2 large / V1 small, default); 1 = force V2; 0 = force V1.
_ALGO = int(os.environ.get("RWKV_W8A8_ALGO", "-1"))


def gemm_w8a8_tc(x_q, w_t, x_scale, w_scale, out_dtype, bias=None, algo=None):
    """out[m,n] = (Σ_k x_q[m,k]·w[n,k]) · x_scale[m] · w_scale[n] (+ bias[n]),
    fp16/bf16 out; the bias add happens in fp32 before the output rounding.

    Contract mirrors sgl_kernel.int8_scaled_mm: `w_t` is the [K,N] .t() view of the
    loader's contiguous [N,K] int8 tensor; scales fp32 [M]/[M,1] and [N]/[N,1].
    """
    return torch.ops.rwkv7_w8a8.gemm_w8a8_tc(
        x_q, w_t, x_scale.reshape(-1), w_scale.reshape(-1), out_dtype, bias,
        _ALGO if algo is None else algo,
    )


def _needs_own_kernel() -> bool:
    """True where upstream cutlass int8_scaled_mm is absent (sm100/sm120)."""
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        return False
    return major >= 10


_PATCHED = False


def maybe_patch_w8a8_linear_method():
    """Route sglang's W8A8Int8LinearMethod.apply through our kernel on arches where
    the cutlass op is missing. No-op (upstream untouched) on sm80–90. Idempotent."""
    global _PATCHED
    if _PATCHED or os.environ.get("RWKV_W8A8_TC", "1") == "0":
        return
    if not _needs_own_kernel() or not _ensure_loaded():
        return
    try:
        from sglang.srt.layers.quantization import w8a8_int8 as _w8a8_mod

        def _apply(self, layer, x, bias=None):
            from sglang.srt.layers.quantization.int8_kernel import (
                per_token_quant_int8,
            )

            x_q, x_scale = per_token_quant_int8(x)
            x_q_2d = x_q.view(-1, x_q.shape[-1])
            k = x_q_2d.shape[-1]
            if k % 64:
                # Zero-pad K to the kernel's tile (RWKV LoRA-up ranks: 96/128/...).
                # int8 zeros contribute exact zeros to the int32 sums, so this is
                # bit-identical to the unpadded product. Weight pad cached per layer.
                pad = 64 - k % 64
                w_pad = getattr(layer, "_rwkv_w8a8_wpad", None)
                if w_pad is None:
                    w_pad = torch.nn.functional.pad(
                        layer.weight.t().contiguous(), (0, pad)
                    ).contiguous()
                    layer._rwkv_w8a8_wpad = w_pad
                x_q_2d = torch.nn.functional.pad(x_q_2d, (0, pad))
                w_t = w_pad.t()
            else:
                w_t = layer.weight
            out = gemm_w8a8_tc(
                x_q_2d,
                w_t,
                x_scale.view(-1),
                layer.weight_scale,
                x.dtype,
                bias,
            )
            return out.view(*x_q.shape[:-1], layer.weight.shape[1])

        _w8a8_mod.W8A8Int8LinearMethod.apply = _apply
        _PATCHED = True
        print("[rwkv7_w8a8] sm120 w8a8 path enabled (s8 wmma GEMM in place of cutlass)")
    except Exception as e:  # pragma: no cover - upstream layout drift
        print(f"[rwkv7_w8a8] patch skipped ({e}); w8a8 stays on upstream paths")
