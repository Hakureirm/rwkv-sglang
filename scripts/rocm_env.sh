#!/usr/bin/env bash
# Source this file before deploying or launching RWKV-7 on PyTorch ROCm.
# It selects the active gfx target and keeps NVIDIA-only extensions disabled.

_rwkv_rocm_python="${PYTHON:-${SGLANG_PYTHON:-python}}"
_rwkv_rocm_info="$($_rwkv_rocm_python - <<'PY' 2>/dev/null || true
import torch
if torch.version.hip is not None and torch.cuda.is_available():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = getattr(props, "gcnArchName", "").split(":", 1)[0]
    print(f"{torch.version.hip}|{arch}|{torch.cuda.get_device_name(0)}")
PY
)"

if [[ -z "$_rwkv_rocm_info" ]]; then
  echo "[rwkv7-rocm] PyTorch ROCm device is not available via $_rwkv_rocm_python" >&2
  return 1 2>/dev/null || exit 1
fi

IFS='|' read -r RWKV_ROCM_VERSION RWKV_ROCM_ARCH RWKV_ROCM_DEVICE <<<"$_rwkv_rocm_info"
if [[ -n "$RWKV_ROCM_ARCH" ]]; then
  export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$RWKV_ROCM_ARCH}"
fi

# Consumer RDNA installations commonly lack an AITER build for their gfx target.
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-0}"
export USE_ROCM_AITER_ROPE_BACKEND="${USE_ROCM_AITER_ROPE_BACKEND:-0}"

# These paths compile CUDA C++ containing PTX, WMMA, cp.async, or NVIDIA warp
# assumptions. The portable Triton/Torch path remains available when they are off.
export RWKV_FAST_LINEAR="${RWKV_FAST_LINEAR:-0}"
export RWKV_SPARSE_FFN="${RWKV_SPARSE_FFN:-0}"
export RWKV_FUSED_LORA="${RWKV_FUSED_LORA:-0}"
export RWKV_FUSED_GLUE="${RWKV_FUSED_GLUE:-0}"
export RWKV_GEMV_AUTOTUNE="${RWKV_GEMV_AUTOTUNE:-0}"
export RWKV_FUSED_GATES="${RWKV_FUSED_GATES:-0}"
export RWKV_FUSED_LORA_GATED="${RWKV_FUSED_LORA_GATED:-0}"
export RWKV_FUSED_SQRELU="${RWKV_FUSED_SQRELU:-0}"
export RWKV_FUSED_ADDLN="${RWKV_FUSED_ADDLN:-0}"
export RWKV_ADDLN_WIDE="${RWKV_ADDLN_WIDE:-0}"
export RWKV_FUSED_GNGC="${RWKV_FUSED_GNGC:-0}"
export RWKV_FUSED_RELUSQ="${RWKV_FUSED_RELUSQ:-0}"
export RWKV_FUSED_VRESGATE="${RWKV_FUSED_VRESGATE:-0}"
export RWKV_FAST_LMHEAD="${RWKV_FAST_LMHEAD:-0}"
export RWKV_WKV_CUDA="${RWKV_WKV_CUDA:-0}"
export RWKV_W8A8_TC="${RWKV_W8A8_TC:-0}"

# Keep prefill projection GEMM shapes stable across scheduler chunk boundaries.
# rocBLAS can otherwise choose a different reduction path for M=full_prompt and
# M=chunk, and RWKV's recurrent state can amplify the resulting few-ULP drift.
# Set to 0 to disable; multiples of 256 are recommended for serving chunks.
export RWKV_ROCM_PREFILL_TILE="${RWKV_ROCM_PREFILL_TILE:-256}"

# Fused group-64 dequant + GEMM for quantized M=9..256 projection tiles. W4
# covers the complete window; W8 uses a conservative measured-shape gate.
# Set to 0 for an immediate dequant->rocBLAS fallback during deployment triage.
export RWKV_ROCM_QUANT_PREFILL="${RWKV_ROCM_QUANT_PREFILL:-1}"

export RWKV_ROCM_VERSION RWKV_ROCM_ARCH RWKV_ROCM_DEVICE
printf '[rwkv7-rocm] device=%s arch=%s rocm=%s\n' \
  "$RWKV_ROCM_DEVICE" "$RWKV_ROCM_ARCH" "$RWKV_ROCM_VERSION" >&2
unset _rwkv_rocm_python _rwkv_rocm_info
