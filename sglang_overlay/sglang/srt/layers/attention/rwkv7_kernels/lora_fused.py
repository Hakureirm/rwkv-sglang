# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""JIT loader + packer for the fused 4-chain LoRA decode op (rwkv7_lora.cu, M9).

On the bsz1 fp16 decode path each RWKV-7 layer runs 4 LoRA chains
(w/a/g[, v]): down-GEMV -> act -> up-GEMV(+bias), i.e. ~12+ tiny kernel
launches per layer whose latency dominates over their bandwidth. lora4_m1
packs all chains into ONE custom op with TWO kernel launches:

  stage1  h[r]   = act(dot_fp32(d_cat[r,:], xs[chain_of(r),:]))   (fp32 scratch)
  stage2  y[c,n] = bias_cat[c,n] + dot_fp32(u_cat[n, roff:+rank], h[roff:+rank])

Packed layouts (built once by ``pack_loras`` from the loaded weights):
  d_cat  fp16 [R_total, H]  down weights row-stacked (nn.Linear [rank, H] rows).
  u_cat  fp16 [H, R_total]  up weights column-stacked (== torch.cat(w_up, dim=1));
                            the rank dim is innermost so stage2's warp-per-output
                            reads are coalesced along R.
  bias_cat fp16 [C, H]      zeros where a chain has no bias (g).
  meta   int32 [C, 3]       (rank_offset, rank, act_code) per chain.

Numerics: fp32 accumulate, IEEE (built WITHOUT --use_fast_math), deterministic
reduction order, and the torch chain's fp16 intermediate roundings are
reproduced exactly (see the .cu header) — output matches the torch reference
chain to ~1 fp16 ULP, the same class as gemv_m1. cuda-graph safe (static
shapes, current stream, no host sync); fake registered for piecewise capture.
Enabled via the model (RWKV_FUSED_LORA env, default off).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch

ACT_IDENTITY = 0
ACT_TANH = 1
ACT_SIGMOID = 2

_EXT_LOADED = False
_LOAD_FAILED = False


def _ensure_loaded():
    """JIT-build + register torch.ops.rwkv7_lora.lora4_m1 on first use. Idempotent."""
    global _EXT_LOADED, _LOAD_FAILED
    if _EXT_LOADED:
        return True
    if _LOAD_FAILED:
        return False
    try:
        from torch.utils.cpp_extension import load

        cuda_dir = Path(__file__).parent / "cuda"
        load(
            name="rwkv7_lora",
            sources=[str(cuda_dir / "rwkv7_lora.cu")],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        )
        _register_fakes()
        _EXT_LOADED = True
        return True
    except Exception as e:  # pragma: no cover - build env dependent
        print(f"[rwkv7_lora] JIT load failed, falling back to torch: {e}")
        _LOAD_FAILED = True
        return False


def _register_fakes():
    """FakeTensor (meta) impl so torch.dynamo / piecewise-cuda-graph can trace the
    custom op without a graph break (same pattern as w4_linear)."""
    try:
        @torch.library.register_fake("rwkv7_lora::lora4_m1")
        def _lora4_m1_fake(xs, d_cat, u_cat, bias_cat, meta):
            return xs.new_empty((xs.shape[0], xs.shape[1]))

        @torch.library.register_fake("rwkv7_lora::lora4_mn")
        def _lora4_mn_fake(xs, d_cat, u_cat, bias_cat, meta):
            return xs.new_empty((xs.shape[0], xs.shape[1], xs.shape[2]))

        @torch.library.register_fake("rwkv7_lora::lora4_m1_gated")
        def _lora4_m1_gated_fake(xs, d_cat, u_cat, bias_cat, meta, v, vfirst,
                                 inv_sqrt_e):
            return xs.new_empty((xs.shape[0], xs.shape[1]))
    except Exception:
        pass  # older torch without register_fake -> caller must disable piecewise cuda graph


def available() -> bool:
    return _ensure_loaded()


def pack_loras(
    chains: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the packed (d_cat, u_cat, bias_cat, meta) tensors for lora4_m1.

    chains: per chain (in xs order) a tuple
      (down_weight fp16 [rank, H], up_weight fp16 [H, rank],
       up_bias fp16 [H] or None, act_code).
    Returns contiguous CUDA tensors; raises on any shape/dtype mismatch (the
    model caller catches and falls back to the torch path).
    """
    assert len(chains) >= 1
    H = chains[0][0].shape[1]
    device = chains[0][0].device
    downs, ups, biases, meta = [], [], [], []
    roff = 0
    for dw, uw, b, act in chains:
        rank = dw.shape[0]
        assert dw.dtype == torch.float16 and dw.shape == (rank, H), "bad down weight"
        assert uw.dtype == torch.float16 and uw.shape == (H, rank), "bad up weight"
        assert act in (ACT_IDENTITY, ACT_TANH, ACT_SIGMOID), "bad act code"
        if b is None:
            b = torch.zeros(H, dtype=torch.float16, device=device)
        assert b.dtype == torch.float16 and b.shape == (H,), "bad up bias"
        downs.append(dw)
        ups.append(uw)
        biases.append(b)
        meta.append([roff, rank, act])
        roff += rank
    d_cat = torch.cat(downs, dim=0).contiguous()          # [R_total, H]
    u_cat = torch.cat(ups, dim=1).contiguous()            # [H, R_total]
    bias_cat = torch.stack(biases, dim=0).contiguous()    # [C, H]
    meta_t = torch.tensor(meta, dtype=torch.int32, device=device)
    return d_cat, u_cat, bias_cat, meta_t


def lora4_m1(
    xs: torch.Tensor,
    d_cat: torch.Tensor,
    u_cat: torch.Tensor,
    bias_cat: torch.Tensor,
    meta: torch.Tensor,
) -> torch.Tensor:
    """y[C,H]: all C LoRA chains for one decode token in 2 kernel launches.

    xs fp16 [C, H] = the chains' lerped inputs stacked in pack order. Caller
    (models/rwkv7.py) guarantees eligibility (fp16, M==1, packed weights built)."""
    return torch.ops.rwkv7_lora.lora4_m1(xs, d_cat, u_cat, bias_cat, meta)


def lora4_m1_gated(
    xs: torch.Tensor,
    d_cat: torch.Tensor,
    u_cat: torch.Tensor,
    bias_cat: torch.Tensor,
    meta: torch.Tensor,
    v: torch.Tensor,
    vfirst: torch.Tensor,
    inv_sqrt_e: float,
) -> torch.Tensor:
    """F0066c: lora4_m1 with the gate epilogue folded into stage2.

    Returns [C,H] with rows (w_log, a, g_raw[, v_new]) — byte-identical to
    lora4_m1 followed by fused_lora_gates (bench/test_lora_gated.py); the
    standalone _lora_gates_kernel launch and the raw-lo round trip are gone.
    v/vfirst are [H] fp16 contiguous; only read when C==4 (layer>0)."""
    return torch.ops.rwkv7_lora.lora4_m1_gated(
        xs, d_cat, u_cat, bias_cat, meta, v, vfirst, inv_sqrt_e)


def lora4_mn(
    xs: torch.Tensor,
    d_cat: torch.Tensor,
    u_cat: torch.Tensor,
    bias_cat: torch.Tensor,
    meta: torch.Tensor,
) -> torch.Tensor:
    """y[M,C,H]: all C LoRA chains for M decode tokens in 2 kernel launches
    (batched-M variant, ADR-0005 R3). Per-token result is byte-identical to
    lora4_m1(xs[m]) — see bench/test_lora_mn.py. xs fp16 [M, C, H]. Same packed
    weights as lora4_m1 (they are M-independent)."""
    return torch.ops.rwkv7_lora.lora4_mn(xs, d_cat, u_cat, bias_cat, meta)
