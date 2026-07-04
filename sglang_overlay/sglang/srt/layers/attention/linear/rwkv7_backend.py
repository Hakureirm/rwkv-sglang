# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""RWKV-7 (Goose) linear-attention backend for sglang (M1c/M1d).

Unlike GDN, the RWKV-7 model module does ALL projections / LoRAs / gating in plain
torch and hands the backend already-projected per-head tensors. This backend owns
the two pieces of recurrent state in the MambaPool:

  * conv[0] / conv[1] : the two width-2 (prev-token) token-shift states
                        (attn / ffn), shape (size+1, hidden, 1), fp32.
  * temporal          : the WKV recurrent state S, shape (size+1, H, K, V), fp32.

The model calls `token_shift(...)` (before projections) and `recurrence(...)`
(after projections) directly on this backend instance, which it obtains via
`forward_batch.attn_backend.linear_attn_backend`. We therefore bypass the
RadixLinearAttention / HybridLinearAttnBackend.forward dispatch (whose fixed
mixed_qkv/a/b signature does not fit RWKV-7's r/w/k/v/kk/a). `init_forward_metadata`
is still driven through HybridLinearAttnBackend, so `self.forward_metadata`
(query_start_loc + mamba_cache_indices) is populated normally.
"""

import os
from typing import Optional

import torch

# R2 (ADR-0005): fuse the paged token-shift + lerp into one kernel (shift_lerp*),
# keeping the shifted intermediate on-chip. Default off until serving-validated.
_FUSED_GLUE = os.environ.get("RWKV_FUSED_GLUE", "0") == "1"

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# M3b (ADR-0004): the WKV recurrence is now OUR own FLA-free triton kernel for
# BOTH the decode (T==1) AND the extend/prefill (packed varlen via cu_seqlens)
# path. The deliverable carries zero flash-linear-attention dependency.
from sglang.srt.layers.attention.rwkv7_kernels import wkv_recurrent
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class Rwkv7NoOpFullAttnBackend(AttentionBackend):
    """A trivial full-attention backend for the all-linear RWKV-7 case.

    RWKV-7 has ZERO full-attention layers, so HybridLinearAttnBackend never routes
    any layer to the full backend. But it still calls `init_forward_metadata` (and a
    few cuda-graph hooks) on the full backend each step. Real backends
    (triton/flashinfer) either probe the empty full KV pool at construction or reject
    fp32 planning, so we substitute this no-op instead.
    """

    # HybridLinearAttnBackend (sglang main) copies these off the full backend at
    # construction; there is no full-attn KV pool here, and the req/token pools
    # come from the runner when one is given.
    token_to_kv_pool = None
    req_to_token_pool = None
    needs_cpu_seq_lens = False

    def __init__(self, model_runner: Optional[ModelRunner] = None):
        if model_runner is not None:
            self.req_to_token_pool = model_runner.req_to_token_pool
            self.token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
            self.max_context_len = model_runner.model_config.context_len

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def init_cpu_graph_state(self, *args, **kwargs):
        pass

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        pass

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        pass

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("RWKV-7 has no full-attention layers.")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("RWKV-7 has no full-attention layers.")


class Rwkv7AttnBackend(MambaAttnBackendBase):
    """Linear-attention backend for RWKV-7.

    Both the decode and the extend/prefill paths run through OUR own FLA-free
    `wkv_recurrent` triton kernel (exact, ~1e-6 vs the numpy oracle). scale=1.0 to
    match the numpy oracle, which applies no scaling to r before GroupNorm.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # conv[0] shape = (num_layers, size+1, hidden, 1)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        self.scale = 1.0
        import os
        if os.environ.get("RWKV_PAR_DEBUG") == "1":
            import sys
            mc = model_runner.req_to_token_pool.mamba_pool.mamba_cache
            print(f"[par-debug] backend: conv0={tuple(mc.conv[0].shape)} "
                  f"temporal={tuple(mc.temporal.shape)}", file=sys.stderr, flush=True)

    # ---- token-shift (width-2 causal shift via the conv state) ----
    def token_shift(
        self,
        x: torch.Tensor,
        layer_id: int,
        conv_idx: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Return the previous-token hidden state for each position in x.

        x: [num_tokens, hidden] (post attn_norm / ffn_norm). Updates the stored
        conv state with the last token of each sequence for the next step.
        """
        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv = cache.conv[conv_idx]  # [size+1, hidden, 1]
        md = self.forward_metadata
        cache_indices = md.mamba_cache_indices

        if forward_batch.forward_mode.is_decode_or_idle():
            # one token per request: shifted = stored prev; store current.
            # Padded cuda-graph replay fills the tail of mamba_cache_indices with
            # PAD_SLOT_ID = -1; torch advanced indexing would WRAP -1 to row `size`
            # (an allocatable live slot) and corrupt a real request's shift state.
            # Route pads to row 0, the MambaPool's reserved never-allocated slot
            # (free_slots = arange(1, size+1)); pad outputs are discarded anyway.
            safe_idx = torch.clamp_min(cache_indices, 0)
            prev = conv[safe_idx, :, 0].clone()  # [n, hidden]
            conv[safe_idx, :, 0] = x.to(conv.dtype)
            return prev.to(x.dtype)

        # extend (packed B=1, varlen via query_start_loc)
        qsl = md.query_start_loc.to(torch.long)
        starts = qsl[:-1]
        ends = qsl[1:]
        shifted = torch.empty_like(x)
        if x.shape[0] > 1:
            shifted[1:] = x[:-1]
        # Route cuda-graph pad indices (-1) to the pool reserved row 0, same
        # convention as the decode path.
        safe_idx = torch.clamp_min(cache_indices, 0)
        # The Mamba pool does not guarantee zeroed slots on free/alloc; RWKV-7
        # reads the stored state directly, so zero the shift state of sequences
        # starting from scratch (prefix 0), keeping chunked-prefill / state-cache
        # carry-ins (prefix > 0). Branch-free + host-copy-free (capture-safe on
        # engines that capture extend): fresh rows -> own slot, others -> row 0.
        prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
        if prefix_lens is not None:
            zero_target = torch.where(
                prefix_lens == 0, safe_idx, torch.zeros_like(safe_idx)
            )
            conv.index_fill_(0, zero_target.to(torch.long), 0.0)
        # first token of each sequence reads the stored prev-token (0 for fresh
        # reqs; the correct carry-in for chunked prefill / prefix continuation).
        shifted[starts] = conv[safe_idx, :, 0].to(x.dtype)
        # store last token of each sequence for the next chunk / decode.
        conv[safe_idx, :, 0] = x[ends - 1].to(conv.dtype)
        return shifted

    def _fused_glue_conv(self, layer_id, conv_idx, normed, forward_batch):
        """Shared eligibility check for the fused shift+lerp glue (R2). Returns
        (conv, cache_indices) when eligible (RWKV_FUSED_GLUE, decode, fp16 normed,
        fp32 contiguous conv, int32 contiguous cache_indices, glue built), else
        None so the caller falls back to token_shift + fused_lerp*."""
        if not (_FUSED_GLUE and forward_batch.forward_mode.is_decode_or_idle()
                and normed.dtype == torch.float16):
            return None
        conv = self.req_to_token_pool.mamba2_layer_cache(layer_id).conv[conv_idx]
        if conv.dtype != torch.float32 or not conv.is_contiguous():
            return None
        ci = self.forward_metadata.mamba_cache_indices
        if ci.dtype != torch.int32 or not ci.is_contiguous():
            return None
        from sglang.srt.layers.attention.rwkv7_kernels import glue
        if not glue.available():
            return None
        return conv, ci

    def try_fused_shift_lerp6(self, normed, layer_id, conv_idx, mix6, forward_batch):
        """Fused paged token-shift + 6-way lerp -> lp6[6,T,H], or None (fallback).
        Byte-exact vs token_shift + fused_lerp6 (bench/test_glue.py)."""
        e = self._fused_glue_conv(layer_id, conv_idx, normed, forward_batch)
        if e is None or mix6.dtype != torch.float16:
            return None
        conv, ci = e
        from sglang.srt.layers.attention.rwkv7_kernels import glue
        return glue.shift_lerp6(normed.contiguous(), mix6, ci, conv)

    def try_fused_shift_lerp1(self, normed, layer_id, conv_idx, x_k, forward_batch):
        """Fused paged token-shift + 1-way lerp -> xk[T,H], or None (fallback)."""
        e = self._fused_glue_conv(layer_id, conv_idx, normed, forward_batch)
        if e is None or x_k.dtype != torch.float16:
            return None
        conv, ci = e
        from sglang.srt.layers.attention.rwkv7_kernels import glue
        return glue.shift_lerp1(normed.contiguous(), x_k.reshape(-1).contiguous(), ci, conv)

    # ---- WKV recurrence (decode + extend both -> OUR wkv_recurrent kernel) ----
    def recurrence(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """r,w,k,kk,a: [num_tokens, H, K]; v: [num_tokens, H, V]. Returns [num_tokens, H, V]."""
        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        temporal = cache.temporal  # [size+1, H, K, V] fp32
        md = self.forward_metadata
        cache_indices = md.mamba_cache_indices

        if forward_batch.forward_mode.is_decode_or_idle():
            # [bs, H, *] -> [bs, 1, H, *]
            r4 = r.unsqueeze(1).contiguous()
            w4 = w.unsqueeze(1).contiguous()
            k4 = k.unsqueeze(1).contiguous()
            v4 = v.unsqueeze(1).contiguous()
            kk4 = kk.unsqueeze(1).contiguous()
            a4 = a.unsqueeze(1).contiguous()
            # In-place indexed state: the kernel reads/writes temporal[cache_indices]
            # directly (skips the gather+scatter copies; ~3x less state traffic at large
            # bsz). Same reduction math + bits -> greedy-EXACT + verify_batch preserved.
            o, _ = wkv_recurrent(
                r4, w4, k4, v4, kk4, a4,
                scale=self.scale,
                state_pool=temporal,
                cache_indices=cache_indices,
            )
            return o.squeeze(1)  # [bs, H, V]

        # extend: packed B=1, varlen -> OUR recurrent kernel (de-FLA, ADR-0004).
        # Same pad + slot-reuse handling as token_shift: pads -> reserved row 0,
        # fresh sequences recurrent state zeroed before snapshotting.
        safe_idx = torch.clamp_min(cache_indices, 0)
        prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
        if prefix_lens is not None:
            zero_target = torch.where(
                prefix_lens == 0, safe_idx, torch.zeros_like(safe_idx)
            )
            temporal.index_fill_(0, zero_target.to(torch.long), 0.0)
        init_state = temporal[safe_idx].contiguous().float()  # [N, H, K, V]
        cu = md.query_start_loc.to(torch.int64)
        r1 = r.unsqueeze(0).contiguous()
        w1 = w.unsqueeze(0).contiguous()
        k1 = k.unsqueeze(0).contiguous()
        v1 = v.unsqueeze(0).contiguous()
        kk1 = kk.unsqueeze(0).contiguous()
        a1 = a.unsqueeze(0).contiguous()
        o, final_state = wkv_recurrent(
            r1, w1, k1, v1, kk1, a1,
            scale=self.scale,
            initial_state=init_state,
            output_final_state=True,
            cu_seqlens=cu,
        )
        temporal[safe_idx] = final_state.to(temporal.dtype)
        return o.squeeze(0)  # [total_T, H, V]

    # The model calls token_shift/recurrence directly; these are not used.
    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Rwkv7AttnBackend.recurrence/token_shift are called directly by the model."
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Rwkv7AttnBackend.recurrence/token_shift are called directly by the model."
        )
