# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""RWKV-7 (Goose) model for sglang (M1c/M1d).

All elementwise math (token-shift lerp, projections, LoRAs, gating, GroupNorm,
gate-correction) is plain torch and matches `bench/oracle_numpy.py` exactly; only
the WKV recurrence is our own kernel (via Rwkv7AttnBackend). Module /
parameter names mirror the fla-format checkpoint so `load_weights` uses
`default_weight_loader` with no remapping.

M4 quantization: the linear projections (r/k/v/o_proj, ffn key/value) and the
LoRA down/up projections are sglang quant-aware `ReplicatedLinear` (tp=1) threaded
with `quant_config`. With `quant_config=None` they are unquantized `F.linear`
(bit-identical to the previous `nn.Linear`, so greedy stays EXACT). With
`--quantization w8a8_int8` (per-channel int8 weight, per-token dynamic int8
activation, sgl_kernel `int8_scaled_mm`) the weights drop to int8 — VRAM halves
and the int8 tensor cores keep decode at-least as fast as bf16 on Ampere. The WKV
recurrence/state and the small per-channel params (x_*, k_k, k_a, r_k, g_norm)
are NEVER quantized — they stay bf16/fp32.

Tensor parallelism is head-parallel: head_dim stays whole and whole heads are
split across ranks (r/k/v + LoRA-up column-parallel with no gather, per-channel
params / g_norm / WKV state on the local head slice, o_proj and ffn.value
row-parallel with a single allreduce each). The token-shift mix vectors and the
conv (prev-token) state stay full-width — they act on the replicated hidden
before the column-parallel projections. tp=1 keeps the exact original path.

Per-layer time-mix (att):
  shifted = prev_token(x);  x* = x + x_*·(shifted - x)
  r = r_proj(xr); k = k_proj(xk); v = v_proj(xv)
  w_log = -e^-0.5 * sigmoid( w_up(tanh(w_down(xw))) + w_bias )       # log decay
  a = sigmoid( a_up(a_down(xa)) + a_bias )
  g = g_up( sigmoid(g_down(xg)) )                                    # no bias
  v-residual (layer>0): v += (v_first - v) * sigmoid( v_up(v_down(xv)) + v_bias )
  kk = k * k_k ; k = k + k*(a-1)*k_a ; kk = L2norm(kk) over head_dim
  y = WKV(r, w_log, k, v, kk, a)                                     # backend kernel
  y = g_norm(y) + (r*k*r_k).sum * v ; out = o_proj(y * g)
Channel-mix (ffn): shifted=prev(x); xk = x + x_k·(shifted-x); out = value(relu(key(xk))**2)
"""

from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.configs.rwkv7 import Rwkv7Config
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.attention.rwkv7_kernels import fast_linear
from sglang.srt.layers.attention.rwkv7_kernels import sparse_cmix
from sglang.srt.layers.attention.rwkv7_kernels import w4_linear
from sglang.srt.layers.attention.rwkv7_kernels.fused import (
    fused_gate_corr,
    fused_kk_kmix,
    fused_lerp6,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers

import os

# e^-0.5 = 1/sqrt(e); w_log = -this * sigmoid(w_raw)  =>  decay = exp(w_log).
_INV_SQRT_E = 0.6065306597126334


# M6 CUDA endgame: route the big r/k/v/o + ffn projections through a hand-tuned
# fp16 GEMV (rwkv7_fast.gemv_m1, adapted from albatross, Apache-2.0) on the M==1
# (bsz1 decode) path. Standalone-benchmarked 1.09-1.61x faster than cuBLAS at M=1
# on the 3090 (0.1B/1.5B r/k/v/o ~1.6x; 7.2B ~1.1x), fp32-accurate to the same
# ULP as torch's fp16 matmul (bench/verify_fast_linear.py). fp16-only (the kernel
# reads at::Half; our precision-matched target is ours-fp16 vs albatross-fp16, and
# Ampere fp16==bf16). bf16/fp32/quantized + any M>1 keep the ReplicatedLinear path.
# Gate: greedy-EXACT (verify_m1d) before it can be the default. Default OFF.
_FAST_LINEAR = os.environ.get("RWKV_FAST_LINEAR", "0") == "1"

# M6 measurement gate: log the per-token zero-fraction of the ffn sqrelu activation
# (relu(k)^2 == 0 iff k<=0). Reproduces the 86-90% figure in bench/results/sparse_ffn/
# sparsity.log. Diagnostic only, env-gated, off by default. NOTE: it calls .item() (a
# device->host sync) so enabling it forces eager / disables cuda-graph — never leave it on
# for serving or benchmarking.
_LOG_SPARSITY = os.environ.get("RWKV_LOG_SPARSITY", "0") == "1"

# M6 phase-2: sparse channel-mix value-projection. relu(k)^2 is 86-90% exact-zero on real
# prompts (measured), so the hand-written sparse kernel skips ~9/10 of the value-weight
# reads — a TRUE bandwidth win past the dense ceiling, greedy-EXACT (0*w=0; fp32 accum),
# cuda-graph safe. bsz1 (M==1) + fp16 + unquantized + conforming shapes only; else dense.
# Default OFF (opt-in), gated on verify_m1d + verify_batch. See docs/design/m6-sparse-ffn.md.
_SPARSE_FFN = os.environ.get("RWKV_SPARSE_FFN", "0") == "1"

# M7 (req#5): weight-only int4 for the big r/k/v/o + ffn key/value projections. When on,
# those projections load as W4Linear (packed int4 + group scales) and decode (M==1) runs
# the hand-written bandwidth-optimal GEMV (rwkv7_w4.cu) — faster than fp16 + ~4x less
# weight VRAM. LoRA/norms/emb/head stay full precision. Opt-in; the checkpoint must be
# produced by bench/quant_w4.py (carries .qweight/.scale instead of .weight). Default OFF.
_W4 = os.environ.get("RWKV_W4", "0") == "1"

# M8: weight-only int8 (w8a16) — same hand-written kernel family as w4 but 8-bit:
# near-lossless (per-group int8 RTN), faster than fp16 at small M (1/2 the weight
# bytes), and — unlike the cutlass w8a8 path (sm80–90 only) — JIT-builds and runs on
# EVERY arch (Turing→Blackwell). Checkpoint from `bench/quant_w4.py --bits 8`.
_W8 = os.environ.get("RWKV_W8", "0") == "1"

# M7 calibration: capture per-projection input Hessians (X^T X) for GPTQ. Env-gated,
# zero cost when off. Run the fp16 model (RWKV_W4 off) through calibration prompts with
# RWKV_CALIB=1 + RWKV_CALIB_OUT=<dir>; Hessians dump to disk (dual trigger: token-count
# target AND atexit, so it survives the Engine subprocess teardown). Offline GPTQ
# (bench/gptq_w4.py) then reads them to produce a better int4 checkpoint (same
# .qweight/.scale format the kernel already serves — no kernel/model change).
_CALIB = os.environ.get("RWKV_CALIB", "0") == "1"
_CALIB_OUT = os.environ.get("RWKV_CALIB_OUT", "")
_CALIB_TOKENS = int(os.environ.get("RWKV_CALIB_TOKENS", "20000"))
_HESS: dict = {}
_NSAMP: dict = {}
_calib_state = {"dumped": False, "trigger": None}


def _calib_dump():
    if not _CALIB_OUT or not _HESS:
        return
    os.makedirs(_CALIB_OUT, exist_ok=True)
    payload = {"hessian": {k: v.detach().cpu() for k, v in _HESS.items()},
               "nsamp": dict(_NSAMP)}
    torch.save(payload, os.path.join(_CALIB_OUT, "calib_hessians.pt"))
    import sys
    print(f"[rwkv7 calib] dumped {len(_HESS)} Hessians ({_NSAMP.get(_calib_state['trigger'],0)} "
          f"tokens) -> {_CALIB_OUT}", file=sys.stderr, flush=True)


def _calib_accumulate(qname: str, x: torch.Tensor):
    xf = x.reshape(-1, x.shape[-1]).float()
    if qname not in _HESS:
        _HESS[qname] = xf.t() @ xf
        _NSAMP[qname] = xf.shape[0]
        if _calib_state["trigger"] is None:
            _calib_state["trigger"] = qname
    else:
        _HESS[qname].add_(xf.t() @ xf)
        _NSAMP[qname] += xf.shape[0]
    if (not _calib_state["dumped"] and qname == _calib_state["trigger"]
            and _NSAMP[qname] >= _CALIB_TOKENS):
        _calib_dump()
        _calib_state["dumped"] = True


if _CALIB:
    import atexit
    atexit.register(_calib_dump)


class W4Linear(nn.Module):
    """Weight-only group-wise symmetric int4 replacement for a bias-free ReplicatedLinear.

    Stores `qweight` (uint8 [N, K/2]) + `scale` (fp16 [N, K/GROUP]); decode (M==1, fp16)
    runs the hand-written int4 GEMV, everything else dequantizes to the activation dtype
    and uses F.linear (correctness-first; prefill is compute-bound). Buffers are named to
    match the bench/quant_w4.py checkpoint keys."""

    def __init__(self, in_features: int, out_features: int, group: int = w4_linear.GROUP):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        self.register_buffer(
            "qweight", torch.empty(out_features, in_features // 2, dtype=torch.uint8),
            persistent=True)
        self.register_buffer(
            "scale", torch.empty(out_features, in_features // group, dtype=torch.float16),
            persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = x.shape[0]
        if (
            x.dtype == torch.float16
            and (x.shape[-1] % self.group) == 0
            and w4_linear.available()
        ):
            if M == 1:
                return w4_linear.gemv_w4_m1(x, self.qweight, self.scale)
            # small batched decode: one int4 weight read feeds all M rows; each row
            # is bit-identical to the M==1 kernel (batch-invariant by construction).
            if 2 <= M <= 8 and (self.out_features % 2) == 0:
                return w4_linear.gemm_w4_small(x, self.qweight, self.scale)
            # medium batched decode: tensor-core GEMM with in-smem int4 dequant
            # (weight HBM traffic = 1/4 of cuBLAS fp16; wmma fp32 accumulate).
            if 8 < M <= 64 and (self.out_features % 64) == 0:
                return w4_linear.gemm_w4_tc(x, self.qweight, self.scale)
        # M>64 / prefill: dequant -> cuBLAS (compute-bound regime; weight read amortized)
        w = w4_linear.dequant(self.qweight, self.scale, self.group).to(x.dtype)
        return torch.nn.functional.linear(x, w)


class W8Linear(nn.Module):
    """Weight-only group-wise symmetric int8 (w8a16) bias-free projection — the 8-bit
    sibling of W4Linear (same dispatch shape: M==1 GEMV / 2<=M<=8 small-GEMM /
    M>8 dequant->cuBLAS). Near-lossless; runs on every arch (JIT, no cutlass)."""

    def __init__(self, in_features: int, out_features: int, group: int = w4_linear.GROUP):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        self.register_buffer(
            "qweight", torch.empty(out_features, in_features, dtype=torch.int8),
            persistent=True)
        self.register_buffer(
            "scale", torch.empty(out_features, in_features // group, dtype=torch.float16),
            persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = x.shape[0]
        if (
            x.dtype == torch.float16
            and (x.shape[-1] % self.group) == 0
            and w4_linear.w8_available()
        ):
            if M == 1:
                return w4_linear.gemv_w8_m1(x, self.qweight, self.scale)
            if 2 <= M <= 8 and (self.out_features % 2) == 0:
                return w4_linear.gemm_w8_small(x, self.qweight, self.scale)
            # medium batched decode: tensor-core GEMM with in-smem int8 dequant
            # (weight HBM traffic = 1/2 of cuBLAS fp16; wmma fp32 accumulate).
            if 8 < M <= 64 and (self.out_features % 64) == 0:
                return w4_linear.gemm_w8_tc(x, self.qweight, self.scale)
        # M>64 / prefill: dequant -> cuBLAS (compute-bound regime; weight read amortized)
        w = w4_linear.dequant_w8(self.qweight, self.scale, self.group).to(x.dtype)
        return torch.nn.functional.linear(x, w)


def _make_proj(in_f: int, out_f: int, quant_config, prefix: str, parallel: str = "column"):
    """A bias-free projection: W4Linear under RWKV_W4, W8Linear under RWKV_W8, else the
    quant-aware ReplicatedLinear (unquantized / w8a8-int8). Under tp>1 the projection
    is head-parallel instead: ColumnParallelLinear (output = this rank's head slice,
    no gather) or RowParallelLinear (local-slice input, allreduce inside)."""
    if get_tensor_model_parallel_world_size() > 1:
        if _W4 or _W8:
            raise NotImplementedError(
                "RWKV_W4/RWKV_W8 quantized projections require tp=1 for now"
            )
        if parallel == "row":
            m = RowParallelLinear(
                in_f, out_f, bias=False, input_is_parallel=True,
                reduce_results=True, quant_config=quant_config, prefix=prefix,
            )
        else:
            m = ColumnParallelLinear(
                in_f, out_f, bias=False, gather_output=False,
                quant_config=quant_config, prefix=prefix,
            )
    elif _W4:
        m = W4Linear(in_f, out_f)
    elif _W8:
        m = W8Linear(in_f, out_f)
    else:
        m = ReplicatedLinear(in_f, out_f, bias=False, quant_config=quant_config, prefix=prefix)
    m._qname = prefix  # for GPTQ calibration keying (see _calib_accumulate)
    return m


def _linear_backend(forward_batch: ForwardBatch):
    """The RWKV-7 linear-attention backend, across sglang versions: v0.5.10 hangs
    it off forward_batch.attn_backend; main moved it to the global forward context."""
    ab = getattr(forward_batch, "attn_backend", None)
    if ab is None:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        ab = get_attn_backend()
    return ab.linear_attn_backend


def _proj_gemv(layer, x: torch.Tensor, fast: bool) -> torch.Tensor:
    """r/k/v/o/ffn projection. W4Linear self-dispatches (int4 GEMV at M==1). Otherwise
    uses the fused fp16 GEMV ONLY on the eligible single-row decode path; anything the
    kernel can't handle falls back to the quant-aware sglang linear (never crashes).
    All these projections are bias-free, so gemv_m1 (no bias) is a drop-in. Eligibility
    mirrors the kernel's requirements so an odd-shaped checkpoint degrades gracefully:
    fast + M==1 + fp16 activation + fp16 contiguous weight + K%4==0 + N even."""
    if _CALIB and getattr(layer, "_qname", None):
        _calib_accumulate(layer._qname, x)
    if isinstance(layer, (W4Linear, W8Linear)):
        return layer(x)
    if (
        fast
        and x.shape[0] == 1
        and x.dtype == torch.float16
        and (x.shape[-1] % 4) == 0
    ):
        w = layer.weight
        if (
            w.dtype == torch.float16
            and w.is_contiguous()
            and (w.shape[0] % 2) == 0
        ):
            return fast_linear.gemv_m1(x, w)
    return layer(x)[0]


class Rwkv7LoRA(nn.Module):
    """fla low-rank block: up(act(down(x))) [+ bias].

    Keys: lora.0.weight (down), lora.2.weight (up), lora.2.bias (up bias).

    The down/up projections are sglang ``ReplicatedLinear`` (tp=1) so they are
    quant-aware (M4): with ``quant_config=None`` they fall through to an
    unquantized ``F.linear`` (bit-identical to ``nn.Linear``); with a quant
    config they carry int8/4-bit weights. The ``nn.Sequential`` is kept purely as
    a name container so checkpoint keys stay ``lora.0`` / ``lora.2`` (we drive the
    forward manually because ReplicatedLinear returns a ``(out, bias)`` tuple).

    Under tp>1 the down proj stays replicated (its input is the full replicated
    hidden and the rank-dim output is tiny, so every rank computes it locally,
    no comm) while the up proj is ColumnParallelLinear (no gather): its output —
    and its bias, sharded by the ColumnParallelLinear bias loader — is exactly
    this rank's head slice, matching the head-parallel r/k/v projections.
    """

    def __init__(
        self,
        hidden_size: int,
        low_rank: int,
        activation: str,
        bias: bool,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        else:
            act = nn.Identity()
        if get_tensor_model_parallel_world_size() > 1:
            up = ColumnParallelLinear(
                low_rank,
                hidden_size,
                bias=bias,
                gather_output=False,
                quant_config=quant_config,
                prefix=add_prefix("lora.2", prefix),
            )
        else:
            up = ReplicatedLinear(
                low_rank,
                hidden_size,
                bias=bias,
                quant_config=quant_config,
                prefix=add_prefix("lora.2", prefix),
            )
        self.lora = nn.Sequential(
            ReplicatedLinear(
                hidden_size,
                low_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("lora.0", prefix),
            ),
            act,
            up,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lora[0](x)
        h = self.lora[1](h)
        out, _ = self.lora[2](h)
        return out


class Rwkv7Attention(nn.Module):
    """RWKV-7 time-mixing block."""

    def __init__(
        self,
        config: Rwkv7Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        # WKV heads tile the channel dim exactly; g_norm(num_groups=num_heads,
        # num_channels=H) and every [T, nh, hd] reshape below silently corrupt if
        # this is violated, so fail loudly at construction instead.
        assert self.num_heads * self.head_dim == self.hidden_size, (
            f"RWKV-7 head geometry mismatch: num_heads({self.num_heads}) * "
            f"head_dim({self.head_dim}) != hidden_size({self.hidden_size})"
        )
        # Head-parallel TP: head_dim stays whole, whole heads are split across
        # ranks. Everything downstream of the r/k/v/LoRA-up projections (per-
        # channel params, g_norm, the WKV recurrence and its state) lives on
        # this rank's head slice; o_proj (row-parallel) restores the full H.
        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0, (
            f"RWKV-7 TP requires num_heads({self.num_heads}) divisible by "
            f"tp_size({tp_size})"
        )
        self.local_num_heads = self.num_heads // tp_size
        self.local_hidden_size = self.local_num_heads * self.head_dim

        H = self.hidden_size
        Hl = self.local_hidden_size
        # token-shift mix vectors (lerp coefficients)
        self.x_r = nn.Parameter(torch.zeros(1, 1, H))
        self.x_w = nn.Parameter(torch.zeros(1, 1, H))
        self.x_k = nn.Parameter(torch.zeros(1, 1, H))
        self.x_v = nn.Parameter(torch.zeros(1, 1, H))
        self.x_a = nn.Parameter(torch.zeros(1, 1, H))
        self.x_g = nn.Parameter(torch.zeros(1, 1, H))

        # Projections are quant-aware ReplicatedLinear (tp=1), or W4Linear under RWKV_W4.
        self.r_proj = _make_proj(H, H, quant_config, add_prefix("r_proj", prefix))
        self.k_proj = _make_proj(H, H, quant_config, add_prefix("k_proj", prefix))
        self.v_proj = _make_proj(H, H, quant_config, add_prefix("v_proj", prefix))
        self.o_proj = _make_proj(H, H, quant_config, add_prefix("o_proj", prefix),
                                 parallel="row")

        self.w_lora = Rwkv7LoRA(
            H, config.decay_low_rank_dim, "tanh", bias=True,
            quant_config=quant_config, prefix=add_prefix("w_lora", prefix),
        )
        self.a_lora = Rwkv7LoRA(
            H, config.a_low_rank_dim, "identity", bias=True,
            quant_config=quant_config, prefix=add_prefix("a_lora", prefix),
        )
        self.g_lora = Rwkv7LoRA(
            H, config.gate_low_rank_dim, "sigmoid", bias=False,
            quant_config=quant_config, prefix=add_prefix("g_lora", prefix),
        )
        if layer_id > 0:
            self.v_lora = Rwkv7LoRA(
                H, config.v_low_rank_dim, "identity", bias=True,
                quant_config=quant_config, prefix=add_prefix("v_lora", prefix),
            )

        self.k_k = nn.Parameter(torch.zeros(Hl))
        self.k_a = nn.Parameter(torch.zeros(Hl))
        self.r_k = nn.Parameter(torch.zeros(self.local_num_heads, self.head_dim))

        self.g_norm = nn.GroupNorm(
            num_groups=self.local_num_heads,
            num_channels=Hl,
            eps=self.head_dim * config.norm_eps,
            affine=True,
        )

        # M5 fusion: stacked token-shift mix vectors, lazily built (post weight-load)
        # on first forward and cached. Order [x_r, x_k, x_w, x_a, x_g, x_v].
        self._mix6 = None
        # M6: build the fp16 GEMV extension at load time (CUDA is up; graceful
        # fallback if the build fails). Only for the unquantized tp=1 path (the
        # kernel is fp16 dense, not int8-aware, and wraps the ReplicatedLinear
        # weight — under tp>1 the parallel linears run instead).
        self._fast = (
            _FAST_LINEAR and (quant_config is None) and tp_size == 1
            and fast_linear.available()
        )
        if self._fast and layer_id == 0:
            import sys
            print("[rwkv7] M6 fused fp16 GEMV projection path ENABLED "
                  "(bsz1 decode, fp16)", file=sys.stderr, flush=True)

    def _mix6_buf(self) -> torch.Tensor:
        if self._mix6 is None:
            self._mix6 = torch.stack(
                [
                    self.x_r.reshape(-1), self.x_k.reshape(-1), self.x_w.reshape(-1),
                    self.x_a.reshape(-1), self.x_g.reshape(-1), self.x_v.reshape(-1),
                ],
                dim=0,
            ).contiguous()
        return self._mix6


    def forward(
        self,
        forward_batch: ForwardBatch,
        x: torch.Tensor,
        v_first: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        T = x.shape[0]
        if T == 0:
            return x, v_first

        be = _linear_backend(forward_batch)
        # Local (per-rank) head slice; == the full width at tp=1.
        H, hd, nh = self.local_hidden_size, self.head_dim, self.local_num_heads

        # Fused triton elementwise path: bit-identical to the torch reference at
        # bf16/fp16 (verified), so it stacks with cuda-graph + int8. fp32 keeps the
        # original torch path (1-ULP reduction-order drift would risk the fp32 gate).
        fused = x.dtype != torch.float32

        shifted = be.token_shift(x, self.layer_id, 0, forward_batch)
        if fused:
            # [6,T,H] in order xr,xk,xw,xa,xg,xv
            lp = fused_lerp6(x, shifted, self._mix6_buf())
            xr, xk, xw, xa, xg, xv = lp[0], lp[1], lp[2], lp[3], lp[4], lp[5]
        else:
            d = shifted - x
            xr = x + self.x_r.view(-1) * d
            xw = x + self.x_w.view(-1) * d
            xk = x + self.x_k.view(-1) * d
            xv = x + self.x_v.view(-1) * d
            xa = x + self.x_a.view(-1) * d
            xg = x + self.x_g.view(-1) * d

        r = _proj_gemv(self.r_proj, xr, self._fast)
        k = _proj_gemv(self.k_proj, xk, self._fast)
        v = _proj_gemv(self.v_proj, xv, self._fast)

        if self.layer_id == 0:
            v_first = v

        # LoRA gates: w=decay, a=in-context-lr, g=output-gate, v=v-residual (layer>0).
        w_log = -torch.sigmoid(self.w_lora(xw)) * _INV_SQRT_E
        a = torch.sigmoid(self.a_lora(xa))
        g = self.g_lora(xg)
        if self.layer_id != 0:
            v = v + (v_first - v) * torch.sigmoid(self.v_lora(xv))

        if fused:
            # kk = L2norm(k·k_k) over hd; k <- k + k·(a-1)·k_a  (one launch)
            kk, k = fused_kk_kmix(k, a, self.k_k, self.k_a, nh)
            r = r.view(T, nh, hd)
            w_log = w_log.view(T, nh, hd)
            k = k.view(T, nh, hd)
            v = v.view(T, nh, hd)
            a = a.view(T, nh, hd)
        else:
            kk = k * self.k_k
            k = k + k * (a - 1.0) * self.k_a
            r = r.view(T, nh, hd)
            w_log = w_log.view(T, nh, hd)
            k = k.view(T, nh, hd)
            v = v.view(T, nh, hd)
            a = a.view(T, nh, hd)
            kk = kk.view(T, nh, hd)
            kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        o = be.recurrence(r, w_log, k, v, kk, a, self.layer_id, forward_batch)
        # o: [T, nh, hd]
        o = self.g_norm(o.reshape(T, H))
        if fused:
            # o = (g_norm(o) + (r*k*r_k).sum(-1)*v) * g   (one launch)
            o = fused_gate_corr(o, r, k, self.r_k, v, g, nh)
        else:
            gate_corr = ((r * k * self.r_k).sum(dim=-1, keepdim=True) * v).reshape(T, H)
            o = o + gate_corr
            o = o * g
        out = _proj_gemv(self.o_proj, o, self._fast)
        return out, v_first


class Rwkv7FeedForward(nn.Module):
    """RWKV-7 channel-mixing block (sqrelu)."""

    def __init__(
        self,
        config: Rwkv7Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        H = config.hidden_size
        self.hidden_size = H
        inter = config.intermediate_size
        self.x_k = nn.Parameter(torch.zeros(H))
        # tp>1: key is column-parallel (local inter slice; sqrelu is elementwise so
        # it acts per-slice), value is row-parallel (allreduce restores the full H).
        tp_size = get_tensor_model_parallel_world_size()
        self.key = _make_proj(H, inter, quant_config, add_prefix("key", prefix))
        self.value = _make_proj(inter, H, quant_config, add_prefix("value", prefix),
                                parallel="row")
        self._fast = (
            _FAST_LINEAR and (quant_config is None) and tp_size == 1
            and fast_linear.available()
        )
        # M6 sparse value-proj: eligible only unquantized tp=1 (not int8, not W4Linear;
        # it wraps the ReplicatedLinear weight); the tiled weight is built lazily on
        # the first (eager warmup) forward, once loaded.
        self._sparse = (
            _SPARSE_FFN and (quant_config is None) and tp_size == 1
            and not (_W4 or _W8)
        )
        self._value_tiled = None

    def forward(self, forward_batch: ForwardBatch, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        be = _linear_backend(forward_batch)
        shifted = be.token_shift(x, self.layer_id, 1, forward_batch)
        xk = x + self.x_k * (shifted - x)
        k = _proj_gemv(self.key, xk, self._fast)
        # M6 sparse value-projection on the eligible bsz1-decode path (kernel applies
        # relu()^2 to k internally, then a sparse fp32-accum SpMV skipping zero rows).
        if self._sparse and k.shape[0] == 1 and k.dtype == torch.float16:
            if self._value_tiled is None:
                if sparse_cmix.available() and sparse_cmix.conforms(self.value.weight):
                    self._value_tiled = sparse_cmix.tile_value_weight(
                        self.value.weight.detach()
                    )
                    if self.layer_id == 0:
                        import sys
                        print("[rwkv7] M6 sparse channel-mix value-proj ENABLED "
                              "(bsz1 decode, fp16)", file=sys.stderr, flush=True)
                else:
                    self._sparse = False  # not buildable → dense from here on
            if self._value_tiled is not None:
                return sparse_cmix.sparse_cmix(k, self._value_tiled, self.hidden_size)
        act = torch.relu(k) ** 2
        if _LOG_SPARSITY:
            import sys
            zf = (act == 0).float().mean().item()
            print(f"[sparsity] L{self.layer_id} rows={act.shape[0]} zero_frac={zf:.4f}",
                  file=sys.stderr, flush=True)
        out = _proj_gemv(self.value, act, self._fast)
        return out


class Rwkv7DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Rwkv7Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        H = config.hidden_size
        eps = config.norm_eps
        bias = config.norm_bias
        if layer_id == 0:
            # ln0: applied ONCE to the embeddings (driven from Rwkv7Model.forward).
            self.pre_norm = nn.LayerNorm(H, eps=eps, bias=bias)
        self.attn_norm = nn.LayerNorm(H, eps=eps, bias=bias)
        self.ffn_norm = nn.LayerNorm(H, eps=eps, bias=bias)
        self.attn = Rwkv7Attention(
            config, layer_id, quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.ffn = Rwkv7FeedForward(
            config, layer_id, quant_config=quant_config,
            prefix=add_prefix("ffn", prefix),
        )

    def forward(
        self,
        forward_batch: ForwardBatch,
        x: torch.Tensor,
        v_first: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, v_first = self.attn(forward_batch, self.attn_norm(x), v_first)
        x = x + attn_out
        x = x + self.ffn(forward_batch, self.ffn_norm(x))
        return x, v_first


class Rwkv7Model(nn.Module):
    def __init__(
        self,
        config: Rwkv7Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Rwkv7DecoderLayer(
                config, idx, quant_config=quant_config, prefix=prefix
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embeddings(input_ids)

        if x.shape[0] > 0:
            # ln0 on the embeddings (once), then the recurrent stack.
            x = self.layers[0].pre_norm(x)

        v_first = None
        for layer in self.layers:
            x, v_first = layer(forward_batch, x, v_first)

        x = self.norm(x)
        return x


class Rwkv7ForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    # ---- BitsAndBytes (4-bit nf4 / 8-bit) support metadata ----
    # RWKV-7 has no fused/stacked projections (r/k/v/o are separate linears), so
    # the stacked-params mapping is empty. The target modules list the linear
    # sub-modules the bnb loader should quantize on the fly (substring match on
    # the checkpoint weight name); it mirrors the ReplicatedLinear layers above.
    bitsandbytes_stacked_params_mapping = {}
    default_bitsandbytes_target_modules = [
        ".r_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".key.",
        ".value.",
        ".lora.0.",
        ".lora.2.",
    ]

    def __init__(
        self,
        config: Rwkv7Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        # M1: no pipeline parallelism / no speculative decoding.
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank, (
            "RWKV-7 (M1) does not support pipeline parallelism."
        )
        self.model = Rwkv7Model(config, quant_config, prefix=add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embeddings.weight, self.lm_head.weight

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Set[str]:
        params_dict = dict(self.named_parameters())
        # W4Linear (RWKV_W4) stores int4 qweight + group scale as BUFFERS, not params —
        # include them so the .qweight/.scale checkpoint keys resolve.
        params_dict.update(dict(self.named_buffers()))
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        # Head-sharded per-channel params (tp>1): the checkpoint stores the full
        # tensor; narrow dim 0 (channels resp. heads) to this rank's head slice
        # before the plain copy. Parallel linears shard via their own weight_loader.
        _head_sharded = (".k_k", ".k_a", ".r_k", ".g_norm.weight", ".g_norm.bias")
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                raise KeyError(
                    f"[rwkv7.load_weights] unexpected checkpoint key: {name}"
                )
            param = params_dict[name]
            if tp_size > 1 and name.endswith(_head_sharded):
                shard = param.shape[0]
                loaded_weight = loaded_weight.narrow(0, tp_rank * shard, shard)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # Assert every model parameter was loaded (catches naming mismatches).
        missing = set(params_dict.keys()) - loaded_params
        if missing:
            raise RuntimeError(
                f"[rwkv7.load_weights] {len(missing)} params not loaded, e.g. "
                f"{sorted(missing)[:8]}"
            )
        return loaded_params


# config.json architectures = ["RWKV7ForCausalLM"]; the registry keys by class
# __name__, so expose that spelling too (thin subclass).
class RWKV7ForCausalLM(Rwkv7ForCausalLM):
    pass


EntryClass = [Rwkv7ForCausalLM, RWKV7ForCausalLM]
