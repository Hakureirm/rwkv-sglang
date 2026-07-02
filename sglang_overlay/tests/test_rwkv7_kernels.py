#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kernel GATE: OUR FLA-free RWKV-7 WKV recurrence triton kernel (`wkv_recurrent`,
ADR-0004 / M3b) must match a pure-torch fp32 naive recurrence on random tensors.

`wkv_recurrent` serves BOTH production WKV paths:
  - DECODE  (cu_seqlens=None, T==1, batched over requests)
  - EXTEND / recurrent-PREFILL (cu_seqlens set, packed B==1 varlen)
plus the general batched multi-step recurrence (cu_seqlens=None, T>1).

Ground-truth math: refs/fla/.../rwkv7/RWKV7(Goose).md `naive_recurrent_rwkv7`
(== bench/oracle_numpy.py `time_mixing`).

Convention notes (load-bearing):
  * `w` passed to the kernel and the reference is the LOG decay; the per-step
    decay factor is `exp(w)` (the triton kernel applies a single exp). A realistic
    log-decay is generated as `w = -0.6065 * sigmoid(randn)` so exp(w) in (~0.545, 1).
  * `kk` = L2-normalized (over K) key-key product. `a_lr` = in-context learning
    rate (sigmoid in (0,1)). The kernel forms a_kernel=-kk, b_kernel=kk*a_lr
    INSIDE, so its signature is `wkv_recurrent(r, w, k, v, kk, a_lr, ...)`.
  * Tensor layout for all kernel inputs: [B, T, H, K] (r,w,k,kk,a_lr) and
    [B, T, H, V] (v). NOT head-first. State/initial/final state: [N, H, K, V].
"""

import sys

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.rwkv7_kernels import wkv_recurrent

DTYPE = torch.float32
TOL = 1e-2  # fp32 gate: max|Δ| / ref.std() < 1e-2 (aim < 1e-3)


# ----------------------------------------------------------------------------- #
# Reference: pure-torch fp32 naive recurrence (kernel state layout [H, K, V]).
# Per step (per b,h): with S of shape [K, V]
#   sa[v]    = sum_k a_kernel[k] * S[k, v]
#   S[k, v]  = exp(w[k]) * S[k, v] + b_kernel[k] * sa[v] + k[k] * v[v]
#   o[v]     = sum_k S[k, v] * (r[k] * scale)
# Matches RWKV7(Goose).md (its state is the [V,K] transpose; output is identical).
# ----------------------------------------------------------------------------- #
def _naive_one_seq(r, w, k, v, a_kernel, b_kernel, scale, S):
    # r,w,k,a_kernel,b_kernel: [T, H, K]; v: [T, H, V]; S: [H, K, V]
    T = r.shape[0]
    outs = []
    for t in range(T):
        rt, wt, kt = r[t], w[t], k[t]
        vt, at, bt = v[t], a_kernel[t], b_kernel[t]
        sa = (at.unsqueeze(-1) * S).sum(dim=1)               # [H, V]
        S = (
            torch.exp(wt).unsqueeze(-1) * S
            + bt.unsqueeze(-1) * sa.unsqueeze(1)
            + kt.unsqueeze(-1) * vt.unsqueeze(1)
        )                                                    # [H, K, V]
        ot = (S * (rt * scale).unsqueeze(-1)).sum(dim=1)     # [H, V]
        outs.append(ot)
    return torch.stack(outs, dim=0), S                       # [T, H, V], [H, K, V]


def naive_recurrent_rwkv7(r, w, k, v, a_kernel, b_kernel, scale,
                          initial_state=None, cu_seqlens=None):
    B, T_total, H, K = r.shape
    V = v.shape[-1]
    dev = r.device
    r, w, k, v, a_kernel, b_kernel = (
        x.to(DTYPE) for x in (r, w, k, v, a_kernel, b_kernel)
    )

    def init_state(idx):
        if initial_state is not None:
            return initial_state[idx].to(DTYPE).clone()
        return torch.zeros(H, K, V, dtype=DTYPE, device=dev)

    if cu_seqlens is None:
        o = torch.empty(B, T_total, H, V, dtype=DTYPE, device=dev)
        ht = torch.empty(B, H, K, V, dtype=DTYPE, device=dev)
        for bi in range(B):
            ob, S = _naive_one_seq(
                r[bi], w[bi], k[bi], v[bi], a_kernel[bi], b_kernel[bi],
                scale, init_state(bi),
            )
            o[bi], ht[bi] = ob, S
        return o, ht

    N = len(cu_seqlens) - 1
    o = torch.empty(1, T_total, H, V, dtype=DTYPE, device=dev)
    ht = torch.empty(N, H, K, V, dtype=DTYPE, device=dev)
    for n in range(N):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        ob, S = _naive_one_seq(
            r[0, bos:eos], w[0, bos:eos], k[0, bos:eos], v[0, bos:eos],
            a_kernel[0, bos:eos], b_kernel[0, bos:eos], scale, init_state(n),
        )
        o[0, bos:eos], ht[n] = ob, S
    return o, ht


# ----------------------------------------------------------------------------- #
# Input synthesis
# ----------------------------------------------------------------------------- #
def make_inputs(B, T, H, K, V, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=DTYPE)

    r = rn(B, T, H, K) * 0.5
    k = rn(B, T, H, K) * 0.5
    v = rn(B, T, H, V) * 0.5
    # realistic log-decay: exp(w) in (~0.545, 1)
    w = -0.6065306597126334 * torch.sigmoid(rn(B, T, H, K))
    # kk: L2-normalized over K
    kk = F.normalize(rn(B, T, H, K), dim=-1)
    # a_lr: in-context learning rate in (0, 1)
    a_lr = torch.sigmoid(rn(B, T, H, K))

    a_kernel = -kk
    b_kernel = kk * a_lr
    return r, w, k, v, kk, a_lr, a_kernel, b_kernel


def rel_err(out, ref):
    out = out.to(DTYPE)
    ref = ref.to(DTYPE)
    denom = ref.std().item()
    denom = denom if denom > 0 else 1.0
    return (out - ref).abs().max().item() / denom


# ----------------------------------------------------------------------------- #
# Cases
# ----------------------------------------------------------------------------- #
def run_case(name, B, T, H, K, V, seed, varlen, decode, use_init, device):
    cu = None
    initial_state = None
    N = B
    if varlen:
        # packed B==1
        assert B == 1
        # build segment lengths
        cu = torch.tensor([0, 30, 77, 128], dtype=torch.long, device=device)
        T = int(cu[-1])
        N = len(cu) - 1

    r, w, k, v, kk, a_lr, a_kernel, b_kernel = make_inputs(B, T, H, K, V, seed, device)

    if use_init:
        gi = torch.Generator(device=device).manual_seed(seed + 999)
        initial_state = torch.randn(N, H, K, V, generator=gi, device=device, dtype=DTYPE) * 0.1

    scale = K ** -0.5

    o_ref, ht_ref = naive_recurrent_rwkv7(
        r, w, k, v, a_kernel, b_kernel, scale,
        initial_state=initial_state, cu_seqlens=cu,
    )

    # ---- OUR wkv_recurrent kernel (decode / varlen-prefill / multi-step) ----
    o_rec, ht_rec = wkv_recurrent(
        r, w, k, v, kk, a_lr,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu,
    )
    results = {
        "rec_o": rel_err(o_rec, o_ref),
        "rec_ht": rel_err(ht_rec, ht_ref),
    }

    tag = "DECODE" if decode else ("VARLEN" if varlen else "PREFILL")
    print(
        f"[{tag:7s}] {name:22s} B={B} T={T} H={H} K={K} V={V} "
        f"init={int(use_init)} | "
        f"rec_o={results['rec_o']:.3e} rec_ht={results['rec_ht']:.3e}"
    )
    results["decode"] = decode
    return results


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; this gate requires a GPU.")
        sys.exit(2)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"torch={torch.__version__}  device={torch.cuda.get_device_name(0)}")
    import triton
    print(f"triton={triton.__version__}  tol={TOL}")
    print("-" * 110)

    cases = [
        # name, B, T, H, K, V, seed, varlen, decode, use_init
        dict(name="prefill_basic",     B=2, T=64,  H=4, K=64, V=64, seed=0,  varlen=False, decode=False, use_init=False),
        dict(name="prefill_nonmult",   B=2, T=130, H=4, K=64, V=64, seed=1,  varlen=False, decode=False, use_init=False),
        dict(name="prefill_init",      B=2, T=96,  H=4, K=64, V=64, seed=2,  varlen=False, decode=False, use_init=True),
        dict(name="varlen_packed",     B=1, T=0,   H=4, K=64, V=64, seed=3,  varlen=True,  decode=False, use_init=False),
        dict(name="varlen_packed_init", B=1, T=0,  H=4, K=64, V=64, seed=4,  varlen=True,  decode=False, use_init=True),
        dict(name="decode_t1",         B=4, T=1,   H=4, K=64, V=64, seed=5,  varlen=False, decode=True,  use_init=True),
        dict(name="decode_t1_noinit",  B=3, T=1,   H=4, K=64, V=64, seed=6,  varlen=False, decode=True,  use_init=False),
    ]

    # wkv_recurrent serves every regime (decode T==1, varlen-prefill, multi-step),
    # so its output AND final-state are gated across ALL cases.
    worst = {"rec_o": 0.0, "rec_ht": 0.0}
    for c in cases:
        res = run_case(device=device, **c)
        worst["rec_o"] = max(worst["rec_o"], res["rec_o"])
        worst["rec_ht"] = max(worst["rec_ht"], res["rec_ht"])

    print("-" * 110)
    print(f"WORST: rec_o={worst['rec_o']:.3e} rec_ht={worst['rec_ht']:.3e}")

    fail = []
    if worst["rec_o"] >= TOL:
        fail.append(f"wkv_recurrent output err {worst['rec_o']:.3e} >= {TOL}")
    if worst["rec_ht"] >= TOL:
        fail.append(f"wkv_recurrent final-state err {worst['rec_ht']:.3e} >= {TOL}")

    if fail:
        print("GATE: FAIL")
        for f in fail:
            print("  -", f)
        sys.exit(1)
    print("GATE: PASS  (wkv_recurrent matches the naive fp32 recurrence within tol)")
    sys.exit(0)


if __name__ == "__main__":
    main()
