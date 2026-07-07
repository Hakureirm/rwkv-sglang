#!/usr/bin/env python3
"""RWKV-7 per-component GPU-time profiler (READ-ONLY; standalone, no engine).

Builds the REAL deployed modules (Rwkv7Attention / Rwkv7FeedForward) with the
real config dims, plus a stub backend whose token_shift/recurrence replicate the
deployed rwkv7_backend.py hot-path verbatim (so the math/kernels are identical to
production). Random weights -> matmul/kernel GPU time is value-independent.

Per-component timing via CUDA events, two ways:
  * eager  : N back-to-back launches, timed -> includes per-op launch overhead
  * graphed: same N launches captured in a CUDAGraph + replayed -> pure GPU busy
The gap (eager-graphed) is the launch/overhead attributable to that op; graphed is
what production (cuda-graph ON) actually pays. We report graphed as primary.

Subcommands: decode | prefill | params
"""
import argparse, json, os, sys
import torch
from torch import nn

from sglang.srt.configs.rwkv7 import Rwkv7Config
from sglang.srt.models.rwkv7 import Rwkv7Attention, Rwkv7FeedForward, Rwkv7DecoderLayer
from sglang.srt.layers.attention.rwkv7_kernels import wkv_recurrent

_INV_SQRT_E = 0.6065306597126334


def P(m, x):
    """Call a projection; deployed ReplicatedLinear returns (out, bias) tuples."""
    r = m(x)
    return r[0] if isinstance(r, tuple) else r


# ----------------------------- stub plumbing -----------------------------
class _Cache:
    def __init__(self, conv0, conv1, temporal):
        self.conv = [conv0, conv1]
        self.temporal = temporal


class _Pool:
    def __init__(self, caches):
        self.caches = caches

    def mamba2_layer_cache(self, layer_id):
        return self.caches[layer_id]


class _Meta:
    def __init__(self, cache_indices, query_start_loc=None):
        self.mamba_cache_indices = cache_indices
        self.query_start_loc = query_start_loc


class _Mode:
    def __init__(self, decode):
        self._d = decode

    def is_decode_or_idle(self):
        return self._d


class _Backend:
    """Replicates rwkv7_backend.py token_shift + recurrence (decode + extend)."""

    def __init__(self, pool, meta, decode, scale=1.0):
        self.req_to_token_pool = pool
        self.forward_metadata = meta
        self.scale = scale
        self._decode = decode

    def token_shift(self, x, layer_id, conv_idx, forward_batch):
        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv = cache.conv[conv_idx]
        md = self.forward_metadata
        ci = md.mamba_cache_indices
        if self._decode:
            prev = conv[ci, :, 0].clone()
            conv[ci, :, 0] = x.to(conv.dtype)
            return prev.to(x.dtype)
        qsl = md.query_start_loc.to(torch.long)
        starts = qsl[:-1]
        ends = qsl[1:]
        shifted = torch.empty_like(x)
        if x.shape[0] > 1:
            shifted[1:] = x[:-1]
        shifted[starts] = conv[ci, :, 0].to(x.dtype)
        conv[ci, :, 0] = x[ends - 1].to(conv.dtype)
        return shifted

    def recurrence(self, r, w, k, v, kk, a, layer_id, forward_batch):
        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        temporal = cache.temporal
        md = self.forward_metadata
        ci = md.mamba_cache_indices
        if self._decode:
            r4 = r.unsqueeze(1).contiguous(); w4 = w.unsqueeze(1).contiguous()
            k4 = k.unsqueeze(1).contiguous(); v4 = v.unsqueeze(1).contiguous()
            kk4 = kk.unsqueeze(1).contiguous(); a4 = a.unsqueeze(1).contiguous()
            o, _ = wkv_recurrent(r4, w4, k4, v4, kk4, a4, scale=self.scale,
                                 state_pool=temporal, cache_indices=ci)
            return o.squeeze(1)
        init_state = temporal[ci].contiguous().float()
        cu = md.query_start_loc.to(torch.int64)
        r1 = r.unsqueeze(0).contiguous(); w1 = w.unsqueeze(0).contiguous()
        k1 = k.unsqueeze(0).contiguous(); v1 = v.unsqueeze(0).contiguous()
        kk1 = kk.unsqueeze(0).contiguous(); a1 = a.unsqueeze(0).contiguous()
        o, fs = wkv_recurrent(r1, w1, k1, v1, kk1, a1, scale=self.scale,
                              initial_state=init_state, output_final_state=True,
                              cu_seqlens=cu)
        temporal[ci] = fs.to(temporal.dtype)
        return o.squeeze(0)

    # ---- R2 fused paged token-shift + lerp glue (mirrors the real backend so the
    # `kernels`/`decode` profiles exercise the CURRENT deployed fused stack; the
    # standalone stub predated the glue and would AttributeError on fp16/bf16). ----
    def _glue_conv(self, layer_id, conv_idx, normed):
        if normed.dtype != torch.float16 or not self._decode:
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
        e = self._glue_conv(layer_id, conv_idx, normed)
        if e is None or mix6.dtype != torch.float16:
            return None
        conv, ci = e
        from sglang.srt.layers.attention.rwkv7_kernels import glue
        return glue.shift_lerp6(normed.contiguous(), mix6, ci, conv)

    def try_fused_shift_lerp1(self, normed, layer_id, conv_idx, x_k, forward_batch):
        e = self._glue_conv(layer_id, conv_idx, normed)
        if e is None or x_k.dtype != torch.float16:
            return None
        conv, ci = e
        from sglang.srt.layers.attention.rwkv7_kernels import glue
        return glue.shift_lerp1(normed.contiguous(), x_k.reshape(-1).contiguous(), ci, conv)


class _FB:
    def __init__(self, backend, decode):
        class _AB: pass
        self.attn_backend = _AB()
        self.attn_backend.linear_attn_backend = backend
        self.forward_mode = _Mode(decode)


# ----------------------------- timing helpers -----------------------------
def time_eager(fn, n_iter, warmup=30, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        s.record()
        for _ in range(n_iter):
            fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / n_iter)
    return best  # ms per op


def time_graph(fn, n_iter, warmup=30, reps=5):
    # warm + capture n_iter launches, replay, time
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            for _ in range(n_iter):
                fn()
    except Exception as ex:
        return None  # not capturable
    torch.cuda.synchronize()
    best = float("inf")
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / n_iter)
    return best  # ms per op


# ----------------------------- model build -----------------------------
def load_cfg(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        j = json.load(f)
    return Rwkv7Config(**{k: v for k, v in j.items() if k != "architectures"}), j


def count_params(cfg):
    """Param counts (numel) by group for the whole model, from real modules."""
    L = cfg.num_hidden_layers
    H = cfg.hidden_size
    dev = "cpu"
    l0 = Rwkv7Attention(cfg, 0)
    l1 = Rwkv7Attention(cfg, 1)
    ff = Rwkv7FeedForward(cfg, 1)
    def n(m): return sum(p.numel() for p in m.parameters())
    attn0 = n(l0); attn1 = n(l1); ffn = n(ff)
    ln = H * 2  # LayerNorm w+b
    # per-layer: attn + ffn + attn_norm + ffn_norm
    per_layer_attn = attn1  # layers 1..L-1
    layers_total = attn0 + (L - 1) * attn1 + L * ffn + L * (2 * ln)
    pre_norm = ln  # layer0 ln0
    final_norm = ln
    emb = cfg.vocab_size * H
    lm_head = cfg.vocab_size * H
    total = layers_total + pre_norm + final_norm + emb + lm_head
    return {
        "H": H, "L": L, "attn_layer0": attn0, "attn_layer1": attn1, "ffn": ffn,
        "layers_total": layers_total, "pre_norm": pre_norm, "final_norm": final_norm,
        "emb": emb, "lm_head": lm_head, "total": total,
        "total_minus_emb": total - emb,
        "per_layer_no_emb": attn1 + ffn + 2 * ln,
    }


def build(cfg, dtype, dev="cuda"):
    H, nh, hd = cfg.hidden_size, cfg.num_heads, cfg.head_dim
    K = V = hd
    attn = Rwkv7Attention(cfg, 1).to(dev).to(dtype).eval()
    ff = Rwkv7FeedForward(cfg, 1).to(dev).to(dtype).eval()
    attn_norm = nn.LayerNorm(H, eps=cfg.norm_eps, bias=cfg.norm_bias).to(dev).to(dtype).eval()
    ffn_norm = nn.LayerNorm(H, eps=cfg.norm_eps, bias=cfg.norm_bias).to(dev).to(dtype).eval()
    final_norm = nn.LayerNorm(H, eps=cfg.norm_eps, bias=cfg.norm_bias).to(dev).to(dtype).eval()
    lm_head = nn.Linear(H, cfg.vocab_size, bias=False).to(dev).to(dtype).eval()
    emb = nn.Embedding(cfg.vocab_size, H).to(dev).to(dtype).eval()
    # g_norm runs in fp32-ish? deployed keeps module dtype = bf16. Keep as-is (cast).
    return dict(attn=attn, ff=ff, attn_norm=attn_norm, ffn_norm=ffn_norm,
                final_norm=final_norm, lm_head=lm_head, emb=emb, H=H, nh=nh, hd=hd, K=K, V=V)


def make_state(cfg, T, decode, dtype, dev="cuda", nreq=1):
    H, nh, hd = cfg.hidden_size, cfg.num_heads, cfg.head_dim
    size = nreq + 1  # one cache slot per request (batched decode)
    conv0 = torch.zeros(size, H, 1, dtype=torch.float32, device=dev)
    conv1 = torch.zeros(size, H, 1, dtype=torch.float32, device=dev)
    temporal = torch.zeros(size, nh, hd, hd, dtype=torch.float32, device=dev)
    caches = {0: _Cache(conv0, conv1, temporal),
              1: _Cache(conv0.clone(), conv1.clone(), temporal.clone())}
    pool = _Pool(caches)
    # int32 cache indices == production (mamba_cache_indices); also the dtype the
    # fused-glue eligibility check requires. token_shift / wkv_recurrent both accept it.
    ci = torch.arange(nreq, dtype=torch.int32, device=dev)
    if decode:
        meta = _Meta(ci)  # nreq sequences, 1 token each
    else:
        qsl = torch.tensor([0, T], dtype=torch.int32, device=dev)  # 1 seq, T tokens
        meta = _Meta(ci, qsl)
    be = _Backend(pool, meta, decode)
    fb = _FB(be, decode)
    return be, fb


# ----------------------------- decode breakdown -----------------------------
def run_decode(cfg, dtype, n_iter, bsz=1):
    m = build(cfg, dtype)
    attn, ff = m["attn"], m["ff"]
    H, nh, hd, K, V = m["H"], m["nh"], m["hd"], m["K"], m["V"]
    be, fb = make_state(cfg, bsz, True, dtype, nreq=bsz)
    T = bsz  # batched decode = bsz sequences, 1 token each -> T rows
    x = torch.randn(T, H, dtype=dtype, device="cuda")
    v_first = torch.randn(T, H, dtype=dtype, device="cuda")

    # Precompute representative intermediates for isolated component loops.
    shifted = be.token_shift(x.clone(), 1, 0, fb)
    d = shifted - x
    xr = x + attn.x_r.view(-1) * d
    xw = x + attn.x_w.view(-1) * d
    xk = x + attn.x_k.view(-1) * d
    xv = x + attn.x_v.view(-1) * d
    xa = x + attn.x_a.view(-1) * d
    xg = x + attn.x_g.view(-1) * d
    r = P(attn.r_proj, xr); k = P(attn.k_proj, xk); v = P(attn.v_proj, xv)
    rk = r.view(T, nh, hd); kkv = k.view(T, nh, hd); vv = v.view(T, nh, hd)
    a = torch.sigmoid(attn.a_lora(xa))
    w_log = (-torch.sigmoid(attn.w_lora(xw)) * _INV_SQRT_E)
    rv = r.view(T, nh, hd); wv = w_log.view(T, nh, hd)
    kv = k.view(T, nh, hd); av = a.view(T, nh, hd)
    kkn = (k * attn.k_k).view(T, nh, hd)
    kkn = kkn / kkn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    g = attn.g_lora(xg)
    o_wkv = be.recurrence(rv, wv, kv, vv.clone(), kkn, av, 1, fb)  # [T,nh,V]
    o_flat = o_wkv.reshape(T, H)
    o_norm = attn.g_norm(o_flat)
    hs = torch.randn(T, H, dtype=dtype, device="cuda")  # generic hidden for norms/lmhead
    tok = torch.zeros(T, dtype=torch.long, device="cuda")

    comps = {}

    def add(name, fn):
        eg = time_eager(fn, n_iter)
        gr = time_graph(fn, n_iter)
        comps[name] = (eg, gr)

    # 1 token-shift (state gather/scatter)
    add("token_shift(attn)", lambda: be.token_shift(x, 1, 0, fb))
    # 2 lerp (6 mixes)
    def _lerp():
        dd = shifted - x
        return (x + attn.x_r.view(-1) * dd, x + attn.x_w.view(-1) * dd,
                x + attn.x_k.view(-1) * dd, x + attn.x_v.view(-1) * dd,
                x + attn.x_a.view(-1) * dd, x + attn.x_g.view(-1) * dd)
    add("lerp(6x)", _lerp)
    # 3 r/k/v proj (3 GEMM HxH)
    def _rkv():
        return P(attn.r_proj, xr), P(attn.k_proj, xk), P(attn.v_proj, xv)
    add("rkv_proj(3 GEMM)", _rkv)
    # 4 LoRAs (8 matmuls) + gating math (sigmoids, w_log, v-residual)
    def _loras():
        wl = -torch.sigmoid(attn.w_lora(xw)) * _INV_SQRT_E
        aa = torch.sigmoid(attn.a_lora(xa))
        gg = attn.g_lora(xg)
        vv2 = v + (v_first - v) * torch.sigmoid(attn.v_lora(xv))
        return wl, aa, gg, vv2
    add("loras(8 matmul)+gate-math", _loras)
    # 5 kk/k mix + L2 norm
    def _kkmix():
        kk = k * attn.k_k
        k2 = k + k * (a - 1.0) * attn.k_a
        kkr = kk.view(T, nh, hd)
        kkr = kkr / kkr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return k2, kkr
    add("kk/k-mix+l2norm", _kkmix)
    # 6 WKV recurrence
    add("wkv_recurrence", lambda: be.recurrence(rv, wv, kv, vv.clone(), kkn, av, 1, fb))
    # 7 g_norm
    add("g_norm", lambda: attn.g_norm(o_flat))
    # 8 gate-correction
    def _gc():
        gcorr = ((rk * kkv * attn.r_k).sum(dim=-1, keepdim=True) * vv).reshape(T, H)
        return o_norm + gcorr
    add("gate-correction", _gc)
    # 9 o_proj (incl gate mul)
    def _oproj():
        return P(attn.o_proj, o_norm * g)
    add("o_proj(1 GEMM)+gate", _oproj)
    # 10 ffn (token-shift + lerp + key + sqrelu + value)
    add("ffn(2 GEMM+shift)", lambda: ff(fb, hs))
    # 11 norms (attn_norm + ffn_norm)
    def _norms():
        return m["attn_norm"](hs), m["ffn_norm"](hs)
    add("layernorms(attn+ffn)", _norms)
    # 12 lm_head
    add("lm_head(1 GEMM)", lambda: m["lm_head"](hs))
    # 13 embedding + final norm (per step, once)
    def _embfinal():
        return m["emb"](tok), m["final_norm"](hs)
    add("emb+final_norm", _embfinal)

    return comps


# ----------------------------- prefill breakdown -----------------------------
def run_prefill(cfg, dtype, T, n_iter):
    m = build(cfg, dtype)
    attn, ff = m["attn"], m["ff"]
    H, nh, hd, K, V = m["H"], m["nh"], m["hd"], m["K"], m["V"]
    be, fb = make_state(cfg, T, False, dtype)
    x = torch.randn(T, H, dtype=dtype, device="cuda")
    xr = torch.randn(T, H, dtype=dtype, device="cuda")
    r = torch.randn(T, nh, hd, dtype=dtype, device="cuda")
    w = (-torch.rand(T, nh, hd, dtype=dtype, device="cuda") * _INV_SQRT_E)
    k = torch.randn(T, nh, hd, dtype=dtype, device="cuda")
    v = torch.randn(T, nh, hd, dtype=dtype, device="cuda")
    kk = torch.randn(T, nh, hd, dtype=dtype, device="cuda")
    kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    a = torch.sigmoid(torch.randn(T, nh, hd, dtype=dtype, device="cuda"))
    comps = {}

    def add(name, fn, ng=None):
        eg = time_eager(fn, n_iter, warmup=10, reps=4)
        gr = time_graph(fn, n_iter, warmup=10, reps=4)
        comps[name] = (eg, gr)

    add("rkv_proj(3 GEMM)", lambda: (P(attn.r_proj, xr), P(attn.k_proj, xr), P(attn.v_proj, xr)))
    add("loras(8 matmul)", lambda: (attn.w_lora(xr), attn.a_lora(xr), attn.g_lora(xr), attn.v_lora(xr)))
    add("o_proj(1 GEMM)", lambda: P(attn.o_proj, xr))
    add("ffn(2 GEMM)", lambda: ff(fb, x))
    add("wkv_recurrence(scan)", lambda: be.recurrence(r, w, k, v.clone(), kk, a, 1, fb))
    add("token_shift(attn)", lambda: be.token_shift(x, 1, 0, fb))
    return comps


def fmt(comps):
    # graphed primary; fall back to eager if not capturable
    rows = []
    for k, (eg, gr) in comps.items():
        prim = gr if gr is not None else eg
        rows.append((k, prim, eg, gr))
    tot = sum(r[1] for r in rows)
    rows.sort(key=lambda r: -r[1])
    out = []
    out.append(f"{'component':28s} {'graphed_us':>11s} {'eager_us':>10s} {'launch_us':>10s} {'%graphed':>9s}")
    for k, prim, eg, gr in rows:
        gu = (gr if gr is not None else float('nan')) * 1000
        eu = eg * 1000
        lu = (eg - (gr if gr is not None else eg)) * 1000
        out.append(f"{k:28s} {gu:11.2f} {eu:10.2f} {lu:10.2f} {100*prim/tot:8.1f}%")
    out.append(f"{'TOTAL(sum-of-parts)':28s} {tot*1000:11.2f}")
    return "\n".join(out), tot


def _profile_launches(fn, n_iter, warmup=30):
    """Run fn() n_iter times under torch.profiler; return (rows, total_calls, total_us)
    where rows = [(kernel_name, count, cuda_time_us), ...]. Counts DEVICE kernels only."""
    from torch.profiler import profile, ProfilerActivity
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
    total_calls, total_us, rows = 0, 0.0, []
    for e in prof.key_averages():
        ct = getattr(e, "cuda_time_total", 0) or getattr(e, "device_time_total", 0) or 0
        if ct <= 0:
            continue
        rows.append((e.key, e.count, ct))
        total_calls += e.count
        total_us += ct
    rows.sort(key=lambda r: -r[1])  # sort by COUNT (launch-overhead hypothesis)
    return rows, total_calls, total_us


def run_kernels(cfg, dtype, n_iter):
    """Count CUDA kernel launches for ONE decoder layer and (separately) lm_head, then
    extrapolate the full decode step = per-layer x L + lm_head + emb/final. torch.profiler
    (eager). Rows sorted by LAUNCH COUNT (the launch-overhead-on-fast-cards hypothesis)."""
    layer = Rwkv7DecoderLayer(cfg, 1).to("cuda").to(dtype).eval()
    H, L = cfg.hidden_size, cfg.num_hidden_layers
    lm_head = nn.Linear(H, cfg.vocab_size, bias=False).to("cuda").to(dtype).eval()
    be, fb = make_state(cfg, 1, True, dtype)
    x = torch.randn(1, H, dtype=dtype, device="cuda")
    v_first = torch.randn(1, H, dtype=dtype, device="cuda")
    hs = torch.randn(1, H, dtype=dtype, device="cuda")

    lay_rows, lay_calls, lay_us = _profile_launches(
        lambda: layer(fb, x, v_first), n_iter)
    head_rows, head_calls, head_us = _profile_launches(lambda: lm_head(hs), n_iter)

    lpl = lay_calls / n_iter          # launches per layer
    hpl = head_calls / n_iter         # launches for lm_head
    full = lpl * L + hpl + 2          # +emb +final_norm (~1 each, measured elsewhere)
    print(f"## KERNEL LAUNCH COUNT (eager, {n_iter} iters, dtype={dtype})  L={L}")
    print(f"# per-layer: distinct={len(lay_rows):3d}  launches/layer={lpl:6.1f}  "
          f"GPU-busy/layer={lay_us/n_iter:8.1f}us")
    print(f"# lm_head  : distinct={len(head_rows):3d}  launches={hpl:6.1f}  "
          f"GPU-busy={head_us/n_iter:8.1f}us")
    print(f"# ==> FULL DECODE STEP launches ~= {lpl:.1f}*{L} + {hpl:.1f} + 2 = {full:.0f}")
    print(f"\n{'kernel (per LAYER, by count)':52s} {'#/layer':>8s} {'us/layer':>9s} {'us/launch':>10s}")
    for key, cnt, ct in lay_rows[:40]:
        c = cnt / n_iter
        print(f"{key[:52]:52s} {c:8.2f} {ct/n_iter:9.2f} {ct/max(cnt,1):10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["decode", "prefill", "params", "kernels"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=1, help="batched-decode sequences (decode cmd)")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    cfg, raw = load_cfg(args.model)
    print(f"# model={args.model} dtype={args.dtype} H={cfg.hidden_size} L={cfg.num_hidden_layers} "
          f"nh={cfg.num_heads} hd={cfg.head_dim}")

    if args.cmd == "kernels":
        run_kernels(cfg, dtype, args.n_iter)
        return

    if args.cmd == "params":
        p = count_params(cfg)
        for k, v in p.items():
            print(f"{k:22s} {v:,}" if isinstance(v, int) else f"{k:22s} {v}")
        return

    if args.cmd == "decode":
        comps = run_decode(cfg, dtype, args.n_iter, bsz=args.bsz)
        body, tot = fmt(comps)
        print(f"## DECODE per-layer component breakdown (T=1, bsz={args.bsz})")
        print(body)
        L = cfg.num_hidden_layers
        # per-layer = everything except lm_head, emb+final_norm
        per_layer = sum((gr if gr is not None else eg)
                        for kk, (eg, gr) in comps.items()
                        if kk not in ("lm_head(1 GEMM)", "emb+final_norm"))
        head = next((gr if gr is not None else eg) for kk, (eg, gr) in comps.items() if kk == "lm_head(1 GEMM)")
        ef = next((gr if gr is not None else eg) for kk, (eg, gr) in comps.items() if kk == "emb+final_norm")
        full = per_layer * L + head + ef
        print(f"\n# per-layer (graphed sum) = {per_layer*1000:.2f} us; x{L} = {per_layer*L*1000:.2f} us")
        print(f"# + lm_head {head*1000:.2f} us + emb/final {ef*1000:.2f} us")
        print(f"# SYNTH full decode step (graphed) = {full*1000:.2f} us  => "
              f"{args.bsz*1000/full:.1f} tok/s ceiling(GPU-busy, bsz={args.bsz})")
        return

    if args.cmd == "prefill":
        comps = run_prefill(cfg, dtype, args.T, args.n_iter)
        L = cfg.num_hidden_layers
        print(f"## PREFILL per-layer component breakdown (T={args.T}, bsz=1)")
        body, tot = fmt(comps)
        print(body)
        wkv = next((gr if gr is not None else eg) for kk, (eg, gr) in comps.items() if "wkv" in kk)
        gemm = sum((gr if gr is not None else eg) for kk, (eg, gr) in comps.items() if kk in
                   ("rkv_proj(3 GEMM)", "loras(8 matmul)", "o_proj(1 GEMM)", "ffn(2 GEMM)"))
        print(f"\n# per-layer WKV scan = {wkv*1000:.1f} us ; linear-GEMM (rkv+lora+o+ffn) = {gemm*1000:.1f} us")
        print(f"# WKV / (WKV+GEMM) = {100*wkv/(wkv+gemm):.1f}%  (x{L} layers)")
        return


if __name__ == "__main__":
    main()
