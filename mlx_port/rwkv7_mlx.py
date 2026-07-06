#!/usr/bin/env python3
"""
RWKV-7 (Goose) native MLX inference port (Apple Silicon, single-stream).

Math ground truth: `bench/oracle_numpy.py` (pure-numpy fp32 oracle). Every op
below mirrors the oracle's `rwkv7_forward` / `time_mixing` / `channel_mixing`
semantics exactly; layout conventions (fp32 state S[K, V], log-decay `w`,
caller-normalized `kk`) follow the project's Triton kernel
`sglang_overlay/.../rwkv7_kernels/wkv_recurrent.py`.

Weights: fla-format safetensors (tools/convert_rwkv7_blinkdl_to_fla.py layout):
LoRA down/up are nn.Linear convention ([low,in]/[out,low], bias=W0), the full
r/k/v/o + ffn projections are [out, in], `r_k` is [n_head, head_dim].
`num_heads` is derived from the r_k shape, NOT from config.json (the fla-hub
0.1B config says 32 heads; the checkpoint has 12 — same trap the sglang config
documents).

Precision policy (what passes the oracle gate, see mlx_port/README.md):
  * big projections (emb, head, r/k/v/o_proj, ffn key/value) keep the shipped
    bf16 weights and run bf16 GEMMs (MLX steel GEMM accumulates fp32, same as
    cuBLAS bf16 — the regime the sglang gate was passed under);
  * EVERYTHING else is fp32: the residual stream, all LayerNorms, the LoRA
    chains (down/act/up + bias), token-shift lerps, k_k/k_a/r_k elementwise,
    kk L2-normalization, the WKV recurrence + its state, GroupNorm, and the
    (r*k*r_k)-bonus. This is strictly closer to the fp32 oracle than the bf16
    activation stream the CUDA backends gate with.

WKV state is S[n_head, K, V] fp32 (== numpy oracle's S transposed, matching
wkv_recurrent.py). Per token (all-old-S on the RHS, exactly like the oracle):

    sa[v]   = sum_k (-kk[k]) * S[k, v]
    S[k, v] = exp(w[k]) * S[k, v] + (kk[k]*a[k]) * sa[v] + k[k] * v[v]
    y[v]    = sum_k S[k, v] * r[k]

Two WKV paths, selected by RWKV_MLX_WKV (gate both before trusting either):
  * "pure"  (default): vectorized-over-heads MLX ops, sequential over T.
  * "metal": one fused `mx.fast.metal_kernel` per (layer, chunk) — a
    threadgroup per head, one thread per V-column, K-loop in registers —
    the whole T-scan in a single dispatch (the Triton kernel's mapping).

Zero external deps beyond mlx; no fla, no torch, no transformers.
"""
import glob
import json
import math
import os

import mlx.core as mx

# Oracle: w = exp(-sigmoid(lora) / e**0.5). Keep the DIVISION form (fp32 /
# fp32(sqrt(e))) so the op sequence matches oracle_numpy.py line-for-line.
_SQRT_E = math.e ** 0.5

# WKV path default: "metal" (fused scan kernel) or "pure" (plain MLX ops).
# Per-instance (constructor arg) because the compiled decode step bakes the
# traced path — switching modes requires a fresh model instance.
#
# Default is "metal": at equal peak memory and equal (within-noise) bsz1 decode
# it prefills 5-8x faster than "pure" (whole-chunk scan in one dispatch vs a
# Python-level T-loop), and its decode tok/s is tighter run-to-run (no fp
# reduction difference — both paths are oracle-exact 24/24). "pure" stays a
# dependency-free fallback (no Metal JIT) via RWKV_MLX_WKV=pure. The metal
# kernel uses only portable Metal (threadgroup mem, barriers, precise::exp), so
# it is expected to build on M1/M2/M3/M4 as well; measured here on M5.
WKV_DEFAULT = os.environ.get("RWKV_MLX_WKV", "metal")


def _layer_norm(x, w, b, eps=1e-5):
    """Oracle `layer_norm`: (x - mean) / (var + eps)**0.5 * w + b, fp32.

    mx.fast.layer_norm computes mean/var in fp32 internally; with fp32 x/w/b
    it is the oracle expression exactly (population variance, ddof=0)."""
    return mx.fast.layer_norm(x, w, b, eps)


def _group_norm(x, w, b, n_head, head_dim, eps):
    """Oracle `group_norm`: per-head mean/var over head_dim, eps = head_dim *
    norm_eps (64e-5 for hd=64 — matches the oracle's hardcoded 64e-5 and the
    sglang GroupNorm eps). MLX has no GroupNorm fast-op; written out in fp32.
    x: [T, H] -> [T, H] with the affine (full-width w/b) applied post-flatten,
    exactly like the oracle's `.flatten() * w + b`."""
    T = x.shape[0]
    xh = x.reshape(T, n_head, head_dim)
    mean = mx.mean(xh, axis=-1, keepdims=True)
    var = mx.mean(mx.square(xh - mean), axis=-1, keepdims=True)
    xh = (xh - mean) / mx.sqrt(var + eps)
    return xh.reshape(T, n_head * head_dim) * w + b


def _wkv_scan_pure(S, r, w, k, v, kk, a):
    """Sequential WKV scan in plain MLX ops, fp32 throughout.

    S: [H, K, V] fp32; r/w/k/v/kk/a: [T, H, D] fp32 (w = log-decay, kk already
    L2-normalized). Returns (y [T, H, D] fp32, S' [H, K, V] fp32). The [H,1,K]
    @ [H,K,V] matmuls contract only K, so head-batching is exact."""
    T = r.shape[0]
    decay = mx.exp(w)  # [T, H, D]
    b = kk * a         # [T, H, D]  (the Triton kernel's b_kernel)
    ys = []
    for t in range(T):
        sa = -mx.matmul(kk[t][:, None, :], S)                # [H, 1, V]
        S = decay[t][:, :, None] * S + b[t][:, :, None] * sa \
            + k[t][:, :, None] * v[t][:, None, :]            # all-old-S RHS
        ys.append(mx.matmul(r[t][:, None, :], S)[:, 0, :])   # [H, V]
    return mx.stack(ys, axis=0), S


# ---------------------------------------------------------------------------
# Fused Metal WKV scan (RWKV_MLX_WKV=metal). Mapping mirrors wkv_recurrent.py:
# one threadgroup per head, one thread per V-column; the thread keeps its
# state column S[:, v] (D fp32 registers) and walks the sequence in time. The
# per-step K-vectors (r/w/k/kk/a) are staged in threadgroup memory by the D
# threads cooperatively; both reductions (sa, y) contract only K, so each
# thread's column is independent — no cross-thread accumulation, and the fp32
# summation order (sequential k=0..D-1) is fixed and deterministic.
# ---------------------------------------------------------------------------
_METAL_SRC = """
    uint v_i = thread_position_in_threadgroup.x;   // V column, 0..D-1
    uint h = threadgroup_position_in_grid.y;       // head
    const int T = tlen[0];

    // state column: s[k] = S[h, k, v_i]
    float s[D];
    const device float* s0h = s0 + h * D * D;
    for (uint kx = 0; kx < D; kx++) {
        s[kx] = s0h[kx * D + v_i];
    }

    threadgroup float shr[D], shdecay[D], shk[D], shkk[D], sha[D];
    for (int t = 0; t < T; t++) {
        const uint off = (uint(t) * H + h) * D;
        shr[v_i] = r[off + v_i];
        // decay = precise::exp(w): computed ONCE per K-element here (D exps per
        // step) instead of once per (v-column, K-element) inside the k-loop (was
        // D*D exps per step — the D V-threads each recomputed the same D decays).
        // Same metal::precise::exp on the same fp32 input, so bit-identical to
        // the old in-loop exp — only the call count drops (verified by the
        // oracle gate: 24/24 unchanged). precise:: keeps it matching mx.exp.
        shdecay[v_i] = metal::precise::exp(w[off + v_i]);
        shk[v_i] = k[off + v_i];
        shkk[v_i] = kk[off + v_i];
        sha[v_i] = a[off + v_i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float vv = v[off + v_i];
        float sa = 0.0f;
        for (uint kx = 0; kx < D; kx++) {
            sa -= shkk[kx] * s[kx];                 // sum_k (-kk[k]) S[k,v]
        }
        float out = 0.0f;
        for (uint kx = 0; kx < D; kx++) {
            // all-old-S RHS: decay*S + (kk*a)*sa + k*v
            float sk = shdecay[kx] * s[kx]
                     + shkk[kx] * sha[kx] * sa + shk[kx] * vv;
            s[kx] = sk;
            out += sk * shr[kx];                    // y[v] = sum_k S[k,v] r[k]
        }
        o[off + v_i] = out;
        // shr..sha are rewritten next step; keep the group in lockstep.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* s1h = s1 + h * D * D;
    for (uint kx = 0; kx < D; kx++) {
        s1h[kx * D + v_i] = s[kx];
    }
"""

_metal_kernel_cache = {}


def _wkv_scan_metal(S, r, w, k, v, kk, a):
    """Fused Metal scan: same math/layout as _wkv_scan_pure, one dispatch for
    the whole [T, H, D] chunk. T rides in as a tiny int32 array (NOT a template
    arg) so decode (T=1) and any prefill chunk share one compiled kernel."""
    T, H, D = r.shape
    key = (H, D)
    if key not in _metal_kernel_cache:
        _metal_kernel_cache[key] = mx.fast.metal_kernel(
            name=f"rwkv7_wkv_scan_h{H}_d{D}",
            input_names=["r", "w", "k", "v", "kk", "a", "s0", "tlen"],
            output_names=["o", "s1"],
            source=_METAL_SRC,
        )
    kern = _metal_kernel_cache[key]
    tlen = mx.array([T], dtype=mx.int32)
    o, S1 = kern(
        inputs=[r, w, k, v, kk, a, S, tlen],
        template=[("H", H), ("D", D)],
        grid=(D, H, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(T, H, D), (H, D, D)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return o, S1


def _wkv_scan(mode, S, r, w, k, v, kk, a):
    if mode == "metal":
        return _wkv_scan_metal(S, r, w, k, v, kk, a)
    return _wkv_scan_pure(S, r, w, k, v, kk, a)


class Rwkv7MLX:
    """RWKV-7 model on MLX. Weights from an fla-format safetensors dir.

    dtype applies to the BIG projections only (emb/head/r/k/v/o/ffn); all
    small per-channel params, norms and LoRAs are promoted to fp32 (lossless
    from bf16) per the precision policy in the module docstring."""

    def __init__(self, model_dir, dtype=mx.bfloat16, wkv_mode=None):
        self.wkv_mode = wkv_mode or WKV_DEFAULT
        assert self.wkv_mode in ("pure", "metal"), self.wkv_mode
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        weights = {}
        for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            weights.update(mx.load(f))

        # Head geometry from the checkpoint (config num_heads is unreliable).
        r_k0 = weights["model.layers.0.attn.r_k"]
        self.n_head, self.head_dim = r_k0.shape
        self.n_embd = int(cfg["hidden_size"])
        self.n_layer = int(cfg["num_hidden_layers"])
        self.vocab = int(cfg["vocab_size"])
        self.norm_eps = float(cfg.get("norm_eps", 1e-5))
        self.gn_eps = self.head_dim * self.norm_eps  # oracle's 64e-5
        assert self.n_head * self.head_dim == self.n_embd, (
            f"head geometry mismatch: {self.n_head}*{self.head_dim} "
            f"!= {self.n_embd}"
        )
        self.dtype = dtype

        consumed = set()

        def take(name, cast_dtype):
            consumed.add(name)
            t = weights[name]
            return t.astype(cast_dtype) if t.dtype != cast_dtype else t

        big = lambda n: take(n, dtype)
        f32 = lambda n: take(n, mx.float32)

        self.emb = big("model.embeddings.weight")
        self.head = big("lm_head.weight")
        self.ln_out = (f32("model.norm.weight"), f32("model.norm.bias"))
        self.ln0 = (
            f32("model.layers.0.pre_norm.weight"),
            f32("model.layers.0.pre_norm.bias"),
        )

        self.layers = []
        for i in range(self.n_layer):
            A = f"model.layers.{i}.attn"
            F = f"model.layers.{i}.ffn"
            L = {
                "ln1": (f32(f"model.layers.{i}.attn_norm.weight"),
                        f32(f"model.layers.{i}.attn_norm.bias")),
                "ln2": (f32(f"model.layers.{i}.ffn_norm.weight"),
                        f32(f"model.layers.{i}.ffn_norm.bias")),
                # token-shift lerp coefficients, [1,1,D] -> [D]
                **{x: f32(f"{A}.{x}").reshape(-1)
                   for x in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]},
                "Wr": big(f"{A}.r_proj.weight"),
                "Wk": big(f"{A}.k_proj.weight"),
                "Wv": big(f"{A}.v_proj.weight"),
                "Wo": big(f"{A}.o_proj.weight"),
                "k_k": f32(f"{A}.k_k"),
                "k_a": f32(f"{A}.k_a"),
                "r_k": f32(f"{A}.r_k"),  # [n_head, head_dim]
                "gn": (f32(f"{A}.g_norm.weight"), f32(f"{A}.g_norm.bias")),
                "fx_k": f32(f"{F}.x_k"),
                "Wfk": big(f"{F}.key.weight"),
                "Wfv": big(f"{F}.value.weight"),
            }
            # LoRA chains (fp32): down [low,H], up [H,low], bias [H].
            # w=tanh(+bias), a=identity(+bias), g=sigmoid(no bias),
            # v=identity(+bias, layers>0 only).
            for nm, has_bias in [("w", True), ("a", True), ("g", False)]:
                L[f"{nm}_down"] = f32(f"{A}.{nm}_lora.lora.0.weight")
                L[f"{nm}_up"] = f32(f"{A}.{nm}_lora.lora.2.weight")
                if has_bias:
                    L[f"{nm}_bias"] = f32(f"{A}.{nm}_lora.lora.2.bias")
            if i > 0:
                L["v_down"] = f32(f"{A}.v_lora.lora.0.weight")
                L["v_up"] = f32(f"{A}.v_lora.lora.2.weight")
                L["v_bias"] = f32(f"{A}.v_lora.lora.2.bias")
            self.layers.append(L)

        # Strict both ways, like sglang load_weights: no silent extra/missing.
        extra = set(weights) - consumed
        if extra:
            raise KeyError(f"unconsumed checkpoint keys: {sorted(extra)[:8]}")
        mx.eval(
            self.emb, self.head,
            *[p for L in self.layers for p in L.values()
              if isinstance(p, mx.array)],
        )
        # decode step, compiled once per model (T==1 shapes are static). NB only
        # the T==1 decode path is compiled: compiling the T>1 prefill reorders fp
        # ops and breaks oracle bit-exactness on 0.1B (see prefill()).
        self._step = mx.compile(self._forward_seq)

    # ---- state ----------------------------------------------------------
    def new_state(self):
        """Per layer: (att token-shift [D], ffn token-shift [D], WKV S
        [H, K, V]) — ALL fp32, zero-init, exactly oracle `new_state`."""
        D, H, hd = self.n_embd, self.n_head, self.head_dim
        return [
            (
                mx.zeros((D,), dtype=mx.float32),
                mx.zeros((D,), dtype=mx.float32),
                mx.zeros((H, hd, hd), dtype=mx.float32),
            )
            for _ in range(self.n_layer)
        ]

    # ---- big projections: bf16 GEMM boundary ----------------------------
    def _proj(self, x32, W):
        """x (fp32 [T, in]) @ W.T with W in the big-proj dtype. The input is
        rounded to the weight dtype at the GEMM boundary (same rounding point
        as the sglang bf16 path); MLX accumulates the GEMM in fp32; the output
        is promoted back to fp32 for the elementwise math."""
        return mx.matmul(x32.astype(W.dtype), W.T).astype(mx.float32)

    @staticmethod
    def _lora(x, down, up, bias, act):
        """Oracle LoRA: act(x @ W1) @ W2 [+ bias], all fp32. `down`/`up` are
        nn.Linear-convention (converter transposed), so x @ down.T == x @ W1."""
        h = mx.matmul(x, down.T)
        if act == "tanh":
            h = mx.tanh(h)
        elif act == "sigmoid":
            h = mx.sigmoid(h)
        out = mx.matmul(h, up.T)
        return out + bias if bias is not None else out

    # ---- one layer over a [T, D] chunk -----------------------------------
    def _time_mix(self, L, x, v_first, shift, S):
        """Oracle `time_mixing`, vectorized over T (scan sequential inside
        _wkv_scan). Returns (dx, v_first, new_shift, new_S)."""
        T = x.shape[0]
        H, hd = self.n_head, self.head_dim
        # token-shift: last_x for row 0 is the carried state, then x[t-1]
        last = mx.concatenate([shift[None, :], x[:-1]], axis=0)
        d = last - x
        xr, xw, xk, xv, xa, xg = (
            x + L[m] * d for m in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]
        )

        r = self._proj(xr, L["Wr"])
        # log-decay, oracle form: -sigmoid(tanh(xw@W1)@W2 + bias) / sqrt(e)
        w = -mx.sigmoid(self._lora(xw, L["w_down"], L["w_up"],
                                   L["w_bias"], "tanh")) / _SQRT_E
        k = self._proj(xk, L["Wk"])
        v = self._proj(xv, L["Wv"])
        if v_first is None:
            v_first = v  # layer 0 publishes its value projection
        else:
            v = v + (v_first - v) * mx.sigmoid(
                self._lora(xv, L["v_down"], L["v_up"], L["v_bias"], None)
            )
        a = mx.sigmoid(self._lora(xa, L["a_down"], L["a_up"],
                                  L["a_bias"], None))
        g = self._lora(xg, L["g_down"], L["g_up"], None, "sigmoid")
        kk = k * L["k_k"]
        k = k + k * (a - 1.0) * L["k_a"]

        rh, wh, kh, vh, ah = (t.reshape(T, H, hd) for t in (r, w, k, v, a))
        kkh = kk.reshape(T, H, hd)
        # oracle: kk /= max(||kk||, 1e-12), L2 over head_dim
        kkh = kkh / mx.maximum(
            mx.linalg.norm(kkh, axis=-1, keepdims=True), 1e-12
        )

        y, S = _wkv_scan(self.wkv_mode, S, rh, wh, kh, vh, kkh, ah)

        y = _group_norm(y.reshape(T, H * hd), L["gn"][0], L["gn"][1],
                        H, hd, self.gn_eps)
        # bonus: ((r*k*r_k).sum over head_dim) * v, flattened back to [T, D]
        bonus = mx.sum(rh * kh * L["r_k"], axis=-1, keepdims=True) * vh
        y = y + bonus.reshape(T, H * hd)
        dx = self._proj(y * g, L["Wo"])
        return dx, v_first, x[-1], S

    def _channel_mix(self, L, x, shift):
        """Oracle `channel_mixing`: value(relu(key(lerp))**2)."""
        last = mx.concatenate([shift[None, :], x[:-1]], axis=0)
        xk = x + L["fx_k"] * (last - x)
        kf = self._proj(xk, L["Wfk"])
        return self._proj(mx.square(mx.maximum(kf, 0.0)), L["Wfv"]), x[-1]

    # ---- full forward over a token chunk ---------------------------------
    def _forward_seq(self, tokens, state):
        """tokens: [T] int32. Returns (last-token logits [vocab] fp32,
        new state). Functional in `state` (mx.compile-safe)."""
        x = self.emb[tokens].astype(mx.float32)
        x = _layer_norm(x, *self.ln0, self.norm_eps)
        v_first = None
        new_state = []
        for i, L in enumerate(self.layers):
            sa, sf, S = state[i]
            dx, v_first, sa, S = self._time_mix(
                L, _layer_norm(x, *L["ln1"], self.norm_eps), v_first, sa, S
            )
            x = x + dx
            dxf, sf = self._channel_mix(
                L, _layer_norm(x, *L["ln2"], self.norm_eps), sf
            )
            x = x + dxf
            new_state.append((sa, sf, S))
        x = _layer_norm(x[-1:], *self.ln_out, self.norm_eps)
        logits = mx.matmul(x.astype(self.head.dtype), self.head.T)
        return logits[0].astype(mx.float32), new_state

    # ---- public API -------------------------------------------------------
    def prefill(self, tokens, state, chunk=None):
        """Run `tokens` (list[int]) through the model, returning (last-token
        logits, state). Chunked so the lazy graph (T scan steps x layers) is
        evaluated in bounded pieces; chunking is exact (the recurrence carries
        all cross-chunk context in `state`)."""
        chunk = chunk or (256 if self.wkv_mode == "metal" else 32)
        # Prefill runs the EAGER forward (not the compiled self._step): mx.compile
        # fuses/reorders the fp ops, which shifts rounding by ~1 ULP (state diff
        # ~2e-7). Harmless for the big models but enough to flip a greedy token on
        # 0.1B (logits are closer together), so compiling the T>1 prefill breaks
        # the oracle bit-exactness — measured +13% prefill but 0.1B fell to 5/24.
        # Bit-exactness is the red line; prefill stays eager. (Decode's compiled
        # T==1 step is separately gate-validated as exact.)
        logits = None
        for s in range(0, len(tokens), chunk):
            part = mx.array(tokens[s:s + chunk], dtype=mx.int32)
            logits, state = self._forward_seq(part, state)
            mx.eval(logits, *[t for lay in state for t in lay])
        return logits, state

    def step(self, token, state):
        """One greedy-decode step (compiled). Returns (logits, state)."""
        return self._step(mx.array([token], dtype=mx.int32), state)

    def greedy_loop(self, logits, state, n):
        """Async-pipelined greedy decode: the argmax token array is fed
        straight back as the next input (no per-step host sync; the host
        enqueues step t+1 while the GPU runs step t — the standard mlx-lm
        generation pattern). The compiled step executes atomically, so
        async_eval(tok) materializes the carried state too. Returns
        (tokens as 1-elem mx.arrays, logits, state); caller mx.eval's."""
        toks = []
        tok = mx.argmax(logits).reshape(1).astype(mx.int32)
        mx.async_eval(tok)
        for _ in range(n):
            toks.append(tok)
            logits, state = self._step(tok, state)
            tok = mx.argmax(logits).reshape(1).astype(mx.int32)
            mx.async_eval(tok)
        return toks, logits, state

    def generate(self, prompt_tokens, n, state=None, prefill_mode="seq"):
        """Greedy-generate `n` tokens after `prompt_tokens`. prefill_mode
        "seq" uses the chunked vectorized prefill; "step" feeds the prompt
        token-by-token through the decode step (oracle-style) — the gate
        cross-checks both produce identical continuations."""
        state = state or self.new_state()
        if prefill_mode == "step":
            for t in prompt_tokens:
                logits, state = self.step(t, state)
        else:
            logits, state = self.prefill(prompt_tokens, state)
        toks, _, state = self.greedy_loop(logits, state, n)
        mx.eval(*toks)
        return [int(t[0]) for t in toks], state


def load_model(model_dir, dtype="bfloat16", wkv=None):
    dt = {"bfloat16": mx.bfloat16, "float16": mx.float16,
          "float32": mx.float32}[dtype]
    return Rwkv7MLX(model_dir, dtype=dt, wkv_mode=wkv)
