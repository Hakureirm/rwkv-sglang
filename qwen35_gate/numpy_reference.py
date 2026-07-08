"""Independent numpy fp32 reference for Qwen3.5, adapted from Bo Peng's
run_rwkv7_qwen35.py for the 2B and 9B checkpoint sizes.

Source: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py
Fetched: 2026-07-07. Unmodified copy kept at vendor/run_rwkv7_qwen35.py for diffing.

This module contains ONLY the Qwen35 half of the upstream script (the RWKV7 class
and its comparison loop are out of scope for this gate) plus the shared math
helpers it depends on, copied verbatim. Deltas from upstream, all forced by
generalizing from the script's original target (Qwen/Qwen3.5-0.8B), first to
Qwen3.5-2B (F0050) and now to Qwen3.5-9B (F0054):

1. `self.C` (hidden_size) is derived from the checkpoint's embedding tensor
   shape instead of hardcoded `1024` — that value is specific to the 0.8B
   checkpoint. Qwen3.5-2B reports `hidden_size: 2048`, Qwen3.5-9B reports 4096.
2. `self.n_layer` is derived from the checkpoint's layer keys (same pattern
   upstream already uses for its RWKV7 class) instead of hardcoded `24`. 2B
   happens to still be 24; 9B is 32 (verified against each tier's config.json).
3. The tokenizer is loaded from a local HF checkpoint directory (passed in)
   instead of the hardcoded network id `"Qwen/Qwen3.5-0.8B"` — loading the
   0.8B tokenizer to decode a bigger tier's logits would still "work" (same
   tokenizer family/vocab) but is fragile to rely on implicitly, and this
   avoids a network dependency.
4. (F0054, added for 9B) `self.Hk`/`self.Hv`/`self.N`/`self.conv_len` (linear-attn
   head geometry) and `self.aH`/`self.aKV`/`self.aN` (full-attn head geometry) are
   now DERIVED from checkpoint tensor shapes (see `__init__`) instead of hardcoded
   `H=16, N=128, conv_len=4, aH=8, aKV=2, aN=256`. F0050's 2B-only version left
   these hardcoded after hand-verifying them against 2B's config.json, and
   explicitly warned that a future tier "could easily differ again" and must be
   re-checked, not assumed. Checking 9B's config.json found it DOES differ:
   `num_attention_heads` 8->16 and `num_key_value_heads` 2->4 (both used verbatim
   as aH/aKV — an easy fix), but also something F0050 had no way to anticipate:
   `linear_num_key_heads=16` but `linear_num_value_heads=32` — 9B's linear
   attention uses MORE value heads than key/query heads (2x), which upstream's own
   script never handles (it unconditionally does `q, k, v = np.split(qkv, 3)` and
   reshapes v to the SAME head count as q/k, which is only valid when
   num_k_heads == num_v_heads, true for 0.8B/2B by coincidence, false for 9B, and
   not even integer-splittable for 9B's actual tensor width so this fails loud
   rather than silent). Fixed per HF's own `Qwen3_5GatedDeltaNet.forward()`
   (`transformers/models/qwen3_5/modeling_qwen3_5.py`): split qkv into
   `[key_dim, key_dim, value_dim]` (not three equal thirds), reshape q/k at
   `Hk` heads and v at `Hv` heads, and when `Hv > Hk`, `repeat_interleave` q/k
   onto v's head grid before the recurrence (`np.repeat(..., axis=0)` is the
   numpy equivalent — verified algebraically equivalent to HF's official
   `torch_recurrent_gated_delta_rule` step-by-step recurrence). Deriving these
   from tensor shapes (rather than a per-tier hardcoded number pair) means this
   generalizes to any future Qwen3.5 dense tier without a new manual code patch;
   each derived value is still independently cross-checked against 2B's AND 9B's
   own `config.json` by hand in F0054 (see that finding for the full table),
   and for 2B this refactor is provably behavior-preserving (`Hk==Hv` there, so
   the new split/repeat path reduces exactly to the old `np.split(qkv, 3)` path).

Two constants remain genuinely hardcoded (not derivable from tensor shapes, since
they aren't stored as weights) and were hand-verified against BOTH 2B's and 9B's
`config.json` rather than assumed to carry over: the GQA rope base `10000000.0`
(`rope_parameters.rope_theta`, identical in both configs) and the hybrid-layer
pattern `i % 4 != 3` (`full_attention_interval=4`, identical in both configs —
confirmed against the literal `layer_types` array in each config, not just the
`full_attention_interval` scalar). `rope_dim = aN // 4` (`partial_rotary_factor
= 0.25` in both configs) is a formula, not a value, and stays correct automatically
now that `aN` is derived. If a future tier changes any of these three, this file
would need another manual patch — do not assume they hold without checking.
"""

import numpy as np
import torch
from transformers import AutoTokenizer

PROBE_TEXT = " Eiffel"


def SIGMOID(x):
    return 1.0 / (1.0 + np.exp(-x))


def SILU(x):
    return x * SIGMOID(x)


def L2_QWEN(x):
    return x * (np.sum(x * x, axis=-1, keepdims=True) + 1e-6) ** -0.5


def RMS_NORM(x, w):
    return w * x * (np.mean(x * x, axis=-1, keepdims=True) + 1e-6) ** -0.5


def DPLR(S, R, W, K, V, A, B):
    S = np.einsum("hk,hkv->hkv", W, S) + np.einsum("hb,ha,hav->hbv", B, A, S) + np.einsum("hk,hv->hkv", K, V)
    return np.einsum("hk,hkv->hv", R, S), S


def GQA(S, q, k, v, H, KV, N):
    S = np.concatenate((S, np.stack((k, v), axis=0).reshape(2, KV, 1, N)), axis=2)
    k, v = S
    k, v = np.repeat(k, H // KV, axis=0), np.repeat(v, H // KV, axis=0)
    a = SOFTMAX(np.einsum("hd,htd->ht", q, k) * (N**-0.5), axis=-1)
    return np.einsum("ht,htd->hd", a, v).reshape(H * N), S


def SOFTMAX(x, axis=-1):
    y = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return y / np.sum(y, axis=axis, keepdims=True)


def top_logits(logits, k=10):
    probs = SOFTMAX(logits)
    ids = np.argpartition(logits, -k)[-k:]
    ids = ids[np.argsort(logits[ids])[::-1]]
    return [(int(i), float(logits[i]), float(probs[i])) for i in ids]


class Qwen35:
    def __init__(self, checkpoint, tokenizer_dir):
        print(f"loading checkpoint: {checkpoint}")
        pth = torch.load(checkpoint, map_location="cpu", mmap=True)
        self.W = {k: v.detach().cpu().float().numpy().astype(np.float32, copy=False).squeeze() for k, v in pth.items()}
        print(f"loaded: keys={len(self.W):,} params={sum(v.size for v in self.W.values()):,}")

        W = self.W
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.n_layer = 1 + max(int(k.split(".")[1]) for k in W if k.startswith("layers."))
        self.C = W["embed_tokens.weight"].shape[1]
        self.gdn_layers = tuple(i % 4 != 3 for i in range(self.n_layer))

        # Linear-attn (GDN) and full-attn (GQA) head geometry: derived from actual
        # checkpoint tensor shapes rather than hardcoded — see module docstring
        # (point 4) for why 9B forces this (asymmetric linear-attn key/value head
        # counts that upstream's script never handles). `gdn0`/`full0` are sample
        # layer indices guaranteed to be a GDN layer and a full-attn layer
        # respectively, per the `i % 4 != 3` pattern already computed above.
        gdn0 = next(i for i in range(self.n_layer) if self.gdn_layers[i])
        full0 = next(i for i in range(self.n_layer) if not self.gdn_layers[i])

        self.N = W[f"layers.{gdn0}.linear_attn.norm.weight"].shape[0]  # linear_key_head_dim == linear_value_head_dim
        self.Hv = W[f"layers.{gdn0}.linear_attn.dt_bias"].shape[0]  # linear_num_value_heads
        conv_dim = W[f"layers.{gdn0}.linear_attn.conv1d.weight"].shape[0]  # = 2*Hk*N + Hv*N (q,k,v concatenated)
        self.conv_len = W[f"layers.{gdn0}.linear_attn.conv1d.weight"].shape[-1]  # linear_conv_kernel_dim
        key_dim, rem = divmod(conv_dim - self.Hv * self.N, 2)
        assert rem == 0 and key_dim % self.N == 0, f"unexpected linear_attn conv_dim={conv_dim} Hv={self.Hv} N={self.N}"
        self.Hk = key_dim // self.N  # linear_num_key_heads
        assert self.Hv % self.Hk == 0, f"num_v_heads={self.Hv} must be an integer multiple of num_k_heads={self.Hk}"

        self.aN = W[f"layers.{full0}.self_attn.q_norm.weight"].shape[0]  # head_dim
        self.aH = W[f"layers.{full0}.self_attn.q_proj.weight"].shape[0] // (2 * self.aN)  # q_proj fused with output gate
        self.aKV = W[f"layers.{full0}.self_attn.k_proj.weight"].shape[0] // self.aN  # num_key_value_heads

        self.emb, self.ln_outW = W["embed_tokens.weight"], W["norm.weight"] + 1
        self.head = W["lm_head.weight"].T if "lm_head.weight" in W else self.emb.T
        self.TM = tuple(self.make_GDN(i) if gdn else self.make_GQA(i) for i, gdn in enumerate(self.gdn_layers))
        self.CM = tuple(self.make_FFN(i) for i in range(self.n_layer))
        print(
            f"resolved shape: n_layer={self.n_layer} C={self.C} Hk={self.Hk} Hv={self.Hv} N={self.N} "
            f"conv_len={self.conv_len} aH={self.aH} aKV={self.aKV} aN={self.aN} "
            f"(upstream 0.8B hardcode was n_layer=24 C=1024 H=16 N=128 conv_len=4 aH=8 aKV=2 aN=256)"
        )

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def S0(self):
        S = []
        for gdn in self.gdn_layers:
            if gdn:
                conv_dim = 2 * self.Hk * self.N + self.Hv * self.N
                S.append([{"conv": np.zeros((conv_dim, self.conv_len - 1), np.float32), "rnn": np.zeros((self.Hv, self.N, self.N), np.float32)}, None])
            else:
                S.append([{"kv": np.zeros((2, self.aKV, 0, self.aN), np.float32)}, None])
        return S

    def EMB(self, token):
        return self.emb[token]

    def NORM(self, X):
        return RMS_NORM(X, self.ln_outW)

    def HEAD(self, X):
        return X @ self.head

    def run_one(self, token, S):
        X = self.EMB(int(token))
        for TM, CM, s in zip(self.TM, self.CM, S):
            X, s[0] = TM(X, s[0])
            X, s[1] = CM(X, s[1])
        return self.HEAD(self.NORM(X)), S

    def forward(self, tokens, S=None):
        S = self.S0() if S is None else S
        logits = None
        for token in tokens:
            logits, S = self.run_one(token, S)
        return logits, S

    def report(self, probe_text=PROBE_TEXT):
        probe_tokens = self.encode(probe_text)
        logits, _ = self.forward(probe_tokens)
        rows = top_logits(logits)
        print(f"\n== Qwen3.5 (numpy fp32 reference) top-10 logits ==")
        print(f"text: {probe_text!r}")
        print(f"tokens: {probe_tokens}")
        for rank, (token, logit, prob) in enumerate(rows, 1):
            print(f"{rank}: token={token} logit={logit:.6f} prob={prob:.8f} text={self.decode([token])!r}")
        return probe_tokens, rows

    def make_GDN(self, i):
        p, W, Hk, Hv, N = f"layers.{i}.linear_attn.", self.W, self.Hk, self.Hv, self.N
        lnW, qkvW, convW = W[f"layers.{i}.input_layernorm.weight"] + 1, W[p + "in_proj_qkv.weight"].T, W[p + "conv1d.weight"]
        gW, aW, wW, wB = W[p + "in_proj_z.weight"].T, W[p + "in_proj_b.weight"].T, W[p + "in_proj_a.weight"].T, W[p + "dt_bias"]
        wP, oNorm, oW = np.exp(W[p + "A_log"]), W[p + "norm.weight"], W[p + "out_proj.weight"].T
        key_dim, value_dim, rep = Hk * N, Hv * N, Hv // Hk

        def layer(X, state):
            x = RMS_NORM(X, lnW)
            conv = np.concatenate((state["conv"], (x @ qkvW).reshape(2 * key_dim + value_dim, 1)), axis=-1)
            state["conv"] = conv[:, 1:].copy()
            qkv = SILU(np.sum(conv * convW, axis=-1))
            q, k, v = qkv[:key_dim], qkv[key_dim:2 * key_dim], qkv[2 * key_dim:]

            q = L2_QWEN(q.reshape(Hk, N)) * (N**-0.5)
            k = L2_QWEN(k.reshape(Hk, N))
            v = v.reshape(Hv, N)
            if rep > 1:
                # Qwen3.5-9B only (Hk=16, Hv=32, rep=2 here; 2B has Hk==Hv so this
                # is skipped and the function is byte-for-byte the old 2B path).
                # np.repeat(..., axis=0) duplicates each head contiguously
                # (head0,head0,head1,head1,...), matching HF's
                # torch.repeat_interleave(dim=2) exactly (NOT np.tile's
                # head0,head1,head0,head1,... layout). Doing this after L2-norm
                # instead of before (HF's literal order is repeat-then-norm) is
                # equivalent: norm is a per-vector op, and repeat only ever
                # duplicates a vector, so norm-then-duplicate == duplicate each
                # identical copy then norm it independently.
                q, k = np.repeat(q, rep, axis=0), np.repeat(k, rep, axis=0)
            w = np.pow(1.0 + np.exp(wB + x @ wW), -wP).reshape(Hv, 1)
            a = SIGMOID(x @ aW).reshape(Hv, 1)

            y, state["rnn"] = DPLR(state["rnn"], q, w, k, a * v, -a * w * k, k)
            y = RMS_NORM(y, oNorm).reshape(Hv * N)
            g = SILU(x @ gW)
            return X + (y * g) @ oW, state
        return layer

    def make_GQA(self, i):
        p, W, C, H, KV, N = f"layers.{i}.self_attn.", self.W, self.C, self.aH, self.aKV, self.aN
        lnW, qgW, q_norm = W[f"layers.{i}.input_layernorm.weight"] + 1, W[p + "q_proj.weight"].T, W[p + "q_norm.weight"] + 1
        kW, vW, oW, k_norm = W[p + "k_proj.weight"].T, W[p + "v_proj.weight"].T, W[p + "o_proj.weight"].T, W[p + "k_norm.weight"] + 1
        qgW = qgW.reshape(C, H, N * 2)
        qW, gW = qgW[:, :, :N].reshape(C, H * N), qgW[:, :, N:].reshape(C, H * N)
        rope_dim = N // 4
        inv_freq = 1.0 / (10000000.0 ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))

        def rotate_half(x):
            h = x.shape[-1] // 2
            return np.concatenate((-x[..., h:], x[..., :h]), axis=-1)

        def layer(X, state):
            x = RMS_NORM(X, lnW)
            q = RMS_NORM((x @ qW).reshape(H, N), q_norm)
            k = RMS_NORM((x @ kW).reshape(KV, N), k_norm)
            v = (x @ vW).reshape(KV, N)

            freq = state["kv"].shape[2] * inv_freq
            cos, sin = np.concatenate((np.cos(freq), np.cos(freq))), np.concatenate((np.sin(freq), np.sin(freq)))
            q0, k0 = q[..., :rope_dim], k[..., :rope_dim]
            q = np.concatenate((q0 * cos + rotate_half(q0) * sin, q[..., rope_dim:]), axis=-1)
            k = np.concatenate((k0 * cos + rotate_half(k0) * sin, k[..., rope_dim:]), axis=-1)

            y, state["kv"] = GQA(state["kv"], q, k, v, H, KV, N)
            g = SIGMOID(x @ gW)
            return X + (y * g) @ oW, state
        return layer

    def make_FFN(self, i):
        W, p = self.W, f"layers.{i}.mlp."
        lnW = W[f"layers.{i}.post_attention_layernorm.weight"] + 1
        gW, kW, vW = W[p + "gate_proj.weight"].T, W[p + "up_proj.weight"].T, W[p + "down_proj.weight"].T

        def layer(X, state):
            x = RMS_NORM(X, lnW)
            return X + ((SILU(x @ gW) * (x @ kW)) @ vW), state
        return layer
