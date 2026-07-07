"""Independent numpy fp32 reference for Qwen3.5, adapted from Bo Peng's
run_rwkv7_qwen35.py for the 2B checkpoint size.

Source: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py
Fetched: 2026-07-07. Unmodified copy kept at vendor/run_rwkv7_qwen35.py for diffing.

This module contains ONLY the Qwen35 half of the upstream script (the RWKV7 class
and its comparison loop are out of scope for this gate) plus the shared math
helpers it depends on, copied verbatim. Three deltas from upstream, all forced by
generalizing from the script's original target (Qwen/Qwen3.5-0.8B) to Qwen3.5-2B:

1. `self.C` (hidden_size) is now derived from the checkpoint's embedding tensor
   shape instead of hardcoded `1024` — that value is specific to the 0.8B
   checkpoint. Qwen3.5-2B's `config.json` reports `hidden_size: 2048`.
2. `self.n_layer` is derived from the checkpoint's layer keys (same pattern
   upstream already uses for its RWKV7 class) instead of hardcoded `24`. For
   2B this happens to still be 24 (verified against config.json), but deriving
   it costs nothing and removes one more silent-mismatch risk.
3. The tokenizer is loaded from a local HF checkpoint directory (passed in)
   instead of the hardcoded network id `"Qwen/Qwen3.5-0.8B"` — loading the
   0.8B tokenizer to decode 2B's logits would still "work" (same tokenizer
   family/vocab) but is fragile to rely on implicitly, and this avoids a
   network dependency.

Every other hardcoded architecture constant in the upstream `Qwen35.__init__`
(H=16, N=128, conv_len=4, aH=8, aKV=2, aN=256, full_attention_interval=4 via
`i % 4 != 3`) was cross-checked BY HAND against Qwen3.5-2B's `config.json`
(`linear_num_key_heads`, `linear_key_head_dim`, `linear_conv_kernel_dim`,
`num_attention_heads`, `num_key_value_heads`, `head_dim`, `full_attention_interval`)
and the GQA rope constants (`rope_theta=10000000`, `partial_rotary_factor=0.25`
=> `rope_dim = head_dim // 4`) and found to match exactly — i.e. verified, not
assumed, per the task that produced this file. They are LEFT hardcoded here
because they were confirmed correct for 2B, not because they're assumed to
generalize. If this script is ever pointed at a different Qwen3.5 dense size
(4B/9B), these must be re-verified against that size's config.json first —
do not assume they hold.
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
        self.H, self.N, self.conv_len = 16, 128, 4
        self.aH, self.aKV, self.aN = 8, 2, 256
        self.gdn_layers = tuple(i % 4 != 3 for i in range(self.n_layer))
        self.emb, self.ln_outW = W["embed_tokens.weight"], W["norm.weight"] + 1
        self.head = W["lm_head.weight"].T if "lm_head.weight" in W else self.emb.T
        self.TM = tuple(self.make_GDN(i) if gdn else self.make_GQA(i) for i, gdn in enumerate(self.gdn_layers))
        self.CM = tuple(self.make_FFN(i) for i in range(self.n_layer))
        print(f"resolved shape: n_layer={self.n_layer} C={self.C} (upstream 0.8B hardcode was n_layer=24 C=1024)")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def S0(self):
        S = []
        for gdn in self.gdn_layers:
            if gdn:
                S.append([{"conv": np.zeros((3 * self.H * self.N, self.conv_len - 1), np.float32), "rnn": np.zeros((self.H, self.N, self.N), np.float32)}, None])
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
        p, W, H, N = f"layers.{i}.linear_attn.", self.W, self.H, self.N
        lnW, qkvW, convW = W[f"layers.{i}.input_layernorm.weight"] + 1, W[p + "in_proj_qkv.weight"].T, W[p + "conv1d.weight"]
        gW, aW, wW, wB = W[p + "in_proj_z.weight"].T, W[p + "in_proj_b.weight"].T, W[p + "in_proj_a.weight"].T, W[p + "dt_bias"]
        wP, oNorm, oW = np.exp(W[p + "A_log"]), W[p + "norm.weight"], W[p + "out_proj.weight"].T

        def layer(X, state):
            x = RMS_NORM(X, lnW)
            conv = np.concatenate((state["conv"], (x @ qkvW).reshape(3 * H * N, 1)), axis=-1)
            state["conv"] = conv[:, 1:].copy()
            qkv = SILU(np.sum(conv * convW, axis=-1))
            q, k, v = np.split(qkv, 3)

            q = L2_QWEN(q.reshape(H, N)) * (N**-0.5)
            k = L2_QWEN(k.reshape(H, N))
            v = v.reshape(H, N)
            w = np.pow(1.0 + np.exp(wB + x @ wW), -wP).reshape(H, 1)
            a = SIGMOID(x @ aW).reshape(H, 1)

            y, state["rnn"] = DPLR(state["rnn"], q, w, k, a * v, -a * w * k, k)
            y = RMS_NORM(y, oNorm).reshape(H * N)
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
