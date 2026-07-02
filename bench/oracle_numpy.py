#!/usr/bin/env python3
"""
RWKV-7 (Goose) pure-numpy reference oracle — the M1 correctness gate.

Bit-level ground truth for verifying the sglang RWKV-7 implementation. Adapted
faithfully from BlinkDL/RWKV-LM `RWKV-v7/rwkv_v7_numpy.py` (numpy port by
johanwind), generalized to: infer model dims from the checkpoint, greedy-generate
N tokens, and dump prompt-final logits for comparison.

NOTE (F0004): `fla` is NOT aligned with the official reference and must NOT be the
oracle. This numpy forward + the `rwkv` pip `cpu fp32` strategy are the oracles.
Neither needs nvcc.

Usage:
  python oracle_numpy.py --model /path/to/rwkv7-g1a-0.1b-xxxx.pth \
      --prompt "The Eiffel Tower is in" --n 20 [--head-size 64] [--dump-logits out.npy]
"""
import argparse
import os

import numpy as np
from torch import load as torch_load

layer_norm = lambda x, w, b: (x - x.mean()) / (x.var() + 1e-5) ** 0.5 * w + b
group_norm = lambda x, w, b: (
    (x - x.mean(axis=1, keepdims=1)) / (x.var(axis=1, keepdims=1) + 64e-5) ** 0.5
).flatten() * w + b
sigmoid = lambda x: 1 / (1 + np.exp(-x))


def time_mixing(x, v0, last_x, S, params, n_head, head_size):
    # Positional unpacking matches BlinkDL .pth key order (see reference).
    mr, mw, mk, mv, ma, mg, w_bias, r_k, Ww1, Ww2, Wa1, Wa2, a_bias, Wg1, Wg2 = params[:15]
    k_k, k_a, Wr, Wk, Wv, Wo, ln_w, ln_b = params[-8:]

    xr, xw, xk, xv, xa, xg = [x + m * (last_x - x) for m in [mr, mw, mk, mv, ma, mg]]

    r = Wr @ xr
    w = np.exp(-sigmoid(np.tanh(xw @ Ww1) @ Ww2 + w_bias) / np.e ** 0.5)
    k = Wk @ xk
    v = Wv @ xv
    if v0 is None:
        v0 = v
    else:
        Wv2, Wv1, v_bias = params[15:18]
        v += (v0 - v) * sigmoid(xv @ Wv1 @ Wv2 + v_bias)
    a = sigmoid(xa @ Wa1 @ Wa2 + a_bias)
    g = sigmoid(xg @ Wg1) @ Wg2
    kk = k * k_k
    k += k * (a - 1) * k_a

    r, w, k, v, kk, a, r_k = [
        i.reshape(n_head, head_size, 1) for i in [r, w, k, v, kk, a, r_k]
    ]
    kk /= np.maximum(np.linalg.norm(kk, axis=1, keepdims=1), 1e-12)

    S = S * w.mT - S @ kk * (kk * a).mT + v * k.mT
    y = S @ r

    y = group_norm(y, ln_w, ln_b)
    y += ((r * k * r_k).sum(axis=1, keepdims=1) * v).flatten()
    return Wo @ (y * g), v0, x, S


def channel_mixing(x, last_x, mix, Wk, Wv):
    k = Wk @ (x + mix * (last_x - x))
    v = Wv @ np.maximum(k, 0) ** 2
    return v, x


def rwkv7_forward(params, token, state, n_layer, n_head, head_size):
    x = params("emb")[0][token]
    x = layer_norm(x, *params("blocks.0.ln0"))
    v0 = None
    for i in range(n_layer):
        x_ = layer_norm(x, *params(f"blocks.{i}.ln1"))
        dx, v0, state[0][i, 0], state[1][i] = time_mixing(
            x_, v0, state[0][i, 0], state[1][i], params(f"blocks.{i}.att"), n_head, head_size
        )
        x = x + dx
        x_ = layer_norm(x, *params(f"blocks.{i}.ln2"))
        dx, state[0][i, 1] = channel_mixing(x_, state[0][i, 1], *params(f"blocks.{i}.ffn"))
        x = x + dx
    x = layer_norm(x, *params("ln_out"))
    logits = params("head")[0] @ x
    return logits, state


def load_model(path, lazy_fp32=False):
    """Load a BlinkDL RWKV-7 .pth.

    Default: promote every tensor to fp32 numpy up-front (simple, but ~2x the
    checkpoint size in RAM — fine for <=1.5B). lazy_fp32: keep the (bf16)
    checkpoint as torch tensors and promote to fp32 numpy on demand per access.
    Promotion bf16->fp32 is LOSSLESS, so lazy_fp32 is bit-identical to the default
    while keeping peak RAM ~= checkpoint size (needed for 7.2B on a 31GB box).
    """
    if lazy_fp32:
        try:
            weights = torch_load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except Exception:
            weights = torch_load(path, map_location="cpu", weights_only=True)
        n_layer = 1 + max(
            int(k.split(".")[1]) for k in weights if k.startswith("blocks.")
        )
        n_embd = weights["emb.weight"].shape[1]
        return weights, n_layer, n_embd
    weights = torch_load(path, map_location="cpu", weights_only=True)
    weights = {k: v.squeeze().float().numpy() for k, v in weights.items()}
    n_layer = 1 + max(int(k.split(".")[1]) for k in weights if k.startswith("blocks."))
    n_embd = weights["emb.weight"].shape[1]
    return weights, n_layer, n_embd


def new_state(n_layer, n_embd, n_head, head_size):
    return [
        np.zeros((n_layer, 2, n_embd), dtype=np.float32),
        np.zeros((n_layer, n_head, head_size, head_size), dtype=np.float32),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to RWKV-7 .pth (no extension also ok)")
    ap.add_argument("--prompt", default="\nThe Eiffel Tower is located in the city of")
    ap.add_argument("--n", type=int, default=20, help="greedy tokens to generate")
    ap.add_argument("--head-size", type=int, default=64)
    ap.add_argument("--dump-logits", default=None, help="save prompt-final logits to .npy")
    ap.add_argument(
        "--lazy-fp32",
        action="store_true",
        help="keep checkpoint bf16, promote to fp32 numpy per access (bit-identical, "
        "low RAM; use for 7.2B on a 31GB box).",
    )
    ap.add_argument(
        "--dump-fixture",
        default=None,
        help="write a verify-style JSON fixture (prompt_tokens, greedy_tokens, ...).",
    )
    args = ap.parse_args()

    model_path = args.model if args.model.endswith(".pth") else args.model + ".pth"
    weights, n_layer, n_embd = load_model(model_path, lazy_fp32=args.lazy_fp32)
    head_size = args.head_size
    n_head = n_embd // head_size
    print(f"model: n_layer={n_layer} n_embd={n_embd} n_head={n_head} head_size={head_size}")

    if args.lazy_fp32:
        # weights are torch tensors; promote (losslessly) per access.
        params = lambda prefix: [
            weights[k].squeeze().float().numpy()
            for k in weights
            if k.startswith(prefix)
        ]
    else:
        params = lambda prefix: [weights[k] for k in weights if k.startswith(prefix)]

    os.environ["RWKV_V7_ON"] = "1"
    from rwkv.utils import PIPELINE  # World trie tokenizer ships with `rwkv` pip

    class _Dummy:  # PIPELINE only needs .args for some paths; we use encode/decode only
        pass

    pipeline = PIPELINE(_Dummy(), "rwkv_vocab_v20230424")
    tokens = pipeline.encode(args.prompt)
    print(f"prompt tokens ({len(tokens)}): {tokens}")

    state = new_state(n_layer, n_embd, n_head, head_size)
    logits = None
    for tok in tokens:
        logits, state = rwkv7_forward(params, tok, state, n_layer, n_head, head_size)

    if args.dump_logits:
        np.save(args.dump_logits, logits)
        print(f"saved prompt-final logits -> {args.dump_logits} (shape {logits.shape})")

    gen = []
    for _ in range(args.n):
        tok = int(np.argmax(logits))
        gen.append(tok)
        logits, state = rwkv7_forward(params, tok, state, n_layer, n_head, head_size)
    print(f"greedy tokens: {gen}")
    print(f"greedy text: {pipeline.decode(gen)!r}")

    if args.dump_fixture:
        import json

        fixture = {
            "_comment": "Correctness-gate fixture from bench/oracle_numpy.py (pure-numpy "
            "RWKV-7, fp32). sglang impl MUST reproduce greedy_tokens exactly.",
            "model": os.path.basename(model_path),
            "arch": {
                "n_layer": n_layer,
                "n_embd": n_embd,
                "n_head": n_head,
                "head_size": head_size,
                "vocab": int(weights["emb.weight"].shape[0]),
            },
            "tokenizer": "rwkv_vocab_v20230424 (World trie, via rwkv pip)",
            "prompt": args.prompt,
            "prompt_tokens": list(tokens),
            "greedy_tokens": gen,
            "greedy_text": pipeline.decode(gen),
        }
        with open(args.dump_fixture, "w") as fh:
            json.dump(fixture, fh, indent=2)
        print(f"wrote fixture -> {args.dump_fixture}")


if __name__ == "__main__":
    main()
