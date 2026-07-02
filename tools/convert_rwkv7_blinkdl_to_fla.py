#!/usr/bin/env python3
"""
Convert a BlinkDL RWKV-7 `.pth` checkpoint to fla-format (safetensors + config.json)
so sglang's fla-naming `load_weights` can consume it (matches vLLM PR #41060 /
fla-hub layout). Verified against fla-hub/rwkv7-0.1B-g1 tensor shapes.

Key facts baked in:
- head count derived from `att.r_k` shape [n_head, head_dim] (NOT config num_heads).
- LoRA: BlinkDL stores `x @ W1 @ W2` so W1=[in,low], W2=[low,out]; fla uses
  nn.Linear (weight [out,in]) → TRANSPOSE both: lora.0.weight = W1.T [low,in],
  lora.2.weight = W2.T [out,low]; lora.2.bias = W0 (squeezed).
- g_lora has NO bias. v_lora exists for layers>0 ONLY (block 0's v0/v1/v2 are dead
  weights and are dropped, matching fla-hub).
- full projections (receptance/key/value/output) + r_k + ln_x are NOT transposed.
- activations are applied in the MODEL, not here (w=tanh, a/v=identity, g=sigmoid).

Usage:
  python convert_rwkv7_blinkdl_to_fla.py --pth IN.pth --out OUT_DIR
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file


def _sq(t):  # squeeze (1,1,D)->(D)
    return t.squeeze()


def convert(pth_path, out_dir):
    w = torch.load(pth_path, map_location="cpu", weights_only=True)
    ks = list(w.keys())
    n_layer = 1 + max(int(k.split(".")[1]) for k in ks if k.startswith("blocks."))
    n_embd = w["emb.weight"].shape[1]
    n_head, head_dim = w["blocks.0.att.r_k"].shape
    decay_lr = w["blocks.0.att.w1"].shape[1]
    a_lr = w["blocks.0.att.a1"].shape[1]
    gate_lr = w["blocks.0.att.g1"].shape[1]
    # v_lora only exists for layers>0; read its dim from layer 1
    v_lr = w["blocks.1.att.v1"].shape[1]
    ffn_inter = w["blocks.0.ffn.key.weight"].shape[0]
    vocab = w["emb.weight"].shape[0]
    print(f"n_layer={n_layer} n_embd={n_embd} n_head={n_head} head_dim={head_dim} "
          f"decay_lr={decay_lr} a_lr={a_lr} v_lr={v_lr} gate_lr={gate_lr} "
          f"ffn_inter={ffn_inter} vocab={vocab}")

    out = {}
    # top-level
    out["model.embeddings.weight"] = w["emb.weight"].contiguous()
    out["lm_head.weight"] = w["head.weight"].contiguous()
    out["model.norm.weight"] = w["ln_out.weight"].contiguous()
    out["model.norm.bias"] = w["ln_out.bias"].contiguous()

    for i in range(n_layer):
        b = f"blocks.{i}"
        L = f"model.layers.{i}"
        # norms: ln1->attn_norm, ln2->ffn_norm, (layer0 ln0)->pre_norm
        out[f"{L}.attn_norm.weight"] = w[f"{b}.ln1.weight"].contiguous()
        out[f"{L}.attn_norm.bias"] = w[f"{b}.ln1.bias"].contiguous()
        out[f"{L}.ffn_norm.weight"] = w[f"{b}.ln2.weight"].contiguous()
        out[f"{L}.ffn_norm.bias"] = w[f"{b}.ln2.bias"].contiguous()
        if i == 0:
            out[f"{L}.pre_norm.weight"] = w[f"{b}.ln0.weight"].contiguous()
            out[f"{L}.pre_norm.bias"] = w[f"{b}.ln0.bias"].contiguous()

        a = f"{b}.att"
        A = f"{L}.attn"
        # token-shift mix vectors keep [1,1,D]
        for x in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]:
            out[f"{A}.{x}"] = w[f"{a}.{x}"].contiguous()
        # squeezed scalars
        out[f"{A}.k_k"] = _sq(w[f"{a}.k_k"]).contiguous()
        out[f"{A}.k_a"] = _sq(w[f"{a}.k_a"]).contiguous()
        out[f"{A}.r_k"] = w[f"{a}.r_k"].contiguous()  # [n_head, head_dim]
        # full projections (no transpose)
        out[f"{A}.r_proj.weight"] = w[f"{a}.receptance.weight"].contiguous()
        out[f"{A}.k_proj.weight"] = w[f"{a}.key.weight"].contiguous()
        out[f"{A}.v_proj.weight"] = w[f"{a}.value.weight"].contiguous()
        out[f"{A}.o_proj.weight"] = w[f"{a}.output.weight"].contiguous()
        # g_norm <- ln_x
        out[f"{A}.g_norm.weight"] = w[f"{a}.ln_x.weight"].contiguous()
        out[f"{A}.g_norm.bias"] = w[f"{a}.ln_x.bias"].contiguous()
        # LoRAs: transpose down/up; bias = W0 squeezed
        # w_lora (tanh), a_lora (identity), g_lora (sigmoid, no bias), v_lora (identity, layers>0)
        out[f"{A}.w_lora.lora.0.weight"] = w[f"{a}.w1"].t().contiguous()
        out[f"{A}.w_lora.lora.2.weight"] = w[f"{a}.w2"].t().contiguous()
        out[f"{A}.w_lora.lora.2.bias"] = _sq(w[f"{a}.w0"]).contiguous()
        out[f"{A}.a_lora.lora.0.weight"] = w[f"{a}.a1"].t().contiguous()
        out[f"{A}.a_lora.lora.2.weight"] = w[f"{a}.a2"].t().contiguous()
        out[f"{A}.a_lora.lora.2.bias"] = _sq(w[f"{a}.a0"]).contiguous()
        out[f"{A}.g_lora.lora.0.weight"] = w[f"{a}.g1"].t().contiguous()
        out[f"{A}.g_lora.lora.2.weight"] = w[f"{a}.g2"].t().contiguous()
        if i > 0:
            out[f"{A}.v_lora.lora.0.weight"] = w[f"{a}.v1"].t().contiguous()
            out[f"{A}.v_lora.lora.2.weight"] = w[f"{a}.v2"].t().contiguous()
            out[f"{A}.v_lora.lora.2.bias"] = _sq(w[f"{a}.v0"]).contiguous()

        # ffn
        f = f"{b}.ffn"
        F = f"{L}.ffn"
        out[f"{F}.x_k"] = _sq(w[f"{f}.x_k"]).contiguous()
        out[f"{F}.key.weight"] = w[f"{f}.key.weight"].contiguous()
        out[f"{F}.value.weight"] = w[f"{f}.value.weight"].contiguous()

    os.makedirs(out_dir, exist_ok=True)
    save_file(out, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})

    config = {
        "architectures": ["RWKV7ForCausalLM"],
        "model_type": "rwkv7",
        "hidden_size": n_embd,
        "num_hidden_layers": n_layer,
        "head_dim": head_dim,
        "num_heads": n_head,  # derived from r_k: hidden//head_dim
        "decay_low_rank_dim": decay_lr,
        "a_low_rank_dim": a_lr,
        "v_low_rank_dim": v_lr,
        "gate_low_rank_dim": gate_lr,
        "intermediate_size": ffn_inter,
        "hidden_ratio": ffn_inter / n_embd,
        "hidden_act": "sqrelu",
        "norm_eps": 1e-5,
        "norm_bias": True,
        "norm_first": True,
        "vocab_size": vocab,
        "tie_word_embeddings": False,
        "attn": None,
        "attn_mode": "chunk",
        "bos_token_id": 0,
        "eos_token_id": 0,
        "use_cache": True,
        "torch_dtype": "float32",
    }
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"wrote {len(out)} tensors + config.json -> {out_dir}")
    return out, config


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    convert(args.pth, args.out)
