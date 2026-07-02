#!/usr/bin/env python3
"""Offline W8A8-int8 quantizer for RWKV-7 (fla-format) checkpoints (M4).

Produces a sglang-loadable `--quantization w8a8_int8` checkpoint: each quantized
linear weight W [out, in] is stored as

  * `<name>.weight`        int8   [out, in]   (per-output-channel symmetric)
  * `<name>.weight_scale`  float32[out, 1]    scale[o] = max|W[o,:]| / 127

Activations are quantized dynamically per-token at runtime (sglang
`per_token_quant_int8` + `int8_scaled_mm`), so NO calibration data is needed —
the weight scales are a closed-form per-channel max. The WKV recurrence/state and
the small per-channel params (x_*, k_k, k_a, r_k, g_norm, all norms, biases,
embeddings, lm_head) are copied through unchanged in their original dtype.

Quantized modules (matches the ReplicatedLinear layers in models/rwkv7.py):
  attn r/k/v/o_proj, ffn key/value, and every LoRA down (lora.0) / up (lora.2).

Usage (on the box):
  ~/envs/rwkv-sgl/bin/python tools/quantize_w8a8_int8.py \
      --src /home/user/rwkv_models/rwkv7-1.5b-fla \
      --dst /home/user/rwkv_models/rwkv7-1.5b-w8a8 \
      [--ignore-lora]
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROJ_SUFFIXES = (
    ".r_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".ffn.key.weight",
    ".ffn.value.weight",
)
LORA_SUFFIXES = (".lora.0.weight", ".lora.2.weight")


def is_quantizable(key: str, quant_lora: bool) -> bool:
    if key.endswith(PROJ_SUFFIXES):
        return True
    if quant_lora and key.endswith(LORA_SUFFIXES):
        return True
    return False


def quantize_per_channel(w: torch.Tensor):
    """w: [out, in] float -> (int8 [out,in], fp32 scale [out,1])."""
    wf = w.to(torch.float32)
    scale = wf.abs().amax(dim=1, keepdim=True) / 127.0
    scale = scale.clamp_min(1e-8)
    q = torch.round(wf / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source fla-format model dir")
    ap.add_argument("--dst", required=True, help="output int8 model dir")
    ap.add_argument(
        "--ignore-lora",
        action="store_true",
        help="keep LoRA down/up in bf16 (only quantize r/k/v/o_proj + ffn). The "
        "config.json 'ignore' list is set accordingly so the runtime matches.",
    )
    args = ap.parse_args()
    quant_lora = not args.ignore_lora

    os.makedirs(args.dst, exist_ok=True)
    src_st = os.path.join(args.src, "model.safetensors")

    out = {}
    n_quant = 0
    n_copy = 0
    quant_bytes = 0
    orig_bytes = 0
    ignore_prefixes = set()
    with safe_open(src_st, "pt") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            if is_quantizable(k, quant_lora):
                q, scale = quantize_per_channel(t)
                out[k] = q.contiguous()
                out[k[: -len(".weight")] + ".weight_scale"] = scale.contiguous()
                n_quant += 1
                quant_bytes += q.numel() * 1 + scale.numel() * 4
                orig_bytes += t.numel() * t.element_size()
            else:
                out[k] = t.contiguous()
                n_copy += 1
                # record un-quantized linear prefixes for the ignore list
                if args.ignore_lora and k.endswith(LORA_SUFFIXES):
                    ignore_prefixes.add(k[: -len(".weight")])

    save_file(out, os.path.join(args.dst, "model.safetensors"))

    # config.json: add quantization_config so sglang auto-detects w8a8_int8.
    with open(os.path.join(args.src, "config.json")) as f:
        cfg = json.load(f)
    qc = {"quant_method": "w8a8_int8"}
    if args.ignore_lora:
        # sglang should_ignore_layer matches with check_contains=True, i.e. the
        # ignore entries are substrings of the layer prefix. Every LoRA leaf
        # linear prefix (…{w,a,g,v}_lora.lora.0 / .lora.2) contains "lora".
        qc["ignore"] = ["lora"]
    cfg["quantization_config"] = qc
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # copy tokenizer / aux files if present (none expected for skip_tokenizer_init)
    for fn in os.listdir(args.src):
        if fn in ("model.safetensors", "config.json"):
            continue
        sp = os.path.join(args.src, fn)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(args.dst, fn))

    print(f"quantized {n_quant} linear weights, copied {n_copy} tensors")
    print(
        f"quantized-weight bytes: {orig_bytes/1e6:.1f}MB (bf16) -> "
        f"{quant_bytes/1e6:.1f}MB (int8+scale)  "
        f"ratio {orig_bytes/max(quant_bytes,1):.2f}x"
    )
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
