#!/usr/bin/env python3
"""Write a 'fake-quant' RWKV-7 checkpoint: target linears round-tripped through
group-wise int4 (quantize -> dequantize back to bf16), everything else untouched.

This measures the ACCURACY impact of a w4 scheme end-to-end WITHOUT any kernel or
model surgery — the resulting dir loads with the stock model. Because rwkv7_w4.cu is
bit-identical to this dequant (verify_w4.py: rel err ~2e-4), a passing fake-quant here
predicts the real int4-kernel accuracy. De-risks before integration.

  python bench/make_fakequant_w4.py --model ~/rwkv_models/rwkv7-1.5b-fla \
      --out ~/rwkv_models/rwkv7-1.5b-w4fake-g64 --group 64 [--sym|--asym]
"""
import argparse, glob, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

TARGET_SUFFIXES = ("r_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                   ".key.weight", ".value.weight")


def w4_roundtrip(W, group, asym, clip):
    N, K = W.shape
    if K % group:
        return W, False
    orig_dtype = W.dtype
    Wg = W.float().view(N, K // group, group)
    if asym:
        mn = Wg.amin(dim=2, keepdim=True)
        s = ((Wg.amax(dim=2, keepdim=True) - mn) / 15.0).clamp(min=1e-8)
        q = torch.round((Wg - mn) / s).clamp_(0, 15)
        Wdq = (q * s + mn)
    elif clip:
        # per-group MSE-optimal symmetric scale (search the clip ratio)
        maxv = Wg.abs().amax(dim=2, keepdim=True)                      # [N,NG,1]
        best_s = (maxv / 7.0).clamp(min=1e-8)
        best_e = torch.full_like(maxv, float("inf"))
        for c in [x / 100.0 for x in range(60, 101, 2)]:              # 0.60..1.00
            s = (c * maxv / 7.0).clamp(min=1e-8)
            dq = torch.round(Wg / s).clamp_(-7, 7) * s
            e = ((dq - Wg) ** 2).sum(dim=2, keepdim=True)
            better = e < best_e
            best_s = torch.where(better, s, best_s)
            best_e = torch.where(better, e, best_e)
        Wdq = torch.round(Wg / best_s).clamp_(-7, 7) * best_s
    else:
        s = (Wg.abs().amax(dim=2, keepdim=True) / 7.0).clamp(min=1e-8)
        q = torch.round(Wg / s).clamp_(-7, 7)
        Wdq = (q * s)
    return Wdq.view(N, K).to(orig_dtype), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--clip", action="store_true", help="per-group MSE-optimal clip search (sym)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # copy non-weight files (config, tokenizer, etc.)
    for f in os.listdir(args.model):
        if not f.endswith(".safetensors"):
            src = os.path.join(args.model, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.out, f))

    n_q = 0
    n_tot = 0
    for sf in glob.glob(os.path.join(args.model, "*.safetensors")):
        sd = load_file(sf)
        out = {}
        for name, W in sd.items():
            if W.ndim == 2 and name.endswith(TARGET_SUFFIXES):
                n_tot += 1
                Wq, ok = w4_roundtrip(W.cuda(), args.group, args.asym, args.clip)
                out[name] = Wq.cpu().contiguous()
                n_q += int(ok)
            else:
                out[name] = W
        save_file(out, os.path.join(args.out, os.path.basename(sf)),
                  metadata={"format": "pt"})
    print(f"fake-quant w4 ({'asym' if args.asym else 'sym'} g{args.group}): "
          f"quantized {n_q}/{n_tot} target matrices -> {args.out}")


if __name__ == "__main__":
    main()
