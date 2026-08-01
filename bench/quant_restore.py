#!/usr/bin/env python3
"""Restore chosen projections in an already-quantized checkpoint back to source precision.

Checkpoint surgery, not re-quantization: it drops `<name>.qweight`/`<name>.scale` for the
matched suffixes and puts the original `<name>.weight` back from the fp16 source. Everything
else is copied byte-for-byte from the quantized checkpoint.

  python bench/quant_restore.py --ckpt <w4_dir> --src <fla_dir> --restore value --out <dir>
  # then serve with the SAME suffix list: RWKV_W4=1 RWKV_W4_KEEP_TENSORS=value

Why surgery rather than re-running the quantizer: for GPTQ checkpoints there is no way to
re-run it without the calibration Hessians, and the question worth asking is specifically
about the checkpoint we already shipped. F0082 found that GPTQ's weight error is concentrated
in `ffn.value` (relative error up to 0.266, versus 0.131 for RTN's worst matrix of any kind),
which is the tensor the state accumulation runs through. Surgery isolates that: same GPTQ
weights everywhere else, that one projection handed back its original values.

The serve-time flag list must match what was restored here, or the model builds a quantized
layer for a tensor that now carries `.weight` (or vice versa) and load fails on a missing key
-- loudly, which is the intended behaviour.
"""
import argparse, glob, json, os, shutil
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="quantized checkpoint dir to modify")
    ap.add_argument("--src", required=True, help="fp16 source the quantized ckpt came from")
    ap.add_argument("--restore", required=True,
                    help="comma list of projection suffixes to hand back, e.g. 'value' or "
                         "'k_proj,v_proj'. Matched against the name with '.weight' stripped, "
                         "so 'value' hits ffn.value and does NOT hit attn.v_proj.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sfx = tuple(t for t in args.restore.replace(" ", "").split(",") if t)
    if not sfx:
        raise SystemExit("--restore is empty; nothing to do")

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.ckpt):
        if not f.endswith(".safetensors"):
            p = os.path.join(args.ckpt, f)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(args.out, f))

    src = {}
    for f in sorted(glob.glob(os.path.join(args.src, "*.safetensors"))):
        src.update(load_file(f))

    n_restored = n_kept = 0
    for sf in sorted(glob.glob(os.path.join(args.ckpt, "*.safetensors"))):
        sd = load_file(sf)
        out = {}
        for name, T in sd.items():
            if name.endswith((".qweight", ".scale")):
                base = name.rsplit(".", 1)[0]
                if base.endswith(sfx):
                    if name.endswith(".qweight"):          # emit once per matrix
                        w = src.get(base + ".weight")
                        if w is None:
                            raise SystemExit(f"{base}.weight missing from --src")
                        out[base + ".weight"] = w
                        n_restored += 1
                    continue
            out[name] = T
            n_kept += 1
        save_file(out, os.path.join(args.out, os.path.basename(sf)), metadata={"format": "pt"})

    cfg_path = os.path.join(args.out, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        info = cfg.get("rwkv7_w4_info", {})
        info["keep_tensors"] = sorted(set(info.get("keep_tensors", [])) | set(sfx))
        info["restored_from_source"] = list(sfx)
        cfg["rwkv7_w4_info"] = info
        json.dump(cfg, open(cfg_path, "w"), indent=2)

    if n_restored == 0:
        raise SystemExit(f"restored nothing -- no quantized matrix ends with {sfx}; "
                         f"check the suffix against the checkpoint's key names")
    print(f"restored {n_restored} matrices from source ({', '.join(sfx)}), "
          f"kept {n_kept} tensors -> {args.out}")


if __name__ == "__main__":
    main()
