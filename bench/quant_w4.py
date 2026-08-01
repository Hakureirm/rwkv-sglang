#!/usr/bin/env python3
"""Offline weight-only int4 quantizer: fla RWKV-7 checkpoint -> w4 checkpoint.

Quantizes ONLY the big linear projections (r/k/v/o + ffn key/value) to group-wise
symmetric int4 (GROUP=64, matching rwkv7_w4.cu), storing for each `<name>.weight`:
  <name>.qweight : uint8 [N, K/2]  (2 signed nibbles/byte, little-endian along K)
  <name>.scale   : fp16  [N, K/64]
All other tensors (LoRA rank matrices, norms, embeddings, lm_head, WKV params) are
kept at original precision — they are tiny and/or precision-sensitive. Writes a
`quantization: rwkv_w4` marker + group size into config.json.

  python bench/quant_w4.py --model <fla_dir> --out <w4_dir> [--group 64]

The packing/scale convention is validated bit-identically by bench/verify_w4.py.
"""
import argparse, glob, json, os, re, shutil
import torch
from safetensors.torch import load_file, save_file

TARGET_SUFFIXES = ("r_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                   ".key.weight", ".value.weight")
LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def pack_w8(W: torch.Tensor, group: int):
    """W [N,K] -> (qweight int8[N,K], scale fp16[N,K/group]). Symmetric int8 (near-lossless)."""
    N, K = W.shape
    NG = K // group
    Wg = W.float().view(N, NG, group)
    scale = (Wg.abs().amax(dim=2) / 127.0).clamp(min=1e-8)
    q = torch.round(Wg / scale[:, :, None]).clamp_(-127, 127).to(torch.int8).view(N, K)
    return q.contiguous(), scale.to(torch.float16)


def pack_w4(W: torch.Tensor, group: int):
    """W [N,K] -> (qweight uint8[N,K/2], scale fp16[N,K/group]). Symmetric int4."""
    N, K = W.shape
    NG = K // group
    Wg = W.float().view(N, NG, group)
    scale = (Wg.abs().amax(dim=2) / 7.0).clamp(min=1e-8)          # [N,NG]
    q = torch.round(Wg / scale[:, :, None]).clamp_(-7, 7).to(torch.int32).view(N, K)
    nib = (q & 0xF).to(torch.uint8)
    qweight = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()   # [N, K/2]
    return qweight, scale.to(torch.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--keep-layers", default="",
                    help="comma list of layer indices to leave at checkpoint precision "
                         "(mixed precision; serve with the SAME list in RWKV_W4_KEEP_LAYERS)")
    ap.add_argument("--keep-tensors", default="",
                    help="comma list of projection kinds to leave at checkpoint precision, "
                         "e.g. 'v_proj,value'. The other axis of the same question: is the "
                         "int4 damage localized in a few LAYERS or in a few PROJECTION KINDS? "
                         "(serve with the SAME list in RWKV_W4_KEEP_TENSORS)")
    args = ap.parse_args()
    keep = {int(t) for t in args.keep_layers.replace(" ", "").split(",") if t}
    keep_t = tuple(t for t in args.keep_tensors.replace(" ", "").split(",") if t)

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.model):
        if not f.endswith(".safetensors"):
            src = os.path.join(args.model, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.out, f))

    n_q = n_skip = 0
    for sf in glob.glob(os.path.join(args.model, "*.safetensors")):
        sd = load_file(sf)
        out = {}
        for name, W in sd.items():
            m = LAYER_IDX_RE.search(name)
            base_name = name[: -len(".weight")] if name.endswith(".weight") else name
            protected = (keep and m is not None and int(m.group(1)) in keep) or (
                keep_t and base_name.endswith(keep_t))
            if (W.ndim == 2 and name.endswith(TARGET_SUFFIXES)
                    and (W.shape[1] % args.group == 0) and not protected):
                qw, sc = (pack_w4 if args.bits == 4 else pack_w8)(W.cuda(), args.group)
                base = name[: -len(".weight")]
                out[base + ".qweight"] = qw.cpu().contiguous()
                out[base + ".scale"] = sc.cpu().contiguous()
                n_q += 1
            else:
                out[name] = W
                n_skip += 1
        save_file(out, os.path.join(args.out, os.path.basename(sf)), metadata={"format": "pt"})

    # provenance marker under a NON-standard key (not `quantization_config`, which sglang
    # would try to interpret as a known quant method). The model is told it's w4 via the
    # RWKV_W4=1 env flag at serve time, not via config.json.
    cfg_path = os.path.join(args.out, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        cfg["rwkv7_w4_info"] = {"quant_method": f"rwkv_w{args.bits}", "group_size": args.group,
                                "bits": args.bits, "sym": True,
                                "target_suffixes": list(TARGET_SUFFIXES),
                                "keep_layers": sorted(keep),
                                "keep_tensors": list(keep_t)}
        json.dump(cfg, open(cfg_path, "w"), indent=2)
    kept = f", protected layers {sorted(keep)}" if keep else ""
    kept += f", protected tensors {list(keep_t)}" if keep_t else ""
    print(f"w{args.bits} quant (sym g{args.group}): packed {n_q} matrices, "
          f"kept {n_skip} tensors{kept} -> {args.out}")


if __name__ == "__main__":
    main()
