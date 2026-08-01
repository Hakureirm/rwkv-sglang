#!/usr/bin/env python3
"""How far did a quantized checkpoint move the weights? Offline, no GPU, no calibration data.

Unpacks int4/int8 checkpoints through the same nibble/scale convention the kernels use
(rwkv7_w4.cu) and reports relative reconstruction error against the fp16 source, overall
and per matrix. Two checkpoints can be compared against the same source at once.

  python bench/quant_weight_error.py --src <fla_dir> --ckpt RTN=<w4_dir> --ckpt GPTQ=<w4gptq_dir>

Why this is worth having: when a quantized checkpoint scores badly, the first question is
whether the quantizer is broken or whether it is working as designed and the design is wrong
for that metric. This separates those cheaply. Note the asymmetry when reading the output --
RTN *minimizes* this quantity by construction, while GPTQ minimizes the activation-weighted
error ||(W-W')X|| instead and will knowingly trade plain weight error away for it. So GPTQ
scoring worse here is expected behaviour, not evidence of a bug; what the number tells you is
how much fidelity it spent. It cannot tell you whether it got fair value, because the thing
GPTQ actually optimizes needs the calibration activations to evaluate (see F0082).
"""
import argparse, glob, os, sys
import torch
from safetensors.torch import load_file

GROUP_DEFAULT = 64


def unpack_w4(qw: torch.Tensor, sc: torch.Tensor, group: int) -> torch.Tensor:
    """uint8 [N,K/2] of two signed nibbles (low then high along K) + fp16 [N,K/group] scales.
    Matches rwkv7_w4.cu's decode, including its two's-complement sign extension to [-8,7]."""
    N, Kh = qw.shape
    K = Kh * 2
    q = torch.empty(N, K, dtype=torch.int16)
    q[:, 0::2] = (qw & 0xF).to(torch.int16)
    q[:, 1::2] = (qw >> 4).to(torch.int16)
    q -= (q & 8) << 1
    return (q.float().view(N, K // group, group) * sc.float()[:, :, None]).view(N, K)


def unpack_w8(qw: torch.Tensor, sc: torch.Tensor, group: int) -> torch.Tensor:
    N, K = qw.shape
    return (qw.float().view(N, K // group, group) * sc.float()[:, :, None]).view(N, K)


def load_dir(d: str) -> dict:
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors in {d}")
    out = {}
    for f in files:
        out.update(load_file(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unquantized fla checkpoint dir")
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=dir (repeatable)")
    ap.add_argument("--group", type=int, default=GROUP_DEFAULT)
    ap.add_argument("--top", type=int, default=0, help="also list the N worst matrices per ckpt")
    args = ap.parse_args()

    src = load_dir(args.src)
    labelled = []
    for spec in args.ckpt:
        if "=" not in spec:
            raise SystemExit(f"expected LABEL=dir, got {spec!r}")
        lab, d = spec.split("=", 1)
        labelled.append((lab, load_dir(d)))

    names = sorted({k[:-8] for _, sd in labelled for k in sd if k.endswith(".qweight")})
    if not names:
        raise SystemExit("no .qweight tensors found — are these quantized checkpoints?")

    per = {lab: {} for lab, _ in labelled}      # label -> {matrix name: relative error}
    tot = {lab: 0.0 for lab, _ in labelled}
    norm = 0.0
    for b in names:
        W = src[b + ".weight"].float()
        wn = W.pow(2).sum().item()
        norm += wn
        for lab, sd in labelled:
            if b + ".qweight" not in sd:
                continue
            qw, sc = sd[b + ".qweight"], sd[b + ".scale"]
            deq = (unpack_w8 if qw.dtype == torch.int8 else unpack_w4)(qw, sc, args.group)
            e = (deq.view(W.shape) - W).pow(2).sum().item()
            tot[lab] += e
            per[lab][b] = (e / max(wn, 1e-30)) ** 0.5

    print(f"{len(names)} quantized matrices, group={args.group}\n")
    for lab, _ in labelled:
        print(f"{lab:8s} relative weight error = {(tot[lab] / norm) ** 0.5:.6f}")

    if len(labelled) == 2:
        la, lb = labelled[0][0], labelled[1][0]
        shared = sorted(set(per[la]) & set(per[lb]))
        worse = sum(1 for b in shared if per[lb][b] > per[la][b])
        print(f"\n{lb} worse than {la} on {worse}/{len(shared)} shared matrices")

    if args.top:
        for lab, _ in labelled:
            print(f"\nworst {args.top} for {lab}:")
            for b, e in sorted(per[lab].items(), key=lambda kv: -kv[1])[: args.top]:
                print(f"  {e:.6f}  {b}")


if __name__ == "__main__":
    main()
