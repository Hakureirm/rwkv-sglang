#!/usr/bin/env python3
"""Write a checkpoint whose weights have been through a 4-bit grid and back to fp16.

The point is to measure a GRID, not a kernel. Both int4 and fp4 land in `.weight` at
fp16, so both serve on the ordinary unquantized path and the only thing that differs
between two runs is which 4-bit lattice the weights were snapped to. No kernel work,
no dispatch differences, no dequant-vs-native accumulation to confound it.

  python grid_dequant.py --model <fla_dir> --grid int4|fp4 --out <dir>

int4 is the symmetric [-7, 7] lattice `bench/quant_w4.py` ships. fp4 is E2M1 (one sign
bit, two exponent, one mantissa): 15 distinct values like int4, but spaced
{0, .5, 1, 1.5, 2, 3, 4, 6} rather than uniformly, so it is finer where weight mass
actually sits and coarser in the tails.
"""
import argparse, glob, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

TARGET_SUFFIXES = ("r_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                   ".key.weight", ".value.weight")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# 15 levels fitted to RWKV-7's own group-normalised weight distribution by Lloyd-Max,
# symmetric and containing zero so it drops into the existing nibble packing. Fitted on
# 1.5B; it transfers to 7.2B unchanged (0.854x vs 0.860x relative weight error), so it is
# a property of the architecture's weights rather than of one checkpoint.
FITTED15 = torch.tensor([-0.9467, -0.7312, -0.5667, -0.4308, -0.3096, -0.2012, -0.0993,
                         -0.0027, 0.0927, 0.1926, 0.3003, 0.4190, 0.5554, 0.7185, 0.9402])


def through_int4(W, group):
    N, K = W.shape
    Wg = W.float().view(N, K // group, group)
    s = (Wg.abs().amax(2) / 7.0).clamp(min=1e-8)
    return (torch.round(Wg / s[:, :, None]).clamp_(-7, 7) * s[:, :, None]).view(N, K)


def through_table(W, group, table):
    """Snap to an arbitrary 15-level table. The kernel already turns each nibble into a
    float before multiplying, so a non-uniform table is a lookup rather than a new format
    and needs no fp4 hardware."""
    N, K = W.shape
    Wg = W.float().view(N, K // group, group)
    s = Wg.abs().amax(2).clamp(min=1e-8)              # table is normalised to +-1
    x = Wg / s[:, :, None]
    t = table.to(W.device)
    idx = (x.unsqueeze(-1) - t).abs().argmin(-1)
    return (t[idx] * s[:, :, None]).view(N, K)


def through_fp4(W, group):
    N, K = W.shape
    Wg = W.float().view(N, K // group, group)
    s = (Wg.abs().amax(2) / 6.0).clamp(min=1e-8)          # 6.0 is E2M1's largest value
    x = Wg / s[:, :, None]
    grid = E2M1.to(W.device)
    idx = (x.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    return (torch.sign(x) * grid[idx] * s[:, :, None]).view(N, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", required=True, choices=["int4", "fp4", "table"])
    ap.add_argument("--group", type=int, default=64)
    args = ap.parse_args()
    if args.grid == "int4":
        snap = through_int4
    elif args.grid == "fp4":
        snap = through_fp4
    else:
        snap = lambda W, g: through_table(W, g, FITTED15)

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.model):
        if not f.endswith(".safetensors"):
            p = os.path.join(args.model, f)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(args.out, f))

    n = 0
    err = norm = 0.0
    for sf in sorted(glob.glob(os.path.join(args.model, "*.safetensors"))):
        sd = load_file(sf)
        out = {}
        for name, W in sd.items():
            if W.ndim == 2 and name.endswith(TARGET_SUFFIXES) and W.shape[1] % args.group == 0:
                Wc = W.cuda().float()
                Q = snap(Wc, args.group)
                err += (Q - Wc).pow(2).sum().item()
                norm += Wc.pow(2).sum().item()
                out[name] = Q.to(W.dtype).cpu().contiguous()
                n += 1
            else:
                out[name] = W
        save_file(out, os.path.join(args.out, os.path.basename(sf)), metadata={"format": "pt"})

    cfg_path = os.path.join(args.out, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        cfg["rwkv7_grid_info"] = {"grid": args.grid, "group_size": args.group,
                                  "stored_as": "fp16 dequantized",
                                  "relative_weight_error": (err / norm) ** 0.5}
        json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"{args.grid} g{args.group}: snapped {n} matrices, "
          f"relative weight error {(err / norm) ** 0.5:.6f} -> {args.out}")


if __name__ == "__main__":
    main()
