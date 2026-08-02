#!/usr/bin/env python3
"""Group-size / scale-dtype sweep, and the checkpoint writer for the arms it picks out.

Two jobs, deliberately in one file so the checkpoint that gets served is built by the
same code whose weight error is quoted:

  --mode sweep   reproduce the relative-weight-error table (no checkpoint written)
  --mode write   snap a checkpoint to one (lattice, group, scale dtype) and store fp16

An fp8 group scale is only honest with the per-tensor fp32 factor NVFP4 uses. e4m3's
smallest normal is 2^-6 and these scales run around 1e-2, so quantizing them directly
would spend most of the format's range on values that never occur. The factor
normalises the largest scale in a tensor onto e4m3's top end first.
"""
import argparse, glob, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

TARGET_SUFFIXES = ("r_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                   ".key.weight", ".value.weight")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FITTED15 = torch.tensor([-0.9467, -0.7312, -0.5667, -0.4308, -0.3096, -0.2012, -0.0993,
                         -0.0027, 0.0927, 0.1926, 0.3003, 0.4190, 0.5554, 0.7185, 0.9402])
E4M3_MAX = 448.0


def cast_scale(s, dtype):
    """s: [N, K/g] fp32 group scales."""
    if dtype == "fp16":
        return s.half().float()
    if dtype == "fp32":
        return s
    if dtype == "fp8":
        per_tensor = s.max() / E4M3_MAX                      # one fp32 number per tensor
        return (s / per_tensor).to(torch.float8_e4m3fn).float() * per_tensor
    raise ValueError(dtype)


def snap(W, lattice, group, scale_dtype):
    N, K = W.shape
    Wg = W.float().view(N, K // group, group)
    absmax = Wg.abs().amax(2).clamp(min=1e-8)
    if lattice == "int4":
        s = cast_scale(absmax / 7.0, scale_dtype)[:, :, None]
        return (torch.round(Wg / s).clamp_(-7, 7) * s).view(N, K)
    if lattice == "fp4":
        s = cast_scale(absmax / 6.0, scale_dtype)[:, :, None]
        x = Wg / s
        g = E2M1.to(W.device)
        idx = (x.abs().unsqueeze(-1) - g).abs().argmin(-1)
        return (torch.sign(x) * g[idx] * s).view(N, K)
    if lattice == "table":
        s = cast_scale(absmax, scale_dtype)[:, :, None]      # table is normalised to +-1
        t = FITTED15.to(W.device)
        idx = ((Wg / s).unsqueeze(-1) - t).abs().argmin(-1)
        return (t[idx] * s).view(N, K)
    raise ValueError(lattice)


def targets(model, group):
    for sf in sorted(glob.glob(os.path.join(model, "*.safetensors"))):
        sd = load_file(sf)
        for name, W in sd.items():
            yield sf, sd, name, W, (W.ndim == 2 and name.endswith(TARGET_SUFFIXES)
                                    and W.shape[1] % group == 0)


def bits(group, scale_dtype):
    return 4.0 + {"fp8": 8, "fp16": 16, "fp32": 32}[scale_dtype] / group


def sweep(model, combos):
    mats = [(n, W.cuda().float()) for _, _, n, W, ok in targets(model, 16) if ok]
    print(f"{len(mats)} matrices\n")
    print(f"{'lattice':8} {'group':>5} {'scale':>5} {'rel.err':>10} {'bits/w':>7}")
    out = {}
    for lattice, group, sd_ in combos:
        err = norm = 0.0
        for _, Wc in mats:
            Q = snap(Wc, lattice, group, sd_)
            err += (Q - Wc).pow(2).sum().item()
            norm += Wc.pow(2).sum().item()
        r = (err / norm) ** 0.5
        out[(lattice, group, sd_)] = r
        print(f"{lattice:8} {group:>5} {sd_:>5} {r:>10.6f} {bits(group, sd_):>7.2f}")
    base = out.get(("int4", 64, "fp16"))
    if base:
        print(f"\nversus int4 g64 fp16 (= what we ship):")
        for k, v in out.items():
            print(f"  {k[0]:8} g{k[1]:<3} {k[2]:<5} {v / base:.3f}x")


def write(model, out_dir, lattice, group, scale_dtype):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(model):
        p = os.path.join(model, f)
        if os.path.isfile(p) and not f.endswith(".safetensors"):
            shutil.copy2(p, os.path.join(out_dir, f))
    n = 0
    err = norm = 0.0
    for sf in sorted(glob.glob(os.path.join(model, "*.safetensors"))):
        sd = load_file(sf)
        res = {}
        for name, W in sd.items():
            if W.ndim == 2 and name.endswith(TARGET_SUFFIXES) and W.shape[1] % group == 0:
                Wc = W.cuda().float()
                Q = snap(Wc, lattice, group, scale_dtype)
                err += (Q - Wc).pow(2).sum().item()
                norm += Wc.pow(2).sum().item()
                res[name] = Q.to(W.dtype).cpu().contiguous()
                n += 1
            else:
                res[name] = W
        save_file(res, os.path.join(out_dir, os.path.basename(sf)), metadata={"format": "pt"})
    rel = (err / norm) ** 0.5
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        cfg["rwkv7_grid_info"] = {"grid": lattice, "group_size": group,
                                  "scale_dtype": scale_dtype, "stored_as": "fp16 dequantized",
                                  "bits_per_weight": bits(group, scale_dtype),
                                  "relative_weight_error": rel}
        json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"{lattice} g{group} {scale_dtype}: snapped {n} matrices, rel err {rel:.6f}, "
          f"{bits(group, scale_dtype):.2f} bits/weight -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["sweep", "write"])
    ap.add_argument("--out")
    ap.add_argument("--grid", choices=["int4", "fp4", "table"])
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--scale", default="fp16", choices=["fp8", "fp16", "fp32"])
    args = ap.parse_args()
    if args.mode == "sweep":
        sweep(args.model, [(l, g, s) for g, s in
                           [(16, "fp8"), (32, "fp8"), (64, "fp16")]
                           for l in ("int4", "fp4", "table")])
    else:
        assert args.out and args.grid, "--mode write needs --out and --grid"
        write(args.model, args.out, args.grid, args.group, args.scale)


if __name__ == "__main__":
    main()
