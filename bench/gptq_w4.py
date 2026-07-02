#!/usr/bin/env python3
"""Offline GPTQ int4 quantizer for RWKV-7 -> w4 checkpoint (same .qweight/.scale format
as bench/quant_w4.py, so rwkv7_w4.cu serves it unchanged — no kernel/model change).

GPTQ (Frantar et al.) is activation-aware error-feedback quantization: it uses per-layer
input Hessians H = X^T X (captured by the RWKV_CALIB hook in models/rwkv7.py) to quantize
columns left-to-right, propagating each column's rounding error to the not-yet-quantized
columns via H^-1. This recovers most of the accuracy RTN loses (RTN g64 was -4.95pt; GPTQ
targets ~int8's -2pt) while keeping true 4-bit weights.

  # 1. capture Hessians (fp16 model, calibration prompts):
  RWKV_CALIB=1 RWKV_CALIB_OUT=<dir> RWKV_CALIB_TOKENS=20000 \
      python bench/calib_run.py --model <fla> --tokens 20000
  # 2. GPTQ -> w4 checkpoint:
  python bench/gptq_w4.py --model <fla> --hessians <dir>/calib_hessians.pt \
      --out <w4_gptq_dir> --group 64
"""
import argparse, glob, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

TARGET_SUFFIXES = ("r_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                   ".key.weight", ".value.weight")


def pack_w4(Q_int: torch.Tensor, scale: torch.Tensor):
    """Q_int [N,K] in [-7,7], scale [N,NG] -> qweight uint8[N,K/2] (matches rwkv7_w4.cu)."""
    nib = (Q_int.to(torch.int32) & 0xF).to(torch.uint8)
    qweight = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    return qweight, scale.to(torch.float16)


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, H: torch.Tensor, group: int, damp: float = 0.01):
    """W [N,K] fp32, H [K,K] fp32 (=X^T X). Returns (Q_int [N,K], scale [N,K/group])."""
    N, K = W.shape
    W = W.clone().float()
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H[range(K), range(K)] += damp * torch.mean(torch.diag(H))
    # inverse Cholesky (upper-triangular factor of H^-1)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    NG = K // group
    scales = torch.zeros(N, NG, device=W.device)
    for i in range(K):
        g = i // group
        if i % group == 0:
            blk = W[:, i:i + group]
            scales[:, g] = (blk.abs().amax(dim=1) / 7.0).clamp(min=1e-8)
        s = scales[:, g]
        w = W[:, i]
        q = torch.round(w / s).clamp_(-7, 7)
        Q[:, i] = q
        err = (w - q * s) / Hinv[i, i]
        if i + 1 < K:
            W[:, i + 1:] -= err[:, None] * Hinv[i, i + 1:][None, :]
    return Q.to(torch.int32), scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hessians", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--damp", type=float, default=0.01)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.model):
        if not f.endswith(".safetensors"):
            src = os.path.join(args.model, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.out, f))

    if os.path.isdir(args.hessians) or os.path.isdir(os.path.join(args.hessians, "hessians")):
        shard_dir = args.hessians if os.path.isdir(args.hessians) and not os.path.exists(
            os.path.join(args.hessians, "hessians")) else os.path.join(args.hessians, "hessians")
        # lazy per-shard loading: 7.2B shards are ~1 GiB each; load to CUDA one at
        # a time inside the quantization loop instead of all upfront
        class _ShardHess(dict):
            def __init__(self, d):
                super().__init__()
                self._dir = d
                for f in os.listdir(d):
                    if f.endswith(".pt"):
                        self[f[:-3]] = None
            def fetch(self, k):
                p = os.path.join(self._dir, k + ".pt")
                try:
                    h = torch.load(p, map_location="cuda")["hessian"]
                except Exception as e:  # truncated/corrupt shard -> RTN fallback
                    print(f"  !! shard {k} unreadable ({e}) -> RTN fallback")
                    return None
                if os.environ.get("GPTQ_CONSUME_SHARDS", "0") == "1":
                    os.unlink(p)  # bound disk: shards shrink as the checkpoint grows
                return h
        hess = _ShardHess(shard_dir)
        print(f"sharded Hessians: {len(hess)} shards in {shard_dir}")
    else:
        hess = torch.load(args.hessians, map_location="cuda")["hessian"]
        print(f"loaded {len(hess)} Hessians")

    n_q = n_skip = n_rtn = 0
    for sf in glob.glob(os.path.join(args.model, "*.safetensors")):
        sd = load_file(sf)
        out = {}
        for name, W in sd.items():
            if W.ndim == 2 and name.endswith(TARGET_SUFFIXES) and (W.shape[1] % args.group == 0):
                base = name[: -len(".weight")]
                Hcur = None
                if base in hess:
                    Hcur = (hess.fetch(base) if hasattr(hess, "fetch")
                            else hess[base].cuda())
                if Hcur is not None:
                    Q, sc = gptq_quantize(W.cuda(), Hcur, args.group, args.damp)
                    del Hcur
                    n_q += 1
                else:  # no/unreadable Hessian -> RTN fallback
                    Wg = W.cuda().float().view(W.shape[0], -1, args.group)
                    sc = (Wg.abs().amax(dim=2) / 7.0).clamp(min=1e-8)
                    Q = torch.round(Wg / sc[:, :, None]).clamp_(-7, 7).to(torch.int32).view(W.shape)
                    n_rtn += 1
                qw, scf = pack_w4(Q, sc)
                out[base + ".qweight"] = qw.cpu().contiguous()
                out[base + ".scale"] = scf.cpu().contiguous()
            else:
                out[name] = W
                n_skip += 1
        save_file(out, os.path.join(args.out, os.path.basename(sf)), metadata={"format": "pt"})

    cfg_path = os.path.join(args.out, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        cfg["rwkv7_w4_info"] = {"quant_method": "rwkv_w4_gptq", "group_size": args.group,
                                "bits": 4, "sym": True}
        json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"GPTQ w4 (g{args.group}): {n_q} GPTQ + {n_rtn} RTN-fallback matrices, "
          f"kept {n_skip} tensors -> {args.out}")


if __name__ == "__main__":
    main()
