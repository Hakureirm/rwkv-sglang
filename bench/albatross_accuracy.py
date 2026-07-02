#!/usr/bin/env python3
"""
Albatross GREEDY accuracy vs the numpy/rwkv-lm reference oracle.

The speed comparison (comparison_clean.md) is ours-fp16 vs albatross-fp16. This script
measures the OTHER axis at that same precision: does albatross's **fp16 WKV state** drift
from the token-exact reference? We greedily roll out albatross on the fixture prompt and
count how many tokens match the numpy-oracle `greedy_tokens` (the same fixture OURS matches
token-for-token). Expectation: albatross-fp16 (fp16 state) drifts; ours (fp32 state) does not
-> the accuracy win at equal weight precision.

Run in the albatross dir env (CUDA_HOME=/usr/local/cuda-12.9, TORCH_CUDA_ARCH_LIST=8.6):
  ~/envs/rwkv-sgl/bin/python bench/albatross_accuracy.py \
      --pth ~/rwkv_models/rwkv7-g1/rwkv7-g1g-1.5b-20260526-ctx8192.pth \
      --fixture bench/fixtures/oracle_rwkv7_15b_eiffel.json \
      --albatross-dir ~/albatross/faster3a_2605 --wkv fp16 \
      --out bench/results/clean/albatross_acc_1.5B_fp16.json
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--albatross-dir", default="~/albatross/faster3a_2605")
    ap.add_argument("--wkv", choices=("fp16", "fp32io16"), default="fp16",
                    help="fp16 = albatross native/speed-benched (fp16 state); "
                         "fp32io16 = its more-accurate fp32-state option")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # resolve data/output paths to ABSOLUTE before chdir'ing into the albatross dir
    fixture_path = os.path.abspath(os.path.expanduser(args.fixture))
    out_path = os.path.abspath(os.path.expanduser(args.out)) if args.out else ""
    pth_path = os.path.abspath(os.path.expanduser(args.pth))

    sys.path.insert(0, os.path.expanduser(args.albatross_dir))
    os.chdir(os.path.expanduser(args.albatross_dir))
    import torch
    import rwkv7_fast_v3a as v3a

    # match the speed-benchmark defaults (rwkv7_fast_v3a.py argparse), varying only WKV state.
    v3a.MODEL_PATH = pth_path
    v3a.WKV_MODE = args.wkv
    v3a.EMB_DEVICE = "cpu"
    v3a.RKV_MODE = "off"
    v3a.CMIX_SPARSE = "no-fc"
    v3a.LOWRANK_WEIGHT = "both"
    v3a.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
    v3a.load_extensions(v3a.WKV_MODE)
    model = v3a.RWKV7()

    fx = json.load(open(fixture_path))
    prompt = fx["prompt_tokens"]
    expected = fx["greedy_tokens"]
    n = len(expected)

    state = model.zero_state(1)
    with torch.no_grad():
        logits = model.forward(torch.tensor([prompt], dtype=torch.long), state)
        got = []
        for _ in range(n):
            tok = int(logits.reshape(-1).float().argmax().item())
            got.append(tok)
            logits = model.forward(torch.tensor([[tok]], dtype=torch.long), state)

    n_match = sum(1 for i in range(n) if got[i] == expected[i])
    div = next((i for i in range(n) if got[i] != expected[i]), None)
    exact = got == expected
    res = {
        "engine": "albatross", "wkv_state": args.wkv, "pth": args.pth,
        "n": n, "n_match": n_match, "exact": exact, "first_divergence_index": div,
        "got": got, "expected": expected,
    }
    print(json.dumps({k: v for k, v in res.items() if k not in ("got", "expected")}, indent=2))
    print("GOT     ", got)
    print("EXPECTED", expected)
    print(f"WKV_STATE {args.wkv}  MATCH {n_match}/{n}  EXACT {exact}  FIRST_DIVERGENCE {div}")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(res, open(out_path, "w"), indent=2)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
