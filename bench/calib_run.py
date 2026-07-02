#!/usr/bin/env python3
"""Run the fp16 RWKV-7 model through calibration text so the RWKV_CALIB hook in
models/rwkv7.py accumulates per-projection input Hessians (X^T X) for GPTQ.

Uses a plain text corpus (one document per line). Prefill of each doc captures the
projection inputs; the hook dumps to $RWKV_CALIB_OUT once RWKV_CALIB_TOKENS is reached.

  RWKV_CALIB=1 RWKV_CALIB_OUT=<dir> RWKV_CALIB_TOKENS=20000 \
      python bench/calib_run.py --model <fla> --corpus <calib.txt> --maxlen 256
"""
import argparse, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, help="text file, one document per line")
    ap.add_argument("--maxlen", type=int, default=256, help="cap tokens/doc (bounds activation mem)")
    ap.add_argument("--mem-fraction", type=float, default=0.55)
    ap.add_argument("--docs", type=int, default=400, help="max docs to feed")
    args = ap.parse_args()

    if os.environ.get("RWKV_CALIB") != "1":
        print("WARNING: RWKV_CALIB!=1 — no Hessians will be captured", file=sys.stderr)

    lines = [ln.strip() for ln in open(args.corpus, encoding="utf-8") if ln.strip()]
    lines = lines[: args.docs]
    print(f"calibration: {len(lines)} docs from {args.corpus}", file=sys.stderr)

    import sglang as sgl
    engine = sgl.Engine(
        model_path=args.model, dtype="float16",
        disable_cuda_graph=True, disable_piecewise_cuda_graph=True,
        disable_radix_cache=True, trust_remote_code=True, tp_size=1,
        mem_fraction_static=args.mem_fraction,
    )
    target = int(os.environ.get("RWKV_CALIB_TOKENS", "20000"))
    done = 0
    for i, text in enumerate(lines):
        engine.generate(
            prompt=text,
            sampling_params={"temperature": 0.0, "max_new_tokens": 1, "truncate": args.maxlen}
            if False else {"temperature": 0.0, "max_new_tokens": 1},
        )
        done += min(len(text.split()), args.maxlen)  # rough token estimate for progress
        if i % 25 == 0:
            print(f"  fed {i+1} docs (~{done} words)", file=sys.stderr, flush=True)
        if done > target * 1.5:  # generous margin over the hook's exact token target
            break
    engine.shutdown()
    print("calibration run complete", file=sys.stderr)


if __name__ == "__main__":
    main()
