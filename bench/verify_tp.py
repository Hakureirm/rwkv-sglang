#!/usr/bin/env python3
"""Load a model dir under tensor parallelism and compare greedy generation to an
oracle fixture. Same gate as bench/greedy_check.py, launched at tp_size>1
(default 2). Exact match proves head-parallel TP reproduces the tp=1 numerics;
note the two row-parallel allreduces (o_proj, ffn.value) sum partial products in
a different order than the single tp=1 GEMM, so a ~1-ULP knife-edge argmax flip
is possible — report the first divergence, don't just eyeball "exact".

  python bench/verify_tp.py --model <dir> \
      --fixture bench/fixtures/oracle_rwkv7_15b_eiffel.json --tp 2
"""
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    args = ap.parse_args()

    fx = json.load(open(args.fixture))
    prompt_tokens = fx["prompt_tokens"]
    expected = fx["greedy_tokens"]
    n = len(expected)

    import sglang as sgl
    engine = sgl.Engine(
        model_path=args.model, skip_tokenizer_init=True,
        disable_cuda_graph=True, disable_piecewise_cuda_graph=True,
        disable_radix_cache=True, dtype=args.dtype, tp_size=args.tp,
        mem_fraction_static=args.mem_fraction,
    )
    out = engine.generate(input_ids=[prompt_tokens],
                          sampling_params={"temperature": 0.0, "max_new_tokens": n})
    rec = out[0] if isinstance(out, list) else out
    got = (rec.get("output_ids") or rec.get("token_ids") or rec.get("output_token_ids")
           or (rec.get("meta_info", {}) or {}).get("output_token_ids"))
    got = list(got)[:n]
    n_match = sum(1 for i in range(min(len(got), n)) if got[i] == expected[i])
    div = next((i for i in range(min(len(got), n)) if got[i] != expected[i]), None)
    print(f"TP={args.tp} GREEDY MATCH {n_match}/{n}  first_div={div}  "
          f"exact={got == expected}")
    print("GOT ", got)
    print("EXP ", expected)
    engine.shutdown()
    sys.exit(0 if got == expected else 1)


if __name__ == "__main__":
    main()
