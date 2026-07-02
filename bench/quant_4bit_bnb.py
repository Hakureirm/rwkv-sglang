#!/usr/bin/env python3
"""M4 4-bit probe: load the bf16 fla checkpoint with sglang on-the-fly
bitsandbytes nf4 (4-bit) quantization and run the greedy fixture check.

  source ~/rwkv_env.sh && ~/envs/rwkv-sgl/bin/python bench/quant_4bit_bnb.py \
      --model /home/user/rwkv_models/rwkv7-1.5b-fla \
      --fixture bench/fixtures/oracle_rwkv7_15b_eiffel.json
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    args = ap.parse_args()

    fx = json.load(open(args.fixture))
    prompt_tokens = fx["prompt_tokens"]
    expected = fx["greedy_tokens"]
    n = len(expected)

    import sglang as sgl

    engine = sgl.Engine(
        model_path=args.model,
        skip_tokenizer_init=True,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        dtype="bfloat16",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        tp_size=1,
        mem_fraction_static=args.mem_fraction,
    )
    out = engine.generate(
        input_ids=[prompt_tokens],
        sampling_params={"temperature": 0.0, "max_new_tokens": n},
    )
    rec = out[0] if isinstance(out, list) else out
    got = (
        rec.get("output_ids")
        or rec.get("token_ids")
        or rec.get("output_token_ids")
        or (rec.get("meta_info", {}) or {}).get("output_token_ids")
    )
    got = list(got)[:n]
    exact = got == expected
    div = next((i for i in range(min(len(got), n)) if got[i] != expected[i]), None)
    n_match = sum(1 for i in range(min(len(got), n)) if got[i] == expected[i])
    print("GOT_IDS ", got)
    print("EXPECTED", expected)
    print(f"BNB-NF4  EXACT_MATCH {exact}  MATCH {n_match}/{n}  DIV {div}")
    engine.shutdown()
    sys.exit(0 if exact else 1)


if __name__ == "__main__":
    main()
