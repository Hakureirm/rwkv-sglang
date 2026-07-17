#!/usr/bin/env python3
"""
M1d independent verification / regression test: sglang RWKV-7 greedy output must
match the numpy-oracle fixture token-for-token. Run on the box (rwkv-sgl env,
with ~/rwkv_env.sh sourced).

  source ~/rwkv_env.sh && ~/envs/rwkv-sgl/bin/python bench/verify_m1d.py \
      --model /home/user/rwkv_models/rwkv7-0.1b-fla \
      --fixture bench/fixtures/oracle_rwkv7_01b_eiffel.json
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    ap.add_argument(
        "--dtype",
        default="float32",
        help="compute dtype (float32|bfloat16|float16). State stays fp32 per config.",
    )
    ap.add_argument(
        "--cuda-graph",
        action="store_true",
        help="enable CUDA graph for decode (M2b). Default off (M1 behavior).",
    )
    args = ap.parse_args()

    fx = json.load(open(args.fixture))
    prompt_tokens = fx["prompt_tokens"]
    expected = fx["greedy_tokens"]
    n = len(expected)

    import dataclasses

    import sglang as sgl
    from sglang.srt.server_args import ServerArgs

    # Version-adaptive cuda-graph kwargs: sglang main (>= ~754524d) replaced the
    # boolean disable_piecewise_cuda_graph with per-phase backends;
    # cuda_graph_backend_prefill="disabled" is the legacy flag's documented mapping.
    sa_fields = {f.name for f in dataclasses.fields(ServerArgs)}
    kw = dict(
        model_path=args.model,
        skip_tokenizer_init=True,
        disable_cuda_graph=not args.cuda_graph,
        dtype=args.dtype,
        tp_size=1,
        mem_fraction_static=args.mem_fraction,
    )
    if "disable_piecewise_cuda_graph" in sa_fields:
        kw["disable_piecewise_cuda_graph"] = True
    elif "cuda_graph_backend_prefill" in sa_fields:
        kw["cuda_graph_backend_prefill"] = "disabled"
    engine = sgl.Engine(**kw)
    out = engine.generate(
        input_ids=[prompt_tokens],
        sampling_params={"temperature": 0.0, "max_new_tokens": n},
    )
    rec = out[0] if isinstance(out, list) else out
    # robustly pull generated ids across sglang return shapes
    got = (
        rec.get("output_ids")
        or rec.get("token_ids")
        or rec.get("output_token_ids")
        or (rec.get("meta_info", {}) or {}).get("output_token_ids")
    )
    if got is None:
        print("COULD NOT FIND output ids; keys:", list(rec.keys()))
        print("rec:", rec)
        engine.shutdown()
        sys.exit(2)
    got = list(got)[:n]

    exact = got == expected
    div = next((i for i in range(min(len(got), n)) if got[i] != expected[i]), None)
    n_match = sum(1 for i in range(min(len(got), n)) if got[i] == expected[i])
    print("GOT_IDS ", got)
    print("EXPECTED", expected)
    print(f"DTYPE {args.dtype}  CUDA_GRAPH {'ON' if args.cuda_graph else 'OFF'}")
    print(f"EXACT_MATCH {exact}  MATCH {n_match}/{n}  FIRST_DIVERGENCE_INDEX {div}")
    engine.shutdown()
    sys.exit(0 if exact else 1)


if __name__ == "__main__":
    main()
