#!/usr/bin/env python3
"""Chunked-prefill correctness gate: a single long sequence prefilled in SMALL chunks
must produce identical greedy output to the same sequence prefilled in ONE shot.

RWKV-7 carries its recurrent state across prefill chunks; if the chunk-boundary state
carry-in is wrong, long-prompt outputs silently corrupt. sglang chunks prefill at
--chunked-prefill-size, so we compare a small-chunk engine vs a single-shot engine on a
prompt longer than the chunk size. Closes the "chunked-prefill correctness is inferred,
not tested" gap.

  source ~/rwkv_env.sh && CUDA_VISIBLE_DEVICES=0 python bench/verify_chunked_prefill.py \
      --model <fla_dir> --prompt-len 2048 --chunk 256 --gen 48
"""
import argparse, sys


def make_prompt(n):
    return [(i % 60000) + 1 for i in range(n)]


def greedy(model, chunk_size, prompt, gen, dtype, mem):
    import sglang as sgl
    eng = sgl.Engine(
        model_path=model, skip_tokenizer_init=True,
        disable_cuda_graph=True, disable_piecewise_cuda_graph=True,
        disable_radix_cache=True, chunked_prefill_size=chunk_size,
        dtype=dtype, tp_size=1, mem_fraction_static=mem,
    )
    out = eng.generate(input_ids=[prompt],
                       sampling_params={"temperature": 0.0, "max_new_tokens": gen})
    rec = out[0] if isinstance(out, list) else out
    toks = (rec.get("output_ids") or rec.get("token_ids") or rec.get("output_token_ids")
            or (rec.get("meta_info", {}) or {}).get("output_token_ids"))
    eng.shutdown()
    return list(toks)[:gen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=256, help="small chunked-prefill-size (forces multi-chunk)")
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    args = ap.parse_args()

    prompt = make_prompt(args.prompt_len)
    n_chunks = -(-args.prompt_len // args.chunk)
    print(f"prompt={args.prompt_len} tok, chunk={args.chunk} -> {n_chunks} chunks; gen={args.gen}")

    single = greedy(args.model, args.prompt_len * 2, prompt, args.gen, args.dtype, args.mem_fraction)
    chunked = greedy(args.model, args.chunk, prompt, args.gen, args.dtype, args.mem_fraction)

    match = sum(1 for a, b in zip(single, chunked) if a == b)
    exact = single == chunked
    div = next((i for i in range(min(len(single), len(chunked))) if single[i] != chunked[i]), None)
    print(f"single-shot : {single}")
    print(f"chunked({args.chunk}): {chunked}")
    print(f"CHUNKED-PREFILL MATCH {match}/{args.gen}  first_div={div}  exact={exact}")
    sys.exit(0 if exact else 1)


if __name__ == "__main__":
    main()
