#!/usr/bin/env python3
"""
Task-1a production-correctness gate: dynamic-batching greedy correctness for RWKV-7.

F0008 found that B>=3 *identical-prompt* requests can DIVERGE because RWKV's
per-request recurrent state is NOT prefix-cacheable, yet sglang's token radix
cache shares identical prefixes across requests and (wrongly, for an RNN) lets one
request inherit another's cached prefix without its recurrent state. The fix is
`disable_radix_cache=True`.

This script asserts the fix: with `disable_radix_cache=True` (+ cuda-graph ON, bf16)
every request in a batch produces the SAME greedy output as if it were run alone,
AND identical-prompt copies all match the numpy-oracle fixture (the bit-level
ground truth). Three batches are exercised:

  1. IDENTICAL   : >=4 exact copies of the fixture prompt. Every output must equal
                   the numpy-oracle `greedy_tokens` (this is the exact F0008 repro).
  2. SHARED-PREFIX: prompts sharing a long common prefix but with divergent tails
                   (stresses radix *prefix* matching, not just full-string sharing).
  3. MIXED       : identical copies + shared-prefix + a wholly distinct prompt, all
                   in one batch. Every request must equal its own B=1 reference.

References for non-fixture prompts are computed by running each unique prompt as a
single (B=1) request through the SAME engine — B=1 has no cross-request prefix to
share, so it is the trustworthy per-prompt oracle.

  source ~/rwkv_env.sh && CUDA_VISIBLE_DEVICES=0 ~/envs/rwkv-sgl/bin/python \
      bench/verify_batch.py \
      --model /home/user/rwkv_models/rwkv7-0.1b-fla \
      --fixture bench/fixtures/oracle_rwkv7_01b_eiffel.json \
      --dtype bfloat16 --cuda-graph

Exit 0 iff ALL batches are exact; non-zero otherwise. `--radix-on` flips the guard
off to demonstrate the bug (divergence is intermittent, so a PASS there is not
proof of safety — only the default radix-OFF run is the gate).
"""
import argparse
import json
import sys


def _extract_ids(rec, n):
    got = (
        rec.get("output_ids")
        or rec.get("token_ids")
        or rec.get("output_token_ids")
        or (rec.get("meta_info", {}) or {}).get("output_token_ids")
    )
    if got is None:
        raise RuntimeError(f"could not find output ids; keys={list(rec.keys())}")
    return list(got)[:n]


def _gen(engine, prompts, n):
    out = engine.generate(
        input_ids=prompts,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": n,
            "ignore_eos": True,
        },
    )
    if not isinstance(out, list):
        out = [out]
    return [_extract_ids(r, n) for r in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument("--cuda-graph-max-bs", type=int, default=None)
    ap.add_argument("--identical-bsz", type=int, default=4,
                    help="number of identical-prompt copies (>=4; the F0008 repro)")
    ap.add_argument("--n", type=int, default=None,
                    help="tokens to generate (default: len(fixture.greedy_tokens))")
    ap.add_argument("--radix-on", action="store_true",
                    help="DEMO ONLY: leave radix cache ON to try to reproduce the bug")
    args = ap.parse_args()

    fx = json.load(open(args.fixture))
    prompt = list(fx["prompt_tokens"])
    oracle = list(fx["greedy_tokens"])
    n = args.n or len(oracle)
    oracle = oracle[:n]
    disable_radix = not args.radix_on

    # ---- build prompt variants ------------------------------------------------
    # shared-prefix prompts: same first ~len-2 tokens as the fixture, divergent tails.
    pre = prompt[:-2] if len(prompt) > 4 else prompt
    sp1 = pre + [4706, 4706]            # eiffel prefix + tail A
    sp2 = pre + [22590, 30449]          # eiffel prefix + tail B
    # a wholly distinct prompt (deterministic, avoids token 0)
    distinct = [(i * 37 % 60000) + 1 for i in range(13)]

    import sglang as sgl

    ekw = dict(
        model_path=args.model,
        skip_tokenizer_init=True,
        disable_cuda_graph=not args.cuda_graph,
        disable_piecewise_cuda_graph=True,
        disable_radix_cache=disable_radix,
        dtype=args.dtype,
        tp_size=1,
        mem_fraction_static=args.mem_fraction,
    )
    if args.cuda_graph and args.cuda_graph_max_bs is not None:
        ekw["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    # keep the same invocation across sglang versions (e.g. main dropped
    # disable_piecewise_cuda_graph): only pass kwargs ServerArgs still accepts
    from sglang.srt.server_args import ServerArgs
    ekw = {k: v for k, v in ekw.items() if k in ServerArgs.__dataclass_fields__}
    engine = sgl.Engine(**ekw)

    # ---- per-prompt B=1 references (trustworthy: no cross-request prefix) ------
    ref = {}
    for name, p in [("eiffel", prompt), ("sp1", sp1), ("sp2", sp2), ("distinct", distinct)]:
        ref[name] = _gen(engine, [p], n)[0]

    # cross-check the fixture's own prompt B=1 against the numpy oracle
    eiffel_b1_exact = ref["eiffel"] == oracle

    failures = []

    # ---- BATCH 1: identical copies (the F0008 repro) --------------------------
    b = max(args.identical_bsz, 4)
    ident = _gen(engine, [list(prompt) for _ in range(b)], n)
    ident_vs_oracle = [o == oracle for o in ident]
    ident_pass = all(ident_vs_oracle)
    if not ident_pass:
        failures.append("IDENTICAL")

    # ---- BATCH 2: shared-prefix, divergent tails -----------------------------
    sp_prompts = [list(prompt), list(sp1), list(prompt), list(sp2), list(prompt)]
    sp_names = ["eiffel", "sp1", "eiffel", "sp2", "eiffel"]
    sp_out = _gen(engine, sp_prompts, n)
    sp_vs_ref = [sp_out[i] == ref[sp_names[i]] for i in range(len(sp_out))]
    sp_pass = all(sp_vs_ref)
    if not sp_pass:
        failures.append("SHARED-PREFIX")

    # ---- BATCH 3: mixed identical + shared-prefix + distinct ------------------
    mix_prompts = [list(prompt), list(prompt), list(sp1), list(distinct),
                   list(prompt), list(sp2)]
    mix_names = ["eiffel", "eiffel", "sp1", "distinct", "eiffel", "sp2"]
    mix_out = _gen(engine, mix_prompts, n)
    mix_vs_ref = [mix_out[i] == ref[mix_names[i]] for i in range(len(mix_out))]
    # identical (eiffel) members of the mix must ALSO equal the numpy oracle
    mix_eiffel_vs_oracle = [
        mix_out[i] == oracle for i in range(len(mix_out)) if mix_names[i] == "eiffel"
    ]
    mix_pass = all(mix_vs_ref) and all(mix_eiffel_vs_oracle)
    if not mix_pass:
        failures.append("MIXED")

    engine.shutdown()

    # ---- report --------------------------------------------------------------
    print("=" * 78)
    print(f"VERIFY_BATCH  model={args.model}")
    print(f"  dtype={args.dtype}  cuda_graph={'ON' if args.cuda_graph else 'OFF'}  "
          f"disable_radix_cache={disable_radix}  n={n}")
    print("-" * 78)
    print(f"ORACLE          {oracle}")
    print(f"eiffel B=1 == numpy-oracle : {eiffel_b1_exact}")
    print("-" * 78)
    print(f"[1] IDENTICAL  bsz={b}  every-output==oracle : {ident_pass}  "
          f"({sum(ident_vs_oracle)}/{b})")
    if not ident_pass:
        for i, ok in enumerate(ident_vs_oracle):
            if not ok:
                print(f"      req#{i} DIVERGED: {ident[i]}")
    print(f"[2] SHARED-PREFIX  bsz={len(sp_out)}  every-output==B1-ref : {sp_pass}  "
          f"({sum(sp_vs_ref)}/{len(sp_out)})  tags={sp_names}")
    if not sp_pass:
        for i, ok in enumerate(sp_vs_ref):
            if not ok:
                print(f"      req#{i}({sp_names[i]}) DIVERGED: {sp_out[i]} != {ref[sp_names[i]]}")
    print(f"[3] MIXED  bsz={len(mix_out)}  every-output==B1-ref : {all(mix_vs_ref)}  "
          f"({sum(mix_vs_ref)}/{len(mix_out)})  eiffel==oracle : {all(mix_eiffel_vs_oracle)}  "
          f"tags={mix_names}")
    if not mix_pass:
        for i, ok in enumerate(mix_vs_ref):
            if not ok:
                print(f"      req#{i}({mix_names[i]}) DIVERGED: {mix_out[i]} != {ref[mix_names[i]]}")
    print("-" * 78)
    overall = (not failures) and eiffel_b1_exact
    print(f"OVERALL: {'PASS (all batches exact)' if overall else 'FAIL: ' + ','.join(failures or ['B1!=oracle'])}")
    print("=" * 78)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
