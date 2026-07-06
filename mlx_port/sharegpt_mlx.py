#!/usr/bin/env python3
"""
Real-workload single-stream (bsz1) benchmark for the MLX RWKV-7 port: feed real
ShareGPT prompts through the model and report the realistic prefill + decode
throughput and the latency distribution (TTFT + per-token ITL) over the actual
prompt-length mix — the same workload family as `docs/BENCHMARKS.md §7c`, but
single-stream on Apple Silicon (MLX has no server / continuous batching).

Per sampled conversation: the first human turn is the prompt.
  * TTFT = prefill(prompt) wall time + first decoded token (what a streaming
    client waits for), reported against the real prompt length;
  * decode: up to --max-new greedy tokens, timed PER TOKEN synchronously (the
    streaming path — each token materialized as delivered), giving the inter-token
    latency (ITL) distribution a user actually sees; decode tok/s is derived from
    that (a touch below the async-pipelined ceiling in bench_mlx.py, by design —
    this measures latency, not peak throughput).

Aggregate prefill throughput = total prompt tokens / total prefill time; aggregate
decode throughput = total decoded tokens / total decode time. Distributions are
reported as p50/p90/p99.

  python mlx_port/sharegpt_mlx.py --model /tmp/mlx_models/rwkv7-1.5b-fla \
      --data /tmp/sharegpt.json --n 150 --max-new 128 [--quant w8] [--out ...]
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_mlx import load_model
from gate_oracle import WorldTokenizer


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
    return xs[k]


def first_human_prompts(convs, n, tok, min_tok, max_tok, seed=0):
    """Extract up to n first-human-turn prompts, filtered to a sane token range
    so the length mix is realistic but a single doc can't dominate wall time."""
    random.seed(seed)
    idx = list(range(len(convs)))
    random.shuffle(idx)
    out = []
    for i in idx:
        turns = convs[i].get("conversations", [])
        if not turns or turns[0].get("from") != "human":
            continue
        text = turns[0].get("value", "").strip()
        if not text:
            continue
        ids = tok.encode(text)
        if min_tok <= len(ids) <= max_tok:
            out.append(ids)
        if len(out) >= n:
            break
    return out


def run_one(model, prompt_ids, max_new):
    """One request. Returns (prompt_len, prefill_s, ttft_s, [itl_s...], out_len).
    prefill_s = pure prefill; ttft_s = prefill + first token (client wait)."""
    state = model.new_state()
    t0 = time.perf_counter()
    logits, state = model.prefill(prompt_ids, state)  # prefill() eval's logits+state
    prefill_s = time.perf_counter() - t0
    tok = mx.argmax(logits).reshape(1).astype(mx.int32)
    mx.eval(tok)
    ttft = time.perf_counter() - t0
    itls = []
    out_len = 1
    for _ in range(max_new - 1):
        t1 = time.perf_counter()
        logits, state = model.step(int(tok[0]), state)
        tok = mx.argmax(logits).reshape(1).astype(mx.int32)
        mx.eval(tok)
        itls.append(time.perf_counter() - t1)
        out_len += 1
    return len(prompt_ids), prefill_s, ttft, itls, out_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/tmp/sharegpt.json")
    ap.add_argument("--quant", default=None, help="None (fp16) | w8 | w4")
    ap.add_argument("--wkv", default="metal")
    ap.add_argument("--n", type=int, default=150, help="conversations to run")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--min-tok", type=int, default=8)
    ap.add_argument("--max-tok", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tok = WorldTokenizer(os.path.join(args.model, "rwkv_vocab_v20230424.txt"))
    convs = json.load(open(args.data))
    prompts = first_human_prompts(convs, args.n + args.warmup, tok,
                                  args.min_tok, args.max_tok)
    assert len(prompts) > args.warmup, "not enough usable conversations"
    model = load_model(args.model, wkv=args.wkv, quant=args.quant)
    print(f"model={args.model} quant={args.quant} wkv={args.wkv} "
          f"convs={len(prompts)-args.warmup} max_new={args.max_new}", flush=True)

    for p in prompts[:args.warmup]:  # warm compile/JIT across a couple shapes
        run_one(model, p, 8)

    plens, ttfts, all_itls, dec_tok, dec_time, pre_tok, pre_time = \
        [], [], [], 0, 0.0, 0, 0.0
    t_all = time.perf_counter()
    for k, p in enumerate(prompts[args.warmup:]):
        plen, prefill_s, ttft, itls, olen = run_one(model, p, args.max_new)
        plens.append(plen)
        ttfts.append(ttft * 1000)  # ms
        all_itls.extend(x * 1000 for x in itls)  # ms
        pre_tok += plen
        pre_time += prefill_s  # pure prefill time (TTFT tracked separately)
        dec_tok += len(itls)
        dec_time += sum(itls)
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(prompts)-args.warmup} done "
                  f"({time.perf_counter()-t_all:.0f}s)", flush=True)

    itl_med = statistics.median(all_itls)
    res = {
        "n": len(plens),
        "prompt_len": {"min": min(plens), "p50": pct(plens, 50),
                       "p90": pct(plens, 90), "max": max(plens),
                       "mean": statistics.mean(plens)},
        "ttft_ms": {"p50": pct(ttfts, 50), "p90": pct(ttfts, 90),
                    "p99": pct(ttfts, 99), "max": max(ttfts)},
        "itl_ms": {"p50": pct(all_itls, 50), "p90": pct(all_itls, 90),
                   "p99": pct(all_itls, 99)},
        "decode_tok_s_stream": dec_tok / dec_time if dec_time else float("nan"),
        "prefill_tok_s_agg": pre_tok / pre_time if pre_time else float("nan"),
        "total_prompt_tokens": pre_tok, "total_decoded_tokens": dec_tok,
        "wall_s": time.perf_counter() - t_all,
    }
    print("\n===== ShareGPT single-stream (bsz1) — real prompt-length mix =====")
    pl = res["prompt_len"]
    print(f"prompts: n={res['n']}  len min/p50/mean/p90/max = "
          f"{pl['min']}/{pl['p50']}/{pl['mean']:.0f}/{pl['p90']}/{pl['max']} tok")
    t = res["ttft_ms"]
    print(f"TTFT ms:  p50={t['p50']:.1f}  p90={t['p90']:.1f}  p99={t['p99']:.1f}  max={t['max']:.1f}")
    it = res["itl_ms"]
    print(f"ITL  ms:  p50={it['p50']:.2f}  p90={it['p90']:.2f}  p99={it['p99']:.2f}  "
          f"(streaming per-token latency)")
    print(f"decode: {res['decode_tok_s_stream']:.1f} tok/s (streaming)  "
          f"prefill(agg): {res['prefill_tok_s_agg']:.1f} tok/s")
    print(f"totals: {res['total_prompt_tokens']} prompt tok, "
          f"{res['total_decoded_tokens']} decoded tok, {res['wall_s']:.0f}s")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
