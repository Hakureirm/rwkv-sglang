#!/usr/bin/env python3
"""
Uncheatable-Eval compression rate for the MLX RWKV-7 port — DIRECT-CALL (no HTTP
server; MLX has none). Faithful to this repo's `bench/uncheatable_eval.py`
methodology (itself a port of Jellyfish042/uncheatable_eval), so the numbers are
directly comparable to the CUDA/sglang column in `docs/BENCHMARKS.md §2`:

  * tokenize each raw doc with the RWKV World tokenizer (no BOS);
  * split into `ctx_len`-token chunks; for each chunk feed input = [0] + chunk
    (token 0 is the RWKV EOD/BOS; state resets per chunk);
  * per-token NLL = -log P(token_i | 0, tokens_<i) in NATS, fp32 log-softmax over
    the exact recurrence (model.score_tokens); every real chunk token is scored;
  * neg_log_prob_sum = mean over docs of total doc NLL (nats)  [REF L456];
  * bpb = neg_log_prob_sum / avg_bytes / ln2;
    compression_rate = bpb * 0.125 * 100  (% of utf-8 size, lower is better).

Also emits the position curve (mean -log2 p bucketed by a token's index within its
document) — the "does the recurrent state keep absorbing context" plot Bo asks for.

Works for fp16 AND quant (--quant w8|w4): the SAME ruler for every precision, so
w8/w4 accuracy is reported on the metric that decides quant quality here (greedy
token match is only a sanity check; this is the real number).

  python mlx_port/compression_mlx.py --model /tmp/mlx_models/rwkv7-1.5b-fla \
      --data '/tmp/uncheatable/*.json' --max-docs 50 --ctx-len 4000 \
      [--quant w8] [--out mlx_port/results/compression_1.5b_w8.json]
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_mlx import load_model
from gate_oracle import WorldTokenizer

INV_LN2 = 1.0 / math.log(2.0)
POS_BUCKETS = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1024),
               (1024, float("inf"))]


def bucket_label(lo, hi):
    return f"[{lo}-{int(hi)})" if hi != float("inf") else f"[{lo}+)"


def load_documents(path):
    """.json -> JSON list of strings; .jsonl -> one {'content':...} per line
    (REF load_data_smart)."""
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l)["content"] for l in f if l.strip()]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"{path}: expected a JSON list of strings"
    return [d if isinstance(d, str) else d["content"] for d in data]


def eval_dataset(name, texts, tok, model, ctx_len, block, pos_sum, pos_cnt):
    doc_tokens = [tok.encode(t) for t in texts]  # World tokenizer, no BOS
    doc_nll = [0.0] * len(texts)
    t0 = time.time()
    n_chunks = done = 0
    for di, toks in enumerate(doc_tokens):
        assert len(toks) > 0, f"{name}: empty doc after tokenization"
        for begin in range(0, len(toks), ctx_len):
            chunk = toks[begin:begin + ctx_len]
            nlls = model.score_tokens([0] + chunk, block=block)  # [len(chunk)]
            nlls = nlls.tolist()
            doc_nll[di] += sum(nlls)
            for j, nll in enumerate(nlls):
                pos = begin + j
                for bi, (lo, hi) in enumerate(POS_BUCKETS):
                    if lo <= pos < hi:
                        pos_sum[bi] += nll * INV_LN2
                        pos_cnt[bi] += 1
                        break
            n_chunks += 1
        done += 1
        if done % 25 == 0:
            print(f"  {name}: {done}/{len(texts)} docs, {n_chunks} chunks "
                  f"({time.time()-t0:.0f}s)", flush=True)
    n = len(texts)
    d = {
        "neg_log_prob_sum": sum(doc_nll) / n,
        "avg tokens": sum(len(t) for t in doc_tokens) / n,
        "avg bytes": sum(len(t.encode("utf-8")) for t in texts) / n,
        "sample_count": n,
        "total_nll_nats": sum(doc_nll),
        "total_bytes": sum(len(t.encode("utf-8")) for t in texts),
    }
    d["bpb"] = d["neg_log_prob_sum"] / d["avg bytes"] * INV_LN2
    d["compression_rate"] = d["bpb"] * 0.125 * 100
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True,
                    help=".json (list of str) / .jsonl ({'content':...}); globs OK")
    ap.add_argument("--quant", default=None, help="None (fp16) | w8 | w4")
    ap.add_argument("--wkv", default="metal")
    ap.add_argument("--max-docs", type=int, default=0, help="first N docs/dataset (0=all)")
    ap.add_argument("--ctx-len", type=int, default=4000, help="chunk size in tokens (REF 4000)")
    ap.add_argument("--block", type=int, default=512, help="scoring block (memory bound)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = []
    for pat in args.data:
        files.extend(sorted(glob.glob(pat)) or [pat])
    for f in files:
        assert os.path.exists(f), f"dataset not found: {f}"

    vocab = os.path.join(args.model, "rwkv_vocab_v20230424.txt")
    tok = WorldTokenizer(vocab)
    model = load_model(args.model, wkv=args.wkv, quant=args.quant)
    print(f"model={args.model} quant={args.quant} wkv={args.wkv} "
          f"ctx_len={args.ctx_len} datasets={len(files)}", flush=True)

    pos_sum = [0.0] * len(POS_BUCKETS)
    pos_cnt = [0] * len(POS_BUCKETS)
    per_dataset = {}
    t_all = time.time()
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        texts = load_documents(f)
        if args.max_docs > 0:
            texts = texts[:args.max_docs]
        print(f"== {name}: {len(texts)} docs", flush=True)
        per_dataset[name] = eval_dataset(name, texts, tok, model, args.ctx_len,
                                         args.block, pos_sum, pos_cnt)
        print(f"   bpb={per_dataset[name]['bpb']:.4f} "
              f"compression={per_dataset[name]['compression_rate']:.3f}%", flush=True)

    tot_nll = sum(d["total_nll_nats"] for d in per_dataset.values())
    tot_bytes = sum(d["total_bytes"] for d in per_dataset.values())
    overall = {
        "mean_compression_rate": sum(d["compression_rate"] for d in per_dataset.values()) / len(per_dataset),
        "pooled_compression_rate": tot_nll / tot_bytes * INV_LN2 * 0.125 * 100,
        "pooled_bpb": tot_nll / tot_bytes * INV_LN2,
        "total_docs": sum(d["sample_count"] for d in per_dataset.values()),
        "wall_time_s": time.time() - t_all,
    }

    print("\n===== compression rate (% of utf-8 size; lower is better) =====")
    print(f"{'dataset':<28} {'docs':>5} {'bpb':>8} {'compression%':>13}")
    for name, d in per_dataset.items():
        print(f"{name:<28} {d['sample_count']:>5} {d['bpb']:>8.4f} {d['compression_rate']:>13.3f}")
    print(f"{'MEAN (unweighted)':<28} {'':>5} {'':>8} {overall['mean_compression_rate']:>13.3f}")
    print(f"POOLED bpb={overall['pooled_bpb']:.4f}  "
          f"compression={overall['pooled_compression_rate']:.3f}%  "
          f"(docs={overall['total_docs']}, {overall['wall_time_s']:.0f}s)")

    print("\n===== compression vs token position (mean -log2 p) =====")
    curve = []
    for (lo, hi), s, c in zip(POS_BUCKETS, pos_sum, pos_cnt):
        mb = s / c if c else float("nan")
        print(f"{bucket_label(lo, hi):<12} {c:>10} tok  {mb:>8.4f} bits")
        curve.append({"bucket": bucket_label(lo, hi), "tokens": c, "mean_neg_log2_p": mb})

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"model": args.model, "quant": args.quant,
                       "ctx_len": args.ctx_len, "max_docs": args.max_docs,
                       "datasets": per_dataset, "overall": overall,
                       "position_curve": curve,
                       "methodology": "uncheatable_eval (Jellyfish042) via MLX "
                                      "score_tokens, fp32 log-softmax"}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
