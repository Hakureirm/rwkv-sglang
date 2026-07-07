#!/usr/bin/env python3
"""
Uncheatable-Eval compression rate for Qwen3.5-2B via mlx-lm -- DIRECT-CALL,
same methodology/formulas as this dir's `compression_mlx.py` (the RWKV-7
ruler), so the numbers are directly comparable on this machine. This is the
accuracy analog of `bench_mlx_qwen35.py` (which matches the SPEED protocol):
this project's "zero fla/torch/transformers" policy governs what it ships as
its OWN RWKV-7 implementation, not the yardstick used on a competitor's own
native implementation (see mlx_port/README.md's note on this file's sibling).

Same formulas as `bench/uncheatable_eval.py` / `compression_mlx.py`:
  * tokenize each raw doc with Qwen3.5's own tokenizer (HF AutoTokenizer,
    add_special_tokens=False -- raw text only, matching the RWKV side's
    "no BOS" policy so neither model gets a tokenization freebie/penalty);
  * split into `ctx_len`-token chunks; for each chunk feed
    input = [EOD_ID] + chunk -- see "Reset token" below for why this is the
    correct analog of RWKV's "prepend token 0", not just a naming coincidence;
  * per-token NLL = -log P(token_i | EOD_ID, tokens_<i) in NATS, fp32
    log-softmax over a single fresh (cache=None) forward pass; every real
    chunk token is scored;
  * neg_log_prob_sum = mean over docs of total doc NLL (nats);
  * bpb = neg_log_prob_sum / avg_bytes / ln2;
    compression_rate = bpb * 0.125 * 100  (% of utf-8 size, lower is better).

Reset token -- verified, not assumed: RWKV's WKV recurrence has a persistent
state that must be explicitly zeroed at a chunk boundary (hence "prepend
token 0"). Qwen3.5 has no such state: each `model(ids[None], cache=None)`
call already starts from nothing (no KV-cache, no mamba-cache carried in) --
prepending a boundary token doesn't create the reset, the fresh call does;
prepending `<|endoftext|>` additionally tells the model'S OWN semantics "a
new document starts here", which is the closest available analog and is
this checkpoint's actual EOD token (checked directly against ITS OWN
tokenizer below, not copied from another tokenizer version -- ids can differ
across retrains/retokenizations, so this project's rule is "verify per
checkpoint").

Scoring is a single whole-chunk forward pass (no block-splitting, unlike
RWKV's O(1)-state chunked scan): a 4000-token forward comfortably fits M5
unified memory (already validated at 1024 tokens' worth of prefill in
bench_mlx_qwen35.py; this is the same code path, just a longer sequence).

  python mlx_port/compression_mlx_qwen35.py \
      --model /private/tmp/qwen35_mlx_test/Qwen3.5-2B \
      --data '/private/tmp/uncheatable_full/*.json' --max-docs 500 \
      --out mlx_port/results/compression_qwen35_2b_bf16.json
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import mlx.core as mx
from mlx_lm.utils import load
from transformers import AutoTokenizer

INV_LN2 = 1.0 / math.log(2.0)
POS_BUCKETS = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1024),
               (1024, float("inf"))]


def bucket_label(lo, hi):
    return f"[{lo}-{int(hi)})" if hi != float("inf") else f"[{lo}+)"


def load_documents(path):
    """.json -> JSON list of strings; .jsonl -> one {'content':...} per line.
    Identical to compression_mlx.py's loader (dataset format is model-agnostic)."""
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l)["content"] for l in f if l.strip()]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"{path}: expected a JSON list of strings"
    return [d if isinstance(d, str) else d["content"] for d in data]


def score_chunk_nlls(model, eod_id, chunk_ids):
    """[EOD_ID] + chunk -> one fresh forward pass -> NLL (nats) for every
    real chunk token. Mirrors rwkv7_mlx.py's score_tokens semantics exactly
    (same input construction, same fp32 log-softmax, same shift-by-one), the
    ONLY difference being no persistent recurrent state to carry across
    `block`-sized sub-pieces -- a hybrid-attention/GDN transformer with
    cache=None IS the reset, so the whole chunk scores in one call."""
    ids = mx.array([eod_id] + chunk_ids, dtype=mx.int32)[None]  # (1, 1+L)
    logits = model(ids, cache=None)  # (1, 1+L, vocab)
    logits = logits[0].astype(mx.float32)  # (1+L, vocab)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    pred_logp = logp[:-1]  # rows 0..L-1: row i predicts ids[i+1] = chunk[i]
    targets = mx.array(chunk_ids, dtype=mx.int32)[:, None]
    nll = -mx.take_along_axis(pred_logp, targets, axis=-1)[:, 0]
    mx.eval(nll)
    return nll.tolist()


def eval_dataset(name, texts, tok, model, eod_id, ctx_len, pos_sum, pos_cnt):
    doc_tokens = [tok.encode(t, add_special_tokens=False) for t in texts]
    doc_nll = [0.0] * len(texts)
    t0 = time.time()
    n_chunks = done = 0
    for di, toks in enumerate(doc_tokens):
        assert len(toks) > 0, f"{name}: empty doc after tokenization"
        for begin in range(0, len(toks), ctx_len):
            chunk = toks[begin:begin + ctx_len]
            nlls = score_chunk_nlls(model, eod_id, chunk)
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
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--max-docs", type=int, default=0, help="first N docs/dataset (0=all)")
    ap.add_argument("--ctx-len", type=int, default=4000, help="chunk size in tokens (REF 4000)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = []
    for pat in args.data:
        files.extend(sorted(glob.glob(pat)) or [pat])
    for f in files:
        assert os.path.exists(f), f"dataset not found: {f}"

    print(f"loading mlx-lm model: {args.model}", flush=True)
    model, _mlx_tok = load(args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Verify, don't assume -- this checkpoint's own EOD token id.
    eod_ids = tok.encode("<|endoftext|>", add_special_tokens=False)
    assert len(eod_ids) == 1, f"<|endoftext|> did not encode to 1 token: {eod_ids}"
    eod_id = eod_ids[0]
    assert eod_id == tok.pad_token_id, (
        f"<|endoftext|> id {eod_id} != tokenizer.pad_token_id {tok.pad_token_id} "
        "-- this checkpoint's convention differs from what was assumed; check "
        "tokenizer_config.json by hand before trusting this run.")
    print(f"model={args.model} eod_id={eod_id} ({tok.decode([eod_id])!r}) "
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
        per_dataset[name] = eval_dataset(name, texts, tok, model, eod_id,
                                         args.ctx_len, pos_sum, pos_cnt)
        print(f"   bpb={per_dataset[name]['bpb']:.4f} "
              f"compression={per_dataset[name]['compression_rate']:.3f}%", flush=True)
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

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
            json.dump({"model": args.model, "eod_id": eod_id,
                       "ctx_len": args.ctx_len, "max_docs": args.max_docs,
                       "datasets": per_dataset, "overall": overall,
                       "position_curve": curve,
                       "methodology": "uncheatable_eval (Jellyfish042) via mlx-lm "
                                      "direct forward, fp32 log-softmax, "
                                      "cache=None per chunk = the reset"}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
