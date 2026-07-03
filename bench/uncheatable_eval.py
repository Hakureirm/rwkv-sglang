#!/usr/bin/env python3
"""
Uncheatable Eval (compression rate) for OUR sglang RWKV-7 server.

Faithful port of Jellyfish042/uncheatable_eval (BlinkDL's decreed accuracy metric):
fresh-corpus documents -> model cross-entropy -> compression rate measured against
BYTES of utf-8 text (tokenizer-independent). Reference copy of the official code is
in scratchpad/official_evals/uncheatable_evaluator.py ("REF" below).

Replicated methodology (line numbers refer to REF):
  * RWKV tokenization + chunking (REF eval_rwkv, L375-L419):
      - tokenize the raw document with the RWKV world tokenizer (no BOS token exists;
        REF load_rwkv7 L210-L227 uses the rwkv pip world tokenizer, no special tokens);
      - split into chunks of `chunk_size` tokens (REF default chunk_size=4000, L38);
      - for each chunk, prepend token 0:  input_chunk = [0] + input_seq[b:b+cs]
        (REF L397 - token 0 is the RWKV "BOS"/EOD; state resets between chunks);
      - loss = cross_entropy(logits[:-1], input_chunk[1:])  (REF L301-L303), i.e. every
        real token of the chunk is scored, conditioned on [0] + preceding chunk tokens.
        (REF casts logits to bf16 before the CE; on our side the server computes the
        logprob from its own logits - a float32 log_softmax - so tiny numerical diffs
        vs REF are expected and documented, the formula is identical.)
  * neg_log_prob_sum = mean over documents of the total NLL in NATS (REF L456).
  * avg bytes = mean utf-8 byte length of the documents (REF L460, L352-L357).
  * Final metrics (REF L761-L763):
      bpc              = (neg_log_prob_sum / avg character count) * (1/ln 2)
      bpb              = (neg_log_prob_sum / avg bytes)           * (1/ln 2)
      compression_rate = (neg_log_prob_sum / avg bytes) * (1/ln 2) * 0.125 * 100
    i.e. compression_rate is "compressed size as % of original utf-8 size"
    (bits-per-byte / 8 * 100).
  * Dataset format (REF load_data_smart, L643-L675): a .json file is a JSON list of
    strings; a .jsonl file has one {"content": ...} object per line. The official
    datasets are HF datasets (e.g. Jellyfish042/UncheatableEval-2026-04, split "test",
    columns category/content) - see bench/data/README.md for how to fetch them on a
    machine with network and convert to local .json.

OUR scoring path: POST {host}/generate with input_ids and
  return_logprob=true, logprob_start_len=0, max_new_tokens=1 (temperature 0).
sglang returns meta_info["input_token_logprobs"] = [[logprob, token_id, text?], ...]
aligned with input_ids, where entry 0 has logprob None (regular requests prepend None
and drop the sampled position - sglang scheduler_output_processor_mixin.py
_process_input_token_logprobs: `[None] + input_token_logprobs[:-1]`). With our
prepended token 0 this yields exactly the REF loss terms: entries [1:] are
log P(token_i | 0, tokens_<i) for every real chunk token.

Extra output (ours, not in REF): a compression-vs-token-position curve - the mean
-log2 p(token) bucketed by the token's index within its document
(buckets [0,64) [64,128) [128,256) [256,512) [512,1024) [1024,+inf)), printed and
written to CSV. NB positions are document-global; with chunking the model context
resets every --ctx-len tokens (like REF).

Usage:
  python bench/uncheatable_eval.py --model <model_dir> --host 127.0.0.1 --port 30000 \
      --data bench/data/uncheatable/*.json --max-docs 0 --ctx-len 4000 \
      --out bench/results/uncheatable_1.5b.json
  # or let it spawn its own server (releases it on exit):
  python bench/uncheatable_eval.py --model <model_dir> --launch --data ...
"""

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# Position buckets for the compression-vs-position curve: [lo, hi)
POS_BUCKETS = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, float("inf"))]


def bucket_label(lo, hi):
    return f"[{lo}-{int(hi)})" if hi != float("inf") else f"[{lo}+)"


def load_documents(path):
    """REF load_data_smart (uncheatable_evaluator.py L643-L652):
    .json -> JSON list of strings; .jsonl -> one {"content": ...} per line."""
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line)["content"] for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"{path}: expected a JSON list of strings"
    return [d if isinstance(d, str) else d["content"] for d in data]


def score_chunk(sess, gen_url, chunk_ids, timeout):
    """One forward over [0]+chunk via /generate; returns per-token NLL (nats) list
    for the real chunk tokens (all entries after the leading token 0)."""
    r = sess.post(
        gen_url,
        json={
            "input_ids": chunk_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    item = r.json()
    if isinstance(item, list):
        item = item[0]
    itl = item["meta_info"]["input_token_logprobs"]  # [[lp, tok_id, (text)], ...]
    assert len(itl) == len(chunk_ids), f"got {len(itl)} logprobs for {len(chunk_ids)} input tokens"
    assert itl[0][0] is None, "first input token should have logprob None"
    for ent, tok in zip(itl, chunk_ids):
        assert ent[1] == tok, "input_token_logprobs token ids do not match request input_ids"
    return [-ent[0] for ent in itl[1:]]  # NLL in nats, one per real chunk token


def eval_dataset(name, texts, tokenizer, sess, gen_url, chunk_size, concurrency, timeout, pos_sum, pos_cnt):
    """Returns the REF-style data_dict for one dataset; accumulates the global
    position-curve sums (pos_sum/pos_cnt, in -log2 p units) as a side effect."""
    # tokenize all docs client-side (REF eval_rwkv L379-L385; world tokenizer, no BOS)
    doc_tokens = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

    # build per-chunk work items: (doc_idx, doc_pos_offset, [0]+chunk)  (REF L395-L397)
    work = []
    for di, toks in enumerate(doc_tokens):
        assert len(toks) > 0, f"{name}: empty document after tokenization"
        for begin in range(0, len(toks), chunk_size):
            work.append((di, begin, [0] + toks[begin : begin + chunk_size]))

    doc_nll = [0.0] * len(texts)  # total NLL (nats) per document
    t0 = time.time()
    done = 0

    def run(item):
        di, begin, chunk_ids = item
        return di, begin, score_chunk(sess, gen_url, chunk_ids, timeout)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for di, begin, nlls in pool.map(run, work):
            doc_nll[di] += sum(nlls)
            inv_ln2 = 1.0 / math.log(2.0)
            for j, nll in enumerate(nlls):
                pos = begin + j  # document-global token index
                for bi, (lo, hi) in enumerate(POS_BUCKETS):
                    if lo <= pos < hi:
                        pos_sum[bi] += nll * inv_ln2  # -log2 p
                        pos_cnt[bi] += 1
                        break
            done += 1
            if done % 50 == 0:
                print(f"  {name}: chunk {done}/{len(work)} ({time.time()-t0:.0f}s)", flush=True)

    n = len(texts)
    # REF data_dict (L455-L462) + final metrics (L761-L763)
    d = {
        "neg_log_prob_sum": sum(doc_nll) / n,
        "avg tokens": sum(len(t) for t in doc_tokens) / n,
        "avg character count": sum(len(t) for t in texts) / n,
        "avg bytes": sum(len(t.encode("utf-8")) for t in texts) / n,
        "sample_count": n,
        "total_nll_nats": sum(doc_nll),
        "total_bytes": sum(len(t.encode("utf-8")) for t in texts),
    }
    d["bpc"] = (d["neg_log_prob_sum"] / d["avg character count"]) * (1 / math.log(2))  # REF L761
    d["bpb"] = (d["neg_log_prob_sum"] / d["avg bytes"]) * (1 / math.log(2))  # REF L762
    d["compression_rate"] = d["neg_log_prob_sum"] / d["avg bytes"] * (1 / math.log(2)) * 0.125 * 100  # REF L763
    return d


def launch_server(args):
    """Spawn a sglang server as a child process and wait for /health."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--host", args.host, "--port", str(args.port),
        "--dtype", args.dtype, "--trust-remote-code",
        "--disable-radix-cache",
        "--mem-fraction-static", str(args.mem_fraction_static),
    ]
    print("launching:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    base = f"http://{args.host}:{args.port}"
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            if requests.get(base + "/health", timeout=2).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(2)
    proc.terminate()
    raise RuntimeError("server did not become healthy within 360s")


def main():
    ap = argparse.ArgumentParser(description="Uncheatable Eval (compression rate) vs sglang server")
    ap.add_argument("--model", required=True, help="model dir (for the tokenizer; and --launch)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--launch", action="store_true", help="spawn our own sglang server on --host/--port")
    ap.add_argument("--dtype", default="bfloat16", help="--launch only")
    ap.add_argument("--mem-fraction-static", type=float, default=0.5, help="--launch only")
    ap.add_argument("--data", nargs="+", required=True, help=".json (list of strings) / .jsonl ({'content':...}) files; globs OK")
    ap.add_argument("--max-docs", type=int, default=0, help="first N docs per dataset (0=all)")
    ap.add_argument("--ctx-len", type=int, default=4000, help="chunk size in tokens (REF chunk_size default 4000)")
    ap.add_argument("--concurrency", type=int, default=8, help="chunks in flight (server batches dynamically)")
    ap.add_argument("--timeout", type=float, default=1200.0, help="per-request timeout (s)")
    ap.add_argument("--out", default="", help="write full JSON results here (CSV curve alongside)")
    args = ap.parse_args()

    files = []
    for pat in args.data:
        hits = sorted(glob.glob(pat)) or [pat]
        files.extend(hits)
    for f in files:
        assert os.path.exists(f), f"dataset not found: {f}"

    from transformers import AutoTokenizer  # same convention as bench/accuracy_eval.py L172
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    proc = launch_server(args) if args.launch else None
    try:
        gen_url = f"http://{args.host}:{args.port}/generate"
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=args.concurrency, pool_maxsize=args.concurrency)
        sess.mount("http://", adapter)

        pos_sum = [0.0] * len(POS_BUCKETS)  # sum of -log2 p per bucket (all datasets)
        pos_cnt = [0] * len(POS_BUCKETS)
        per_dataset = {}
        t_all = time.time()
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            texts = load_documents(f)
            if args.max_docs > 0:
                texts = texts[: args.max_docs]
            print(f"== {name}: {len(texts)} docs", flush=True)
            per_dataset[name] = eval_dataset(
                name, texts, tokenizer, sess, gen_url, args.ctx_len,
                args.concurrency, args.timeout, pos_sum, pos_cnt,
            )
            print(json.dumps({k: v for k, v in per_dataset[name].items()}, indent=2), flush=True)

        # ---- summary ----
        tot_nll = sum(d["total_nll_nats"] for d in per_dataset.values())
        tot_bytes = sum(d["total_bytes"] for d in per_dataset.values())
        overall = {
            # official leaderboard style: unweighted mean of per-dataset compression rates
            "mean_compression_rate": sum(d["compression_rate"] for d in per_dataset.values()) / len(per_dataset),
            # byte-weighted pooled rate over all docs (same formula as REF L763, pooled)
            "pooled_compression_rate": tot_nll / tot_bytes * (1 / math.log(2)) * 0.125 * 100,
            "pooled_bpb": tot_nll / tot_bytes * (1 / math.log(2)),
            "total_docs": sum(d["sample_count"] for d in per_dataset.values()),
            "wall_time_s": time.time() - t_all,
        }

        print("\n===== compression rate (% of utf-8 size; lower is better) =====")
        print(f"{'dataset':<40} {'docs':>5} {'bpb':>8} {'compression%':>13}")
        for name, d in per_dataset.items():
            print(f"{name:<40} {d['sample_count']:>5} {d['bpb']:>8.4f} {d['compression_rate']:>13.3f}")
        print(f"{'MEAN (unweighted over datasets)':<40} {'':>5} {'':>8} {overall['mean_compression_rate']:>13.3f}")
        print(f"{'POOLED (byte-weighted)':<40} {overall['total_docs']:>5} {overall['pooled_bpb']:>8.4f} {overall['pooled_compression_rate']:>13.3f}")

        print("\n===== compression vs token position (mean -log2 p; lower is better) =====")
        curve_rows = []
        print(f"{'token position':<16} {'tokens':>10} {'mean -log2 p':>13}")
        for (lo, hi), s, c in zip(POS_BUCKETS, pos_sum, pos_cnt):
            lbl = bucket_label(lo, hi)
            mean_bits = s / c if c else float("nan")
            # bits-per-TOKEN here (position curve); the headline metric above is per-BYTE
            print(f"{lbl:<16} {c:>10} {mean_bits:>13.4f}")
            curve_rows.append({"bucket": lbl, "tokens": c, "mean_neg_log2_p": mean_bits})

        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            payload = {
                "model": args.model, "ctx_len": args.ctx_len, "max_docs": args.max_docs,
                "datasets": per_dataset, "overall": overall, "position_curve": curve_rows,
                "methodology": "uncheatable_eval (Jellyfish042) via sglang /generate input_token_logprobs",
            }
            with open(args.out, "w") as fh:
                json.dump(payload, fh, indent=2)
            csv_path = os.path.splitext(args.out)[0] + "_curve.csv"
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["bucket", "tokens", "mean_neg_log2_p"])
                w.writeheader()
                w.writerows(curve_rows)
            print(f"\nwrote {args.out} and {csv_path}")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
