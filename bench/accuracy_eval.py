#!/usr/bin/env python3
"""
Accuracy eval for RWKV-7: lambada_openai + MMLU (BlinkDL rwkv_mmlu_eval methodology),
runnable on (a) the BlinkDL `rwkv` pip REFERENCE (--backend rwkv, from the raw .pth) and
(b) OURS via the sglang OpenAI server (--backend server). This establishes that OURS
matches the rwkv-lm reference on STANDARD metrics (not just greedy-exact), and quantifies
int8 drift.

MMLU here = the exact methodology that yields the published "World-1.5B-v3 ~44.87%":
BlinkDL template, prepend token 0, take last-position logits, compare the single-token
choices " A"/" B"/" C"/" D" by log-prob, argmax = prediction (see refs/RWKV-LM/RWKV-v7/
rwkv_mmlu_eval.py). NB our models are the g1/g1g series (a DIFFERENT checkpoint than
World-v3), so the reference number here is rwkv-pip on the SAME .pth our fla model was
converted from - the published 44.87% is cited only as a methodology sanity anchor.

lambada acc/ppl definitions mirror lm-eval `lambada_openai`:
  context   = text.rsplit(' ',1)[0]           (all but last word)
  continuation = ' ' + last word
  full = context+continuation; target tokens = enc(full)[len(enc(context)):]
  acc = fraction where EVERY target token is the greedy argmax
  ppl = exp(-sum target logprobs / #target tokens)

Usage
-----
  # REFERENCE (rwkv pip, from .pth) - oracle env has `rwkv`+torch(cuda)
  ~/envs/rwkv-ref/bin/python bench/accuracy_eval.py --backend rwkv \
      --task mmlu   --pth ~/rwkv_models/rwkv7-g1/rwkv7-g1g-1.5b-20260526-ctx8192.pth \
      --vocab <dir>/rwkv_vocab_v20230424.txt --mmlu <dir>/mmlu_test_dataset \
      --sample 2000 --seed 42 --out bench/results/clean/acc_ref_1.5B_mmlu.json
  ~/envs/rwkv-ref/bin/python bench/accuracy_eval.py --backend rwkv --task lambada \
      --pth <pth> --vocab <vocab> --lambada <parquet> \
      --out bench/results/clean/acc_ref_1.5B_lambada.json

  # OURS (sglang server on :30000, started separately) - lm-eval venv or rwkv-sgl venv
  python bench/accuracy_eval.py --backend server --task mmlu \
      --url http://127.0.0.1:30000/v1 --model-name <served-name> \
      --tokenizer <model_dir> --mmlu <dir>/mmlu_test_dataset --sample 2000 --seed 42 \
      --out bench/results/clean/acc_ours_1.5B_bf16_mmlu.json
"""
import argparse
import json
import math
import os
import random

TEMPLATE = '''User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: The answer is'''
CHOICES = [" A", " B", " C", " D"]


def _log_softmax_np(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    z = x - m
    return z - math.log(np.exp(z).sum())


def mmlu_prompt(sample):
    return (TEMPLATE.replace("<Q>", sample["question"])
            .replace("<|A|>", sample["choices"][0])
            .replace("<|B|>", sample["choices"][1])
            .replace("<|C|>", sample["choices"][2])
            .replace("<|D|>", sample["choices"][3])
            .replace("<SUBJECT>", sample["subject"].replace("_", " ")))


def load_mmlu(path, sample, seed):
    from datasets import load_from_disk
    d = load_from_disk(path)
    idx = list(range(len(d)))
    if sample and sample < len(d):
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:sample])
    return d, idx


def load_lambada(path, limit):
    from datasets import load_dataset
    d = load_dataset("parquet", data_files=path, split="train")
    rows = [d[i]["text"] for i in range(len(d))]
    if limit:
        rows = rows[:limit]
    return rows


def split_lambada(text):
    text = text.strip()
    ctx, last = text.rsplit(" ", 1)
    return ctx, " " + last


# --------------------------------------------------------------------------- #
# rwkv pip REFERENCE backend                                                   #
# --------------------------------------------------------------------------- #
def rwkv_backend(args):
    import numpy as np
    os.environ.setdefault("RWKV_V7_ON", "1")
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ.setdefault("RWKV_CUDA_ON", args.rwkv_cuda_on)
    from rwkv.model import RWKV
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(args.vocab)
    model_path = args.pth[:-4] if args.pth.endswith(".pth") else args.pth
    print(f"[ref] loading {model_path} strategy={args.strategy!r} cuda_on={os.environ['RWKV_CUDA_ON']}")
    model = RWKV(model=model_path, strategy=args.strategy)

    if args.task == "mmlu":
        d, idx = load_mmlu(args.mmlu, args.sample, args.seed)
        choice_tok = [tok.encode(c)[0] for c in CHOICES]
        correct = 0
        for n, i in enumerate(idx):
            s = d[i]
            ids = [0] + tok.encode(mmlu_prompt(s).replace("\r\n", "\n").strip())
            logits, _ = model.forward(ids, None, full_output=False)
            lg = logits.float().cpu().numpy() if hasattr(logits, "float") else np.asarray(logits)
            pred = int(np.argmax([lg[t] for t in choice_tok]))
            correct += int(pred == s["answer"])
            if (n + 1) % 500 == 0:
                print(f"  {n+1}/{len(idx)} acc={correct/(n+1):.4f}")
        acc = correct / len(idx)
        res = {"task": "mmlu", "backend": "rwkv", "n": len(idx), "acc": acc}
    else:
        rows = load_lambada(args.lambada, args.limit)
        correct = 0
        tot_ll = 0.0
        tot_tok = 0
        for n, text in enumerate(rows):
            ctx, cont = split_lambada(text)
            ctx_ids = tok.encode(ctx)
            full_ids = tok.encode(ctx + cont)
            tgt = full_ids[len(ctx_ids):]
            if not tgt:
                continue
            logits, _ = model.forward(full_ids, None, full_output=True)
            lg = logits.float().cpu().numpy() if hasattr(logits, "float") else np.asarray(logits)
            greedy = True
            for j, t in enumerate(tgt):
                pos = len(ctx_ids) + j - 1
                row = lg[pos]
                tot_ll += float(_log_softmax_np(row)[t])
                tot_tok += 1
                if int(np.argmax(row)) != t:
                    greedy = False
            correct += int(greedy)
            if (n + 1) % 1000 == 0:
                print(f"  {n+1}/{len(rows)} acc={correct/(n+1):.4f}")
        acc = correct / len(rows)
        res = {"task": "lambada", "backend": "rwkv", "n": len(rows),
               "acc": acc, "perplexity": math.exp(-tot_ll / max(tot_tok, 1))}
    res.update({"pth": args.pth, "strategy": args.strategy})
    _write(args.out, res)


# --------------------------------------------------------------------------- #
# sglang server backend (OURS) - MMLU via echo logprobs                        #
# --------------------------------------------------------------------------- #
def server_backend(args):
    # sglang disables echo+logprobs on /v1/completions, so we score MMLU choices via the
    # native /generate API: for each question, one forward over the context requesting the
    # log-probs of the 4 single-token choices at the next position (token_ids_logprob),
    # argmax = prediction (BlinkDL rwkv_mmlu_eval methodology, served through OUR sglang).
    import numpy as np
    import requests
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    base = args.url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    gen_url = base + "/generate"

    if args.task != "mmlu":
        raise SystemExit("server backend implements MMLU only; use lm-eval for lambada on ours")

    d, idx = load_mmlu(args.mmlu, args.sample, args.seed)
    choice_tok = [tok.encode(c)[0] for c in CHOICES]
    correct = 0
    sess = requests.Session()

    # NB sglang 0.5.10 crashes if `token_ids_logprob` requests are BATCHED together
    # (server-side get_token_ids_logprobs_raw sees token_ids=None for a batched slot).
    # So we send ONE request at a time and wait for it - only ever one in flight, so the
    # scheduler never batches two token_ids_logprob requests. Serial but robust (~2000 fast
    # single-token forwards). token_ids_logprob is a single flat list (the working form).
    for n, i in enumerate(idx):
        s = d[i]
        ctx = [0] + tok.encode(mmlu_prompt(s).replace("\r\n", "\n").strip())
        r = sess.post(gen_url, json={
            "input_ids": ctx,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True,
            "token_ids_logprob": list(choice_tok),
        }, timeout=600)
        r.raise_for_status()
        item = r.json()
        otl = item["meta_info"]["output_token_ids_logprobs"][0]  # [[lp, tokid, txt], ...]
        lp_by_tok = {tid: lp for lp, tid, _ in otl if lp is not None}
        lps = [lp_by_tok.get(t, -1e9) for t in choice_tok]
        if int(np.argmax(lps)) == s["answer"]:
            correct += 1
        if (n + 1) % 200 == 0:
            print(f"  {n+1}/{len(idx)} acc={correct/(n+1):.4f}")
    acc = correct / len(idx)
    res = {"task": "mmlu", "backend": "server", "n": len(idx), "acc": acc,
           "model": args.tokenizer, "dtype": args.dtype}
    _write(args.out, res)


def _write(path, obj):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(obj, open(path, "w"), indent=2)
        print("wrote", path)
    print(json.dumps(obj, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=("rwkv", "server"))
    ap.add_argument("--task", required=True, choices=("mmlu", "lambada"))
    ap.add_argument("--out", default="")
    # rwkv
    ap.add_argument("--pth", default="")
    ap.add_argument("--vocab", default="")
    ap.add_argument("--strategy", default="cuda fp16")
    ap.add_argument("--rwkv-cuda-on", default="1")
    # server
    ap.add_argument("--url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model-name", default="rwkv")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--dtype", default="")
    ap.add_argument("--batch", type=int, default=32, help="server: #questions per flush")
    # data
    ap.add_argument("--mmlu", default="")
    ap.add_argument("--lambada", default="")
    ap.add_argument("--sample", type=int, default=0, help="mmlu: random subsample size (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="lambada: first N (0=all)")
    args = ap.parse_args()
    (rwkv_backend if args.backend == "rwkv" else server_backend)(args)


if __name__ == "__main__":
    main()
