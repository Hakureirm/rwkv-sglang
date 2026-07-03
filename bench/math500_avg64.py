#!/usr/bin/env python3
"""
MATH500 avg@N (default avg@64) for OUR sglang RWKV-7 server.

Faithful port of BlinkDL's albatross eval script, reference copy at
scratchpad/official_evals/eval_math500_albatross.py ("REF" below). Everything that
defines the metric is copied from REF; only the inference transport differs
(REF drives the albatross rwkv7_fast_v3a engine directly; we POST to our sglang
server's /generate and let its dynamic batching do the rollouts).

Replicated from REF (line numbers refer to that file):
  * Prompt template (REF build_prefill_cache L128-L134, default --prompt-style
    fake_think L56-L60):
        problem = task.problem.strip().replace("\r\n", "\n")
        prompt  = f"User: {problem}\n\nAssistant: <think></think"      # NB no closing '>'
    and ids = [0] + tokenizer.encode(prompt)   (REF L135; token 0 prepended).
    Context clamp (REF L136-L137): if len(ids)+max_new_tokens > ctx_limit,
    keep the LAST max(1, ctx_limit-max_new_tokens) ids.
  * Sampling params (REF L50-L55, L419): temperature=1.0, top_p=0.28, top_k=32,
    max_new_tokens=1500, ctx_limit=8192, sampler order temperature -> top_k -> top_p
    (sglang's sampler applies temperature, then top_k, then top_p - same order).
    No repetition penalty (REF L421 "penalty": "off").
  * Stop conditions (REF process_next_token L257-L276 / finish_row L187-L199):
      - token 0 sampled  -> "eod"   (we pass stop_token_ids=[0]);
      - "\nUser:" in the decoded text -> "user_stop", completion is the text BEFORE
        it (we pass stop=["\nUser:"]; sglang trims the stop string, same result);
      - max_new_tokens reached -> "max_tokens" (truncated).
    Completion post-processing (REF L196-L199, L219):
        completion = text.split("\nUser:", 1)[0]
        if completion.startswith(">"): completion = completion[1:]   # closes fake think tag
        completion = completion.strip()
  * Grading: verify_one() below is copied VERBATIM from REF L390-L402 (math_verify
    parse/verify; gold wrapped as $\\boxed{answer}$, strict=False). pip install math_verify.
  * avg@N semantics (REF run_master L499-L512): rollout_accuracy =
    correct_generations / total_generations over (num problems x N samples); we also
    report pass_at_rollout_accuracy (any-correct per problem) like REF.
  * Dataset (REF load_tasks L73-L90): MATH500.jsonl, one JSON object per line with
    fields problem/answer/subject/level/unique_id. See bench/data/README.md - the
    upstream set is HuggingFaceH4/MATH-500 (test split, 500 problems); the box has no
    HF access so fetch it on the Mac and pass --data <local jsonl>.

Known deltas vs REF (documented, metric-neutral or unavoidable server-side):
  - REF samples with its own fp32 top-k/top-p kernel and torch seed; sglang sampling
    is not seed-controlled per request, so individual rollouts are not bit-identical
    (avg@64 is a distributional metric; this is expected).
  - REF checks "\nUser:" on incrementally decoded text and drops pending bytes with
    U+FFFD; sglang's stop-string matcher is equivalent for the final completion.

Usage (smoke): python bench/math500_avg64.py --model <dir> --host 127.0.0.1 --port 30000 \
                   --data bench/data/MATH500.jsonl --limit 5 --samples 2
Full avg@64:   ... --limit 0 --samples 64 --concurrency 256
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class Task:  # REF L31-L38
    index: int
    problem: str
    answer: str
    subject: str = ""
    level: str = ""
    unique_id: str = ""


def load_tasks(dataset):  # REF load_tasks L73-L90 (verbatim modulo Task import)
    rows = []
    with open(dataset, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                Task(
                    index=index,
                    problem=str(item["problem"]),
                    answer=str(item["answer"]),
                    subject=str(item.get("subject", "")),
                    level=str(item.get("level", "")),
                    unique_id=str(item.get("unique_id", index)),
                )
            )
    return rows


def verify_one(item):  # REF verify_one L390-L402, copied verbatim
    from math_verify import parse, verify

    try:
        gold = parse(f"$\\boxed{{{item['answer']}}}$")
        pred = parse(str(item["completion"]))
        correct = bool(pred and verify(gold, pred, strict=False))
        error = ""
    except Exception as exc:
        correct = False
        error = f"{type(exc).__name__}: {exc}"
    out = dict(item)
    out["correct"] = correct
    out["verify_error"] = error
    return out


def build_prompt_ids(task, tokenizer, prompt_style, max_new_tokens, ctx_limit):
    # REF build_prefill_cache L128-L137
    problem = task.problem.strip().replace("\r\n", "\n")
    if prompt_style == "fake_think":
        prompt = f"User: {problem}\n\nAssistant: <think></think"
    elif prompt_style == "plain":
        prompt = f"User: {problem}\n\nAssistant:"
    else:
        raise ValueError(f"unknown prompt style: {prompt_style}")
    ids = [0] + tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) + max_new_tokens > ctx_limit:
        ids = ids[-max(1, ctx_limit - max_new_tokens):]
    return ids


def postprocess_completion(text):
    # REF finish_row L196-L199 + L219: split at "\nUser:", drop the '>' that closes
    # the fake think tag, strip. (sglang already trims the stop string, split is a
    # no-op safety net.)
    completion = text.split("\nUser:", 1)[0]
    if completion.startswith(">"):
        completion = completion[1:]
    return completion.strip()


def one_rollout(sess, gen_url, ids, args):
    r = sess.post(
        gen_url,
        json={
            "input_ids": ids,
            "sampling_params": {
                # REF L50-L54 defaults: temperature=1.0 top_p=0.28 top_k=32 max_new=1500
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
                "stop": ["\nUser:"],       # REF user_stop (L270-L272)
                "stop_token_ids": [0],     # REF eod (L263-L265)
            },
        },
        timeout=args.timeout,
    )
    r.raise_for_status()
    item = r.json()
    if isinstance(item, list):
        item = item[0]
    meta = item["meta_info"]
    return {
        "text": item["text"],
        "completion_tokens": meta.get("completion_tokens", 0),
        "finish_reason": (meta.get("finish_reason") or {}).get("type", ""),
        "matched": (meta.get("finish_reason") or {}).get("matched", None),
    }


async def run_all(tasks, prompt_ids, args, gen_url):
    from concurrent.futures import ThreadPoolExecutor
    # asyncio's default thread pool is ~32 threads; size it to --concurrency so the
    # semaphore (not the pool) is the actual cap.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=args.concurrency))
    sem = asyncio.Semaphore(args.concurrency)
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.concurrency, pool_maxsize=args.concurrency)
    sess.mount("http://", adapter)
    done = 0
    total = len(tasks) * args.samples
    t0 = time.time()

    async def one(task, sample_id):
        nonlocal done
        async with sem:
            out = await asyncio.to_thread(one_rollout, sess, gen_url, prompt_ids[task.index], args)
        done += 1
        if done % max(1, total // 20) == 0:
            print(f"  {done}/{total} rollouts ({time.time()-t0:.0f}s)", flush=True)
        # stop_reason mapping to REF vocabulary (finish_row L187-L219)
        fr, matched = out["finish_reason"], out["matched"]
        if fr == "stop" and matched == 0:
            stop_reason = "eod"
        elif fr == "stop":
            stop_reason = "user_stop"
        else:
            stop_reason = "max_tokens"
        return {
            "task_index": task.index,
            "sample_id": sample_id,
            "problem": task.problem,
            "answer": task.answer,
            "subject": task.subject,
            "level": task.level,
            "unique_id": task.unique_id,
            "prompt_tokens": len(prompt_ids[task.index]),
            "generated_tokens": out["completion_tokens"],
            "stop_reason": stop_reason,
            "ended_eod": stop_reason == "eod",
            "ended_user_stop": stop_reason == "user_stop",
            "truncated": stop_reason == "max_tokens",
            "completion": postprocess_completion(out["text"]),
        }

    coros = [one(t, s) for t in tasks for s in range(args.samples)]
    results = await asyncio.gather(*coros)
    return list(results), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="MATH500 avg@N against a running sglang server")
    ap.add_argument("--model", required=True, help="model dir (tokenizer)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--data", required=True, help="MATH500.jsonl (problem/answer per line)")
    ap.add_argument("--samples", type=int, default=64, help="rollouts per problem (avg@N)")
    ap.add_argument("--concurrency", type=int, default=128, help="rollouts in flight")
    ap.add_argument("--limit", type=int, default=0, help="first N problems (0=all 500)")
    # REF defaults L49-L54
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--ctx-limit", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.28)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--prompt-style", choices=("fake_think", "plain"), default="fake_think")
    ap.add_argument("--verify-workers", type=int, default=8)  # REF L63
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default="", help="summary JSON path (generations JSONL alongside)")
    args = ap.parse_args()

    tasks = load_tasks(args.data)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"problems={len(tasks)} samples={args.samples} total_generations={len(tasks)*args.samples}", flush=True)

    from transformers import AutoTokenizer  # same convention as bench/accuracy_eval.py L172
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = {t.index: build_prompt_ids(t, tokenizer, args.prompt_style, args.max_new_tokens, args.ctx_limit) for t in tasks}

    gen_url = f"http://{args.host}:{args.port}/generate"
    raw_rows, gen_wall = asyncio.run(run_all(tasks, prompt_ids, args, gen_url))
    raw_rows.sort(key=lambda x: (x["task_index"], x["sample_id"]))  # REF L482

    # grading (REF run_master L484-L490)
    print(f"verifying rows={len(raw_rows)} workers={args.verify_workers}", flush=True)
    if args.verify_workers <= 1:
        verified = [verify_one(row) for row in raw_rows]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.verify_workers) as pool:
            verified = list(pool.map(verify_one, raw_rows, chunksize=16))
    verified.sort(key=lambda x: (x["task_index"], x["sample_id"]))

    # summary (REF run_master L496-L531)
    by_task = {}
    for row in verified:
        by_task.setdefault(int(row["task_index"]), []).append(row)
    total = len(verified)
    correct_generations = sum(int(row["correct"]) for row in verified)
    pass_tasks = sum(1 for rows in by_task.values() if any(row["correct"] for row in rows))
    gen_tokens = sum(row["generated_tokens"] for row in verified)
    summary = {
        "num_tasks": len(tasks),
        "rollout": args.samples,
        "total_generations": total,
        "correct_generations": correct_generations,
        "rollout_accuracy": correct_generations / max(total, 1),        # <- avg@N (REF L511)
        "pass_at_rollout_accuracy": pass_tasks / max(len(tasks), 1),    # REF L512
        "ended_eod_rate": sum(int(r["ended_eod"]) for r in verified) / max(total, 1),
        "ended_user_stop_rate": sum(int(r["ended_user_stop"]) for r in verified) / max(total, 1),
        "truncated_rate": sum(int(r["truncated"]) for r in verified) / max(total, 1),
        "mean_generated_tokens": gen_tokens / max(total, 1),
        "generated_tokens_total": gen_tokens,
        "gen_wall_time_s": gen_wall,
        "throughput_gen_tok_per_s": gen_tokens / max(gen_wall, 1e-9),
        "sample_per_sec": total / max(gen_wall, 1e-9),
        "config": {
            "model": args.model, "data": args.data,
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens, "ctx_limit": args.ctx_limit,
            "sampler_order": "temperature -> top_k -> top_p",  # REF L419
            "penalty": "off",                                   # REF L421
            "prompt_style": args.prompt_style,
            "concurrency": args.concurrency,
        },
    }
    print("\n===== MATH500 avg@%d =====" % args.samples)
    print(f"avg@{args.samples} (rollout_accuracy): {summary['rollout_accuracy']*100:.2f}%  "
          f"({correct_generations}/{total})")
    print(f"pass@{args.samples}: {summary['pass_at_rollout_accuracy']*100:.2f}%")
    print(f"eod {summary['ended_eod_rate']*100:.1f}%  user_stop {summary['ended_user_stop_rate']*100:.1f}%  "
          f"truncated {summary['truncated_rate']*100:.1f}%")
    print(f"throughput: {gen_tokens} gen tokens / {gen_wall:.1f}s = "
          f"{summary['throughput_gen_tok_per_s']:.1f} tok/s (client wall; see server log for its own count)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        gens_path = os.path.splitext(args.out)[0] + "_generations.jsonl"
        with open(gens_path, "w", encoding="utf-8") as f:
            for row in verified:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {args.out} and {gens_path}")
    print("MATH500_SGLANG_RESULT " + json.dumps(summary, ensure_ascii=False))  # REF L533 style


if __name__ == "__main__":
    main()
