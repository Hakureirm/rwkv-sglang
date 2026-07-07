#!/usr/bin/env python3
"""
MATH500 avg@K for RWKV-7 on MLX -- DIRECT-CALL (no server; MLX has none),
same metric/prompt/sampler/grader as `bench/math500_avg64.py` (our sglang
harness, itself a faithful port of BlinkDL's albatross `eval_math500.py` --
see that file's docstring for the REF line citations). This file reuses its
dataset loader, prompt builder, completion post-processing and math_verify
grader UNCHANGED (imported, not copied) so grading can never drift between
the CUDA/sglang and Apple-Silicon/MLX legs of this project's MATH500 numbers.

Sampler: albatross/sglang apply temperature -> top_k -> top_p, in that
order (`bench/math500_avg64.py` docstring). mlx_lm's own `make_sampler`
composes top_p BEFORE top_k, which is a DIFFERENT algorithm when both are
simultaneously restrictive -- so this file hand-wires mlx_lm's individual
`apply_top_k`/`apply_top_p`/`categorical_sampling` primitives in the
REF order instead of using `make_sampler`. In practice temperature=1.0 (the
REF default, unchanged here) makes the temperature step a no-op scale, so
only the top_k-before-top_p ordering is actually load-bearing; documented
for the record in case a future run changes temperature.

Realistic-scale note (read before citing this as "the" RWKV MATH500 number):
this is bsz1, no batching (MLX has no continuous-batching server here, unlike
the sglang harness this mirrors, which gets its throughput from serving many
concurrent rollouts) -- full 500-problem x 64-sample avg@64 would take on the
order of days on a single Mac. `--limit`/`--samples` default to a reduced,
explicitly-labeled subset; see the finding doc for the exact wall-clock
justification.

    python mlx_port/math500_mlx.py --model /tmp/mlx_models/rwkv7-1.5b-fla \
        --data bench/data/MATH500.jsonl --limit 60 --samples 4 \
        --out mlx_port/results/math500_rwkv7_1.5b.json
"""
import argparse
import json
import os
import random
import sys
import time

import mlx.core as mx
from mlx_lm.sample_utils import apply_top_k, apply_top_p, categorical_sampling

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_mlx import load_model
from gate_oracle import WorldTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import math500_avg64 as ref  # load_tasks / build_prompt_ids / postprocess_completion / verify_one

DEFAULT_SEED = 20260707  # fixed so the RWKV and Qwen3.5 MLX harnesses draw the
                          # IDENTICAL problem subset at a given --limit (paired
                          # comparison, not two independent samples)


class _TokAdapter:
    """Adapts WorldTokenizer (no add_special_tokens kwarg) to the
    `ref.build_prompt_ids` call convention (`tok.encode(prompt,
    add_special_tokens=False)`), so the REF prompt-building code runs
    unmodified for RWKV's own tokenizer."""

    def __init__(self, tok):
        self.tok = tok

    def encode(self, s, add_special_tokens=False):
        return self.tok.encode(s)


def select_subset(tasks, limit, seed):
    if limit <= 0 or limit >= len(tasks):
        return tasks
    idx = sorted(random.Random(seed).sample(range(len(tasks)), limit))
    return [tasks[i] for i in idx]


def sample_token(logits, top_k, top_p, temp):
    """REF order: temperature -> top_k -> top_p (bench/math500_avg64.py
    docstring). temp=1.0 (REF default) is a no-op scale either way."""
    logits = mx.reshape(logits, (-1,)).astype(mx.float32)
    logp = logits - mx.logsumexp(logits)
    if temp != 1.0:
        logp = logp * (1.0 / temp)
    if top_k and top_k > 0:
        logp = apply_top_k(logp, top_k)
    if top_p and 0 < top_p < 1.0:
        logp = apply_top_p(logp, top_p)
    tok = categorical_sampling(logp, 1.0)  # temp already applied above
    return int(tok.item())


def rollout(model, tok, prompt_ids, max_new_tokens, top_k, top_p, temp,
            eod_id=0, stop_str="\nUser:"):
    state = model.new_state()
    logits, state = model.prefill(prompt_ids, state)
    out_ids = []
    stop_reason = "max_tokens"
    for _ in range(max_new_tokens):
        next_id = sample_token(logits, top_k, top_p, temp)
        if next_id == eod_id:
            stop_reason = "eod"
            break
        out_ids.append(next_id)
        text = tok.decode(out_ids)
        if stop_str in text:
            stop_reason = "user_stop"
            break
        logits, state = model.step(next_id, state)
    text = tok.decode(out_ids)
    if stop_reason == "user_stop":
        text = text.split(stop_str, 1)[0]
    return text, len(out_ids), stop_reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "bench", "data", "MATH500.jsonl"))
    ap.add_argument("--limit", type=int, default=60, help="0 = all 500")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--samples", type=int, default=4, help="rollouts per problem (avg@N)")
    ap.add_argument("--max-new-tokens", type=int, default=1500)  # REF default
    ap.add_argument("--ctx-limit", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.28)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--wkv", default="metal")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--verify-workers", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--progress-every", type=int, default=5)
    args = ap.parse_args()

    tasks = ref.load_tasks(args.data)
    tasks = select_subset(tasks, args.limit, args.seed)
    print(f"problems={len(tasks)} (seed={args.seed}, from 500) samples={args.samples} "
          f"total_generations={len(tasks) * args.samples}", flush=True)

    vocab = os.path.join(args.model, "rwkv_vocab_v20230424.txt")
    world_tok = WorldTokenizer(vocab)
    tok_adapter = _TokAdapter(world_tok)
    model = load_model(args.model, wkv=args.wkv, quant=args.quant)

    prompt_ids = {
        t.index: ref.build_prompt_ids(t, tok_adapter, "fake_think",
                                       args.max_new_tokens, args.ctx_limit)
        for t in tasks
    }

    raw_rows = []
    t0 = time.time()
    n_done = 0
    n_total = len(tasks) * args.samples
    for t in tasks:
        for s in range(args.samples):
            text, n_gen, stop_reason = rollout(
                model, world_tok, prompt_ids[t.index], args.max_new_tokens,
                args.top_k, args.top_p, args.temperature)
            completion = ref.postprocess_completion(text)
            raw_rows.append({
                "task_index": t.index, "sample_id": s, "problem": t.problem,
                "answer": t.answer, "subject": t.subject, "level": t.level,
                "unique_id": t.unique_id,
                "prompt_tokens": len(prompt_ids[t.index]),
                "generated_tokens": n_gen, "stop_reason": stop_reason,
                "ended_eod": stop_reason == "eod",
                "ended_user_stop": stop_reason == "user_stop",
                "truncated": stop_reason == "max_tokens",
                "completion": completion,
            })
            n_done += 1
            if n_done % args.progress_every == 0:
                dt = time.time() - t0
                print(f"  {n_done}/{n_total} rollouts ({dt:.0f}s, "
                      f"{dt/n_done:.1f}s/rollout, ETA {(n_total-n_done)*dt/n_done:.0f}s)",
                      flush=True)
    gen_wall = time.time() - t0

    print(f"verifying rows={len(raw_rows)} workers={args.verify_workers}", flush=True)
    if args.verify_workers <= 1:
        verified = [ref.verify_one(row) for row in raw_rows]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.verify_workers) as pool:
            verified = list(pool.map(ref.verify_one, raw_rows, chunksize=8))

    total = len(verified)
    correct = sum(int(r["correct"]) for r in verified)
    by_task = {}
    for r in verified:
        by_task.setdefault(r["task_index"], []).append(r)
    pass_tasks = sum(1 for rs in by_task.values() if any(r["correct"] for r in rs))
    gen_tokens = sum(r["generated_tokens"] for r in verified)

    summary = {
        "engine": "mlx_port (direct-call, no server)",
        "model": args.model, "wkv": args.wkv, "quant": args.quant,
        "num_tasks": len(tasks), "rollout": args.samples,
        "total_generations": total, "correct_generations": correct,
        "rollout_accuracy": correct / max(total, 1),
        "pass_at_rollout_accuracy": pass_tasks / max(len(tasks), 1),
        "ended_eod_rate": sum(int(r["ended_eod"]) for r in verified) / max(total, 1),
        "ended_user_stop_rate": sum(int(r["ended_user_stop"]) for r in verified) / max(total, 1),
        "truncated_rate": sum(int(r["truncated"]) for r in verified) / max(total, 1),
        "mean_generated_tokens": gen_tokens / max(total, 1),
        "generated_tokens_total": gen_tokens,
        "gen_wall_time_s": gen_wall,
        "throughput_gen_tok_per_s": gen_tokens / max(gen_wall, 1e-9),
        "config": {
            "model": args.model, "data": args.data, "problem_subset_seed": args.seed,
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens, "ctx_limit": args.ctx_limit,
            "sampler_order": "temperature -> top_k -> top_p (hand-wired mlx_lm primitives)",
            "prompt_style": "fake_think",
        },
        "scale_note": (
            f"REDUCED from the canonical avg@64/500-problem CUDA protocol to "
            f"{len(tasks)} problems x {args.samples} samples ({total} total "
            f"generations) -- single-Mac bsz1 realistic time budget, see finding doc."
        ),
    }
    print("\n===== MATH500 avg@%d (MLX, RWKV-7, reduced scale) =====" % args.samples)
    print(f"avg@{args.samples}: {summary['rollout_accuracy']*100:.2f}% ({correct}/{total})")
    print(f"pass@{args.samples}: {summary['pass_at_rollout_accuracy']*100:.2f}%")
    print(f"eod {summary['ended_eod_rate']*100:.1f}%  truncated {summary['truncated_rate']*100:.1f}%")
    print(f"mean_generated_tokens={summary['mean_generated_tokens']:.1f}  "
          f"throughput={summary['throughput_gen_tok_per_s']:.1f} tok/s  wall={gen_wall:.0f}s")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        gens_path = os.path.splitext(args.out)[0] + "_generations.jsonl"
        with open(gens_path, "w", encoding="utf-8") as f:
            for row in verified:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {args.out} and {gens_path}")
    print("MATH500_MLX_RWKV_RESULT " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
