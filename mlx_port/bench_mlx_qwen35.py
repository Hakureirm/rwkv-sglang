#!/usr/bin/env python3
"""
Qwen3.5-2B on MLX: bsz1 decode + long-prompt prefill throughput, benchmarked
with the SAME protocol as this dir's bench_mlx.py (RWKV-7's own bench), so
the numbers are directly, honestly comparable to the RWKV-7 1.5B MLX numbers
in docs/BENCHMARKS.md §12 -- not just superficially similar.

Protocol (mirrors bench_mlx.py's bench_decode/bench_prefill line-for-line):
  * decode: prefill a short seed prompt, run `--warmup` untimed greedy steps
    (async-pipelined: mx.async_eval per step, no host round-trip mid-loop,
    a single mx.eval to drain), then time `--decode-tokens` further greedy
    steps the same way. Median of `--decode-runs` runs; report median + best.
  * prefill: a `--prefill-tokens`-token prompt (seed prompt tokens tiled --
    deterministic and reproducible by construction, same trick bench_mlx.py
    uses), fresh cache, timed end-to-end including forcing logits + cache
    state to materialize. Median of `--prefill-runs` runs after 1 discarded
    warmup run.

Difference from bench_mlx.py (disclosed, not hidden): this benchmarks
Qwen3.5 through mlx_lm 0.31.3 -- the opponent's own native, actively
maintained MLX implementation (real hand-written Metal delta-rule kernel for
its Gated-DeltaNet layers, see F0044) -- not a from-scratch port. mlx_port/'s
"zero fla/torch/transformers" policy governs what THIS PROJECT ships as its
own RWKV-7 implementation; it was never a requirement to also hand-port the
competitor's architecture just to benchmark it (F0044's Decision section --
mirrors how the GPU/cloud tier benchmarks Qwen3.5 through sglang's own
native support, not a hand-rolled mirror port). No RWKV-7 code, weights, or
oracle fixtures are touched by this file.

There is no numerical oracle for Qwen3.5 in this repo (out of scope, see
F0044's honest-limits section) -- a coherence sample (greedy continuation,
eyeballed for on-topic, non-garbled prose) substitutes, same bar F0044 used.

    python bench_mlx_qwen35.py
    python bench_mlx_qwen35.py --bf16-model /path/to/bf16 --int4-model /path/to/int4
"""
import argparse
import gc
import json
import os
import platform
import statistics
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load

DEFAULT_BF16 = "/private/tmp/qwen35_mlx_test/Qwen3.5-2B"
DEFAULT_INT4 = "/private/tmp/qwen35_mlx_test/Qwen3.5-2B-mlx-4bit"
DEFAULT_PROMPT = "The capital of France is"  # same prompt as F0044's smoke test


def _greedy_loop(model, cache, logits, n):
    """Mirrors rwkv7_mlx.py's Model.greedy_loop exactly: async-pipelined
    greedy decode over `n` steps -- the argmax token array is fed straight
    back as the next input, no per-step host sync (mx.async_eval only).
    Takes last-step logits in, returns (list of n token arrays, final
    logits); caller mx.eval()s the token list to drain."""
    toks = []
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.async_eval(tok)
    for _ in range(n):
        toks.append(tok)
        logits = model(tok[:, None], cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.async_eval(tok)
    return toks, logits


def bench_decode(model, prompt_tokens, n_timed=128, n_warm=16, runs=5):
    """Same shape as bench_mlx.py's bench_decode: fresh cache each run,
    prefill the seed prompt, n_warm untimed steps (drained), then n_timed
    timed steps (drained). Returns (median, best) tok/s over `runs` runs."""
    rates = []
    for _ in range(runs):
        cache = make_prompt_cache(model)
        logits = model(mx.array(prompt_tokens)[None], cache=cache)
        mx.eval(logits, *[c.state for c in cache])
        warm, logits = _greedy_loop(model, cache, logits, n_warm)
        mx.eval(*warm)  # drain: caches warm, queue empty
        t0 = time.perf_counter()
        toks, logits = _greedy_loop(model, cache, logits, n_timed)
        mx.eval(*toks)
        rates.append(n_timed / (time.perf_counter() - t0))
    return statistics.median(rates), max(rates)


def bench_prefill(model, tokens, runs=3):
    """Same shape as bench_mlx.py's bench_prefill: fresh cache each run,
    time end-to-end including full logits+state materialization, median of
    `runs` after 1 discarded warmup run."""
    rates = []
    for i in range(runs + 1):  # +1 warmup (discarded)
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        logits = model(mx.array(tokens)[None], cache=cache)
        mx.eval(logits, *[c.state for c in cache])
        dt = time.perf_counter() - t0
        if i > 0:
            rates.append(len(tokens) / dt)
    return statistics.median(rates)


def coherence_sample(model, tokenizer, prompt_tokens, n=24):
    """Not an oracle gate (none exists for Qwen3.5 in this repo, F0044) --
    a greedy continuation to eyeball for on-topic, non-garbled prose, the
    same bar F0044's feasibility probe used."""
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt_tokens)[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache])
    toks, _ = _greedy_loop(model, cache, logits, n)
    mx.eval(*toks)
    ids = [int(t.item()) for t in toks]
    return ids, tokenizer.decode(ids)


def run_one(tag, model_path, prompt_text, decode_tokens, warmup, decode_runs,
            prefill_tokens, prefill_runs):
    model, tokenizer = load(model_path)
    prompt_tokens = tokenizer.encode(prompt_text)
    long_prompt = (prompt_tokens * (prefill_tokens // len(prompt_tokens) + 1)
                   )[:prefill_tokens]

    cont_ids, cont_text = coherence_sample(model, tokenizer, prompt_tokens)
    print(f"[{tag}] coherence sample (greedy, {len(cont_ids)} tok): "
          f"{cont_text!r}")

    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    dec_med, dec_best = bench_decode(model, prompt_tokens, n_timed=decode_tokens,
                                      n_warm=warmup, runs=decode_runs)
    pre = bench_prefill(model, long_prompt, runs=prefill_runs)
    peak = (mx.get_peak_memory() / 2**30 if hasattr(mx, "get_peak_memory")
            else float("nan"))

    print(f"BENCH_{tag} decode={dec_med:.1f} tok/s (median; best {dec_best:.1f}) "
          f"(bsz1 greedy, {decode_tokens} steady-state, median of {decode_runs}) "
          f"prefill={pre:.1f} tok/s ({prefill_tokens} tokens, median of "
          f"{prefill_runs})  peak_mem={peak:.2f} GiB", flush=True)

    result = {
        "tag": tag,
        "model_path": model_path,
        "prompt_text": prompt_text,
        "prompt_tokens_n": len(prompt_tokens),
        "coherence_sample_ids": cont_ids,
        "coherence_sample_text": cont_text,
        "decode_tok_s_median": dec_med,
        "decode_tok_s_best": dec_best,
        "decode_tokens": decode_tokens,
        "decode_warmup": warmup,
        "decode_runs": decode_runs,
        "prefill_tok_s_median": pre,
        "prefill_tokens": prefill_tokens,
        "prefill_runs": prefill_runs,
        "peak_mem_gib": peak,
    }

    # Release fully before the next config loads -- same discipline
    # bench_mlx.py documents: del + gc + clear_cache, so per-config peak
    # memory isn't polluted by the previous config's retained weights.
    del model, tokenizer
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-model", default=DEFAULT_BF16)
    ap.add_argument("--int4-model", default=DEFAULT_INT4)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--decode-runs", type=int, default=5)
    ap.add_argument("--prefill-tokens", type=int, default=1024)
    ap.add_argument("--prefill-runs", type=int, default=3)
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()

    info = mx.device_info()
    print(f"chip: {info.get('device_name')}  memory: "
          f"{info.get('memory_size', 0) / 2**30:.0f} GB unified  "
          f"macOS {platform.mac_ver()[0]}  mlx {mx.__version__}  "
          f"mlx_lm {mlx_lm.__version__}  python {platform.python_version()}")

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [("QWEN35_2B_bf16", args.bf16_model, "bf16"),
            ("QWEN35_2B_int4", args.int4_model, "int4")]
    for tag, path, precision in jobs:
        if not os.path.isdir(path):
            print(f"SKIP {tag}: {path} not found")
            continue
        result = run_one(tag, path, args.prompt, args.decode_tokens, args.warmup,
                          args.decode_runs, args.prefill_tokens, args.prefill_runs)
        result["precision"] = precision
        out_path = os.path.join(args.out_dir, f"bench_qwen35_2b_{precision}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
