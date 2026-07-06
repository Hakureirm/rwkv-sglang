#!/usr/bin/env python3
"""
MLX-port single-stream benchmark: bsz1 greedy-decode tok/s and prefill tok/s.

Protocol (honest-numbers discipline):
  * GATE FIRST: before any timing, the 24-token oracle gate for the exact
    (model, dtype, wkv) configuration is re-run in-process; on mismatch the
    bench ABORTS — no number can be printed from an ungated configuration.
  * decode: process the fixture prompt, run 16 untimed warmup steps, then
    time 128 greedy steps (each step syncs on argmax, i.e. real decode
    latency including the host round-trip). median of 3 runs.
  * prefill: a 1024-token prompt (fixture prompt tokens tiled — RWKV-7 dense
    prefill cost is content-independent), fresh state, timed end-to-end
    including the final state materialization. median of 3 runs after 1
    warmup (warmup covers mx.compile trace + Metal JIT).

  python bench_mlx.py [--models-root /tmp/mlx_models] [--wkv pure,metal]
      [--dtype bfloat16] [--decode-tokens 128] [--prefill-tokens 1024]

Markers: BENCH_<TAG> lines, one per (model, wkv) pair.
"""
import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_mlx import load_model


def _gate(model, fx, tag, wkv):
    got, _ = model.generate(fx["prompt_tokens"], len(fx["greedy_tokens"]))
    if got != fx["greedy_tokens"]:
        print(f"BENCH_ABORT_{tag} gate FAILED for wkv={wkv} — "
              f"refusing to publish numbers", flush=True)
        sys.exit(1)
    print(f"[{tag}] pre-bench gate re-check PASS 24/24 (wkv={wkv})")


def bench_decode(model, prompt_tokens, n_timed=128, n_warm=16, runs=5):
    """Steady-state bsz1 greedy decode via the async-pipelined loop (the same
    greedy_loop the gate validates token-exactly). The pipeline is drained
    (mx.eval on every produced token) before the clock stops. Returns
    (median, best) tok/s: bsz1 greedy decode is bandwidth-bound on the per-token
    weight read, so on a loaded machine the host-side scheduling adds jitter —
    `best` is the least-contended (closest-to-hardware) rate, `median` the
    typical rate. Both are reported for honesty."""
    rates = []
    for _ in range(runs):
        state = model.new_state()
        logits, state = model.prefill(prompt_tokens, state)
        warm, logits, state = model.greedy_loop(logits, state, n_warm)
        mx.eval(*warm)  # drain: compile/JIT + caches warm, queue empty
        t0 = time.perf_counter()
        toks, logits, state = model.greedy_loop(logits, state, n_timed)
        mx.eval(*toks)
        rates.append(n_timed / (time.perf_counter() - t0))
    return statistics.median(rates), max(rates)


def bench_prefill(model, tokens, runs=3):
    rates = []
    for i in range(runs + 1):  # +1 warmup (discarded)
        state = model.new_state()
        t0 = time.perf_counter()
        logits, state = model.prefill(tokens, state)
        dt = time.perf_counter() - t0  # prefill() eval'd logits + full state
        if i > 0:
            rates.append(len(tokens) / dt)
    return statistics.median(rates)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fixtures = os.path.join(here, "..", "bench", "fixtures")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="/tmp/mlx_models")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--wkv", default="pure,metal")
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--prefill-tokens", type=int, default=1024)
    args = ap.parse_args()

    info = mx.device_info()
    print(f"chip: {info.get('device_name')}  memory: "
          f"{info.get('memory_size', 0) / 2**30:.0f} GB unified  "
          f"macOS {platform.mac_ver()[0]}  mlx {mx.__version__}  "
          f"python {platform.python_version()}")

    jobs = [
        ("01B", os.path.join(args.models_root, "rwkv7-0.1b-fla"),
         os.path.join(fixtures, "oracle_rwkv7_01b_eiffel.json")),
        ("15B", os.path.join(args.models_root, "rwkv7-1.5b-fla"),
         os.path.join(fixtures, "oracle_rwkv7_15b_eiffel.json")),
    ]
    for tag, model_dir, fixture in jobs:
        fx = json.load(open(fixture))
        prompt = fx["prompt_tokens"]
        # 1024-token prefill prompt: fixture tokens tiled (dense-cost model,
        # content-independent; keeps the bench fully deterministic).
        long_prompt = (prompt * (args.prefill_tokens // len(prompt) + 1))[
            : args.prefill_tokens]
        for wkv in args.wkv.split(","):
            model = load_model(model_dir, dtype=args.dtype, wkv=wkv)
            _gate(model, fx, tag, wkv)
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()  # drop the gate's transient buffers
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            dec_med, dec_best = bench_decode(model, prompt,
                                             n_timed=args.decode_tokens)
            pre = bench_prefill(model, long_prompt)
            peak = (mx.get_peak_memory() / 2**30
                    if hasattr(mx, "get_peak_memory") else float("nan"))
            print(f"BENCH_{tag} wkv={wkv} dtype={args.dtype} "
                  f"decode={dec_med:.1f} tok/s (median; best {dec_best:.1f}) "
                  f"(bsz1 greedy, {args.decode_tokens} steady-state, median of 5)"
                  f"  prefill={pre:.1f} tok/s ({args.prefill_tokens} tokens, "
                  f"median of 3)  peak_mem={peak:.2f} GiB", flush=True)
            # Honest per-config peak needs the PREVIOUS config fully released
            # before the next one loads. `del model` is not enough: the compiled
            # decode step (mx.compile(self._forward_seq)) captures every weight
            # by closure, so the ~3 GiB stays live until the compiled callable
            # is dropped too. Null it, gc the cycle, then return the pool to the
            # OS — otherwise pure-then-metal reports metal's peak as ~2x (it was
            # reading the retained pure weights). Confirmed: active mem -> 0.
            model._step = None
            del model
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()


if __name__ == "__main__":
    main()
