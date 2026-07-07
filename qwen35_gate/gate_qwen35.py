#!/usr/bin/env python3
"""Qwen3.5 correctness gate: independent numpy fp32 reference vs an actually-
running serving backend, on the same probe text.

This is the Qwen3.5 analogue of `mlx_port/gate_oracle.py` and
`bench/oracle_numpy.py`'s bit-exact RWKV-7 discipline, adapted for a
third-party model whose serving code (mlx-lm, sglang) this project doesn't
control: exact bit-identity across fp32-CPU/bf16-Metal/bf16-CUDA backends is
not a realistic bar, so the gate checks (a) top-1 token agreement, (b) top-5
token SET agreement, (c) probability closeness on the shared tokens. See
`docs/findings/00XX-qwen35-oracle-gate.md` for the full writeup and the
rationale for these specific thresholds.

Usage:
    python gate_qwen35.py \\
        --pth /tmp/qwen35_gate_work/qwen35_2b_text.pth \\
        --hf-dir /private/tmp/qwen35_mlx_test/Qwen3.5-2B \\
        [--sglang-url http://192.168.x.x:30070] \\
        [--probe " Eiffel"]

`--sglang-url`, if given, is queried live via sglang's native /generate
endpoint using the SAME input_ids the other two paths use (bypassing sglang's
own tokenizer, for an apples-to-apples comparison). If omitted or
unreachable, the sglang leg is skipped and the verdict is based on the
numpy-vs-mlx comparison alone.
"""
import argparse
import json
import sys

import numpy as np

from numpy_reference import PROBE_TEXT, Qwen35
import mlx_probe


def rows_to_dict(rows):
    """[(token, logit_or_logprob, prob), ...] -> {token: prob}"""
    return {t: p for t, _, p in rows}


def compare(name_a, rows_a, name_b, rows_b, prob_tol=0.02):
    a, b = rows_to_dict(rows_a), rows_to_dict(rows_b)
    top1_a, top1_b = rows_a[0][0], rows_b[0][0]
    top1_match = top1_a == top1_b
    top5_a, top5_b = set(list(a)[:5]), set(list(b)[:5])
    top5_set_match = top5_a == top5_b
    shared = set(a) & set(b)
    max_abs_diff = max((abs(a[t] - b[t]) for t in shared), default=float("nan"))
    verdict = "PASS" if top1_match and top5_set_match and max_abs_diff <= prob_tol else "WARN"
    print(f"\n== compare: {name_a} vs {name_b} ==")
    print(f"top-1: {name_a}={top1_a} {name_b}={top1_b} -> {'MATCH' if top1_match else 'MISMATCH'}")
    print(f"top-5 set: {'MATCH' if top5_set_match else 'MISMATCH'} ({name_a}={sorted(top5_a)} {name_b}={sorted(top5_b)})")
    print(f"shared tokens in both top-10: {len(shared)}/10")
    print(f"max abs prob diff on shared tokens: {max_abs_diff:.6f} (tol={prob_tol})")
    print(f"VERDICT: {verdict}")
    return {
        "top1_match": top1_match, "top5_set_match": top5_set_match,
        "shared_count": len(shared), "max_abs_prob_diff": max_abs_diff, "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="flat text-only .pth from run_qwen35_make_pth.py")
    ap.add_argument("--hf-dir", required=True, help="original HF checkpoint dir (tokenizer + mlx-lm source)")
    ap.add_argument("--sglang-url", default=None, help="e.g. http://host:30070 (native /generate API)")
    ap.add_argument("--probe", default=PROBE_TEXT)
    ap.add_argument("--out", default=None, help="write full JSON result here")
    args = ap.parse_args()

    result = {"probe": args.probe, "pth": args.pth, "hf_dir": args.hf_dir}

    print("### 1/3: numpy fp32 reference ###")
    llm = Qwen35(args.pth, args.hf_dir)
    tokens, np_rows = llm.report(args.probe)
    result["probe_tokens"] = tokens
    result["numpy_top10"] = np_rows

    print("\n### 2/3: mlx-lm live (bf16) ###")
    _, mlx_rows = mlx_probe.report(args.hf_dir, args.probe)
    result["mlx_top10"] = mlx_rows
    result["numpy_vs_mlx"] = compare("numpy_fp32", np_rows, "mlx_bf16", mlx_rows)

    if args.sglang_url:
        print(f"\n### 3/3: sglang live ({args.sglang_url}) ###")
        try:
            import urllib.request
            payload = json.dumps({
                "input_ids": tokens,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_logprob": True, "top_logprobs_num": 10, "logprob_start_len": 0,
            }).encode()
            req = urllib.request.Request(
                args.sglang_url.rstrip("/") + "/generate", data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            top = data["meta_info"]["output_top_logprobs"][0]
            sgl_rows = [(tid, lp, float(np.exp(lp))) for lp, tid, _ in top]
            result["sglang_top10"] = sgl_rows
            result["numpy_vs_sglang"] = compare("numpy_fp32", np_rows, "sglang_bf16", sgl_rows)
        except Exception as e:
            print(f"sglang leg skipped/failed: {e!r}")
            result["sglang_error"] = repr(e)
    else:
        print("\n### 3/3: sglang live -- skipped (no --sglang-url given) ###")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")

    ok = result["numpy_vs_mlx"]["verdict"] == "PASS"
    if "numpy_vs_sglang" in result:
        ok = ok and result["numpy_vs_sglang"]["verdict"] == "PASS"
    print(f"\nGATE_QWEN35_{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
