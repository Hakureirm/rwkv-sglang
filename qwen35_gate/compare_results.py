#!/usr/bin/env python3
"""Standalone reimplementation of gate_qwen35.py's compare() logic, for combining
two results captured on separate machines (F0054: numpy leg on the 5090 tower,
sglang leg on the 3090 box, since the 3090 box lacks the RAM/disk for the fp32
conversion step). Same tolerance policy as F0050/gate_qwen35.py: top-1 exact
match, top-5 token-SET exact match, max abs prob diff on shared tokens <= 0.02.
"""
import json
import sys


def rows_to_dict(rows):
    return {t: p for t, _, p in rows}


def compare(name_a, rows_a, name_b, rows_b, prob_tol=0.02):
    a, b = rows_to_dict(rows_a), rows_to_dict(rows_b)
    top1_a, top1_b = rows_a[0][0], rows_b[0][0]
    top1_match = top1_a == top1_b
    top5_a, top5_b = set(list(a)[:5]), set(list(b)[:5])
    top5_set_match = top5_a == top5_b
    shared = set(a) & set(b)
    max_abs_diff = max((abs(a[t] - b[t]) for t in shared), default=float("nan"))
    verdict = "PASS" if top1_match and top5_set_match and max_abs_diff <= prob_tol else "WARN/FAIL"
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


if __name__ == "__main__":
    # numpy_json: {"tokens": [...], "numpy_top10": [[tok, logit, prob], ...]}
    # sglang_json: {"tokens": [...], "sglang_top10": [[tok, logp, prob], ...]}
    numpy_path, sglang_path = sys.argv[1], sys.argv[2]
    with open(numpy_path) as f:
        npy = json.load(f)
    with open(sglang_path) as f:
        sgl = json.load(f)
    assert npy["tokens"] == sgl["tokens"], f"TOKEN MISMATCH: numpy={npy['tokens']} sglang={sgl['tokens']}"
    result = compare("numpy_fp32", npy["numpy_top10"], "sglang_bf16", sgl["sglang_top10"])
    out = {**result, "probe_tokens": npy["tokens"], "numpy_top10": npy["numpy_top10"], "sglang_top10": sgl["sglang_top10"]}
    with open("combined_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nGATE_QWEN35_9B_{'PASS' if result['verdict'] == 'PASS' else 'FAIL'}")
