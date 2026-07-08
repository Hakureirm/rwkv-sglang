#!/usr/bin/env python3
"""Query a live sglang server's /generate endpoint with a fixed token-id list and
print the top-10 next-token distribution as JSON. Mirrors the sglang leg of
qwen35_gate/gate_qwen35.py exactly (same payload shape), but standalone so it can
run directly on the box hosting the sglang server without needing the numpy
reference / torch / transformers stack also present in the same process (F0054:
the numpy-reference leg runs on a different machine for this 9B gate, since this
box does not have enough free RAM/disk for the fp32 conversion step -- see finding
doc). Token IDs are passed in explicitly (as produced by the numpy reference's own
AutoTokenizer.encode call elsewhere) rather than re-tokenized here, so both legs
are guaranteed to run forward on the identical input.
"""
import argparse
import json
import sys
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="e.g. http://127.0.0.1:30071")
    ap.add_argument("--tokens", required=True, help="JSON list of input token ids, e.g. '[242476, 300]'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tokens = json.loads(args.tokens)
    payload = json.dumps({
        "input_ids": tokens,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True, "top_logprobs_num": 10, "logprob_start_len": 0,
    }).encode()
    req = urllib.request.Request(
        args.url.rstrip("/") + "/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    top = data["meta_info"]["output_top_logprobs"][0]
    import math
    rows = [(tid, lp, math.exp(lp)) for lp, tid, _ in top]
    rows.sort(key=lambda r: -r[1])

    result = {"url": args.url, "tokens": tokens, "sglang_top10": rows}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
