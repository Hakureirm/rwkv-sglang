"""Speed A/B for RWKV_SPEC: real measured tok/s, spec-on vs spec-off.

bench/spec_gate.py already proves correctness (spec-on == spec-off token-
identical); it does not measure speed. This script measures speed only,
against a SINGLE already-running server (either plain or RWKV_SPEC -- run it
once per server, same as spec_gate.py's two-phase pattern, since two sglang
servers can't cleanly share one GPU).

tok/s = meta_info["completion_tokens"] / meta_info["e2e_latency"] (the
server's own measured latency, not a client-side wall clock -- avoids HTTP
round-trip noise). One untimed warmup request first (cuda graph capture /
lazy setup should not count against the first real prompt).

Usage:
  python bench/spec_speed.py --port 30000 --gen-len 128 --tag "spec-off"
  python bench/spec_speed.py --port 30000 --gen-len 128 --tag "spec-on"
"""
import argparse
import json
import statistics

import requests

PROMPTS = [
    ("capital of France (high accept)", "User: What is the capital of France?\n\nAssistant:"),
    ("haiku about autumn (creative)", "User: Write a haiku about autumn.\n\nAssistant:"),
    ("why sky is blue (explain)", "User: Explain why the sky is blue in one sentence.\n\nAssistant:"),
    ("list three primes (medium)", "User: List three prime numbers.\n\nAssistant:"),
    ("translate good morning (short)", "User: Translate 'good morning' to Spanish.\n\nAssistant:"),
    ("largest planet (factual)", "User: Name the largest planet in the solar system.\n\nAssistant:"),
    ("continue sequence (pattern)", "User: Continue the sequence: 2, 4, 8, 16,\n\nAssistant:"),
]


def _post(sess, url, body):
    r = sess.post(url, json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d[0] if isinstance(d, list) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--tag", required=True, help="label for this run, e.g. spec-on / spec-off")
    ap.add_argument("--dump", default=None, help="optional path to append JSON results")
    a = ap.parse_args()
    sess = requests.Session()
    url = f"http://{a.host}:{a.port}/generate"

    # untimed warmup (cuda graph capture / lazy setup shouldn't count)
    _post(sess, url, {"text": PROMPTS[0][1], "sampling_params": {"temperature": 0.0, "max_new_tokens": 16}})

    rows = []
    for name, p in PROMPTS:
        rec = _post(
            sess, url,
            {"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": a.gen_len}},
        )
        mi = rec["meta_info"]
        n = mi["completion_tokens"]
        lat = mi["e2e_latency"]
        tps = n / lat if lat > 0 else float("nan")
        accept = mi.get("spec_accept_length") or mi.get("accept_length")
        rows.append({"name": name, "prompt": p, "n": n, "latency": lat, "tps": tps, "accept": accept})
        acc_s = f"  accept={accept:.2f}" if accept is not None else ""
        print(f"  [{a.tag}] {name:35s} n={n:4d}  latency={lat:6.3f}s  {tps:7.1f} tok/s{acc_s}", flush=True)

    tps_vals = [r["tps"] for r in rows]
    print(f"\n=== {a.tag}: median {statistics.median(tps_vals):.1f} tok/s  "
          f"mean {statistics.fmean(tps_vals):.1f} tok/s  over {len(rows)} prompts (gen_len={a.gen_len}) ===")

    if a.dump:
        try:
            with open(a.dump) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        data[a.tag] = rows
        with open(a.dump, "w") as f:
            json.dump(data, f, indent=2)
        print(f"appended -> {a.dump}")


if __name__ == "__main__":
    main()
