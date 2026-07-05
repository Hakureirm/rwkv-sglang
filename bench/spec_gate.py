"""Hard gate + speed A/B for RWKV-7 chain speculative decoding (ADR-0006).

Greedy chain-verify is exact by construction, so the gate is absolute: a server
running --speculative-algorithm RWKV_CHAIN must emit BYTE-IDENTICAL output_ids
to the plain server for the same greedy requests. Any divergence = state
snapshot/rollback bug (not "acceptable spec-decode drift" — there is no such
thing at temperature 0 in this design).

Two phases against two sequentially-booted servers (same GPU):
  --mode baseline --port P --dump G.json   plain server: record output_ids + timing
  --mode spec     --port P --dump G.json   spec server: compare token-exact, report
                                           accept-length stats + tok/s ratio.

Timing: bsz1 sequential (spec targets single-stream first), one untimed warmup
request, then per-prompt wall clock over gen-len tokens. The A/B ratio is
decode-dominated at gen-len>=128; TTFT rides along equally in both modes.

Usage:
  python bench/spec_gate.py --mode baseline --port 30080 --dump /tmp/gate.json
  python bench/spec_gate.py --mode spec     --port 30081 --dump /tmp/gate.json
"""
import argparse, json, statistics, time
import requests

PROMPTS = [
    "User: What is the capital of France?\n\nAssistant:",
    "User: Write a haiku about autumn.\n\nAssistant:",
    "User: Explain why the sky is blue in one sentence.\n\nAssistant:",
    "User: Solve for x: 2x + 6 = 14.\n\nAssistant: <think></think",
    "User: List three prime numbers.\n\nAssistant:",
    "User: Translate 'good morning' to Spanish.\n\nAssistant:",
    "User: Who wrote Romeo and Juliet?\n\nAssistant:",
    "User: What is 15 percent of 200?\n\nAssistant: <think></think",
    "The quick brown fox",
    "def fibonacci(n):",
]


def _gen(sess, url, prompt, gen_len):
    t0 = time.perf_counter()
    r = sess.post(url, json={"text": prompt,
                             "sampling_params": {"temperature": 0.0, "max_new_tokens": gen_len}},
                  timeout=600)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    d = d[0] if isinstance(d, list) else d
    return d, dt


def run(sess, url, gen_len):
    _gen(sess, url, PROMPTS[0], 8)  # warmup
    rows = []
    for p in PROMPTS:
        d, dt = _gen(sess, url, p, gen_len)
        n = d["meta_info"]["completion_tokens"]
        rows.append({"prompt": p, "output_ids": d["output_ids"], "n": n,
                     "tps": n / dt, "meta": d["meta_info"]})
        print(f"  |out|={n:4d}  {n/dt:7.1f} tok/s  {p[:40]!r}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "spec"], required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/generate"
    sess = requests.Session()

    rows = run(sess, url, args.gen_len)
    tps = [r["tps"] for r in rows]
    print(f"pooled: median {statistics.median(tps):.1f} tok/s  "
          f"mean {statistics.fmean(tps):.1f} tok/s over {len(rows)} prompts")

    if args.mode == "baseline":
        json.dump(rows, open(args.dump, "w"))
        print(f"baseline dumped -> {args.dump}")
        return

    base = json.load(open(args.dump))
    assert len(base) == len(rows), "prompt set mismatch vs baseline dump"
    ok = True
    for b, s in zip(base, rows):
        same = b["output_ids"] == s["output_ids"]
        ok &= same
        if not same:
            ids_b, ids_s = b["output_ids"], s["output_ids"]
            div = next((i for i, (x, y) in enumerate(zip(ids_b, ids_s)) if x != y),
                       min(len(ids_b), len(ids_s)))
            print(f"  MISMATCH at pos {div} (|base|={len(ids_b)} |spec|={len(ids_s)}) "
                  f"{b['prompt'][:40]!r}")
            print(f"    base[{div}:{div+6}]={ids_b[div:div+6]}  spec[...]={ids_s[div:div+6]}")
    # accept-length stats, if the server exposes them in meta_info
    accepts = [s["meta"].get("spec_accept_length") for s in rows]
    if any(a is not None for a in accepts):
        vals = [a for a in accepts if a is not None]
        print(f"accept-length/round: mean {statistics.fmean(vals):.2f} over {len(vals)} prompts")
    med_b = statistics.median(r["tps"] for r in base)
    med_s = statistics.median(tps)
    print(f"speed: baseline {med_b:.1f} -> spec {med_s:.1f} tok/s  ({med_s/med_b:.2f}x)")
    print("GATE:", "PASS (token-identical)" if ok else "FAIL (divergence above)")
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
