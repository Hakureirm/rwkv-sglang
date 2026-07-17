#!/usr/bin/env python3
"""#50 flagship framing-2: decode-step GPU timing from a torch-profiler trace.

Parses the chrome trace (json/.gz) sglang's /start_profile writes during a c=1
decode run. Empirical structure on this stack (7.2B bsz1, overlap scheduler):
the GPU timeline is CONTIGUOUS — there is no inter-step idle to cluster on
(p99 inter-kernel gap ~0.3 us) — so steps are COUNTED, not clustered:
  steps = #cudaGraphLaunch runtime events (fallback: wkv_decode kernels / L).

Reports:
  * span/step = kernel-window wall / steps  -> tok/s (the matched kernel-loop
    framing; with zero inter-step idle this equals the serving decode rate);
  * busy/step (sum kernel durs / steps) and gap/step (span - busy);
  * the inter-kernel gap distribution INCLUDING NEGATIVE gaps — same-stream
    consecutive-kernel overlap is the direct PDL signature (impossible without
    programmatic launch on one stream): overlap share + gained us/step;
  * per-kernel table (count/step, us/step) — the F0060 §2 analog on sm120.

Usage:
  python3 bench/step_span_from_trace.py TRACE.json[.gz] [--layers 32] [--top 24]
"""
import argparse
import gzip
import json
from collections import defaultdict


def load_events(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--layers", type=int, default=32,
                    help="decoder layers (fallback step counter)")
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--trim", type=float, default=0.04,
                    help="fraction of the window trimmed at each edge "
                         "(drops partial first/last steps)")
    args = ap.parse_args()

    evs = load_events(args.trace)
    ks = [e for e in evs
          if e.get("ph") == "X" and "kernel" in e.get("cat", "").lower()]
    assert ks, "no GPU kernel events in trace"
    ks.sort(key=lambda e: e["ts"])

    # trim partial steps at the window edges
    w0 = ks[0]["ts"]
    w1 = max(e["ts"] + e["dur"] for e in ks)
    lo = w0 + (w1 - w0) * args.trim
    hi = w1 - (w1 - w0) * args.trim
    ks = [e for e in ks if e["ts"] >= lo and (e["ts"] + e["dur"]) <= hi]

    t0 = ks[0]["ts"]
    t1 = max(e["ts"] + e["dur"] for e in ks)
    span = t1 - t0

    # step count: cudaGraphLaunch in the same window, else wkv/L, else lerp6/L
    graph_launches = [e for e in evs
                      if e.get("ph") == "X"
                      and e.get("cat") == "cuda_runtime"
                      and "GraphLaunch" in e.get("name", "")
                      and lo <= e["ts"] <= hi]
    n_steps = len(graph_launches)
    src = "cudaGraphLaunch"
    if n_steps < 4:
        wkv = sum(1 for e in ks if "wkv_decode" in e.get("name", ""))
        n_steps = round(wkv / args.layers)
        src = f"wkv_decode/{args.layers}"
    if n_steps < 4:
        sl6 = sum(1 for e in ks if "shift_lerp6" in e.get("name", ""))
        n_steps = round(sl6 / args.layers)
        src = f"shift_lerp6/{args.layers}"
    assert n_steps >= 4, "could not count steps"

    busy = sum(e["dur"] for e in ks)
    # inter-kernel transitions (sorted by start; same GPU, one serving stream)
    gaps = []
    for a, b in zip(ks, ks[1:]):
        gaps.append(b["ts"] - (a["ts"] + a["dur"]))
    pos = sum(g for g in gaps if g > 0)
    neg = -sum(g for g in gaps if g < -0.02)  # overlap gained (us)
    n_over = sum(1 for g in gaps if g < -0.02)
    gs = sorted(gaps)

    def pct(v, p):
        return v[min(len(v) - 1, int(p * len(v)))]

    span_step = span / n_steps
    print(f"window {span/1e3:.2f} ms, steps={n_steps} (by {src}), "
          f"kernels/step={len(ks)/n_steps:.1f}")
    print(f"SPAN/step {span_step:.1f} us -> {1e6/span_step:.2f} tok/s "
          f"(kernel-loop framing)")
    print(f"BUSY/step {busy/n_steps:.1f} us   gap/step "
          f"{(span-busy)/n_steps:+.1f} us (pos {pos/n_steps:.1f}, "
          f"overlap-gained {neg/n_steps:.1f})")
    print(f"inter-kernel gap us: p10 {pct(gs,0.10):+.3f}  p50 "
          f"{pct(gs,0.50):+.3f}  p90 {pct(gs,0.90):+.3f}  p99 "
          f"{pct(gs,0.99):+.3f}   overlapped transitions: "
          f"{n_over/len(gaps)*100:.1f}%")

    agg = defaultdict(lambda: [0, 0.0])
    for e in ks:
        a = agg[e.get("name", "?")]
        a[0] += 1
        a[1] += e["dur"]
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
    print(f"\nper-kernel (per step over {n_steps}): count  us  name")
    for name, (c, d) in rows[: args.top]:
        print(f"  {c/n_steps:7.2f}  {d/n_steps:9.2f}  {name[:105]}")


if __name__ == "__main__":
    main()
