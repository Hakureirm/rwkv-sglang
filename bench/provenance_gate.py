#!/usr/bin/env python3
"""Which tree produced this number? Refuse to guess.

`sglang_mainline/README.md` has said since it was written that measuring one
artifact while publishing another is the defect. It said so because it had
already happened once. On 2026-08-09 it happened twice more in a single
afternoon -- BENCHMARKS §6c was measured on the port-patched `sglang_overlay/`
line and read as if it were this project's stack, and a runtime notice written
to fix that mistake quoted the same misattributed number. A paragraph telling
people to be careful did not stop the person who wrote the paragraph.

So this is a gate rather than a note. Point it at the `models/rwkv7.py` that a
box actually loaded, and it says which tree in this repository that file came
from -- or that it matches none of them. `--require mainline` makes a mismatch
an exit code, which is the only form of "be careful" that survives a hurried
afternoon.

    # what is the box running?
    docker exec $C cat /sgl-workspace/sglang/python/sglang/srt/models/rwkv7.py > /tmp/live.py
    python bench/provenance_gate.py /tmp/live.py

    # refuse to benchmark unless it is the tree we publish
    python bench/provenance_gate.py /tmp/live.py --require mainline || exit 1

    # stamp to embed next to a result
    python bench/provenance_gate.py /tmp/live.py --json

Matching is by content hash first (exact), then by a feature fingerprint, so a
tree carried onto a new base by a port patch -- edited, therefore never
hash-equal -- is still identified rather than reported as "unknown". The
features are the kernel modules the file wires in: that is what actually
differs between the lines here, and it is what a reader of a benchmark number
needs, since a tree missing `mega` is missing the whole F0063-F0066c ladder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# Every candidate this repository carries. `sglang_overlay/` is retired but stays
# in the table on purpose: the point of the gate is to name what a box is running,
# and a retired tree is exactly the answer that has been missed twice.
CANDIDATES = {
    "mainline": REPO / "sglang_mainline/srt/models/rwkv7.py",
    "overlay(retired)": REPO / "sglang_overlay/sglang/srt/models/rwkv7.py",
}

# The kernel modules whose presence separates the lines. Counted, not just
# present/absent: a tree that mentions `mega` once in a comment is not a tree
# that wires the megakernel in.
FEATURES = ("mega", "glue", "fused_lora", "w8a8", "w4", "sparse_cmix", "fast_linear")


def features(text: str) -> dict[str, int]:
    return {f: len(re.findall(re.escape(f), text)) for f in FEATURES}


def distance(a: dict[str, int], b: dict[str, int]) -> float:
    """Relative feature distance, so a big file does not dominate a small one."""
    total = 0.0
    for f in FEATURES:
        hi = max(a[f], b[f], 1)
        total += abs(a[f] - b[f]) / hi
    return total / len(FEATURES)


def describe(path: pathlib.Path) -> dict:
    text = path.read_text(errors="replace")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "lines": text.count("\n") + 1,
        "features": features(text),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("deployed", help="the models/rwkv7.py the box actually loaded")
    ap.add_argument(
        "--require",
        choices=sorted(CANDIDATES),
        help="exit non-zero unless the deployed file is this tree",
    )
    ap.add_argument("--json", action="store_true", help="emit the stamp only")
    args = ap.parse_args()

    dep_path = pathlib.Path(args.deployed)
    if not dep_path.is_file():
        print(f"REFUSED: {dep_path} is not a file", file=sys.stderr)
        return 2
    dep = describe(dep_path)

    available = {n: p for n, p in CANDIDATES.items() if p.is_file()}
    if not available:
        # Refusing beats reporting "unknown": with no candidate to compare against,
        # a verdict here would carry no information at all.
        print("REFUSED: no candidate tree found in this repo", file=sys.stderr)
        return 2

    rows = []
    for name, path in available.items():
        cand = describe(path)
        rows.append(
            {
                "tree": name,
                "exact": cand["sha256"] == dep["sha256"],
                "distance": round(distance(dep["features"], cand["features"]), 4),
                "lines": cand["lines"],
                "features": cand["features"],
            }
        )
    rows.sort(key=lambda r: (not r["exact"], r["distance"]))
    best = rows[0]
    # Only an exact hash match establishes which tree this is. Anything else is
    # "nearest of what was on disk to compare against", and saying more than that
    # would be inventing lineage: the A800's deployment descends from the retired
    # overlay, but that tree's model file is not checked in here, so the nearest
    # candidate is mainline and reporting "derived from mainline" would be false.
    verdict = "EXACT" if best["exact"] else "NOT-EXACT"
    stamp = {
        "deployed": dep,
        "verdict": verdict,
        "matched_tree": best["tree"] if verdict == "EXACT" else None,
        "nearest_tree": best["tree"],
        "nearest_distance": best["distance"],
        "candidates_compared": sorted(available),
        "candidates_missing": sorted(set(CANDIDATES) - set(available)),
        "candidates": rows,
    }
    if args.json:
        print(json.dumps(stamp, indent=2))
    else:
        print(f"deployed : {dep['path']}")
        print(f"           sha256 {dep['sha256']}  lines {dep['lines']}")
        print(f"           {dep['features']}")
        for r in rows:
            mark = "<<<" if r is best else "   "
            print(f"  {mark} {r['tree']:18} exact={str(r['exact']):5} dist={r['distance']:.4f} lines={r['lines']}")
        if verdict == "EXACT":
            print(f"VERDICT  : EXACT -> {best['tree']}")
        else:
            print(f"VERDICT  : NOT-EXACT (nearest of {sorted(available)}: "
                  f"{best['tree']}, dist {best['distance']:.4f})")
            print("           nearest != lineage. This says only that no tree on disk "
                  "here matches; it does not identify what the box is running.")
        if stamp["candidates_missing"]:
            print(f"NOTE     : not on disk to compare against: "
                  f"{stamp['candidates_missing']}")
        missing = [f for f in FEATURES if dep["features"][f] == 0 and any(r["features"][f] > 0 for r in rows)]
        if missing:
            # Absent is not the same as load-bearing. `mega` is the worked example:
            # it is opt-in (`RWKV_MEGA` defaults to 0, `serve.sh` never sets it) and
            # scoped to fp16 bsz1 decode, so a tree without it produces the same
            # numbers as a tree with it for any run that did not turn it on. Report
            # the fact; let the reader check the run's env before blaming it.
            print(f"MISSING  : the deployed tree does not wire in {missing}. "
                  f"Whether that changed a number depends on whether the run enabled "
                  f"them -- check the recipe's env before attributing anything to this.")

    if args.require:
        ok = best["tree"] == args.require and verdict == "EXACT"
        if not ok:
            print(
                f"GATE FAILED: required '{args.require}' exactly, got "
                f"{verdict}" + (f" of '{best['tree']}'" if best["tree"] else ""),
                file=sys.stderr,
            )
            return 1
        print(f"GATE PASSED: deployed tree is {args.require}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
