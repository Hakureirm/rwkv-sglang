#!/usr/bin/env python3
"""Compare MATH500 arms with a confidence interval that respects the sampling design.

Why this exists: `rollout_accuracy` in the summary JSON is a point estimate over
problems x rollouts, and the naive binomial error bar on it is wrong — rollouts of the
SAME problem are strongly correlated (a problem the model can do, it does most of the
time; one it can't, it never does). Treating 500x8 as 4000 independent trials
understates the true error by roughly the square root of the rollout count, which is
exactly the regime where two quantization arms look separated and are not.

So resample **problems** with replacement, not generations, and carry every rollout of
a resampled problem along with it (a cluster bootstrap). That is the unit the design
actually randomizes over.

  python bench/math500_compare.py --baseline NONE=out/NONE.math500_generations.jsonl \
      --arm L0=out/L0.math500_generations.jsonl --arm ALL4=out/ALL4.math500_generations.jsonl

Reports each arm's accuracy, its 95% CI, and the paired difference against the baseline
with a CI. A difference whose CI spans zero has not been shown, however pretty the
point estimates look side by side.
"""
import argparse, collections, json, random, sys


def load(path, label):
    """task_index -> list of 0/1. Loud on anything that would silently become 'no data'."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{label}: {path}:{lineno} is not JSON ({exc})")
    if not rows:
        raise SystemExit(f"{label}: {path} has no generations — refusing to report 0.0000 "
                         f"as if it were a measurement")
    missing = [k for k in ("task_index", "correct") if k not in rows[0]]
    if missing:
        raise SystemExit(f"{label}: {path} rows lack {missing}; this file was probably "
                         f"written before grading ran")
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task_index"]].append(1 if r["correct"] else 0)
    return by_task


def acc(by_task, tasks):
    hit = tot = 0
    for t in tasks:
        v = by_task[t]
        hit += sum(v)
        tot += len(v)
    return hit / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="LABEL=path.jsonl")
    ap.add_argument("--arm", action="append", default=[], help="LABEL=path.jsonl (repeatable)")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    def split(spec):
        if "=" not in spec:
            raise SystemExit(f"expected LABEL=path, got {spec!r}")
        lab, path = spec.split("=", 1)
        return lab, load(path, lab)

    base_label, base = split(args.baseline)
    arms = [split(s) for s in args.arm]

    # Only problems every arm actually ran are comparable; a paired test on a ragged set
    # silently compares different problem mixes.
    common = set(base)
    for _, d in arms:
        common &= set(d)
    if not common:
        raise SystemExit("arms share no problems — nothing to compare")
    for lab, d in [(base_label, base)] + arms:
        if len(d) != len(common):
            print(f"note: {lab} has {len(d)} problems, comparing on the {len(common)} shared",
                  file=sys.stderr)
    tasks = sorted(common)

    rng = random.Random(args.seed)
    draws = [[tasks[rng.randrange(len(tasks))] for _ in tasks] for _ in range(args.reps)]

    def ci(vals):
        v = sorted(vals)
        return v[int(0.025 * len(v))], v[int(0.975 * len(v))]

    print(f"{len(tasks)} problems, {args.reps} cluster-bootstrap resamples over problems\n")
    base_pt = acc(base, tasks)
    base_boot = [acc(base, d) for d in draws]
    lo, hi = ci(base_boot)
    print(f"{base_label:8s} acc={base_pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]   (baseline)")

    for lab, d in arms:
        pt = acc(d, tasks)
        diffs = [acc(d, dr) - acc(base, dr) for dr in draws]
        dlo, dhi = ci(diffs)
        boot = [acc(d, dr) for dr in draws]
        alo, ahi = ci(boot)
        verdict = "SEPARATED" if (dlo > 0 or dhi < 0) else "not separated"
        print(f"{lab:8s} acc={pt:.4f}  95% CI [{alo:.4f}, {ahi:.4f}]   "
              f"vs {base_label}: {pt - base_pt:+.4f}  95% CI [{dlo:+.4f}, {dhi:+.4f}]  {verdict}")


if __name__ == "__main__":
    main()
