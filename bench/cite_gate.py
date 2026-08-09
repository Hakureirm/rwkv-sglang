#!/usr/bin/env python3
"""Every published number should be findable in a raw artifact. Check it.

On 2026-08-09 BENCHMARKS §6c was found to publish `369.6` as the fp16
single-stream figure. The sweep backing every other cell in that table reads
`336.2`; `369.6` appears in no artifact on the box, in no log, and nowhere in
this repository. It survived review because the section *looked* sourced -- it
cited six raw JSON files by name. Five of those six were not in the repository
either: they existed only in a container's /tmp.

Two failure modes, one check. Point it at a section heading:

    python bench/cite_gate.py --section "6c"
    python bench/cite_gate.py --section "6c" --strict      # exit 1 on either

It reports:

  MISSING   -- a raw file the section cites that is not in bench/results/
  UNBACKED  -- a number printed in the section that appears in none of the
               artifacts that section cites

UNBACKED is advisory by nature and will list derived values -- percentages,
ratios, deltas, dates, section numbers -- because those are computed from the
artifacts rather than stored in them. That is why the report prints them for a
human to clear rather than failing on each one. What it is for is the case that
is not derived and not present either: a headline throughput or accuracy figure
with nothing behind it. `369.6` is exactly that, and this catches it.

Numbers below MIN_MAGNITUDE and inside `--ignore` are skipped; the defaults are
tuned so that a run over the current document is short enough to actually read.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench/results"
DOCS = [REPO / "docs/BENCHMARKS.md"]

# Below this, a number is far more likely to be a version, a count, a section
# reference or a tolerance than a measurement worth tracing.
MIN_MAGNITUDE = 100.0


def load_numbers(path: pathlib.Path) -> set[float]:
    """Every numeric leaf in a JSON artifact, flattened."""
    out: set[float] = set()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            out.add(float(node))

    try:
        walk(json.loads(path.read_text()))
    except Exception:
        # A citation can point at a directory or a .log; not being able to read
        # numbers out of it is not the same as the file being absent, so this is
        # silent here and the file still counts as present.
        pass
    return out


def expand_citations(sec: str) -> set[str]:
    """The citations as this document actually writes them, not as one wishes it did.

    BENCHMARKS.md uses three forms, and a checker that only understands the first
    is worse than useless -- it invents a missing file for the second and stays
    silent about the third:

      `bsz_sweep_fullstack_5090.json`              plain
      `..._cliff_stage1_w4a8.json`                 elided common prefix
      `..._cliff_stage1_{base,w4a8}.json`          brace set, and `{,_fine}`

    The elided form is resolved by suffix against what is on disk, which is the
    only thing it can mean. Brace sets are expanded before anything else, so the
    members are checked individually rather than skipped -- the previous regex
    did not admit `{` at all, so those citations were never checked and a genuine
    absence there would have passed silently.
    """
    def expand_braces(s: str) -> list[str]:
        # Recursive, because the document nests them:
        # `bsz_sweep_{1.5b,7.2b}_{fp16,w4gptq,w4rtn}_3090.json` is six files, and
        # expanding only the leftmost group leaves `{` in the name, which then
        # reports as a missing file that was never cited. That false positive is
        # worse than the gap it replaced -- a checker that cries wolf gets muted.
        m = re.search(r"\{([^{}]*)\}", s)
        if not m:
            return [s]
        return [
            out
            for part in m.group(1).split(",")
            for out in expand_braces(s[: m.start()] + part.strip() + s[m.end():])
        ]

    out: set[str] = set()
    for raw in re.findall(r"`([A-Za-z0-9_./{},-]+\.json)`", sec):
        for name in expand_braces(raw):
            if name.startswith("..."):
                tail = name[3:]
                hits = [p.name for p in RESULTS.rglob(f"*{tail}")]
                # An elided citation that resolves to exactly one file on disk is
                # that file. Zero or several, and the shorthand is genuinely
                # ambiguous -- keep it verbatim so it is reported rather than
                # quietly resolved to a guess.
                out.add(hits[0] if len(hits) == 1 else name)
            else:
                out.add(name)
    return out


def section_text(doc: pathlib.Path, section: str) -> str | None:
    text = doc.read_text()
    m = re.search(rf"^##+\s*{re.escape(section)}\b.*?$", text, re.M)
    if not m:
        return None
    start = m.start()
    nxt = re.search(r"^##\s", text[m.end():], re.M)
    return text[start:m.end() + nxt.start()] if nxt else text[start:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True, help="heading to check, e.g. 6c")
    ap.add_argument("--strict", action="store_true", help="exit 1 on MISSING or UNBACKED")
    ap.add_argument("--ignore", default="", help="comma-separated numbers to skip")
    # So the gate can be run against an arbitrary copy of the document -- which is
    # how it gets tested: point it at a past revision (`git show REV:path`) that is
    # known to have carried an unsourced number and confirm it still says so.
    ap.add_argument("--doc", help="check this file instead of the tracked BENCHMARKS.md")
    args = ap.parse_args()

    docs = [pathlib.Path(args.doc)] if args.doc else DOCS
    sec = None
    for doc in docs:
        sec = section_text(doc, args.section)
        if sec:
            break
    if sec is None:
        print(f"REFUSED: no section '{args.section}' in {[str(d) for d in docs]}", file=sys.stderr)
        return 2

    cited = sorted(expand_citations(sec))
    if not cited:
        # A section with no citations cannot be checked, and passing it silently
        # would be the worst outcome: it would read as verified.
        print(f"REFUSED: section '{args.section}' cites no raw .json artifact; "
              f"nothing to check against", file=sys.stderr)
        return 2

    present, missing = {}, []
    for name in cited:
        hits = list(RESULTS.rglob(pathlib.Path(name).name))
        if hits:
            present[name] = hits[0]
        else:
            missing.append(name)

    backing: set[float] = set()
    for p in present.values():
        backing |= load_numbers(p)

    ignore = {float(x) for x in args.ignore.split(",") if x.strip()}
    printed = {
        float(t.replace(",", ""))
        for t in re.findall(r"(?<![\w.])(\d[\d,]*\.?\d*)(?![\w.])", sec)
    }
    unbacked = sorted(
        n for n in printed
        if n >= MIN_MAGNITUDE and n not in ignore
        and not any(abs(n - b) <= max(0.05, abs(b) * 0.001) for b in backing)
    )

    print(f"section  : {args.section}")
    print(f"cited    : {len(cited)} artifact(s); {len(present)} present, {len(missing)} missing")
    for m in missing:
        print(f"  MISSING  {m}")
    if missing:
        # §4 is the worked example of a MISSING that is correct as it stands: the
        # caption cites the `{fp16,w8,w8a8,w4}_{1.5b,7.2b}` family and the very next
        # clause says "7.2B has no landed w8g64 raw -- that slot is left empty, not
        # filled". A family shorthand with a gap the prose owns is not the defect
        # this gate is looking for. Read the sentence around the citation before
        # touching anything.
        print("  (a brace-family member can be absent on purpose -- check whether the "
              "surrounding prose already declares the gap before 'fixing' it)")
    print(f"backing  : {len(backing)} distinct numeric values across present artifacts")
    if unbacked:
        print(f"UNBACKED : {len(unbacked)} printed value(s) not found in any cited artifact")
        for n in unbacked:
            print(f"  {n:g}")
        print("  (derived values -- percentages, ratios, deltas, dates -- land here by "
              "construction. Clear them by eye; what matters is a headline measurement "
              "with nothing behind it.)")
    else:
        print("UNBACKED : none")

    if args.strict and (missing or unbacked):
        print("GATE FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
