#!/usr/bin/env python3
"""Regenerate docs/FINDINGS.md from the findings' front-matter.

Edit the findings, then re-run this; never edit the table by hand. Titles come from
front-matter `title:` in any of its historical quoting styles, falling back to the
body's first heading for the early findings that predate the front-matter convention.
"""
import glob
import os
import re

rows = []
for p in sorted(glob.glob("docs/findings/*.md")):
    s = open(p, encoding="utf-8").read()[:3000]
    fid = re.search(r'finding_id:\s*["\']?(F\d+[a-z]?)', s)
    t = re.search(r'title:\s*"([^"]+)"', s, re.S) or re.search(r"title:\s*'([^']+)'", s, re.S)
    if not t:
        t = re.search(r'title:\s*>-?\s*\n\s+(.+)', s) or re.search(r'^title:\s*([^\n]+)', s, re.M)
    if not t:
        t = re.search(r'^#\s+(?:Finding\s+F?\d+[a-z]?:?\s*)?(.+)', s, re.M)
    status = re.search(r'status:\s*(\S+)', s)
    name = os.path.basename(p)
    base = name.rsplit('.md', 1)[0]
    fid_s = fid.group(1) if fid else "F" + base.split('-')[0]
    title = re.sub(r'\s+', ' ', (t.group(1).strip().rstrip('"') if t else base))
    if len(title) > 110:
        title = title[:107] + "..."
    rows.append((fid_s, title, status.group(1) if status else "open", name))

out = [
    "# Findings index",
    "",
    "Dated measurement reports, including the negative results. Each is self-contained:",
    "methodology, raw data pointers, and what was retracted when a prediction failed.",
    "Generated from front-matter by `python3 tools/gen_findings_index.py` — edit the",
    "findings, not this table.",
    "",
    "| id | finding | status |",
    "|---|---|---|",
]
for fid, title, status, name in rows:
    out.append(f"| {fid} | [{title.replace('|', chr(92) + '|')}](findings/{name}) | {status} |")
open("docs/FINDINGS.md", "w").write("\n".join(out) + "\n")
print(f"wrote docs/FINDINGS.md with {len(rows)} rows")
