# bench/data — eval datasets (provenance + how to fetch)

The GPU box has no HuggingFace/GitHub access. Fetch everything here on the Mac, then
`rsync -av bench/data/ 3090-devbox:~/rwkv-vllm/bench/data/`.

## MATH500.jsonl (COMPLETE — 500 problems, ready for full runs)

Source: HF dataset `HuggingFaceH4/MATH-500`, split `test` (the exact 500-problem set the
albatross reference `eval_math500_albatross.py` reads as `dataset/MATH500.jsonl`).
Fields per line: `problem`, `answer`, `subject`, `level`, `unique_id`.

Fetched 2026-07-03 via the HF datasets-server rows API (no auth needed):

```python
import json, urllib.request
rows = []
for off in range(0, 500, 100):
    url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
           f"&config=default&split=test&offset={off}&length=100")
    rows += [x["row"] for x in json.load(urllib.request.urlopen(url))["rows"]]
with open("MATH500.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({k: r.get(k, "") for k in
            ("problem", "answer", "subject", "level", "unique_id")}, ensure_ascii=False) + "\n")
```

(or with `datasets`: `load_dataset("HuggingFaceH4/MATH-500")["test"].to_json("MATH500.jsonl")`)

## uncheatable/ (SAMPLE ONLY — 20 docs/category; full set is 500/category)

Source: HF dataset `Jellyfish042/UncheatableEval-2026-04`, split `test`
(the current official Uncheatable Eval corpus: 15 categories x 500 fresh documents,
columns `content` (eval text) / `untruncated_content` / `category` / `date` / `url`).
Each `<category>.json` here is a JSON list of the `content` strings — exactly the local
file format the official `evaluator.py load_data_smart` accepts.

The 20-doc samples were fetched 2026-07-03 via the datasets-server rows API; rows are
stored grouped by category in alphabetical blocks of 500 (ao3_english @0,
ao3_nonenglish @500, arxiv_cs @1000, ... wikipedia_nonenglish @7000).

**Full download (run on the Mac, ~215 MB parquet / 444 MB expanded):**

```python
from datasets import load_dataset
from collections import defaultdict
import json
ds = load_dataset("Jellyfish042/UncheatableEval-2026-04")["test"]
by_cat = defaultdict(list)
for row in ds:
    by_cat[row["category"]].append(row["content"])
for cat, texts in by_cat.items():
    json.dump(texts, open(f"uncheatable/{cat}.json", "w"), ensure_ascii=False)
```

NOTE for headline numbers: use the FULL 500-doc categories, not these 20-doc samples.
The official monthly datasets are versioned (`UncheatableEval-YYYY-MM`); record which
month you evaluated.
