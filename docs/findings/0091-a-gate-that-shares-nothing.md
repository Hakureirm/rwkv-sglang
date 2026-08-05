---
finding_id: F0091
title: "Every correctness gate here compared our code to our code; this one compares it to BlinkDL's numpy forward, on prompts nobody chose — 24/24 at four concurrencies"
date: 2026-08-06
status: closed
severity: medium
---

# F0091 — A gate that shares nothing with what it tests

**Date:** 2026-08-06 · **Card:** RTX 5090 · **Model:** rwkv7-1.5b

## Why

Prompted by a one-line criticism that lands: *you cannot write your own tests for
your own code.* Reviewing the day's gates against it, the split is real:

| gate | reference written by | independent? |
|---|---|---|
| `verify_batch.py` + numpy fixture | pre-existing, independent implementation | **yes** |
| `test_lora_mn.py` (used for F0088) | pre-existing; mn vs m1 | **yes** |
| btlqql's `bench_acceptance_matrix.py` | third party | **yes** |
| `test_vresgate_mn.py` (F0087) | **written today, with the change** | reference is the pre-change torch chain, but the shapes and dtypes tested are the author's choice |
| the transformers NaN regression test (F0085) | **written today, with the change** | sabotage-checked, but what it asserts is the author's choice |
| `lora_stage_split.py`'s cuBLAS column (F0088) | **written today** | **it was wrong** |

The last row is the criticism cashing out. That reference was non-monotonic in M,
a ratio taken against it said the batch gate could be widened, and the end-to-end
sweep said the opposite. What caught it was a measurement the author did not
write, not a test.

The structural weakness is not dishonesty, it is coverage: **a test written by the
author of a change tests the cases its author thought of**, and those are exactly
the cases the change already handles.

## The gate

`bench/xcheck_numpy_vs_server.sh`. Nothing in the loop is shared with the code
under test:

- **reference**: `bench/oracle_numpy.py`, a faithful port of BlinkDL's own
  RWKV-v7 numpy forward, invoked through its CLI **unmodified**. Editing it to
  make this pass would end its status as a reference, which the script says in
  its header.
- **prompts**: `lambada_test.parquet` at a fixed stride (24 prompts, every 20th
  row), so the cases are not chosen by whoever wrote the change.
- **coverage**: concurrency 1, 4, 8 and 16, so `T == 1`, `2 <= T <= gate` and
  `T > gate` — the three branches today's changes touched — are each compared to
  the same single-stream ground truth.

Result on the deployed tree, carrying F0086 + F0087 + F0088:

```
 concurrency    match   of
           1       24   24
           4       24   24
           8       24   24
          16       24   24
RESULT: ALL MATCH
```

## The gate refuses

Run unchanged against the int4 model instead — a genuinely different
implementation of the same weights — it reports **84 mismatches of 96 and exits
1**. Three prompts still agree, which is the right amount of agreement to expect
from a lower-precision model on short greedy continuations, and is why a gate
like this has to be run against a known-different artifact before its green is
worth anything.

## Three harness failures on the way to one green light

Every one of them was in code written for this check, not in the model:

1. **Wrong tree.** The first run had neither `PYTHONPATH` to the deployed sglang
   nor the production flags, so it launched the container's own sglang. That
   failed loudly — `model type rwkv7` unrecognised — but the quiet version of the
   same mistake is a green gate against the wrong code. The script now prints
   which tree it loaded and how many `RWKV_*` flags are set, before doing
   anything.
2. **96 mismatches that were quoting.** `oracle_numpy.py` prints its completion as
   a Python repr; the capture stored the repr verbatim and compared it to raw
   text. Every row "failed".
3. **`RWKV_W4` unset** (earlier, F0090): `scripts/serve.sh` does not export it,
   so the int4 server died at startup and the summary read as "the int4 rows are
   missing" rather than "the int4 server never ran".

Three attempts, three bugs, all mine, none in the thing being tested. That ratio
is the argument for keeping the reference external: the harness is as likely to be
wrong as the code, and only an external reference makes the two distinguishable.

## Standing use

Run it after any change to the decode path, before the tok/s tables:

```
PTH=<rwkv7-*.pth> SERVED=<served-model-dir> PARQUET=<lambada_test.parquet> \
SGLANG=<sglang-checkout>/python bash bench/xcheck_numpy_vs_server.sh
```

The reference pass costs ~25 minutes at 1.5B (numpy, fp32, CPU) and is cached in
`$OUT/reference.json`; the server pass is under a minute.
