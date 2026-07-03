---
doc_kind: finding
finding_id: F0022
title: "State prefix cache (req#3): RWKV-7 routed through sglang's state-aware MambaRadixCache — greedy-EXACT on shared-prefix batches (0.1B+1.5B) where the plain token radix corrupted, and non-zero cache hits (~30% cached tokens on a shared-prefix serving load, was 0 with radix off)"
last_verified_commit: "HEAD"
discovered_by: lead (M12), 2026-07-03
severity: info
status: open
related: [F0008]
---

# Finding F0022: state prefix cache via MambaRadixCache

## Why this was open (F0008 recap)
RWKV-7's per-request state is O(1) and **not token-addressable**, so sglang's default
**token** radix cache corrupts shared-prefix batches (F0008 / `radix_correctness.md`:
`--radix-on` gave OVERALL FAIL). We had force-disabled radix → **cache hit rate = 0**,
which fails BlinkDL's requirement #3 ("state cache, with a reasonable cache hit rate").

## What was done (reuse, not new machinery)
sglang has a **state-aware** `MambaRadixCache` (landed for GDN/Mamba/Kimi): it snapshots
the recurrent state at prefix boundaries and restores it for a matching prefix — exactly
right for an RNN. RWKV-7's state (2 conv token-shift + 1 temporal WKV) is natively
`MambaPool`-shaped, so no new checkpoint pool / copy hook / kernel is needed. Two edits:
1. `server_args.py`: RWKV-7 → `_handle_mamba_radix_cache(support_mamba_cache=True,
   support_mamba_cache_extra_buffer=False)` (was `=False`, which set `disable_radix_cache`).
2. `scripts/deploy.sh`: idempotent 1-line patch teaching scheduler `is_hybrid_ssm` about
   `rwkv7_config` (else it builds a plain, RNN-incorrect RadixCache). Kept out of the
   full overlay because scheduler.py is huge + churns; same intent as the main-port patch.

Attention-backend routing is untouched (we do NOT make `mamba2_config` truthy — that
would hijack RWKV-7 to the mamba2 backend at attention_registry).

## Correctness gate (the F0008 case, flipped)
`bench/verify_batch.py --radix-on` (radix ON = MambaRadixCache), greedy vs the numpy oracle:

| model | IDENTICAL | SHARED-PREFIX | MIXED | overall |
|---|---|---|---|---|
| 0.1B bf16 | 4/4 EXACT | 5/5 EXACT | 6/6 EXACT | **PASS** |
| 1.5B bf16 | 4/4 EXACT | 5/5 EXACT | 6/6 EXACT | **PASS** |

The exact shared-prefix case that was `OVERALL FAIL` with the plain token radix is now
`PASS` — proof the state-aware cache is correct (`disable_radix_cache=False`, no
"Disabling Radix" warning, `mamba_scheduler_strategy=no_buffer`, page_size=1, triton).

## Cache hit rate (req#3's literal ask)
1.5B server (radix ON, cuda-graph OFF), `bench_serving --dataset-name generated-shared-prefix`
(4 groups × 8 prompts, 512-token shared system prompt). Scheduler "Prefill batch" lines show
**non-zero, growing `#cached-token`** (538 → 1076 → 1574 across batches) — the state cache is
hitting. Aggregate over the visible prefill batches ≈ **30% cached tokens**
(3744 cached / 12429 total), vs **0% with radix off**. Hit rate scales with prefix sharing
(longer/ more-shared system prompts → higher). RWKV-7's serving win from this is in
**prefill/TTFT** (decode is already O(1)/token). Metric: `#cached-token` in the Prefill-batch
log; Prometheus `sglang:cache_hit_rate` when `--enable-metrics`.

## Notes / follow-ups
- Serving needs `--disable-piecewise-cuda-graph` for now (a first attempt OOM-killed during
  piecewise-graph capture with the mamba pool; graphs-off serves fine). cuda-graph + mamba
  radix co-existence is a follow-up.
- `no_buffer` strategy (extra-buffer off) is the conservative start; extra-buffer / chunk-
  boundary tuning could raise hit rate further.

## Cross-references
[[F0008]] (why plain radix corrupts) · `bench/verify_batch.py --radix-on` ·
`scripts/deploy.sh` (is_hybrid_ssm patch) · `docs/design/state-prefix-cache.md`.
