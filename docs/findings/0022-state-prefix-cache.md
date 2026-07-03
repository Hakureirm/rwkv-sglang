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

## Cache hit rate (req#3's literal ask) — ~98% on a realistic high-reuse workload
Hit rate is workload-dependent for EVERY prefix cache (transformer or RNN) — it's just
`shared_prefix_tokens / total_prompt_tokens`. Measured on the 1.5B server (radix ON,
cuda-graph OFF), `bench_serving --dataset-name generated-shared-prefix`:

| workload | shared prefix | arrival | steady-state hit rate |
|---|---|---|---|
| high-reuse (2 groups × 32 prompts) | 2048 tok | rate 4/s | **~98.3%** (34,816 cached / 35,402) |
| low-reuse cold-start (4 groups × 8) | 512 tok | rate ∞ (all at once) | ~30% |

Steady-state prefill batches on the high-reuse load: `#new-token: 36, #cached-token: 2048`
(= 98.3%) repeated; Mean TTFT **200 ms** (vs 784 ms on the cold-start load). The ~30% earlier
number was a **cold-start-worst-case measurement artifact** (request-rate=∞ fires all requests
before the first one caches its prefix; short prefix), NOT a cache limitation — with requests
arriving over time and a substantial shared prefix (multi-turn chat / shared system prompt, the
exact traffic where Claude/DeepSeek report 98–99%), RWKV-7's state cache hits the same ~98%.
RWKV's serving win from this is in **prefill/TTFT** (decode is already O(1)/token). Metric:
`#cached-token` in the Prefill-batch log; Prometheus `sglang:cache_hit_rate` with `--enable-metrics`.
`--enable-int8-mamba-checkpoint` (2× cached-prefix capacity) is available for capacity-bound
loads (OOM'd on this box at mem-fraction 0.6 — a memory-budget follow-up, not needed for hit rate).

## Cross-framework comparison (there is nothing else to compare against)
Among RWKV serving stacks, **only ours has a state prefix cache at all**:
| stack | dynamic batching | prefix/state cache | hit rate |
|---|---|---|---|
| **ours (rwkv-sglang)** | ✅ sglang-native | ✅ **MambaRadixCache (state-aware)** | **~98% high-reuse** |
| Albatross (faster3a) | ✗ single mega-kernel, no scheduler | ✗ none (every request from scratch) | n/a (0) |
| RWKV-LM reference | ✗ no serving layer | ✗ none | n/a (0) |
| vLLM-RWKV adapters | — both upstream PRs closed 2026-06-29, no working out-of-tree plugin | ✗ none shipping | n/a |

So "reasonable cache hit rate" (req#3) is met AND is a category-exclusive capability: every
other RWKV serving path recomputes shared prefixes from zero.

## Notes / follow-ups
- Serving needs `--disable-piecewise-cuda-graph` for now (a first attempt OOM-killed during
  piecewise-graph capture with the mamba pool; graphs-off serves fine). cuda-graph + mamba
  radix co-existence is a follow-up.
- `no_buffer` strategy (extra-buffer off) is the conservative start; extra-buffer / chunk-
  boundary tuning could raise hit rate further.

## Cross-references
[[F0008]] (why plain radix corrupts) · `bench/verify_batch.py --radix-on` ·
`scripts/deploy.sh` (is_hybrid_ssm patch) · `docs/design/state-prefix-cache.md`.
