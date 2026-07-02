---
doc_kind: finding
finding_id: F0016
title: "Serving-scale measured: decode throughput scales ~50× with concurrency (166→8298 tok/s, bsz 1→128) at flat VRAM (256 concurrency = +202 MiB), and peak VRAM is context-invariant (+4 MiB across 1K→64K) — the O(1)-state wedge, quantified"
last_verified_commit: "HEAD"
discovered_by: lead, 2026-07-01
severity: info
status: open
related: [F0007, F0014, F0015]
---

# Finding F0016: the O(1)-state serving wedge, measured

RWKV-7 carries a **constant** recurrent state per sequence (1.62 M elements, no growing
KV cache). This finding measures the two serving consequences directly on one exclusive
RTX 3090 (`rwkv7-1.5b-fla`, bf16, cuda-graph ON = production decode path, radix cache OFF).
Motivation: the same-precision single-stream comparison (F0014/F0015) anchors on albatross's
strongest, least-serving-relevant axis; the numbers below are the axes a serving engine is
actually chosen for. Artifacts: `bench/results/serving_scale/` (`conc_scale_15b.log`,
`ctx_invariance_15b.log`), scripts `bench/throughput.py` + `bench/serving_scale.py`.

## 1. Concurrency scaling (fixed 512-tok context)
| bsz | decode tok/s | peak VRAM (MiB) |
|----:|-------------:|----------------:|
|   1 |        166.0 |          12,420 |
|  16 |      2,143.2 |          12,622 |
|  64 |      6,444.5 |          12,622 |
| 128 |      8,297.8 |          12,622 |
| 256 |      8,186.7 |          12,622 |

Decode throughput scales **~50×** (166 → 8,298 tok/s) from bsz 1 → 128, then plateaus at
256 (compute-bound). Peak VRAM: **+202 MiB total across a 256× concurrency increase** — each
added sequence is a tiny constant state, so hundreds of concurrent seqs fit in one 24 GB card.

## 2. Context-length invariance (fixed bsz 8)
`rwkv7-1.5b`'s config declares an 8,192-token trained window; RWKV-7's recurrence has no
architectural context limit, so `--max-context 131072`
(+`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`) measures serving **cost** past it (output
quality beyond the trained window is not claimed).

| context | decode ms/step | peak VRAM (MiB) |
|--------:|---------------:|----------------:|
|   1,024 |           7.73 |          12,364 |
|   8,192 |           8.89 |          12,364 |
|  16,384 |          10.69 |          12,364 |
|  32,768 |          10.47 |          12,366 |
|  65,536 |           7.41 |          12,368 |

**Peak VRAM: 12,364 → 12,368 MiB (+4 MiB, +0.03%) across a 64× context increase.** Decode
stays O(1)/token (single-digit ms/step, no growth trend). TTFT grows linearly with context
(581 ms → 47 s) — expected, prefill is O(T) for any model; the wins are decode + memory.

## Flagship confirmation (7.2B, added 2026-07-02)
Same sweeps at 7.2B (bf16, `bench/results/serving_scale/{ctx,conc}_72b.log`): context 1K→32K =
**+0 MiB** peak VRAM (17,866 flat) at 22–32 ms/step; concurrency bsz 1→64 = 46.6→**1,802.7 tok/s
(38.7×)** at **+308 MiB**. The O(1)-state properties hold unchanged at the flagship size.

## Honesty caveats
- **VRAM appears "flat" partly because sglang pre-allocates a static state pool** from
  `mem_fraction_static`. The load-bearing claim is not "the pool is fixed" (that is config) but
  that **the pool can hold hundreds of arbitrary-length RWKV states** because each is a fixed
  constant — hence the +202 MiB / +4 MiB deltas *inside* a fixed budget, and why a KV-cache
  engine with the same budget would OOM at 256×64K.
- **The decode-tok/s column of the context sweep is noise-dominated at long context** (it is a
  difference of two ~47 s prefill-inclusive runs, decode delta sub-second). The robust signals
  are peak VRAM (measured absolutely) and ms/step; we read *no context penalty*, not a speedup.

## Consequence
This reframes the top-of-README benchmark: lead with concurrency/VRAM/int8/accuracy (won),
demote the same-precision single-stream chart (F0014/F0015, albatross's bandwidth-ceiling turf)
to an honest "one axis albatross leads" subsection. See `README.md` §📊 Benchmarks.
