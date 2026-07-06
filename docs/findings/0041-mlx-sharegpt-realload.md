---
doc_kind: finding
finding_id: F0041
title: "MLX Apple-Silicon real-workload (ShareGPT, bsz1 single-stream): realistic prefill+decode throughput and the TTFT / inter-token-latency distribution over the real prompt-length mix"
last_verified_commit: "HEAD"
discovered_by: Opus 4.8 (agent-assisted), 2026-07-06
severity: info
status: open
related: [F0038, F0039]
---

# Finding F0041: MLX ShareGPT real-workload latency/throughput (Apple Silicon)

## Context
F0038/F0039 measured throughput on a synthetic tiled prompt. This finding closes the loop with a
**real** workload — actual ShareGPT conversations — single-stream (bsz1) on M5, reporting what a
streaming client experiences: TTFT and inter-token latency (ITL) over the real prompt-length mix,
plus aggregate prefill/decode throughput. (MLX has no server/continuous batching, so this is the
single-stream analogue of `docs/BENCHMARKS.md §7c`.)

## Methodology
`mlx_port/sharegpt_mlx.py`: sample N conversations, take each **first human turn** as the prompt
(this is where the real length distribution comes from), prefill it (TTFT = prefill + first token),
then greedy-decode up to `--max-new` tokens timed **per token synchronously** (the streaming path —
each token materialized as delivered), yielding the ITL distribution. decode tok/s is derived from
those per-token times (a touch below `bench_mlx.py`'s async-pipelined ceiling, by design: this
measures latency, not peak throughput). p50/p90/p99 reported.

## Results — RWKV-7 1.5B, M5, ShareGPT (150 conversations, first-human-turn prompts, max_new=128)

Real prompt-length mix (tokens): **min 8 / p50 51 / mean 244 / p90 856 / max 1865** — the true
ShareGPT spread (many short turns, a long tail).

| precision | TTFT p50 / p90 / p99 (ms) | ITL p50 / p90 / p99 (ms) | decode tok/s (stream) | prefill tok/s (agg) |
|---|---|---|---:|---:|
| **fp16** | 77.8 / 631.0 / 1309.5 | 38.8 / 48.7 / 57.6 | 25.6 | 1,202 |
| **w8** | 71.8 / 608.1 / 1322.1 | 19.5 / 27.7 / 37.2 | 48.0 | 1,307 |

(Each: 150 convs, 36,551 prompt tokens prefilled + 19,050 tokens decoded; same prompts. Wall time
fp16 774 s → w8 425 s.)

## Reading it
- **TTFT is prompt-length-driven, as expected on a recurrent model:** at the p50 prompt (51 tok)
  the user waits **~78 ms** for the first token; the p90 prompt (856 tok) is **~631 ms**; the longest
  (1865 tok) **~1.4 s**. RWKV-7 prefill is O(prompt) with a constant-size state (no KV blowup), so
  even the long tail stays sub-1.5 s single-stream.
- **Inter-token latency is tight** (fp16 p50 38.8 ms, p99 57.6 ms) — a steady streaming cadence of
  ~26 tok/s that a reader comfortably keeps up with. p99/p50 ≈ 1.5× (the tail is host-load jitter on
  this shared box, not model variance — the state is constant work per token).
- **Streaming decode (25.6 tok/s) is below `bench_mlx.py`'s async ceiling (33–37)** by design: this
  path `mx.eval`s every token to timestamp it (the real streaming-delivery cost), which serializes
  the host round-trip the async pipeline hides. It is the honest number a streaming client sees.
- **w8 turns the F0039 decode win into a visibly smoother stream on the real workload:** ITL p50
  **halves** (38.8 → 19.5 ms), streaming decode **+88%** (25.6 → 48.0 tok/s), TTFT trims a little
  (77.8 → 71.8 ms p50) and aggregate prefill nudges up (1,202 → 1,307 tok/s) — and it is
  greedy-lossless (F0039) + compression-lossless (F0040). The whole 150-conversation replay
  finishes in **425 s vs 774 s**. On Apple Silicon single-stream, w8 is the recommended default for
  interactive use.

## Honesty / host load
Shared M5 (load ~8); single-stream latency carries run-to-run jitter, reported as distributions
(p50/p90/p99), not single points. Prompt-length mix is the real ShareGPT distribution
(min/p50/mean/p90/max reported); a token cap (`--max-tok`) keeps one giant prompt from dominating
wall time. `--quant` runs use the F0039 weights.

## Cross-references
`mlx_port/sharegpt_mlx.py` · F0038 (synthetic throughput + bandwidth ceiling) · F0039 (quant) ·
`docs/BENCHMARKS.md §7c` (the CUDA server-side ShareGPT comparison).
