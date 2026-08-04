# Same-card comparison with btlqql/sglang-rwkv7 (RTX 5090, 2026-08-04)

Both stacks on the same RTX 5090, measured by the **other project's own harness**
(`benchmark/rwkv7/bench_acceptance_matrix.py` from btlqql/sglang-rwkv7) — a plain
`requests` client against each server's standard `/generate` streaming endpoint, so
the yardstick is identical and was written by neither side for this comparison.

Model `rwkv7-g1-1.5b` fp16 · decode 128 tokens · 2 warmups, median of 5 · recurrent
cache flushed before every sample.

- ours: `scripts/serve.sh` production defaults, `MEMFRAC=0.50`
  (`ours-5090-dense-prod.jsonl`; `ours-5090-dense.jsonl` is an earlier run with
  `RWKV_SPARSE_FFN=0` — bs8 identical within noise, kept for the record)
- theirs: image pinned to their stated floors (flashinfer-python 0.6.15.post1,
  sglang-kernel 0.4.5), launched with their documented dense command
  (their `benchmark/rwkv7/README.md` "Dense server": triton backend, fp32 state,
  decode CUDA graph bs 1/2/4/8, prefill graph buckets 128…16384,
  `--max-running-requests 16`), `--mem-fraction-static 0.80`

## Decode tok/s (median)

| batch | prompt | ours | theirs | theirs/ours |
|---:|---:|---:|---:|---:|
| 1 | 128 | 506 | 448 | 0.89 |
| 1 | 512 | 513 | 457 | 0.89 |
| 1 | 2048 | 513 | 444 | 0.87 |
| 8 | 128 | 1,931 | 3,122 | 1.62 |
| 8 | 512 | 1,930 | 3,007 | 1.56 |
| 8 | 2048 | 1,922 | 3,040 | 1.58 |

## Prefill tok/s (median)

| batch | prompt | ours | theirs |
|---:|---:|---:|---:|
| 1 | 128 | 7,370 | 14,407 |
| 1 | 512 | 24,363 | 24,764 |
| 1 | 2048 | 28,270 | 28,079 |
| 8 | 128 | 36,249 | 38,922 |
| 8 | 512 | 45,717 | 49,512 |
| 8 | 2048 | 38,495 | 50,852 |

## Reading

- **bs1 decode is ours by 1.12–1.16×** — the single-stream megakernel path.
- **bs8 decode is theirs by 1.56–1.62×** — their batched kernels and full-graph
  decode; the crossover sits between batch 1 and 8 and is unmeasured.
- **Short-prompt TTFT is theirs** (8.9 ms vs 17.4 ms at 128 tokens, bs1): their
  fixed-shape prefill CUDA-graph buckets. Our prefill graph is disabled at boot
  (ADR-0008); at 512+ tokens prefill is par.
- Their documented launch caps at 16 running requests, so the high-concurrency
  regime (our 1.5B 29,533 tok/s at c=320, BENCHMARKS §5) has no comparable cell
  under their published configuration. The flag is adjustable; "as documented"
  is the only claim made here.
- Neither side's quantized tiers are compared here; dense fp16 only.
