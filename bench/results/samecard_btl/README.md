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

> **The prefill column of this harness does not reproduce.** Running the same
> server at the same setting four times in one session (2026-08-05) gave 13,396
> / 28,752 / 28,047 / 13,439 tok/s at bs8 / prompt 128 — a 2.1x spread with
> nothing changed between runs, alternating with run order rather than with any
> setting. The decode column over those same four runs was stable to 0.4%. Read
> the numbers below as order-of-magnitude only; a difference smaller than ~2x in
> this column is not evidence of anything. This was found while chasing an
> apparent regression that turned out to be this noise.

| batch | prompt | ours | theirs |
|---:|---:|---:|---:|
| 1 | 128 | 7,370 | 14,407 |
| 1 | 512 | 24,363 | 24,764 |
| 1 | 2048 | 28,270 | 28,079 |
| 8 | 128 | 36,249 | 38,922 |
| 8 | 512 | 45,717 | 49,512 |
| 8 | 2048 | 38,495 | 50,852 |

## Re-measured on this harness after the LoRA gate landed (2026-08-06)

The fused-LoRA batch gate is now 8 rather than 4. The crossover had been recorded
as a range, ~M=4→8, and the gate set to the safe end without measuring inside it;
worse, the commit that measured it and raised the default edited `sglang_overlay/`
— the retired line, deleted a few commits later — so **the shipped default stayed
at 4 until 2026-08-06** (F0086).

Our column re-run here on the same harness, both gate values, two rounds each,
decode tok/s (median of 5, 2 warmups). **Their column is not re-run**: it is still
the 2026-08-04 measurement.

| batch | prompt | ours, gate 4 | ours, gate 8 | theirs (08-04) | theirs/ours | was |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | — | 514.6 | 448 | 0.871 | 0.885 |
| 1 | 512 | — | 514.5 | 457 | 0.888 | 0.891 |
| 1 | 2048 | — | 514.4 | 444 | 0.863 | 0.866 |
| 8 | 128 | 1,936.4 | **2,167.3** | 3,122 | **1.441** | 1.617 |
| 8 | 512 | 1,937.3 | **2,168.3** | 3,007 | **1.387** | 1.558 |
| 8 | 2048 | 1,926.2 | **2,159.4** | 3,040 | **1.408** | 1.582 |

**bs8 gap 1.56–1.62x → 1.39–1.44x. bs1 is untouched** (the gate covers T=1 either
way), and that is the control: our gate-4 arm today reads 1,936.4 / 1,937.3 /
1,926.2 against 1,931 / 1,930 / 1,922 on 2026-08-04 — **0.2–0.4% apart**, so the
card and the harness are in the same state and the comparison against their
un-re-run column holds. Raw: `ours-5090-regate-r{1,2}_gate{4,8}.jsonl`.

Two things this does not say. It is a decode-column result only — the prefill
column of this harness does not reproduce (see the caveat above). And 1.39–1.44x
is still their result, not parity: what is left is the batched kernels, and the
gate was the part of it that cost nothing to fix.

## Reading

- **bs1 decode is ours by 1.12–1.16×** — the single-stream megakernel path.
- **bs8 decode is theirs by 1.56–1.62×** as measured on 2026-08-04, **1.39–1.44×
  after the LoRA gate landed** (re-measured section above) — their batched kernels
  and full-graph decode; the crossover sits between batch 1 and 8 and is
  unmeasured.
- **Short-prompt TTFT is theirs** (8.9 ms vs 17.4 ms at 128 tokens, bs1): their
  fixed-shape prefill CUDA-graph buckets. This project disables prefill CUDA
  graphs for RWKV-7 on purpose and the reason is structural, not a missing
  feature: the model calls its linear-attention backend directly, so the
  backend's per-batch varlen metadata would sit *inside* the captured region and
  be frozen at capture. Fixed-shape bucketing is what makes their configuration
  able to capture it at all. Closing this means moving that metadata out of the
  captured region, not flipping a flag. (Subject to the noise caveat above.)
- Their documented launch caps at 16 running requests, so the high-concurrency
  regime (our 1.5B 29,533 tok/s at c=320, BENCHMARKS §5) has no comparable cell
  under their published configuration. The flag is adjustable; "as documented"
  is the only claim made here.
- Neither side's quantized tiers are compared here; dense fp16 only.
