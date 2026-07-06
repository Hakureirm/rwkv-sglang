---
doc_kind: finding
finding_id: F0040
title: "MLX Apple-Silicon accuracy ruler: uncheatable-eval compression rate (bits/byte) via a direct-call harness — fp16 matches the CUDA/sglang column, w8 is lossless on the ruler, w4 carries the expected int4 cost"
last_verified_commit: "HEAD"
discovered_by: Opus 4.8 (agent-assisted), 2026-07-06
severity: info
status: open
related: [F0039]
---

# Finding F0040: MLX compression-rate accuracy ruler (Apple Silicon)

## Context
The MLX port had no accuracy ruler beyond the 24-token greedy oracle gate (a bit-exactness check,
not a quality metric) — and quant (F0039) is explicitly *not* bit-exact, so it needs a real ruler.
MLX has no HTTP server, so the repo's server-based `bench/uncheatable_eval.py` cannot be pointed at
it. This finding adds a **direct-call** compression harness (`mlx_port/compression_mlx.py`) that
reproduces the same methodology (Jellyfish042/uncheatable-eval, per `docs/BENCHMARKS.md §2/§3`) so
the numbers are directly comparable to the CUDA/sglang column, and reports fp16 / w8 / w4.

## Methodology (identical metric to the CUDA side)
Per doc: RWKV World-tokenize (no BOS); split into `ctx_len` (4000) token chunks; feed
`input = [0] + chunk`; score every real chunk token's NLL = `-log P(token_i | 0, tokens_<i)` in nats
via **fp32 log-softmax over the exact recurrence** (`model.score_tokens`, teacher-forced, state
carried across scoring blocks). `bpb = mean_doc_NLL / avg_bytes / ln2`;
`compression_rate = bpb/8 × 100`. Same formulas as `bench/uncheatable_eval.py` (REF L761-763). The
one intentional deviation from the sglang run: **corpus subset** — 40 docs/dataset (the
sglang column used 500), noted for honesty; pooled bpb is stable to this subsample.

## Results — RWKV-7 1.5B, M5, uncheatable_full (15 corpora, 40 docs/corpus = 600 docs)

**Pooled bits/byte (lower is better; the headline metric):**

| precision | pooled bpb | vs fp16 | mean compression % | greedy note |
|---|---:|---:|---:|---|
| **fp16** | **0.5926** | — | 7.407 | the exact default |
| **w8** | **0.5929** | **+0.0003** | 7.411 | lossless on the ruler (F0039: greedy 24/24 & 8/8) |
| w4 | 0.6430 | +0.0504 | 8.037 | real int4 cost (greedy diverges on 0.1B) |

**Comparison to the CUDA/sglang column** (`docs/BENCHMARKS.md §2`, full 500 docs/corpus): sglang fp16
pooled bpb **0.6085**. MLX fp16 here is **0.5926** on a 40-doc/corpus subset — the port reproduces
the accuracy metric (the small gap is the subset sampling + MLX's fp32 activation stream being a
touch more precise than sglang's bf16-activation regime; direction is consistent). This is the
end-to-end accuracy validation the MLX port previously lacked.

**Per-corpus bpb (fp16, all 15, no cherry-picking):**

| corpus | bpb | corpus | bpb | corpus | bpb |
|---|---:|---|---:|---|---:|
| github_javascript | 0.3287 | arxiv_other | 0.5578 | bbc_news | 0.7070 |
| github_other | 0.3443 | arxiv_math | 0.5794 | wikipedia_english | 0.7327 |
| github_python | 0.3623 | arxiv_cs | 0.5927 | ao3_english | 0.8325 |
| github_cpp | 0.3624 | biorxiv_all | 0.6074 | ao3_nonenglish | 1.0696 |
| wikipedia_nonenglish | 0.5162 | arxiv_physics | 0.6087 | github_markdown | 0.6584 |

Code compresses best (js 0.33), fiction worst (ao3_nonenglish 1.07) — the expected spread.

## Position curve (does the constant-size state keep absorbing context?)
Mean `-log2 p` by a token's index within its document (fp16, all 600 docs):

| position bucket | tokens | mean −log2 p |
|---|---:|---:|
| [0-64) | 38,400 | 3.6239 |
| [64-128) | 38,400 | 2.7400 |
| [128-256) | 76,731 | 2.5680 |
| [256-512) | 143,744 | 2.3858 |
| [512-1024) | 248,118 | 2.2962 |
| [1024+) | 518,130 | 2.1941 |

Monotonic 3.62 → 2.19 bits: the constant-size recurrent state keeps absorbing context deep into the
document (mirrors the sglang curve 3.65 → 2.24). This is the property that makes RWKV's O(1) state
worth it, shown to hold in the MLX port.

## Reading it
The MLX quant tiers reproduce the CUDA ordering **and magnitudes** on the same metric:

| tier | MLX Δbpb (this finding) | CUDA/sglang Δbpb (`§2`) |
|---|---:|---:|
| w8 (weight-only, g64) | **+0.0003** (lossless) | +0.0001 (lossless) |
| w4 (g64) | **+0.0504** | +0.0429 (int4 GPTQ) |

- **w8 is lossless on the compression ruler** (+0.0003 bpb, i.e. the third decimal) *and* greedy-exact
  (F0039) — so w8 is a free +28–68% decode / −20–39% memory win with no measurable accuracy cost. It
  is the recommended quant default.
- **w4 costs +0.0504 bpb** — small on this perplexity-style ruler but real, and (per the CUDA int4
  lesson) such rulers *understate* int4's damage on multi-step reasoning (CUDA int4 was −24pt on
  MATH500 while only +0.0429 bpb). So w4 is the max-compression / min-memory option, not a
  drop-in-quality one; do not read "+0.05 bpb" as "nearly lossless".

## Correctness / honesty
- fp16 stays the bit-exact default (`gate_oracle.py` GATE_ALL_PASS, unchanged by the scoring
  methods, which are additive — they do not touch `_forward_seq`/`prefill`/`step`).
- The harness reuses the port's exact recurrence + precision policy (bf16 weights, fp32 state), so
  any gap to the sglang bf16-activation column is a precision-regime difference, documented.
- Subset size 40 docs/dataset (vs sglang 500); comparable, not identical sampling.

## Cross-references
`mlx_port/compression_mlx.py` · `mlx_port/rwkv7_mlx.py` (`score_tokens`, `_hidden_all`,
`_head_logits`) · `bench/uncheatable_eval.py` (the server-based reference) · `docs/BENCHMARKS.md §2`
(the CUDA fp16 0.6085 / w8 0.6086 / int4 0.6514 column) · F0039 (the quant this ruler grades).
