---
doc_kind: finding
finding_id: F0049
title: "Desktop-GPU tier (RTX 3090, 24GB) of the RWKV-7 vs Qwen3.5 comparison: same-precision bf16 peak concurrency, RWKV wins the 1.5B/2B tier for real (+11.7%); the 7.2B/9B tier is directionally consistent with the cloud-tier finding but the Qwen3.5-9B boundary search was cut short (session ended mid-sweep) — do not cite a final number for that tier yet"
last_verified_commit: "HEAD"
discovered_by: Sonnet 5 (agent-assisted, 3090 box; write-up completed directly after the agent's session ended on a transient API stream error mid-report), 2026-07-07
severity: info
status: open
related: [F0044, F0045, F0047, F0048]
---

# Finding F0049: desktop-tier (RTX 3090) Qwen3.5 vs RWKV-7 comparison

## Context

The cloud tier (RTX 5090, F0044–F0048) found that RWKV-7 wins same-precision (bf16) peak
concurrency at both size tiers (1.5B/2B: +21.9%; 7.2B/9B: +43.7%), reversing an earlier bsz1-only
reading that favored Qwen3.5 (RWKV's hand kernels only help in fp16, not bf16). This finding
repeats the same measurement on the desktop-GPU tier — a single RTX 3090, 24GB, a materially
tighter memory budget than the 32GB 5090 — using the same `--dtype bfloat16` Qwen3.5 boot fix and
the same `bsz_throughput.py`-style concurrency sweep protocol established there.

**Process note**: the session that gathered this data ended on a transient API stream error
partway through its own write-up (not a task failure — 112 real tool calls and multiple completed
sweeps preceded it). All the raw result JSONs it produced are intact on the box; this finding was
written directly from that raw data rather than re-running the session, to avoid a third attempt
at the same class of transient failure.

## 1.5B / 2B tier — complete, same-precision, real result

| model | precision | peak tok/s | concurrency at peak |
|---|---|---:|---:|
| RWKV-7 1.5B | bf16 | **7,058.6** | c=256 |
| Qwen3.5-2B | bf16 | 6,316.9 | c=384 |

**RWKV-7 wins by +11.7%** at this tier on the 3090 — smaller margin than the 5090's +21.9%, but
the same direction. (Bsz1 for reference, not the deciding metric per this project's own
full-spectrum-over-single-stream doctrine: RWKV-7 1.5B bf16 not separately isolated in this
session's data; Qwen3.5-2B bf16 bsz1 = 175.9 tok/s.)

## 7.2B / 9B tier — directionally consistent, but incomplete: do not cite a final number

RWKV-7 7.2B bf16 peaked at **1,796.3 tok/s @ c=128** (the c=96 reading of 1,499.5 is a dip inside
an otherwise plateaued 128–192 region — 1,794.3 at c=192 confirms 128's reading rather than being
a fluke; not chased further since the region is clearly flat, not still climbing).

Qwen3.5-9B's own boundary search followed the same iterative `--cuda-graph-max-bs` escalation
pattern the cloud tier needed (this project's own documented gotcha: sglang's cuda-graph capture
bucket must be sized to cover the top concurrency point tested, or throughput silently degrades to
slow eager mode above the configured cap) — cg8→cg32→cg48, each finding a higher peak than the
last (323.6 → 844.7 → 1,083.2 tok/s), with **the c=48 reading still visibly climbing, not
plateaued**, when the session ended before a cg64 (or higher) attempt could run. Do not read
"RWKV 1,796.3 vs Qwen3.5 1,083.2" as this tier's real result — Qwen3.5-9B's true peak on this
24GB card is unmeasured, likely higher than 1,083.2, and could plausibly land anywhere between
there and RWKV's number depending on how much further headroom the tighter 24GB budget leaves
(9B on a 24GB card is a bigger relative squeeze than on the 32GB tower, so don't assume the
5090's ~4,296 tok/s 9B peak scales down proportionally).

**What's needed to close this out**: continue the `--cuda-graph-max-bs` escalation (cg64, cg80,
...) on Qwen3.5-9B/3090 bf16 until two consecutive points show a genuine decline (the same
bracketing standard used for every other peak in this comparison project), the same way F0047's
fp16-7.2B correction and the cloud tier's Qwen3.5-9B search were closed out.

## Files

Raw JSONs on the 3090 box (not this repo — this box has no GitHub access):
`~/rwkv-vllm/bench/results/{rwkv7_1.5b_bf16_sweep_3090,rwkv7_7.2b_bf16_sweep_3090_cg192,
qwen35_2b_bf16_sweep_3090_cg448,qwen35_9b_bf16_sweep_3090_cg{8,32,48}}.json`. Qwen3.5-2B/9B
weights already resident at `~/rwkv_models/` (or wherever the box's model dir landed — checked
via the earlier session's own recon) for whoever continues the cg64+ sweep.

## Cross-references

F0044 (MLX feasibility) · F0045 (Apple Silicon matched benchmark) · F0047 (fp16 7.2B concurrency
correction — same boundary-search discipline this finding's incomplete half needs) · F0048 (int8
tier gap) · `memory/project-qwen35-benchmark.md` (full round-by-round log).
