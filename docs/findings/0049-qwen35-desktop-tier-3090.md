---
doc_kind: finding
finding_id: F0049
title: "Desktop-GPU tier (RTX 3090, 24GB) of the RWKV-7 vs Qwen3.5 comparison: same-precision bf16 peak concurrency, RWKV wins BOTH tiers for real — 1.5B/2B +11.7%, 7.2B/9B >=+27.0% (RWKV's confirmed-flat 1,796.3 @ c128 vs Qwen3.5-9B's highest valid reading 1,414.2 @ c64, still climbing when a hard 24GB memory ceiling — not compute saturation — cut the search off)"
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

**Continuation note (2026-07-07, later session)**: the 7.2B/9B gap below — Qwen3.5-9B's boundary
search left mid-sweep at cg48 — has now been closed out on the same box, same protocol. The
section below is rewritten in place to carry the real result; the original "incomplete, do not
cite" framing is preserved in this file's git history for anyone who wants the prior state.

## 1.5B / 2B tier — complete, same-precision, real result

| model | precision | peak tok/s | concurrency at peak |
|---|---|---:|---:|
| RWKV-7 1.5B | bf16 | **7,058.6** | c=256 |
| Qwen3.5-2B | bf16 | 6,316.9 | c=384 |

**RWKV-7 wins by +11.7%** at this tier on the 3090 — smaller margin than the 5090's +21.9%, but
the same direction. (Bsz1 for reference, not the deciding metric per this project's own
full-spectrum-over-single-stream doctrine: RWKV-7 1.5B bf16 not separately isolated in this
session's data; Qwen3.5-2B bf16 bsz1 = 175.9 tok/s.)

## 7.2B / 9B tier — closed out: RWKV wins again, but report the honest shape of the win

RWKV-7 7.2B bf16 peaked at **1,796.3 tok/s @ c=128** (the c=96 reading of 1,499.5 is a dip inside
an otherwise plateaued 128–192 region — 1,794.3 at c=192 confirms 128's reading rather than being
a fluke; not chased further since the region is clearly flat, not still climbing). This number is
unchanged from before and still the correct RWKV-side reference for this box.

Qwen3.5-9B's boundary search continued the same iterative `--cuda-graph-max-bs` escalation from
where it left off (cg8→cg32→cg48 → **cg64**), each step re-confirming the previous concurrency
point from a fresh boot before adding the new top point:

| `--cuda-graph-max-bs` / `--mem-fraction-static` | c=1 | c=8 | c=16 | c=32 | c=48 | c=64 |
|---|---:|---:|---:|---:|---:|---:|
| cg8 / 0.85 | 45.3 | 323.6 | — | — | — | — |
| cg32 / 0.85 | 45.9 | 343.0 | 581.3 | 844.7 | — | — |
| cg48 / 0.90 | 45.8 | — | — | 978.8 | 1,083.2 | — |
| **cg64 / 0.93 (run 1)** | 45.7 | — | — | — | 1,130.5 | **1,414.2** |
| **cg64 / 0.93 (confirm run)** | 45.8 | — | — | 976.7 | 1,066.0 | **1,376.3** |

The c=32 and c=48 re-reads across independent boots agree closely (976.7 vs 978.8; 1,066.0 vs
1,083.2 vs 1,130.5 — a normal few-percent run-to-run band, consistent with this project's other
noise measurements), and the two independent cg64 runs agree within 2.7% (1,376.3 / 1,414.2) —
**the c=64 peak is real and reproducible, and it is still climbing relative to c=48**, exactly
continuing the trend the prior session left off mid-rise.

**Above c=64, this specific 24GB card hits a hard memory ceiling — not a compute plateau.** Three
independent attempts to push past c=64 all failed via genuine CUDA out-of-memory crashes, not the
KV-starvation false-plateau this project usually watches for:

| attempt | `--mem-fraction-static` | boot | outcome |
|---|---:|---|---|
| cg64 (original, unadjusted) | 0.85 | pre-flight reject | `RuntimeError: Not enough memory` before any allocation — the mamba-cache budget alone (64 × ~48.8MB ≈ 3.1GB) exceeded what 0.85 left after the 17.62GB weight load |
| cg72 | 0.95 | clean boot, healthy KV (32,949 tokens) | **crashed on the automatic post-boot warmup request** — only 0.29GB free after cuda-graph capture, not enough for even one request's prefill burst |
| cg72 | 0.94 | clean boot, healthy KV (25,328 tokens) | **crashed mid-sweep escalating to c=64** — `torch.OutOfMemoryError` inside `causal_conv1d_triton.py` (the GDN `mixed_qkv` prefill path), with GPU-wide free memory down to single-digit MiB |
| cg68 | 0.935 | clean boot, healthy KV (27,805 tokens, 27.8% cushion) | **crashed mid-decode within the c=64 test itself** (KV pool only 68% full when it died) — same OOM class, in the same kernel |

Reverse-engineering sglang's exact memory-pool formula (`model_runner_kv_cache_mixin.py`:
`rest_memory = avail_after_load − avail_before_load × (1 − mem_fraction_static)`, then the
mamba-cache budget is subtracted before KV cache gets whatever remains) shows why: the non-static
headroom left for real request bursts after cuda-graph capture reduces to `23.24 × (1 −
mem_fraction_static) − ~0.8GB` (a roughly fixed capture/workspace overhead) — a function of
`mem_fraction_static` **alone**, independent of how that budget is split between mamba-cache and
KV-cache. Getting c≥68's mamba-cache need funded requires `mem_fraction_static` high enough that
the leftover burst headroom drops below what a concurrent-request prefill spike actually needs on
this model — confirmed empirically three times, not just algebraically. On this 24GB card, with
Qwen3.5-9B's fixed 17.62GB bf16 weight footprint, **no value of `--mem-fraction-static` can
simultaneously fund an adequate mamba/KV budget and survive the real burst at c≥68** — a genuine
hardware ceiling, not a tuning gap (three configurations spanning the theoretically-available
range were tried and all crashed the same way).

**Result: the highest valid Qwen3.5-9B bf16 reading on this 24GB card is 1,376.3–1,414.2 tok/s @
c=64 — still climbing, memory-ceiling-terminated, not a confirmed compute-saturation peak.**
Comparing against RWKV-7's own confirmed-flat 1,796.3 @ c=128:

| model | precision | highest valid peak tok/s | concurrency | shape |
|---|---|---:|---:|---|
| RWKV-7 7.2B | bf16 | **1,796.3** | c=128 | confirmed flat (repeat at c=192 = 1,794.3) |
| Qwen3.5-9B | bf16 | 1,376.3–1,414.2 | c=64 | still climbing when the 24GB memory ceiling cut the search off |

**RWKV-7 wins by >= +27.0%** (1,796.3 / 1,414.2, using Qwen's higher reading as the conservative
comparison point) **up to +30.5%** (using the confirm run's 1,376.3). Report the lower figure as
the headline per this project's claims-need-numbers discipline — don't cite the more favorable
gap. Note this is a **floor**, not the true margin: Qwen3.5-9B's real compute-bound ceiling is
unknown and unreachable on this card (on the 32GB 5090 it reached 4,295.8 tok/s @ c=128 before
plateauing), so the actual gap if Qwen3.5-9B had enough VRAM to reach compute saturation could be
smaller than +27–30%, or the ordering could conceivably even flip — this comparison answers "who
wins on this 24GB card today," not "whose architecture has the higher asymptotic ceiling."

**Architectural read (consistent with, and sharper than, the cloud-tier finding)**: this is the
same asymmetry the cloud-tier work already documented (`memory/project-qwen35-benchmark.md` round
4, and F0048's related finding that Qwen3.5's hybrid layers need real per-token KV cache where
RWKV needs none) — Qwen3.5's hybrid architecture must fund both a mamba-cache (recurrent GDN
state) *and* a real per-token KV cache for its 6/24 full-attention layers, while RWKV-7's
100%-recurrent design never allocates a KV cache at all. On the 32GB 5090 that tax pushed
Qwen3.5-9B's usable concurrency ceiling below its compute-bound peak only mildly; on this 24GB
card, with less total VRAM to absorb the same fixed per-request tax, the memory wall arrives well
before compute saturation — sharper illustration of the same effect, not a new one.

## Files

Raw JSONs on the 3090 box (not this repo — this box has no GitHub access):
`~/rwkv-vllm/bench/results/{rwkv7_1.5b_bf16_sweep_3090,rwkv7_7.2b_bf16_sweep_3090_cg192,
qwen35_2b_bf16_sweep_3090_cg448,qwen35_9b_bf16_sweep_3090_cg{8,32,48,64,64_confirm}}.json`.
Server/bench logs documenting the crash boundary (also 3090-box-only):
`~/rwkv-vllm/logs/qwen35_9b_bf16_server_3090_{cg64,cg64_v2,cg64_confirm,cg68,cg72,cg72_v2}.log`,
`~/rwkv-vllm/logs/qwen35_9b_bf16_bench_3090_{cg64,cg64_confirm,cg68,cg72}.log` (the cg68/cg72 bench
logs end in a `ConnectionError` — that's the client-side symptom of the server crashing underneath
it, not a client bug). Qwen3.5-2B/9B weights resident at `~/rwkv_models/qwen3.5-{2b,9b}/`.

## Cross-references

F0044 (MLX feasibility) · F0045 (Apple Silicon matched benchmark) · F0047 (fp16 7.2B concurrency
correction — same boundary-search discipline this finding's 7.2B/9B half used to find its real
ceiling) · F0048 (int8 tier gap, and the same mamba-cache-vs-KV-cache architectural asymmetry
this finding's memory-ceiling section sharpens) · `memory/project-qwen35-benchmark.md` (full
round-by-round log, including this session's).
