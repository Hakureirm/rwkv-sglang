# docs/ — decision & evidence trail

Two reading tracks:

- 🧑‍💻 **[`human/`](human/) — 人类可读版（中文，由浅入深，多图多表）**：想快速理解项目是什么、
  为什么、架构、性能、怎么用，从这里开始。
- 🤖 **this dir (ADR / findings / snapshot) — the dense engineering record** (for agents / deep dives):
  decisions, dated evidence, per-result verification.

The engineering record below (ADSD: decisions, findings, design). Start with the snapshot; the rest
is the honest, dated trail of how each result was reached + verified.

- **[`snapshot.md`](snapshot.md)** — canonical current state (the single source of truth; the
  READMEs are projections of it).

## ADRs — architecture decisions (`adr/`)
| # | Decision |
|---|---|
| [0001](adr/0001-scope-and-wedge.md) | Scope = **RWKV-7 on sglang**, and the wedge (why this project) |
| [0002](adr/0002-sglang-integration-approach.md) | How RWKV-7 plugs into sglang (Mamba/linear-attn substrate) |
| [0003](adr/0003-m1-scope-and-slicing.md) | Slicing M1 into gated increments |
| [0004](adr/0004-no-fla-dependency.md) | **No flash-linear-attention** dependency on the RWKV-7 path |

## Design docs (`design/`)
- [`m1-implementation-plan.md`](design/m1-implementation-plan.md) — the initial model+backend plan.
- [`m6-sparse-ffn.md`](design/m6-sparse-ffn.md) — the M6 CUDA endgame: the three hand-written
  kernels (in-place WKV, sparse sqrelu FFN, fused GEMV), the profiling that targeted them, gates,
  and the honest ceiling analysis.

## Findings — dated observations, each with method + result (`findings/`)
| # | Finding |
|---|---|
| [F0001](findings/0001-dev-box-and-env-recon.md) | Dev-box & environment recon |
| [F0002](findings/0002-rwkv7-architecture-and-vllm-mapping.md) | RWKV-7 architecture ↔ serving-framework mapping |
| [F0003](findings/0003-parity-baselines-and-acceptance.md) | Parity baselines, oracle & acceptance test |
| [F0004](findings/0004-latest-upstream-reanalysis.md) | Verified latest-upstream re-analysis (why sglang) |
| [F0005](findings/0005-m1-complete.md) | M1 — 0.1B exact greedy-match in sglang |
| [F0006](findings/0006-m2-baseline-throughput.md) | M2 baseline — bf16 + 1.5B exact; throughput |
| [F0007](findings/0007-albatross-3090-baseline.md) | Albatross 3090 baseline |
| [F0008](findings/0008-m2b-cudagraph.md) | cuda-graph — big decode speedup, still exact |
| [F0009](findings/0009-7.2b-comparison-radix.md) | 7.2B exact + dynamic-batch correctness (radix auto-off) |
| [F0010](findings/0010-m3b-de-fla-complete.md) | de-FLA complete — own WKV kernel, RWKV path FLA-free |
| [F0011](findings/0011-m4-quant.md) | M4 w8a8-int8 quant |
| [F0012](findings/0012-multigpu-coverage.md) | Multi-GPU coverage T4→H100 |
| [F0013](findings/0013-fusion-and-speed-standing.md) | Elementwise fusion + speed standing |
| [F0014](findings/0014-clean-same-precision-standing.md) | Clean same-precision standing (honest: albatross wins raw speed) |
| [F0015](findings/0015-cuda-endgame-result.md) | CUDA endgame result — fused GEMV + the honest ceiling |
| [F0016](findings/0016-serving-scale-wedge.md) | Serving-scale measured — ~50× concurrency throughput at flat VRAM; context-invariant memory (the O(1)-state wedge) |
| [F0017](findings/0017-w4-int4-quant.md) | Hand-written weight-only int4 — faster than (or ties) fp16 at every bsz≤32 (1.03–1.56×; RTX 3090, 1.5B — off-3090 the verified win is bsz1); 7.2B: 102.8 tok/s (1.29× albatross-fp16), 9.8GB total, lambada −2.64pt; GPTQ 1.5B −3.34pt; M>8 fused-GEMM = endgame |
| [F0018](findings/0018-w8-weight-only.md) | Hand-written weight-only int8 (w8a16) — greedy-EXACT 24/24, ≥fp16 at every bsz≤32 (1.02–1.37×), JIT-runs on every arch (vs cutlass w8a8 sm80–90) |
| [F0019](findings/0019-tp-pp-parallel.md) | TP+PP: full matrix greedy-EXACT on real L4 fleets (tp 2/4/8, pp 2/4/8, mixed tp2×pp2 after the v_first full-width fix); documents sglang's PP chunk-send pitfall for non-replicated proxy tensors |
| [F0020](findings/0020-fused-lora.md) | Fused LoRA (2 launches vs ~12/layer) — fp16 bsz1 203.0→226.5 tok/s (+11.6%), greedy EXACT; profile: lm_head now 58.5% of the step (fp16 bandwidth wall) |

See **[`../bench/results/`](../bench/results/)** for the committed measurement artifacts each
finding cites (`comparison_clean.md`, `lm_eval.md`, `sparse_ffn/`, `best2/`, …).
