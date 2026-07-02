---
doc_kind: finding
finding_id: F0019
title: "TP + PP multi-GPU: head-parallel tensor parallelism and layer-partition pipeline parallelism, both greedy token-EXACT on real 2-GPU hardware (24/24), zero tp=1/pp=1 regression"
last_verified_commit: "HEAD"
discovered_by: lead (M10), 2026-07-02
severity: info
status: open
related: [F0008, F0018]
---

# Finding F0019: tensor + pipeline parallelism

## TP design (head-parallel)
head_dim (64) stays whole; whole heads split across ranks. r/k/v projections and
every LoRA up-projection are ColumnParallelLinear (no gather — outputs stay on
the local head slice); o_proj and ffn.value are RowParallelLinear (one allreduce
each, 2/layer total); ffn.key column-parallel. Per-channel params (k_k, k_a,
r_k), GroupNorm (per-head groups → mathematically exact under head splits), the
WKV recurrence and its state live on the local head slice. The token-shift mix
vectors AND the conv (prev-token) state stay **full-width per rank** — they feed
the replicated hidden that the column-parallel projections consume (state cost:
2×H×fp32 per layer per seq — negligible vs the (nh,64,64) WKV state, which IS
divided by tp). LoRA down-projections stay replicated (full replicated input,
tiny rank output — no comm). W4/W8 quantized projections require tp=1 for now
(explicit NotImplementedError).

## PP design (layer partition)
llama-pattern: make_layers partition, embeddings on the first rank, final norm +
logits on the last, PPMissingLayer elsewhere. RWKV-7 wrinkle: layer 0's value
projection **v_first** is consumed by every later layer (v-residual mixing), so
it travels across stages in the PPProxyTensors dict next to hidden_states
(where llama sends `residual`). The mamba state pool needed NO changes: sglang
already allocates only the local rank's layers and remaps global layer_id →
local pool index (HybridReqToTokenPool.mamba2_layer_cache), and our backend
passes global layer ids. (The GDN reference model qwen3_next asserts PP
unsupported — RWKV-7 is ahead here.)

## Measured (all greedy vs the numpy oracle fixture)
| config | hardware | greedy | decode bsz1/8/32 tok/s |
|---|---|---|---|
| tp=1 / pp=1 regression (0.1B + 1.5B bf16 + 1.5B w8 fp16) | RTX 3090 | **EXACT 24/24 ×3** | unchanged |
| **tp=2** (1.5B bf16) | **2× L4 (real)** | **EXACT 24/24** | 20.6 / 161.9 / 644.4 |
| **pp=2** (1.5B bf16) | **2× L4 (real)** | **EXACT 24/24** | 17.3 / 133.1 / 525.1 |

Notes: 2-GPU runs use the correctness-gate config (cuda-graph OFF, bf16) — these
are functional-verification numbers, not perf numbers; single-L4 WITH graph does
76 tok/s bsz1 (multigpu grid), so do not quote the gate numbers as speed. The
tp=2 exactness is stronger than expected: the two row-parallel allreduces
reorder fp reductions vs the single tp=1 GEMM, yet the fixture still matched
token-for-token (bf16, 24 tokens — knife-edge argmax drift remains possible on
other prompts; lm-eval parity is the right long-form gate if it ever appears).

## Known limits / follow-ups
- W4/W8 × tp>1 (shard qweight rows/scale columns — group=64 divides cleanly).
- tp2/pp2 throughput-tuned numbers (cuda-graph ON) + tp4/pp4 + 7.2B multi-GPU.
- dp-attention (get_attention_tp_size vs TP world size) untested — unsupported.

## Cross-references
[[F0008]] (radix off) · [[F0018]] (w8) · `bench/verify_tp.py` ·
`sglang_overlay/sglang/srt/models/rwkv7.py` · `configs/mamba_utils.py`.
