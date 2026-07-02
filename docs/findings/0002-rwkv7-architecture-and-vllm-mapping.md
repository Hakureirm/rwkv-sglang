---
doc_kind: finding
finding_id: F0002
title: "RWKV-7 architecture & vLLM component mapping"
last_verified_commit: (initial)
discovered_by: recon (P10, 2026-06-30)
severity: info
status: open
related: [F0001, F0003]
---

# Finding F0002: RWKV-7 architecture & vLLM component mapping

## Hypothesis
RWKV-7's time-mixing is a gated/generalized delta rule, so it should map onto
vLLM's existing linear-attention / SSM stateful-model machinery (Mamba2, Gated
DeltaNet) with **no new C++/CUDA** (triton-JIT kernels).

## Method
Read the authoritative references cloned on Mac:
- `refs/fla/fla/ops/rwkv7/RWKV7(Goose).md` (math derivation + naive recurrence).
- `refs/fla/fla/models/rwkv7/{modeling,configuration}_rwkv7.py` (HF impl).
Probed the box `vllm` site-packages for the available linear-attn substrate.

## Result

### RWKV-7 time-mixing (per head, state `S ∈ ℝ^{V×K}`) — DPLR delta rule
```
w_t  = exp(-exp(w))                       # diagonal decay (per key dim)
sa   = (state * a_t).sum(dim=K)           # state-modulated removal  [V]
state = w_t · state + sa ⊗ b_t + v_t ⊗ k_t   # decay + low-rank + rank-1 update
o_t  = (state * q_t).sum(dim=K)           # output  [V]   (q a.k.a. r; scale 1/√K)
```
Per-token projections: `q(=r), k, v, w(decay), a, b` (6 vectors/head). This is
the "diagonal-plus-low-rank" (DPLR) generalized delta rule → reduces to
`fla/ops/generalized_delta_rule`.

### Model shape (RWKV7Config)
- Pure-recurrent: **every layer** = token-shift → time-mix (WKV7) → channel-mix.
  No full attention, no KV cache (optional `attn` dict enables hybrid layers).
- Defaults: `hidden_size=2048`, `num_hidden_layers=24`, `head_dim=64`,
  `hidden_ratio=4`, `hidden_act="sqrelu"` (squared-ReLU channel mixing).
- Low-rank projections produce decay/gate/a/v: `decay_low_rank_dim=64`,
  `gate_low_rank_dim=128`, `a_low_rank_dim=64`, `v_low_rank_dim=16`.
- `model_type='rwkv7'`; `value_dim` may exceed hidden_size (multi-head v).

### Per-sequence state to cache (the "state cache" requirement)
- WKV state matrix: `[num_layers, num_heads, K, V]` (constant size, ctx-independent).
- Token-shift buffers: 2 per layer (time-mix + channel-mix), each `[hidden]`.

### vLLM substrate available (templates / kernels)
- **Models**: `qwen3_next.py` (GatedDeltaNet, closest layer template),
  `mamba2.py` (pure-stateful shape), KimiLinear, MiniMax lightning-attn.
- **State backends (v1)**: `gdn_attn.py` (Gated DeltaNet — closest),
  `mamba2_attn.py`, `linear_attn.py`, `short_conv_attn.py`.
- **Vendored `fla` ops**: delta-rule subset (`chunk_delta_h`,
  `chunk_scaled_dot_kkt`, `solve_tril`, `wy_fast`, `kda`, `fused_recurrent`) —
  **NO rwkv7 wrapper** → must port/adapt `fla/ops/rwkv7/{chunk,fused_recurrent}`.
- Token-shift maps to `layers/mamba/short_conv.py` / `ops/causal_conv1d.py`.

## Conclusion
Scope = **Python** work: (1) `rwkv7.py` model file; (2) a state attention
backend (model after `gdn_attn`/`mamba2_attn`); (3) port fla rwkv7 triton ops
(chunk = prefill, fused_recurrent = decode); (4) RWKV "World" tokenizer; (5)
config/weight loading; (6) reuse vLLM quant for 8/4-bit. **No new C++/CUDA.**
Prefill/decode split (chunk vs recurrent) aligns with vLLM's mamba/gdn backends'
existing continuous-batching design.

### sglang mapping (the CHOSEN track — confirmed paths, HEAD f920a37)
Same component shape as vLLM (sglang's fla was adapted from vLLM's). Targets:
- Template: `python/sglang/srt/models/qwen3_next.py` → author `models/rwkv7.py`.
- State backend: `python/sglang/srt/layers/attention/linear/{gdn_backend,
  kda_backend,lightning_backend}.py` → add `rwkv7_backend.py`.
- Vendored fla: `python/sglang/srt/layers/attention/fla/` (gated-delta subset;
  **no rwkv7**) → port `chunk.py`/`fused_recurrent.py` + `wy_fast`/`solve_tril`/
  `chunk_delta_h` from `refs/fla/fla/ops/rwkv7`.
- State cache: `python/sglang/srt/mem_cache/mamba_radix_cache.py` +
  `mamba_checkpoint_pool.py` (RWKV-7 `[K,V]` matrix state fit = open risk, ADR-0002).
- Token-shift: `python/sglang/srt/layers/attention/mamba/causal_conv1d.py`.
- Structural blueprint to adapt (NOT clone): `refs/pr41060-rwkv7-goose.diff`.

## Cross-references
- [[F0001]] env, [[F0003]] baselines/oracle, [[F0004]] re-analysis. ADR-0002.
  Refs: `refs/sglang`, `refs/fla`, `refs/RWKV-LM`, `refs/pr41060-rwkv7-goose.diff`.
