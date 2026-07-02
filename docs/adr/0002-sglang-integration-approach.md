---
doc_kind: adr
adr_id: 0002
title: "sglang integration approach for RWKV-7"
status: accepted
date: 2026-06-30
last_verified_commit: (initial)
supersedes: []
superseded_by: []
---

# ADR-0002: sglang integration approach for RWKV-7

## Context
ADR-0001 committed the **sglang track**. RWKV-7 ([[F0002]]) is a pure-recurrent,
every-layer-stateful model (token-shift → WKV7 DPLR delta-rule time-mix →
sqrelu channel-mix; no KV cache). sglang already has the serving substrate
(verified): vendored `fla` kernels (gated-delta subset), `RadixLinearAttention`,
`MambaRadixCache` + `mamba_checkpoint_pool` state cache, chunked prefill, dynamic
batching, spec decode, PD-disagg; `qwen3_next.py` (Gated DeltaNet) is the closest
integration template. The closed vLLM PR **#41060** (clean ~3.9k-LOC RWKV-7 V1
integration) is a structural reference to adapt (not clone).

Exact sglang paths to confirm against the landing clone (`refs/sglang`):
`python/sglang/srt/models/qwen3_next.py`, `.../layers/radix_linear_attention.py`,
`.../layers/attention/fla/` (vendored, gated-delta only), `.../mem_cache/
mamba_radix_cache.py`, `.../layers/attention/mamba/`.

## Options considered
1. **Editable sglang source install + add `models/rwkv7.py`** (CHOSEN for dev).
   Add the model file + register it; vendor RWKV-7 fla kernels into sglang's fla
   tree; wire state into the mamba state-cache machinery. Pro: full control,
   matches how every other linear model lives in sglang, upstreamable. Con:
   editable source build (sgl-kernel/flashinfer deps) without GitHub on the box.
2. **PyPI sglang + runtime monkeypatch/plugin**. Pro: avoids source build. Con:
   sglang has no stable out-of-tree model API like vLLM's; fragile.
3. **transformers-backend fallback** (sglang runs the HF/fla model). Pro: fast to
   stand up. Con: inherits fla's accuracy misalignment + no albatross-parity
   kernels → fails the production bar. Useful only as an early smoke harness.

## Decision
**Editable sglang source install; implement `models/rwkv7.py` following
`qwen3_next.py`; bind RWKV-7 kernels; use the mamba state-cache machinery for the
recurrent state.** Build order = ADR-0001 milestones. Concretely:

1. **Kernels**: port `fla/ops/rwkv7/{chunk.py, fused_recurrent.py}` (+ deps
   `wy_fast`, `solve_tril`, `chunk_delta_h`) into sglang's vendored fla tree.
   `chunk` = prefill, `fused_recurrent` = decode — mirroring the mamba/gdn
   prefill/decode split. (sglang vendored only the gated-delta subset; RWKV-7's
   vector-decay DPLR form needs the rwkv7-specific wrappers.)
2. **Model**: `RWKV7ForCausalLM` — token-shift (causal_conv/short_conv), time-mix
   (6 low-rank projections → q/k/v/w/a/b → WKV7 kernel), sqrelu channel-mix;
   weight loader for **fla-format AND raw BlinkDL `.pth`** (HF + ModelScope).
3. **State cache**: represent RWKV-7 per-head `[K,V]` matrix state + 2 token-shift
   vectors/layer in `MambaRadixCache`/`mamba_checkpoint_pool`. **Open risk** (ADR-
   0001): the radix cache assumes Mamba2-shaped state; if RWKV-7 state doesn't fit
   prefix-reuse, add a state-pool adapter and log a finding. M1 may run WITHOUT
   prefix reuse (per-request fresh state) to get correctness first.
4. **Tokenizer**: RWKV World trie (65,536 vocab, not BPE) → HF-compatible wrapper
   (port BBuf/RWKV-World-HF-Tokenizer) into sglang's tokenizer pipeline.
5. **Oracle (correctness gate)**: BlinkDL `rwkv` pip (cuda fp16, `RWKV_V7_ON=1`)
   and `RWKV-v7/rwkv_v7_numpy.py` for bit-level logits. **NOT fla** (misaligned).

## Elegance Law footgun-ledger (ADSD Part 4 — re-design, not clone)
The sglang surface drops, rather than reproduces, predecessor footguns:
- **No silent miscompile / decorative gates**: any kernel/state-shape mismatch
  raises, never returns garbage that "runs" (lesson: vLLM PRs' CI never ran →
  parity unproven; we gate on the numpy oracle, not on "it generates text").
- **fla-as-oracle footgun avoided**: we explicitly forbid fla as the accuracy
  reference (it's misaligned) — a trap an unaware port would fall into.
- **Static-batch footgun avoided**: unlike albatross (static B/T), we deliver
  true dynamic batching + chunked prefill from the state-cache layer up.
- **Quant footgun avoided**: rwkv-pip int8 is *slower* than fp16; our 8/4-bit
  must be VRAM-down AND not-slower (explicit gate, not aspiration).
- **Typed config over option-bag**: a typed `RWKV7Config` mapping (fla-name ↔
  BlinkDL-name) rather than stringly-typed weight-key guessing.

## Done means
ADR-0002 is satisfied when M1 (ADR-0001) passes: `models/rwkv7.py` + ported
kernels + tokenizer load an RWKV-7 0.1B checkpoint in editable sglang and produce
**greedy tokens identical to the numpy oracle** over ≥1000 prompts.

## Consequences
### Positive
Clean, upstreamable model file on a mature substrate; correctness gated on the
true oracle; kernel work reusable for a future vLLM plugin.
### Negative / Risk
Editable sglang build without GitHub on the box (mitigate: clone-on-Mac→rsync,
PyPI for heavy deps incl. `sgl-kernel`/`flashinfer`); state-cache fit risk (#3);
albatross-parity speed needs a tuned WKV7 kernel, not just the triton fwd path.

## Cross-references
ADR-0001 · [[F0002]] · [[F0003]] · refs: `refs/sglang`, `refs/fla`,
`refs/pr41060-rwkv7-goose.diff`, `refs/Albatross`.
