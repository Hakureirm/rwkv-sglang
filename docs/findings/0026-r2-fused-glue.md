---
doc_kind: finding
finding_id: F0026
title: "R2 fused paged layer-boundary glue (shift_lerp6): fuses the paged token-shift (gather+scatter, dropping the .clone()) + 6-way lerp into one on-chip kernel — the albatross mega-fusion technique adapted to sglang's paged conv state. Byte-exact + greedy E2E PASS; +4.6% bsz1 decode (209.3→219.0, clean same-config A/B) on top of the full fast stack"
last_verified_commit: "c4e58e0"
discovered_by: lead (M13), 2026-07-04
severity: info
status: open
related: [F0007, F0023, F0020]
---

# Finding F0026: R2 fused paged layer-boundary glue

## What + why
F0023 §4 established that the remaining bsz1-decode gap vs albatross (F0007 226.5 vs 309.2, both
cuda-graph ON) is **not** kernel quality (our GEMV is a byte-exact vendoring of albatross's) but
**fusion density**: albatross collapses each layer boundary into ~1–2 kernels keeping intermediates
on-chip, while we launched ~7–8 (LN, token_shift gather+scatter, lerp, add …), each round-tripping
HBM. Under cuda-graph the *launch* overhead is captured, so the residual cost is the **HBM
round-trips** — memory-bound at bsz1. R2 attacks that.

`shift_lerp6` (`cuda/rwkv7_glue.cu`) fuses the attn-entry glue: paged token-shift (gather prev from
`conv[cache_indices]` + scatter current normed, **dropping token_shift's full `.clone()`**) + the
6-way lerp, keeping `shifted` **on-chip** (registers) instead of writing it to HBM between a
token_shift kernel and a separate lerp kernel. This is albatross's mega-fusion idea adapted to
sglang's **paged** conv state (the novel part — albatross's is a dense per-batch state).

## Correctness (the non-negotiable gate)
- **Kernel byte-exact** vs `token_shift + fused_lerp6`, including the conv scatter, for T∈{1,2,4,8,32}
  (`bench/test_glue.py`, `torch.equal`). Replicates the exact fp16 rounding of `fused.py:_lerp6_kernel`
  (d=round16(sh−x); prod=round16(mix·d); o=round16(x+prod)) and token_shift's dtype casts — conv is
  **fp32**, so `sh = round_fp16(conv)` (= `prev.to(fp16)`) and `conv ← float(normed)` (= `x.to(fp32)`).
- **End-to-end greedy-EXACT**: `verify_batch --dtype float16 RWKV_FUSED_GLUE=1` on 1.5B → IDENTICAL
  4/4, SHARED-PREFIX 5/5, MIXED 6/6, OVERALL PASS, with "R2 fused glue ENABLED" confirmed firing on
  the decode path (not a silent fallback). So the fusion changes speed, not logits.

## Speed (clean same-config A/B, greedy-exact both arms)
Full fast stack (`RWKV_FAST_LINEAR=1 RWKV_FUSED_LORA=1 RWKV_SPARSE_FFN=1`, fp16, cuda-graph on,
`--cuda-graph-max-bs 8`, `bench/bsz_throughput.py` c=1 in64/out512, n=40), toggling only the glue knob:

| config | bsz1 decode tok/s |
|---|---|
| glue OFF | 209.3 |
| **glue ON** | **219.0** (**+4.6%**) |

The glue removes a fixed per-layer HBM cost (the `shifted` write + the `.clone()`), so its relative
gain shrinks as the baseline gets faster (a glue-only A/B *without* sparse-ffn showed +20% off a
161→194 baseline). +4.6% is the honest gain on top of the already-optimized stack. (This is the
`bsz_throughput` wall-clock metric; not directly comparable to F0007/F0020's steady-state
decode-tok/s methodology — reported as a same-config delta, not a replacement of the 226.5 figure.)

## Scope / remaining
- **attn side shipped** (`Rwkv7Attention.forward`, env `RWKV_FUSED_GLUE`, default off; backend
  `try_fused_shift_lerp6`, falls back when ineligible — non-decode / non-fp16 / non-fp32-conv).
- **ffn side** (`shift_lerp1`): **DONE + verified**. The ffn `x_k` is fp16, so its plain-torch lerp
  (`x + x_k·(shifted−x)`) rounds identically to the kernel; wired in `Rwkv7FeedForward.forward`,
  confirmed FIRING ("R2 fused ...shift+lerp1 (ffn) glue ENABLED") + `verify_batch` OVERALL PASS with
  BOTH attn and ffn glue active. So the entire per-layer token-shift+lerp glue (both boundaries) is
  now fused + paged + on-chip.
- **cuda-graph safety**: cache_indices is int32 and passed directly (no per-call copy); conv pointer
  is the stable cache. A full cuda-graph serving run is the next validation.
- **later**: fold LN into the kernel if its numerics prove matchable (further HBM saving).

## Cross-references
[[F0007]] (bsz1 gap) · [[F0023]] §2/§4 (fusion-density diagnosis) · [[F0020]] (fused LoRA) ·
ADR-0005 R2 · `cuda/rwkv7_glue.cu` · `bench/test_glue.py`.
