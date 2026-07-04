---
doc_kind: adr
adr_id: 0005
title: "Reverse-overtake execution roadmap — the plan to make RWKV-7×SGLang beat albatross even on its home turf"
status: accepted
date: 2026-07-03
last_verified_commit: "ab50b2b"
supersedes: []
superseded_by: []
---

# ADR-0005: Reverse-overtake execution roadmap

## Context
The albatross kernel audit (F0023, source-verified + adversarially checked) + the serving/eval
work (F0024 MATH500 & best-bsz, F0025 PD-mixed & GEMV autotune) map exactly where we lead, where
we're at parity, and the two places albatross still leads. Goal (CTO directive): make the SGLang
adaptation so complete that others — including BlinkDL/Bo — cannot reach it. This ADR sequences the
work into an executable plan with interfaces + test gates, so the build push is fast and correct.

## Standing moats (albatross structurally lacks; keep + surface, don't rebuild)
- **Serving architecture**: sglang scheduler + dynamic batching + paged state pool + state-aware
  MambaRadixCache (F0022, ~98% hit high-reuse). albatross = one static (B,T) mega-graph, no
  scheduler. Peak decode **6885 tok/s @ bsz384** (F0024); PD-mixed tail-latency curve (F0025).
- **Quantization**: w8a8 int8-TC (F0025, decode +15–53%), hand int4+GPTQ, hand w8a16 all-arch.
  albatross = fp16 only, zero int8 path.
- **Multi-GPU** TP 2/4/8 + PP 2/4/8 + mixed (F0019, greedy-exact), **10-GPU cross-arch** coverage.
- **Upstream** sglang-main port + PP-transfer bug (issue #30015).

## The two places albatross still led (F0023 §4) — BOTH now closed (R2, R3)
1. **bsz1 decode latency** — its whole-layer mega-fusion collapses each layer boundary into ~1–2
   kernels vs our ~7–8 launches; this is the entire F0007 0.73× bsz1 gap. NOT a kernel-quality gap
   (our GEMV is a byte-exact vendoring of albatross's; F0023 §2) — a *fusion-density* gap.
2. **LoRA batched-M** (pre-R3 gap) — albatross fuses M≤8/M≤4; our `lora4_m1` was M==1 only. **Closed by R3**: `lora4_mn` covers batched M, now M-gated to ≤4 (loses to cuBLAS above; F0028).

## Execution roadmap (ranked; each with interface + test gate)

### R1 — w8a8 large-M measurement (F0023 #1)  [STATUS: DONE, F0025 Part C]
w8a8 is already realized (sglang-native, F0025). Measured bsz 1–512 with `--cuda-graph-max-bs 512`
(prior tables stopped at 32): **w8a8 peak 9152 tok/s @ 512 vs fp16 6885 @ 384 = +33%**, +30–38% in
the 256–512 band. int8-TC overtake that albatross cannot follow (no int8 path) stacked on top of the
serving-architecture moat. Accuracy: 7.2B 8/8 EXACT; 1.5B free-running diverges at token 12/24
(small-model int8 drift, noted). Artifact: `bench/results/bsz_sweep_w8a8.json`. **R6** (all-arch
hand-written w8 for sm<80) remains.

### R2 — Paged-cache-aware fused layer-boundary kernels (F0023 #2)  [DONE + verified, both boundaries]
**STATUS (2026-07-04): attn + ffn shipped + greedy-verified.** Both boundaries' token-shift+lerp are
now fused: `shift_lerp6` (attn, 6-way) + `shift_lerp1` (ffn, 1-way; ffn `x_k` is fp16 so its
plain-torch lerp rounds identically). Both confirmed FIRING ("R2 fused ...glue ENABLED" attn+ffn) +
`verify_batch` OVERALL PASS greedy-exact. `shift_lerp6` (fuses paged
token-shift gather+scatter — dropping the `.clone()` — with the 6-way lerp, `shifted` stays on-chip)
in `rwkv7_glue.cu`; conv is **fp32** (matched: `sh=round_fp16(conv)`, `conv<-float(normed)`);
cache_indices is int32 → passed directly (cuda-graph-safe, no copy). Kernel gate: byte-exact vs
token_shift+fused_lerp6 (`bench/test_glue.py`, T∈{1..32}). Integrated in `Rwkv7Attention.forward`
(decode, env `RWKV_FUSED_GLUE`, `try_fused_shift_lerp6` in the backend, falls back otherwise).
**E2E gate PASSED**: `verify_batch --dtype float16 RWKV_FUSED_GLUE=1` on 1.5B → IDENTICAL 4/4,
SHARED 5/5, MIXED 6/6 OVERALL PASS, with "R2 fused glue ENABLED" confirmed firing on decode.
**Speed (clean same-config A/B, greedy-exact):** on the full fast stack (fast_linear+fused_lora+
sparse_ffn, fp16, cuda-graph), toggling only the glue: bsz1 **209.3 → 219.0 tok/s = +4.6%** (F0026).
(A glue-only A/B without sparse-ffn showed +20% off a slower 161→194 baseline; +4.6% is the honest
gain on the already-fast stack — the glue removes a fixed per-layer HBM cost that shrinks in % as the
baseline speeds up.) **`shift_lerp1` ffn-side is SHIPPED + verified** (x_k is fp16 so the plain-torch
ffn lerp rounds identically; wired in `Rwkv7FeedForward.forward`, confirmed firing + verify_batch
PASS with both boundaries — F0026/F0028). **Remaining**: (i) full cuda-graph serving run (both
boundaries under production graph capture). (ii) fold LN into the kernel if numerically matchable
(further HBM saving). Original plan below.


Write `add_ln_mix6_shift` (residual-add + LN1 + 6-way time-mix lerp + in-place conv[cache_indices]
shift-store) and `add_ln_cmixmix_shift` (add + LN2 + channel-mix lerp + shift-store → feeds the
sqrelu key GEMV), mirroring albatross `rwkv7_v3a_ops.cu:1745-1825,:1623-1683` BUT doing the
shift-store in-place into the **paged** `conv[cache_indices]` so sglang serving is preserved.
Collapses ~7–8 glue launches/layer → ~2, removes token_shift clone-gather+scatter round-trip.
- **Interface**: new ops in a `rwkv7_glue.cu`; gated by env `RWKV_FUSED_GLUE=1` (default off until
  proven), integrated at `models/rwkv7.py` boundary (replaces the LN+token_shift+lerp+add sequence).
- **Test gate**: greedy token-EXACT vs numpy oracle (24/24) AND vs the current unfused path;
  bsz1 decode tok/s recovers toward albatross 309 (target: close most of 226→309). Feasibility
  proven by our existing `INDEXED_STATE` paged fusion in `wkv_recurrent`.
- **Risk**: high (intricate fused kernel). Build incrementally: (a) add+LN fused first, gate; (b)
  add token-shift+lerp; (c) add in-place paged shift-store. Test each stage token-exact.

**Turnkey scoping (verified 2026-07-03, ready to build).** Boundary is `Rwkv7DecoderLayer.forward`
(`models/rwkv7.py:864-866`): `attn_out,vf = attn(attn_norm(x)); x = x + attn_out; x = x + ffn(ffn_norm(x))`.
Per-layer DECODE glue launches (the bsz1 gap):
  1. `attn_norm(x)` — LayerNorm (weight+bias, eps).
  2. attn entry token_shift (`rwkv7_backend.py:135-137`, decode branch): `prev = conv[cache_indices,:,0].clone()` (gather) + `conv[cache_indices,:,0] = x` (scatter) = **2 paged ops**.
  3. `fused_lerp6(x, shifted, mix6)` → [6,T,H] (already fused).
  4. `x = x + attn_out` — residual add.
  5. `ffn_norm(x)` — LayerNorm.
  6. ffn token_shift (2 paged ops) + ffn lerp (`xk = x + x_k·(shifted−x)`).
  7. `x = x + ffn_out` — residual add.
Two fused kernels to write (decode T==1-per-req first; the paged conv is `[size+1, H, 1]`, indexed by
`md.mamba_cache_indices`):
  - **`add_ln_shift_lerp6`**: takes residual `x_prev` + `attn_out`(=0 at layer entry via the model's
    `x` in) — actually fuse at the ffn→next-attn seam: `x2 = x + attn_out; normed = LN1(x2, w1,b1);
    prev = conv1[ci]; conv1[ci] = x2; lp6 = lerp6(x2, prev, mix6)` → returns (x2, lp6). Removes the
    clone+separate-scatter + the standalone LN + add. gather+scatter fused with the LN read of x2.
  - **`add_ln_shift_lerp1`**: `x3 = x + ffn... ; normed = LN2(x3,w2,b2); prev=conv2[ci]; conv2[ci]=x3;
    xk = x3 + x_k·(prev − x3)` → returns (x3, xk). Feeds the sqrelu key GEMV.
- **Correctness crux**: the in-place `conv[ci] = x` scatter MUST happen after reading `prev` (order),
  and per-request `ci` indexing must match `token_shift`. Stage (a) = do JUST the LN read + fused
  add (no paged touch) and gate token-exact; stage (b) add the lerp; stage (c) fold the paged
  gather/scatter in (highest risk). Test each stage: `verify_batch --dtype float16 RWKV_FUSED_GLUE=1`
  IDENTICAL/SHARED/MIXED must stay EXACT (same gate R3 passed). Prefill (T>1 ragged) path uses the
  `query_start_loc` boundaries (`rwkv7_backend.py:140-147`) — handle after decode path proven.

### R3 — Extend fused LoRA to batched M (F0023 #4)  [DONE + greedy-verified]
`lora4_mn(xs[M,C,H]) -> y[M,C,H]` in `rwkv7_lora.cu` (M grid dim both stages, per-(m) reduction ==
`lora4_m1`). Kernel gate: `lora4_mn[m]` byte-identical to `lora4_m1(xs[m])` ∀m∈{1..32}
(`bench/test_lora_mn.py`). **Integrated** into `models/rwkv7.py` (T>1 → build `xs=lp[2:2+C].permute
(1,0,2).contiguous()`, call `lora4_mn`, index `lo_mn[:,c]`) + `lora_fused.py` wrapper + register_fake,
env-gated `RWKV_FUSED_LORA`. **End-to-end gate PASSED**: `verify_batch --dtype float16` with
`RWKV_FUSED_LORA=1` on 1.5B → IDENTICAL 4/4, SHARED-PREFIX 5/5, MIXED 6/6, **OVERALL PASS** (greedy
token-exact vs numpy oracle at M>1). Bug found+fixed: `g = lo_mn[:,2]` was a strided column slice
(fused_gate_corr assumes contiguous) → `.contiguous()`.
**M-gate (measured, important correction):** a serving A/B revealed `lora4_mn` at large M is
SLOWER than the cuBLAS-batched ReplicatedLinear (it's correctness-first, no smem) — at c=128 the
full stack cratered 6893→3265 tok/s, isolated to fused-LoRA (removing it recovered 6971). Crossover
sweep (fused on vs off, other fast paths on): fused WINS at M≤4 (c1 235 vs 204 +15%, c2/c4 +8-15%)
but LOSES at M≥8 (c16 1445 vs 1859, c32 2009 vs 3129 −36%). So the fused LoRA is **M-gated to
T≤`RWKV_FUSED_LORA_MAX_BS` (default 4)**; above it falls back to cuBLAS. This is a concrete instance
of the (card×precision×bsz) autotune principle: the fused kernel is enabled only in the bsz band
where it wins. Greedy-exact preserved on both paths (verify_batch bsz4=fused / bsz5-6=fallback PASS).

### R4 — Arch-aware GEMV autotune (F0023 #6)  [STATUS: A-seg done, F0025]
`gemv_m1_cfg` + `_select_config` shipped + 3090-seeded (token-exact). B-seg = cross-arch
occupancy pass (task #14) seeds/validates other-arch rows → the portability overtake vs albatross's
per-GPU hand-tune. **Gate**: per-arch best-config selected without manual tuning; no accuracy change.

### R5 — 128-bit vectorized GEMV loads + lm_head-through-gemv (F0023 #3,#5)  [small]
Stride K-loop by 8 halfs, load uint4/float4 (guard K%8==0). Route M==1 lm_head through gemv_m1.
**Gate**: token-exact; measurable-or-neutral (bounded by DRAM floor / cuBLAS quality).

### R6 — w8a8 all-arch extension (F0023 #1b)  [large, later]
Our hand-written `rwkv7_w8.cu` (group-wise-K, all-arch) can carry int8 benefit to sm<80 where
cutlass `int8_scaled_mm` doesn't ship. Needs group-wise-K → per-channel for a single post-accumulate
scale. **Gate**: token-exact + speedup on a Turing (T4) card where cutlass w8a8 is unavailable.

## Sequencing decision
R1 (measuring now) → R3 (medium risk, high confidence, closes a real gap) → R2 (biggest lever,
staged/incremental because high risk) → R4-B DONE (F0027) → R5 (small) → R6 (large,
later). Every kernel gated greedy-token-EXACT vs the numpy oracle before it can be enabled by
default — no accuracy regression is the non-negotiable invariant (per [[feedback-benchmark-rigor]],
[[feedback-elegance-and-adversarial-review]]).

## Consequences
- Closing R2+R3 removes albatross's *only* remaining advantages (bsz1 latency, LoRA-M) while we
  keep every serving/quant/multi-GPU moat it lacks → no axis on which albatross leads.
- Attribution unchanged: perf reference = albatross faster3a / RWKV-LM v7; SGLang design = ours
  ([[reference-rwkv7-code-map]]). Keep public materials free of FLA-dependency implication (ADR-0004)
  and of bounty/requirement framing (serious, formal).

## Cross-references
[[F0023]] audit + roadmap · [[F0024]] MATH500 + best-bsz + cuda_graph_max_bs · [[F0025]] PD-mixed +
[[F0026]] R2 glue · [[F0027]] cross-arch occupancy · [[F0028]] full-stack + M-gate ·
autotune · [[F0022]] state cache · [[F0018]] w8 weight-only negative · [[F0020]] fused LoRA.
