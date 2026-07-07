---
doc_kind: finding
finding_id: F0051
title: "High-bandwidth-card decode gap (reverse-overtake W1): real H100 kernel-launch profile of the deployed fused fp16 stack (699 launches/step, NOT the stale ~144), the single highest-leverage remaining fusion (LoRA-output gate activations, byte-exact incl. the sigmoid transcendental), and the measured bsz1 speed delta + the GPU-busy ceiling that bounds what launch-count fusion can recover"
last_verified_commit: "HEAD"
discovered_by: lead, 2026-07-07
severity: info
status: open
related: [F0007, F0023, F0026, F0028]
---

# Finding F0051: reverse-overtake W1 — LoRA-gate fusion on the high-bandwidth-card decode gap

## 0. Context (what prompted this, attributed honestly)

The ours/Albatross single-stream decode ratio (§7, same 1.5B fp16 weights, same card)
gets **worse as card memory bandwidth rises**: L4 0.900×, RTX 3090 0.746×, RTX 5090
0.717×, H100 0.595×, H200 0.584×, B200 0.513×. **This bandwidth-correlation reading is
this project's own analysis of its own §7 table — it is an observed property of our
data, not a claim made by BlinkDL/Bo.** What Bo did say (relayed): *Albatross itself is
not yet fully optimized — there is still headroom there too — so the gap may be somewhat
understated (the target is still moving), and it is worse on the fast cards; dig deeper.*

The standing hypothesis (F0007/§7): our per-decode-step **kernel-launch overhead grows in
relative terms as compute gets faster** — Albatross runs a whole-layer mega-fused kernel,
we run many small independently-gated hand-written kernels (cleaner integration + per-kernel
correctness gating). On a slow card launch overhead is a small slice of a large per-token
time; on a fast HBM card the bandwidth-bound ideal shrinks, so the same fixed overhead is a
bigger fraction. This finding tests that hypothesis with a **real** profile (not the stale
"~144 kernels/step" internal note — verified below) and builds the single highest-leverage
remaining fusion.

Test environment: profiled + benchmarked on an H100 80GB HBM3 (sm_90) and an L4 (sm_89),
each a real GPU of that type. Profiler = `bench/profile_components.py` (random weights →
value-independent kernel timing; only `config.json` needed).

## 1. The real launch profile (H100, deployed fused fp16 bsz1 decode stack)

Stack = the deployed fused decode path: `RWKV_FUSED_GLUE=1` (paged shift+lerp6/lerp1),
`RWKV_FAST_LINEAR=1` (hand fp16 `gemv_m1`), `RWKV_FUSED_LORA=1` (`lora4_m1`, 2 launches);
sparse-FFN left off (opt-in, per-model). `torch.profiler` CUDA activities, eager, 200 iters,
1.5B (L=24, H=2048).

- **29.0 kernel launches / decoder layer**, 19 distinct kernels.
- **Full decode step ≈ 29.0×24 + 1 (lm_head) + 2 (emb/final) = ~699 launches.**
  → The prior internal "~144/step" note is **stale/wrong**; the M5-era pre-fusion profile
  (`bench/results/profile.md`) was ~1850/step (78/layer eager), and the current fused stack
  is ~699. Neither is 144. (Number now re-derived from source, per the brief.)
- Per-layer GPU-busy = 104.9 µs; lm_head 95.8 µs.

Per-layer kernels, **ranked by launch COUNT** (the launch-overhead hypothesis is about count,
so a fast-but-frequent kernel matters more than a slow-but-rare one):

| kernel (per layer) | #/layer | µs/layer | what it is |
|---|---|---|---|
| `gemv_m1_kernel<128,4>` | 5.0 | 33.2 | r/k/v/o + ffn.key projections (real GEMVs) |
| `vectorized_elementwise` (bundle) | ~12.0 | ~16.6 | **the un-fused pointwise glue** (see below) |
| `vectorized_layer_norm` | 2.0 | 11.2 | attn_norm + ffn_norm |
| `gemv_m1_kernel<256,4>` | 1.0 | 12.9 | ffn.value projection (larger N) |
| `shift_lerp6` / `shift_lerp1` | 1.0 / 1.0 | 5.6 / 5.3 | fused paged token-shift+lerp (R2) |
| `lora_stage1` / `lora_stage2` | 1.0 / 1.0 | 3.1 / 5.0 | fused 4-chain LoRA (M9) |
| `_kk_kmix` / `_gate_corr` / `_wkv_recurrent` | 1.0 each | 1.8 / 1.6 / 2.6 | fused triton kernels |
| GroupNorm (`RowwiseMoment`+`elementwise`) | 2.0 | 5.9 | g_norm |

The **6 `gemv_m1` are irreducible** (real projections; ~44% of per-layer GPU-busy). Everything
labelled "fused" is already one launch. The only large **un-fused** launch-count cluster is the
**~12 `vectorized_elementwise` tiny kernels/layer**, which decompose (verified against the
model source) as:

- **LoRA-output gate math (~7 launches):** `sigmoid(lo[0])`, `sigmoid(lo[1])`, `sigmoid(lo[3])`
  (the 3-count elementwise entry), the `-·*INV_SQRT_E` for `w_log`, and the v-residual
  `(v_first−v)`, `*sigmoid`, `+` chain. **← the single largest fusible cluster.**
- relu()²  in the FFN (2), residual adds `x+=attn_out` / `x+=ffn_out` (2), ≈1 more.

## 2. The fusion built — `fused_lora_gates` (fused.py Kernel D)

Collapses the ~7-launch LoRA-gate cluster (bsz1 fp16 `lora4_m1` path) into **one** triton
launch:

    w_log = -sigmoid(lo[0]) * INV_SQRT_E
    a     =  sigmoid(lo[1])
    v     =  v + (v_first - v) * sigmoid(lo[3])     # layer>0 only
    # g = lo[2] stays a caller-side slice (no kernel)

Wired in `models/rwkv7.py` behind `RWKV_FUSED_GATES` (default OFF, matching the cautious
`RWKV_FUSED_GLUE`/`RWKV_FUSED_LORA` pattern); fires only on the `lo is not None` (T==1 fused
LoRA) path, so bsz>1 (which uses `lo_mn` / torch fallback) is untouched by construction.

**Byte-exactness — the novelty is the transcendental.** Every existing fused kernel uses only
+,−,×,÷,√ and never a transcendental, precisely because triton's `tl.exp`/`tl.sigmoid` could
differ from torch by an fp32 ULP that a strict `torch.equal` gate would catch. Here we compute
sigmoid the way aten does for fp16 — `1/(1+exp(-x))` in fp32 (opmath), then round to fp16 —
using `tl.exp` (libdevice `__nv_expf`, == CUDA `std::exp(float)`), `enable_fp_fusion=False` to
stop ptxas contracting away the intermediate rounds (same trick as the other kernels). Whether
that is bit-exact was an open question resolved by the gate, not assumed.

## 3. Gates (non-negotiable, all passed before any speed claim)

- **Kernel byte-exact gate** (`bench/test_lora_gates.py`, `torch.equal` vs the exact torch op
  sequence): **PASS on L4 and H100**, `max_abs_diff = 0.0` across H∈{768,2048,4096}, C∈{3,4},
  input scale∈{0.5,2,8,30} (incl. saturating tails) and a knife-edge `linspace(-12,12)` sweep
  engineered to land sigmoid near fp16 rounding midpoints. → `tl.exp` matches `expf` tightly
  enough that the fp16-rounded sigmoid is identical to torch. The transcendental fusion **is**
  bit-exact-able on this stack.
- **End-to-end greedy-EXACT** (`bench/verify_batch.py`, 1.5B fp16, cuda-graph, vs numpy oracle):
  gates OFF → `OVERALL: PASS (all batches exact)`; gates ON → `OVERALL: PASS (all batches exact)`.
  The fusion changes zero output tokens.
- **Launch-count drop confirmed** (re-profile, gates ON, H100): per-layer **29.0 → 22.0**
  (−7/layer), full step **~699 → ~531**, GPU-busy/layer **104.9 → 95.5 µs**. `_lora_gates_kernel`
  (1 launch, 2.3 µs) replaces the ~8-launch elementwise cluster (the `vectorized_elementwise`
  count dropped from ~12 to 4/layer). The fusion cut GPU-busy too, not only launch count.

## 4. Speed delta — does it move the 0.595× ratio?

bsz1 decode tok/s, SAME fused baseline (glue+fast-GEMV+fused-LoRA), gates OFF vs ON, median of
3, cuda-graph ON (== how §7 measures). H100 = decisive (worst gap); L4 = the lower-bandwidth
control. bsz8 is the invariant: the fusion fires ONLY on the T==1 `lo` path, so bsz8 (which
uses the `lo_mn`/torch path) must be unchanged — and is, which proves the A/B isolates exactly
this fusion.

| card (sm) | bsz | OFF tok/s | ON tok/s | delta | ratio vs Albatross* |
|---|---|---|---|---|---|
| **H100 (9.0)** | **1** | **359.4** | **392.6** | **+9.24%** | **0.592× → 0.646×** |
| H100 (9.0) | 8 | 2106.8 | 2106.0 | −0.04% (invariant) | — |
| L4 (8.9) | 1 | 82.1 | 83.1 | **+1.22%** | — |
| L4 (8.9) | 8 | 545.7 | 545.6 | −0.02% (invariant) | — |

*Albatross H100 fp16 bsz1 = 607.3 (§7). Our OFF-arm 359.4 lands within run-noise of §7's own
361.1 (independent run) — the measurement setup is representative. (L4 absolute differs from §7's 102.2 because this
A/B uses its own fixed config — prefill 512 / decode 512 / mem-frac 0.85; only the same-config
OFF-vs-ON delta is the claim.)

**A real, well-controlled +9.24% on the worst-gap card — the ratio moves 0.592× → 0.646×.**
And it **confirms the bandwidth hypothesis directly**: the same fusion is only +1.22% on the
low-bandwidth L4 (where per-token time is ~4.4× longer, so the fixed per-kernel overhead is a
much smaller fraction), and never negative — i.e. the win concentrates exactly where the gap is
worst, with no regression on the near-parity card.

### Where the win actually came from (refines the hypothesis)

The naive story is "fewer launches → less CPU launch overhead." Under **full cuda-graph** that
CPU cost is already ~0, and indeed the non-GPU-busy residual barely moved: OFF step = 1/359.4 =
2783 µs with a ~2612 µs GPU-busy floor → ~171 µs overhead; ON step = 2547 µs with a ~2380 µs
floor → ~167 µs overhead. **The overhead was ~unchanged; the win (−236 µs/token) is almost
entirely a GPU-BUSY reduction** (104.9 → 95.5 µs/layer). Fusing the ~8 tiny elementwise kernels
removed their *fixed GPU-side execution cost* — grid setup + the [1,H] HBM read/write of each
intermediate that now stays in registers — which the profiler counts as GPU-busy, not as launch
overhead. So on a fast card the payoff of collapsing many tiny kernels is real but its mechanism
is per-kernel GPU-side overhead + intermediate HBM traffic, **not** CPU/graph launch latency.

### The ceiling that still bounds this axis

Even with the win, the ON GPU-busy floor is ~2380 µs = **~420 tok/s = ~0.69× of Albatross** — so
zeroing *all* remaining overhead on H100 bsz1 caps at ~0.69×. **The bulk of the high-bandwidth
gap is GPU-BUSY time**, dominated by the **6 GEMV projections (~44% of per-layer busy) at
sub-peak bandwidth** plus the elementwise work Albatross folds into its GEMM **epilogues** (≈0
extra GPU time) that we still run as standalone kernels (relu², residual adds, `_gate_corr`,
`_kk_kmix`). Closing past ~0.69× needs GEMV-efficiency + epilogue fusion, not more standalone
elementwise fusion (§5).

## 5. What to try next (redirected by the profile, ranked)

1. **Epilogue-fuse the elementwise INTO the GEMVs** (Albatross's actual technique), not into
   more standalone triton kernels — e.g. relu² into the ffn.value GEMV input, the gate math
   into the `lora_stage2` epilogue. Removes GPU-busy, not just launches. *Higher structural risk
   (touches the hand CUDA GEMV/LoRA kernels + their exactness gates).*
2. **Higher-efficiency M==1 GEMV** (128-bit vectorized loads, roadmap R5; profile.md item 3):
   the GEMVs are the GPU-busy floor, at sub-peak DRAM. This is the lever that actually attacks
   the 0.63× floor on fast cards.
3. **add+LayerNorm fusion** (fold the 2 residual adds into the next norm; Albatross's
   `add_layer_norm_*`): −2 launches/layer, structural (crosses module boundary).
4. r/k/v grouped GEMV (3→1): −2 launches/layer, bandwidth-neutral.

## 6. Attribution / method
SGLang integration + all kernels by this project (Fable, then lead). Performance reference =
Albatross faster3a / RWKV-LM v7 (not RWKV-CUDA). Bandwidth-correlation reading = our own §7
analysis. Long-term iterative work per the brief — this is one measured step, not a full close.

## Cross-references
[[F0007]] (0.746× 3090 baseline mechanism) · [[F0023]] (albatross kernel audit; layer-glue is
the bsz1 gap) · [[F0026]] (R2 glue +4.6% bsz1) · [[F0028]] (full-stack composes greedy-exact) ·
`bench/results/profile.md` (M5-era pre-fusion profile) · [[project-reverse-overtake-progress]].
