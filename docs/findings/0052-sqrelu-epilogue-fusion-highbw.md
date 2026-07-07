---
doc_kind: finding
finding_id: F0052
title: "High-bandwidth-card decode gap (reverse-overtake W1 cont.): epilogue-fusing the FFN relu()**2 activation into the key-projection GEMV (gemv_m1_sqrelu), byte-exact on sm_89/sm_90 incl. quantized-tier blast-radius containment, a real but modest +2.7-2.8% bsz1-decode win on H100 (ratio vs Albatross 0.646x->~0.66x, reproduced across 2 independent runs) and a noise-level +0.24% on L4, plus the F0051 GPU-busy ceiling recomputed (0.69x->~0.71x)"
last_verified_commit: "HEAD"
discovered_by: lead, 2026-07-07
severity: info
status: open
related: [F0051, F0028, F0023]
---

# Finding F0052: reverse-overtake W1 cont. — epilogue-fusing FFN sqrelu into the key-projection GEMV

## 0. Context

F0051 §5 ranked "epilogue-fuse the elementwise INTO the GEMVs (Albatross's actual technique)" as
the #1 next lever after the LoRA-gate cluster fusion, and named `relu(key(xk))**2` (the FFN
channel-mix activation) as one candidate. This finding builds and measures exactly that: fold the
2-launch `relu` + `pow(.,2)` epilogue that follows the `ffn.key` GEMV into the GEMV's own store,
so the intermediate `k[1,inter]` never round-trips to HBM and the 2 downstream elementwise kernels
vanish entirely.

(F0051's own phrasing put the candidate site at "the ffn.value GEMV **input**" — this finding
fuses it at the ffn.key GEMV's **output** instead. The activation is a pure function of `k`,
computed exactly once at the point `k` is produced; fusing it into the *consuming* `ffn.value`
GEMV instead would mean re-deriving `relu(k)**2` redundantly inside every output-tile block of
that GEMV. Fusing at the producer is the same idea, at what turned out to be the more efficient
site.)

This is the smallest of F0051's named epilogue-fusion candidates — one 2-launch elementwise
cluster, versus the ~7-8-launch LoRA-output-gate cluster F0051 fused — chosen specifically to
validate the *technique* (epilogue-fusion into the hand-written GEMV kernels, which is
higher-blast-radius than a standalone triton kernel: `rwkv7_fast.cu` is what every unquantized
fp16 fast-path decode projection, across all quantization tiers' fallback paths, depends on) and
to get an honest read on how much of F0051's ~0.69× ceiling this specific, contained piece is
worth.

Test environment: benchmarked on an H100 80GB HBM3 (sm_90) and an L4 (sm_89), each a real GPU of
that type — same class of hardware F0051 used.

## 1. The fusion built

New kernel `gemv_m1_sqrelu_kernel<Threads,OutTile>` in `rwkv7_fast.cu`: byte-for-byte the same
accumulation loop as `gemv_m1_kernel` (identical per-thread `k`-stride, identical `fmaf` order,
identical `warp_sum` + shared-memory cross-warp reduction) — only the final store differs:

```
// gemv_m1_kernel:         y[n0+j] = __float2half_rn(sum);
// gemv_m1_sqrelu_kernel:  f = __half2float(__float2half_rn(sum));  // == plain store + reload
                           r = f > 0.0f ? f : 0.0f;                 // == torch.relu on that fp16 value
                           y[n0+j] = __float2half_rn(r * r);        // == aten pow(.,2): b*b in fp32, round once
```

This reproduces the exact two rounding points of the torch path it replaces
(`torch.relu(gemv_m1(x, w)) ** 2`) instead of approximating them, per the project's per-kernel
exactness rule. New op `gemv_m1_sqrelu_cfg` (`TORCH_LIBRARY` entry mirroring `gemv_m1_cfg`'s
signature) + Python wrapper `fast_linear.gemv_m1_sqrelu`, which — like `gemv_m1` — resolves
`(threads, out_tile)` via the shared `_select_config(N, K)` autotuner, so both arms always agree
on launch config for a given `(N, K)`. That agreement is load-bearing: fp32 accumulation order in
this kernel family is a function of `Threads`, so a `torch.equal` claim would be meaningless if
the two arms could pick different configs.

Wired into `Rwkv7FeedForward.forward` via a new `_proj_gemv_sqrelu` helper, gated behind
`RWKV_FUSED_SQRELU` (default `"0"`, OFF — left off after this finding, see §7). Eligible only on
the unquantized fp16 `tp=1` bsz1-decode dense path, mutually exclusive by construction with the
M6 sparse-FFN kernel and the sparsity logger (both need the raw, un-activated `k`). W4/W8
quantized layers and any ineligible shape fall back unchanged to the two-step torch path.

## 2. Gates (all green before any speed claim)

- **Kernel byte-exact** (`bench/test_sqrelu_gate.py`, `torch.equal`, not tolerance-based): 5 real
  FFN `(N,K)` shapes (1.5B/7.2B/0.1B-derived), all 8 valid `(threads, out_tile)` configs, 3 input
  scales × 3 seeds op-level (config-matched, isolates the epilogue arithmetic from config choice)
  + 4 scales × 3 seeds adapter-level (the real `models/rwkv7.py` dispatch path) + a knife-edge
  sweep engineered to land `relu(k)**2` on fp16 rounding midpoints. **ALL PASS on both L4 (sm_89)
  and H100 (sm_90)** — literal `torch.equal`, zero diffs, including the saturating tails (`ovf` up
  to 0.50, i.e. half the outputs overflow to `+inf` at the largest scale, where `inf == inf` still
  holds bit-exact). Reproduced independently on the current commit by this finding's author (both
  cards, ALL PASS, ≈$0.05 / about a minute of GPU time) — this is a hard gate per project
  convention, so it was re-run first-hand rather than trusted from the session that wrote the
  kernel.
- **End-to-end greedy-EXACT** (`bench/verify_batch.py`, 1.5B fp16, cuda-graph, vs. the numpy
  oracle, `RWKV_FUSED_SQRELU` OFF then ON): `OVERALL: PASS (all batches exact)` both times,
  identical token IDs across all 3 batch-composition scenarios. The fusion changes zero output
  tokens end-to-end, not only in isolated op tests.
- **Quantized-tier blast-radius containment** (`bench/greedy_check.py`, w4 and w8a8 1.5B, flag OFF
  vs. ON): greedy token IDs bit-identical in both tiers. The new op is unreachable from quantized
  layers by construction (`_proj_gemv_sqrelu` routes `W4Linear`/`W8Linear` straight to the
  unchanged two-step torch path); this confirms it empirically, not only by code inspection.
- **Launch-count / GPU-busy drop confirmed** (`bench/profile_components.py kernels`, H100):
  22.0 → 20.0 launches/layer (−2, exactly the `relu` + `pow` pair), GPU-busy/layer 95.1 → 92.6 µs
  (−2.6%). The per-kernel accounting cross-checks cleanly: the two launches that disappear are
  precisely the two *singleton* `vectorized_elementwise_kernel<8,…>` entries (1.41 + 1.15 =
  2.56 µs combined) — a separate, unrelated 2-count `vectorized_elementwise_kernel<8,…>` bucket is
  untouched (2.78 → 2.77 µs, noise). And the GEMV-family total time is unchanged within noise:
  5×6.541 = 32.71 µs (OFF: a blended average over 4 cheap r/k/v/o launches + 1 larger ffn.key
  launch, all sharing the `<128,4>` template) vs. 4×4.858 + 13.32 = 32.75 µs (ON: the 4 cheap ones
  now average separately once ffn.key is bucketed under its own `gemv_m1_sqrelu_kernel<128,4>`
  label, and that fused kernel's 13.32 µs lands within 0.3% of ffn.key's OFF-state inferred cost
  of ≈13.27 µs). **The fused epilogue itself adds ~0 measurable GPU time** — the entire win is
  eliminating the 2 downstream launches and the `[1,inter]` HBM round-trip between them, not any
  change to the GEMV's own cost. Same story on L4 (459.2 → 457.5 µs/layer, −0.37%; the fused
  kernel costs 129.13 µs vs. an inferred 128.90 µs unfused — again within noise): the removed
  launches cost about the same absolute µs on both cards, but that's a vanishing fraction of L4's
  much larger per-layer GEMV time.

## 3. Speed delta

bsz1 decode tok/s, SAME full deployed fused baseline F0051 measured against (`RWKV_FUSED_GLUE=1`
+ `RWKV_FAST_LINEAR=1` + `RWKV_FUSED_LORA=1` + `RWKV_FUSED_GATES=1` in both arms — i.e. F0051's own
"ON" is this finding's "OFF"), `RWKV_FUSED_SQRELU` OFF vs. ON, median of 3, cuda-graph ON. bsz8 is
the invariant control: this fusion fires only at `M==1`, so bsz8 (torch path) must be unchanged.

| card (sm) | bsz | OFF tok/s | ON tok/s | delta | ratio vs Albatross* |
|---|---|---|---|---|---|
| **H100 (9.0), run 1** | **1** | **393.2** | **404.3** | **+2.82%** | 0.647× → 0.666× |
| **H100 (9.0), run 2 (independent repro)** | **1** | **386.3** | **396.8** | **+2.72%** | 0.636× → 0.653× |
| H100 (9.0), run 1 | 8 | 2109.8 | 2111.1 | +0.06% (invariant) | — |
| H100 (9.0), run 2 | 8 | 2094.7 | 2094.6 | −0.00% (invariant) | — |
| L4 (8.9) | 1 | 83.1 | 83.3 | **+0.24%** (noise-level) | — |
| L4 (8.9) | 8 | 545.2 | 546.0 | +0.15% (invariant) | — |

*Albatross H100 fp16 bsz1 = 607.3 (§7, same reference F0051 used). The two H100 runs were executed
independently (separate container instances, ~30 minutes apart — one by the session that built the
kernel, one by this finding's author as a first-hand re-check) specifically so the delta would have
to survive run-to-run noise before going in this doc. It does: +2.82% and +2.72% agree closely, and
both show the bsz8 control at ≈0%, confirming a clean A/B isolation both times.

**A real, if modest, win on the worst-gap card — smaller than F0051's LoRA-gate fusion (+9.24%),
roughly proportional to the smaller launch-count cluster it collapses (2 launches vs. ~7-8).** It
reproduces F0051's bandwidth-correlation reading once more: the identical code change is a real
win (~2.7-2.8%, several × the observed noise floor) on H100 and a noise-level non-event (+0.24%,
smaller than the bsz8 "invariant" control's own run-to-run wobble) on L4 — because the two removed
launches' fixed GPU-side cost (~2.5 µs combined) is ~2.6% of H100's 95.1 µs/layer but only ~0.4%
of L4's 459.2 µs/layer.

## 4. Updated ceiling (F0051 §"the ceiling that still bounds this axis")

F0051 computed: zeroing all remaining overhead on H100 bsz1 caps at a GPU-busy floor of
≈2380 µs/step ≈ 420 tok/s ≈ **0.69×** of Albatross (from its own measured 95.5 µs/layer +
95.8 µs lm_head). This finding's same-session pre-fusion baseline (95.1 µs/layer, 90.8 µs
lm_head — matching F0051 within ordinary run-to-run noise) reproduces that ceiling almost
exactly: 95.1×24 + 90.8 = 2373.2 µs → 421.4 tok/s → **≈0.69×** (0.694). With the sqrelu fusion
applied (92.6 µs/layer, 90.7 µs lm_head): 92.6×24 + 90.7 = 2313.1 µs → 432.3 tok/s → **≈0.71×**
(0.712).

**So: the F0051 ceiling estimate moves from ~0.69× to ~0.71×** — a small but real upward
revision. This fusion was itself one of the "remaining tiny elementwise kernels" the old ceiling
had implicitly priced in as unavoidable overhead; capturing it mechanically raises the floor that
ceiling describes. The realized ratio this finding measured (0.647–0.666×, run-dependent, §3) is
still below even the *old* 0.69× ceiling, so there is no contradiction between the two numbers —
just headroom identified but not yet fully spent (the gap between "realized" and "ceiling" is the
~167 µs/step of non-GPU-busy residual F0051 measured, which this finding did not touch).

## 5. What's left (F0051 §5, updated)

1. Epilogue-fuse elementwise into the GEMVs — **done for one 2-launch cluster (this finding)**.
   Remaining candidates F0051 named in the same breath: the 2 residual adds (`x += attn_out`,
   `x += ffn_out`) and the `_gate_corr` / `_kk_kmix` fused triton kernels. Naively extrapolating
   this finding's ~1.3 µs/launch removed cost to those ~4 launches suggests perhaps another
   ~5 µs/layer (~5% more GPU-busy reduction) if they turn out equally "free to fuse" — **this is
   an order-of-magnitude guess, not a measurement**. Residual-adds in particular would need to
   fuse into the *next* LayerNorm (crossing a module boundary, F0051 §5 item 3) rather than into a
   producing GEMV, so the mechanism — and therefore whether it is actually free the way this
   finding's epilogue was — is unproven.
2. **Higher-efficiency M==1 GEMV** (128-bit vectorized loads, F0051 §5 item 2) — still the biggest
   remaining lever: the GEMVs are the GPU-busy floor at sub-peak DRAM bandwidth, and this
   finding's own launch-accounting (§2) reconfirms they are ~44-47% of per-layer busy time,
   entirely untouched by any epilogue fusion (§2 showed the fused epilogue adds ~0 cost to the
   GEMV — which also means it cannot make the GEMV itself faster).
3. r/k/v grouped GEMV (F0051 §5 item 4): −2 launches/layer, bandwidth-neutral, not yet built.

## 6. Attribution / method

SGLang integration + all kernels by this project (Fable, then lead). Performance reference =
Albatross faster3a / RWKV-LM v7 (not RWKV-CUDA). Total GPU rental for this finding's
verification (kernelgate ×3 runs across both cards, verify, kprofile, quanteq, speed ×2 runs on
H100 + 1 on L4): ≈$1.3, a short targeted session, no instance left running afterward. Continues
F0051's iterative program (task #5) — one more measured step, not a full close.

## Cross-references
[[F0051]] (the launch profile + ceiling this extends; the larger LoRA-gate sibling win) ·
[[F0028]] (full-stack composes greedy-exact — the pattern `bench/greedy_check.py` /
`verify_batch.py` rely on here) · [[F0023]] (albatross kernel audit; named epilogue-fusion as the
Albatross-specific technique) · `bench/test_sqrelu_gate.py` (the gate) ·
[[project-reverse-overtake-progress]].
