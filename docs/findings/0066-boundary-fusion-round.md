---
doc_kind: finding
finding_id: F0066
title: "Stage-B fusion round (#57, after F0065): (a) fused add_ln+token-shift+lerp boundary kernel — ONE launch replaces the add_ln→shift_lerp6 (attn, J=6) and add_ln→shift_lerp1 (ffn, J=1) pairs, byte-exact composition (WIDE add_ln phases verbatim + glue rounding chain verbatim, `normed` never touches HBM), env RWKV_FUSED_ADDLN_SHIFT; (b) sparse-path finalize — persistent fp32 accumulator (at::zeros once per process, finalize re-zeros in-pass) + hand cast kernel replace the stock FillFunctor-zeros + float16_copy pair, closing BOTH former PDL chain breaks. Targets ~-100-150us/step of the ~710us residual gap to Bo."
status: CLOSED (2026-07-21, round 2, clean single-tenant A/B) — SPLIT VERDICT, both published: **(b) sparse finalize = WIN, banked**: 7.2B 141.3→142.0 (+0.5%) / 1.5B 502.3→509.0 (+1.3%) serving, zeros+cast rows GONE, finalize 1.03us/call, kernels/step −32, **PDL overlap 79.1%→96.8%** (both chain breaks closed), unconditional. **(a) fused add_ln+shift = honest NET REGRESSION as shipped**: E1−E0 = −2.3% 7.2B / −1.8% 1.5B; the launch-count claim delivered (438 kernels/step, −63.6) but the J=6 kernel costs 10.43us/call vs 5.72 composed (+82% — ONE 512-thread block storing 6×8KB output planes can't match the composition's 16-block-parallel shift; single-block store bandwidth is the wall); J=1 is parity on 7.2B (5.48 vs 5.43) and WINS on 1.5B (3.69 vs 4.29). The −60..−90us/step projection is REFUTED (measured +150.6us) — second F0064-class lesson: projections need per-kernel evidence of the actual bound, store-side included. RWKV_FUSED_ADDLN_SHIFT stays default OFF; per-J arming (J=1 only) is supported by the data if wanted; J=6 needs a store-parallel rework before it can pay. Round-1's greedy gate also caught 3 unguarded x.dtype sites (fixed, 6b3e559) — and the audit-grep that missed them had excluded 'x\.dtype' to filter a known line: filters hide their own targets, audit unfiltered
discovered_by: Fable 5, 2026-07-21
severity: info
related: [F0065, F0064, F0063, F0060]
machine: authored on the Mac tree; validation on the 5090 tower
---

# Finding F0066: boundary fusion round (Stage-B proper, cut 1)

## 0. Design

**(a) add_ln_shift{6,1}** (rwkv7_ln.cu + ln_fused.py + rwkv7_backend.py +
models/rwkv7.py): at each per-layer norm boundary the chain ran TWO kernels —
add_ln (residual add + LN, 254.9 us/step post-F0065) then shift_lerp6/1 (paged
token-shift + lerp, 57.0 + 47.5 us/step) — with `normed` doing a full
8KB write + 2 reads through HBM/L2 between them. The fused kernel does
add → Welford stats → LN apply → conv-shift → J-way lerp in ONE launch with
`normed` register-resident:
- add/stats/apply = add_ln's WIDE config VERBATIM (same (32,16) partition,
  MaxVec=2, same trees) ⇒ y bit-identical;
- shift/lerp = shift_lerp*_kernel's exact rounding chain (sh read before
  scatter, conv ← float(y_fp16), d/prod/out rounds) on the register y;
- pads: x_new/stats still computed+written (composition parity), out zeroed,
  conv untouched.
⇒ **byte-exact vs the two-op composition** — gate `bench/test_addln_shift.py`
(torch.equal on x_new + out + conv-after, pads + out-of-range ci included,
48 cases). Wiring: block-level try (env `RWKV_FUSED_ADDLN_SHIFT`, default OFF,
requires `RWKV_FUSED_ADDLN` + fp16 + the glue decode eligibility); attn takes
`lp6=` precomputed lerps (x unused past the lerps on that path — verified),
ffn takes `xk_pre=`. Launch count at the boundaries: ~129/step → ~64.5.

**(b) sparse_out_finalize** (rwkv7_sparse_cmix.cu): the sparse ffn.value path
ran `at::zeros({H})` (FillFunctor, 27.2 us/step of pure launch overhead for
16KB) + cross-tile atomicAdd cmix + `.to(kHalf)` (float16_copy, 29.9 us/step)
— and both stock kernels were PDL chain BREAKS (F0063 §4). Now: a persistent
per-(device,H) fp32 accumulator is allocated+zeroed ONCE per process (warmup,
outside capture); the hand finalize kernel reads f32 → writes f16 (same
__float2half_rn rounding as torch's cast) → **re-zeros the accumulator in the
same pass**, so at::zeros never runs again. One launch instead of two, both
chain breaks close (finalize carries griddepcontrol). Single-stream ordering
makes cross-layer buffer reuse safe; fixed address is capture-friendly.

## 1. Projection (flagged; per-kernel floors behind it)

(a) fused boundary ≈ 4.2-5 us/call × 64.5 ≈ 290-310 us/step vs the current
359.4 (254.9+57.0+47.5) ⇒ −50-70 us/step, plus 64 fewer launches/step
(533→~469) and 2 tighter PDL boundaries. (b) −27 us/step (zeros gone) + 2
chain breaks closed. Combined target ≈ −80-100 us/step ⇒ ~143-144 tok/s
(~92.5% of Bo). Remaining menu after this round: lora_stage2+gates+kmix fold
(~370 us pool), gn_gatecorr prologue (94), stage1 fusion (205).

## 2. Gates

(a) `test_addln_shift.py` 48/48 torch.equal + regression (`test_ln_fused`
WIDE-off byte-PASS, `test_glue.py` untouched) + greedy verify_m1d full stack
+ WIDE + ADDLN_SHIFT: 1.5B 24/24 + 7.2B 8/8 EXACT + the announce-line check
(a silent no-fire measured as a win is the classic self-deception — the agent
brief hard-fails on missing announce). (b) sparse battery + greedy, next round.

## 2b. Next cut banked (design read, not yet implemented): stage2 epilogue fold

CORRECTED after reading both triton kernels in full: `_lora_gates_kernel` is
PURE ELEMENTWISE on `lo` (stage2's output) — foldable; but `_kk_kmix_kernel`
has a **per-head L2 normalize** (tl.sum of kk² over head_dim=64 → sqrt →
fp16-round → clamp 1e-12 → divide) — a 64-wide reduction that stage2's
one-warp-per-output-element shape cannot host without cross-warp/-block
plumbing. Revised scope: fold ONLY the gates chain into stage2's epilogue
(~57 us/step + the lo round trips); `_kk_kmix` stays standalone (already
PDL-armed). Transcription spec for the epilogue (exact chains, from fused.py):
  w row:  s0 = rnd(1/(1+exp(-lo0))); wlog = rnd((-s0) * inv_sqrt_e)
  a row:  a = rnd(1/(1+exp(-lo1)))
  g row:  passthrough (caller-side slice today — keep writing raw lo)
  v row (HAS_V, layer>0): s3 = rnd(sigmoid(lo3)); diff = rnd(vf - v);
          prod = rnd(diff * s3); vnew = rnd(v + prod)   [needs v, v_first]
(each rnd = fp32 op → __float2half_rn → back; sigmoid = 1/(1+expf(-x));
inv_sqrt_e = 0.6065306597126334). Extend stage1/2's meta with an epilogue-act
code per role; byte-exact gate vs the two-op composition. Implement after
(a)+(b) validate.

## 6. F0066c RESULTS (2026-07-21, full round, clean window): WIN — 142.8 = 92.0% of Bo

The stage2+gates fold (§2b) implemented as `lora4_m1_gated` and validated:

**The gate saga (a real audit find):** round-1's composition gate failed 7/9 —
deterministic, single-chain, rare — and the bit-level probe (full 65536-pattern
fp16 census, `bench/results/f0066c/probe_sigmoid_bits.log`) INVERTED the
suspicion: the new CUDA chain is bit-identical to torch.sigmoid on EVERY finite
fp16 pattern (0/65536), while the deployed triton `_lora_gates_kernel`'s
tl.exp deviates 1 ULP from torch/expf/__expf on exactly 2/65536 rare patterns
(its own gate never sampled them). Adjudication: do NOT emulate the anomaly —
the gate re-anchored to the TORCH reference chain (= the model's own non-fused
fallback, the project's original semantic baseline), triton delta kept as an
informational census. The replacement is numerically STRONGER than what it
replaces.

**Gates (all green):** torch-anchored composition 9/9 (census lines fired
exactly on the 2 previously-failing cases, 1 element each — attribution
confirmed); regression suite PASS; greedy full stack + LORA_GATED **1.5B
24/24 + 7.2B 8/8 EXACT** (the 2/65536 anomaly never fired in the fixtures —
no bit change on the pinned decodes); cross-check OFF EXACT.

**Measurement (single-tenant, both instruments concordant):**

| serving c=1 | F0 (off) | F1 (on) | Δ |
|---|---|---|---|
| 7.2B | 142.3 | **142.8** | +0.35% → **92.0% of Bo 155.2** |
| 1.5B | 509.4 | **514.5** | +1.00% |

Per-kernel contract exact: `_lora_gates_kernel` GONE, `lora_stage2_gated`
283.43 us/step vs the composed 315.47 (−32.0 us + −32.3 launches/step;
per-call pair 9.76→8.78 — the epilogue costs +0.80 inside stage2 and kills
the 1.77 standalone launch); stage1 flat (control). kernels/step 502.2→469.1
(prediction matched). 1.5B mirrors (−27.7 us). Promoted to serve.sh defaults.
Ladder: F0065 91.0% → F0066b 91.5% → **F0066c 92.0%**. Raw:
`bench/results/f0066c/`.

## 5. THE BIG ROCK banked (design, not implemented): inline-lerp GEMV — the
## correct successor to the failed J=6 kernel

The J=6 loss teaches that materializing 6 lerp output planes is itself the
waste. The albatross-style design: **the lerp outputs never exist in memory.**
- A compact boundary kernel (one block/row, ~2us) does add + LN + conv-scatter
  and writes only (y, d) — 2×8KB — where d = round_fp16(prev − y) is the
  SHARED lerp delta (it does not depend on the role!).
- The grouped r/k/v GEMV (and the lora stage1 reads, and the ffn.key GEMV via
  x_k) compute their own input ON THE FLY in the x-load path:
  x_role[k] = round_fp16(y[k] + round_fp16(mix_role[k]·d[k])) — the exact
  shift_lerp rounding chain, in registers, per block. Each block reads y, d,
  and ONLY its role's mix row (+24KB L2 per block vs +8KB today); the extra L2
  reads ride under the DRAM weight stream (the bottleneck), so the cost is
  ~hidden, while the wins are: the 6-plane writes+reads GONE, shift_lerp6/1
  GONE, add_ln's apply-store halved. Estimated −(255+57+48) + ~130 ≈
  **−230 us/step**. Consumers to convert: gemv_grouped (rkv), lora_stage1
  (reads xw/xa/xg/xv — 4 more roles), ffn.key gemv_m1 (x_k role).
  Byte-exactness is achievable (same per-element chains); the gate is the
  composition equality on every consumer's output. This is a multi-kernel
  surgery — its own round with its own finding, after F0066c banks.

## 4. RESULTS (2026-07-21 round 2, clean window, 60-sample sentinel zero co-resident)

Gates all green: sparse battery PASS (finalize path), composition 24/24,
greedy full-stack + WIDE + ADDLN_SHIFT **1.5B 24/24 + 7.2B 8/8 EXACT** (the
round-1 crash fixed + re-gated), regression EXACT.

| serving c=1 | E0 (=F0065+b) | E1 (+a) | E1−E0 (a-effect) |
|---|---|---|---|
| 7.2B | **142.0** (+0.5% vs 141.3) = **91.5% of Bo** | 138.8 | **−3.2 (−2.3%)** |
| 1.5B | **509.0** (+1.3% vs 502.3) | 499.9 | −9.1 (−1.8%) |

Kernel-loop concords (E0 141.28 / E1 138.15; BUSY 7202.5 → 7403.7, +201us).
Per-kernel: (b) zeros+cast GONE, finalize 32.26×@1.03us, kernels/step
533.6→501.6, **overlap 79.1→96.8%**; (a) add_ln_shift<2,6> 31.26×@**10.43us**
+ <2,1> 32.26×@5.48 = 502.8us/step vs the composed 352.2 (+150.6);
shift_lerp1 gone, shift_lerp6 layer-0-remnant 1.02× (no pending residual at
layer 0 — structurally can't fuse there), add_ln final-norm remnant 1.02×,
kernels/step 438.0. Root cause of the J=6 loss: single-block 6-plane store
bandwidth (see status). Disposition: (b) merged unconditional; (a) in-tree,
default OFF, honest negative published; J=6 store-parallel rework + per-J
arming are the follow-ups. Raw: `bench/results/f0066/` (c1_*.json E0/E1 both
models + brackets, kerneltable_*, ab_summary.txt; the prefill-poisoned first
E0 trace kept as *_dirty_prefillwindow evidence).

## 3. Artifacts

Branch `stage-b-smallkernels`: 9e883c0 (a: kernel+wiring+test) + this commit
(b + doc). [[F0065]] (the opener this continues) · [[F0064]] §10 (attribution)
· [[F0063]] (chain-break inventory) · task #57.
