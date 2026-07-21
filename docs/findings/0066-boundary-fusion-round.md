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
