---
doc_kind: finding
finding_id: F0066
title: "Stage-B fusion round (#57, after F0065): (a) fused add_ln+token-shift+lerp boundary kernel — ONE launch replaces the add_ln→shift_lerp6 (attn, J=6) and add_ln→shift_lerp1 (ffn, J=1) pairs, byte-exact composition (WIDE add_ln phases verbatim + glue rounding chain verbatim, `normed` never touches HBM), env RWKV_FUSED_ADDLN_SHIFT; (b) sparse-path finalize — persistent fp32 accumulator (at::zeros once per process, finalize re-zeros in-pass) + hand cast kernel replace the stock FillFunctor-zeros + float16_copy pair, closing BOTH former PDL chain breaks. Targets ~-100-150us/step of the ~710us residual gap to Bo."
status: open — implemented on branch stage-b-smallkernels (commits 9e883c0 + this); (a) gate+measure in flight on the tower, (b) queued for the next validation round
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

## 3. Artifacts

Branch `stage-b-smallkernels`: 9e883c0 (a: kernel+wiring+test) + this commit
(b + doc). [[F0065]] (the opener this continues) · [[F0064]] §10 (attribution)
· [[F0063]] (chain-break inventory) · task #57.
