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

Both consumers of lora_stage2's outputs are PURE ELEMENTWISE (verified in
fused.py): `_lora_gates_kernel` runs the w_log/a/v_out rounding chains directly
on `lo` (stage2's output), and `_kk_kmix_kernel` computes kk = rnd(k·k_k) and
k_new = rnd chains from (k, a) at the SAME per-element index. Fold: extend
stage2's meta with a per-role epilogue-act code; after each warp's acc is
final, apply the role's exact rounding chain in-register and write the FINAL
tensors (w_log/a_out/v_out; for the `a` role additionally load k[h] +
kk/ka params and emit kk_out/knew_out) instead of raw `lo`. Kills BOTH triton
kernels (~112 us/step) + the lo/a round trips; extra inputs v/v_first/k/params;
byte-exact gate vs the three-op composition. Implement after (a)+(b) validate.

## 3. Artifacts

Branch `stage-b-smallkernels`: 9e883c0 (a: kernel+wiring+test) + this commit
(b + doc). [[F0065]] (the opener this continues) · [[F0064]] §10 (attribution)
· [[F0063]] (chain-break inventory) · task #57.
