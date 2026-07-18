---
doc_kind: finding
finding_id: F0065
title: "Stage-B opener (#57): the bsz1 small-kernel BUSY is a LATENCY problem before it is a fusion problem — add_ln runs ONE 128-thread block on the whole GPU at decode (grid=dim3(T=1), 6.6us/call × 64.5 = 426us/step) and lora_stage1 sits at ~28% of achievable (few-hundred-block grid, 8 serial load rounds). Two fixes: (1) add_ln WIDE small-T variant ((32,16)=512 thr/row, MaxVec=2 kills the own[16] local-spill risk; env RWKV_ADDLN_WIDE, changes reduction tree ⇒ oracle+greedy gate, NOT bit-parity); (2) lora_stage1 F0064-V1+V2 load treatment (int2 + K-unroll×2 hoist — byte-identical, the latency-bound regime is precisely where V2 helps, unlike the BW-wall GEMVs). Target ~-400us/step of the ~1300us small-kernel excess."
status: open — implemented on branch stage-b-smallkernels; gates + clean-window A/B pending
discovered_by: Fable 5, 2026-07-19
severity: info
related: [F0064, F0063, F0060, F0051, F0055]
machine: authored on the Mac tree; gates+measure target the 5090 tower (currently free)
---

# Finding F0065: small-kernel latency round (Stage-B opener)

## 0. Diagnosis (from source, quantified by F0063/F0064 clean traces)

F0064 §10 pinned the real gap to Bo: ~1.3 ms/step of latency-bound small-kernel
BUSY (Bo 13 kernels/step vs our 533). Reading the three fattest small kernels:

| kernel | us/step | per-call | root cause (source-level) |
|---|---|---|---|
| add_ln<16> | 426.9 | 6.6us | grid=dim3(T)! At T=1 ONE (32,4)=128-thread block owns the whole 4096-elem row on a 170-SM GPU; own[16] dynamically indexed ⇒ local-mem spill risk. Config transcribed torch's per-row block for BIT-PARITY against the LARGE-BATCH profile (W1' was a vllm-rwkv serving fight, T=hundreds) — pathological at T=1. |
| lora_stage2<8> | 256.2 | 7.9us | per-warp short rank-segment reads; deferred (epilogue-fusion candidate later) |
| lora_stage1<128> | 207.3 | 6.4us | grid=Rtot (~few hundred) blocks × 128 thr, 8 serial dependent load rounds; byte floor ~1.8us ⇒ ~28% of achievable = latency-bound |

KEY REGIME DISTINCTION (the F0064 lesson applied, not repeated): these kernels
are in the LATENCY-bound regime (28-40% of achievable), the opposite of the
BW-wall GEMVs (95.6-97.7%). Wider loads + more bytes in flight — useless at the
wall — are exactly the medicine here. Same physics, opposite prescriptions,
told apart by the per-kernel byte-floor arithmetic FIRST (the §10 discipline).

## 1. Fixes implemented (branch stage-b-smallkernels)

1. **add_ln WIDE small-T variant** (rwkv7_ln.cu): same template, instantiated
   `<2>` at (32,16)=512 threads/row for T<=32 && N<=4096, env `RWKV_ADDLN_WIDE`
   (default OFF). 4× fewer serial rounds/thread; own[2] stays in registers.
   NUMERICS: same Welford algorithm, different partition/tree ⇒ NOT bit-parity
   with the (32,4) config (which remains the default and keeps its byte gate).
   Gate = `bench/test_addln_wide.py`: (a) x_new BYTE-identical wide-vs-parity
   (residual add is order-free — any diff is a bug); (b) LN y no farther from
   the fp32 reference than the parity config; (c) binding gate = greedy
   verify_m1d 24/24+8/8 with WIDE armed. (fp16-state WKV precedent bar.)
2. **lora_stage1 wide-load + K-unroll×2** (rwkv7_lora.cu): F0064 V1+V2 pattern,
   per-acc FMA order preserved ⇒ **byte-identical**, unconditional (no env),
   existing byte gates apply as-is. Trip count 8→4 with 2× loads in flight.

## 2. Projection (flagged, to be measured — with per-kernel floors this time)

add_ln 6.6→~1.5-2us ⇒ −300us/step; lora_stage1 6.4→~3.5-4.5us ⇒ −60-90us/step.
Combined ~−360-390us: BUSY 7410→~7020-7050 ⇒ ~141.8-142.4 tok/s serving-equiv
(~91.5% of Bo's 155.2, from 88.0%). Remaining small-kernel excess (~900us:
stage2 256, gn 95, shifts 105, triton pair 112, casts/zeros 57, add_ln residual
~100, stage1 residual ~110, misc) stays for the fusion rounds proper (#57).

## 3. Gate + measure plan (5090 currently FREE — grab the window)

Gates (fast): test_addln_wide (new), test_ln_fused with WIDE **off** (parity
path must stay byte-PASS — proves it untouched), the existing lora byte gates,
greedy verify_m1d 1.5B+7.2B with the full D stack + `RWKV_ADDLN_WIDE=1`.
Measure (same window): matrix legs D (WIDE off) vs D+W (WIDE on) + framing-2
traces both — the per-kernel table must show add_ln 426→? (A/B across legs)
and lora_stage1 207→? (unconditional, reads against F0064 D′V1's 207.30).
Greedy 8/8 hard gate per leg; brackets; sky-yield sentinel; single-tenant only.

## 4. Artifacts

- `rwkv7_ln.cu` (WIDE variant + env), `rwkv7_lora.cu` (stage1 load path),
  `bench/test_addln_wide.py` (new gate) — branch `stage-b-smallkernels`.
- [[F0064]] §10 (the corrected attribution this executes) · [[F0063]] (traces)
  · [[F0051]] (H100 independent cross-confirmation) · task #57.
