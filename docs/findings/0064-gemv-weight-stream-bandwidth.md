---
doc_kind: finding
finding_id: F0064
title: "GEMV weight-stream bandwidth (#50 follow-on): the flagship bsz1 gap to Bo (D=87.9% of 155.2) lives in BUSY/step, not launch overlap — GEMV is ~74% of busy at ~81% of peak BW vs Bo's ~92%. Two bit-exact load-path rewrites of gemv_grouped_m1 (V1: one 64-bit int2 coalesced load per 4-half chunk replacing two strided 32-bit __half2 loads; V2 opt-in RWKV7_GEMV_KUNROLL2: K-unroll x2 load-hoist for 2x memory-level parallelism), both keeping the EXACT Threads*4 partition + FMA order so torch.equal vs the UNCHANGED gemv_m1 reference still holds. STATUS: implemented on branch mega-gemv-bw; bit-exact gate + SASS load-width confirm PENDING; speed measurement HELD for a clean single-tenant 5090 window (never measured dirty)."
status: CLOSED (2026-07-19) — flagship measured clean: NULL RESULT. V1 flat (≤0.15%, inside noise, both instruments), V2 a consistent ~1% REGRESSION (rejected). §10's corrected per-kernel byte-floor accounting shows the GEMVs were ALREADY at 95.6-97.7% of achievable BW — the projected headroom never existed; the real remaining gap to Bo is ~1.3 ms/step of latency-bound small-kernel BUSY → lever redirected to Stage-B fusion. V1 kept as default (bit-exact dual-arch proven, halves LDG count, strictly-no-worse); the falsified projection and the null are published, not buried
discovered_by: Opus 4.8 (1M), 2026-07-18
severity: info
related: [F0063, F0060, F0061, F0056]
machine: 5090 tower (sm120) — target; edit authored on the Mac working tree, branch mega-gemv-bw
---

# Finding F0064: GEMV weight-stream bandwidth — the next flagship lever

## 0. TL;DR

- **Where the flagship gap actually is** (from F0063 §6c, the clean-window
  decomposition): PDL already killed the launch gaps (D net gap/step is
  NEGATIVE, 79.0% overlapped transitions). The residual 87.9%→92.4%-of-ceiling
  gap to Bo is **BUSY/step**: ours ~7410 us/step vs Bo's ~6440 (155.2 tok/s =
  6.44 ms/step). GEMV is ~74% of busy (grouped r/k/v/o 2660 us/step + residual
  gemv_m1 2600 us/step = ~5260 us/step), and at bsz1 GEMV is pure memory-bound:
  we hit ~81% of peak BW (136.4 tok/s = 81.2% of the 168.0 ceiling), Bo ~92%.
  Same bytes moved (10.07 GB/step, disclosed identical) — **Bo's GEMV streams
  them at higher achieved BW.** That is the lever.
- **Why the current kernel leaves BW on the table** (gemv_grouped_m1_kernel,
  rwkv7_mega.cu): each weight row is fetched as **two STRIDED 32-bit `__half2`
  loads** (offsets +0 and +2), and the K-loop is not unrolled, so few loads are
  in flight to cover DRAM latency. Neither maximizes per-transaction bytes nor
  memory-level parallelism — the two things that separate ~81% from ~92% on a
  streaming GEMV (Little's law: achieved BW ∝ bytes-in-flight for a fixed
  latency; transaction width sets the per-load byte count).
- **The rewrite (bit-exact preserved, house-law intact)**:
  - **V1 (default)**: one 64-bit `int2` load = 4 contiguous halves per weight
    row, replacing the two strided 32-bit loads. Across a warp each thread now
    reads a contiguous 8-byte span → a single 256B/warp coalesced stream instead
    of two half-populated sector streams; per-row load-instruction count halves.
  - **V2 (`-DRWKV7_GEMV_KUNROLL2`)**: additionally hoist BOTH consecutive
    k-chunks' loads before either chunk's FMAs → 2× memory-level parallelism.
  - Both keep the **EXACT Threads*4 per-thread partition and the EXACT per-acc
    FMA order** (chunk(k) fully, then chunk(k+kstride) fully) — only how the
    bytes are FETCHED changes, never which fp32 terms are summed or in what
    order. So `y` stays **byte-identical** to gemv_m1.
- **Strongest possible gate**: only `gemv_grouped_m1` is rewritten this round;
  `gemv_m1_kernel` (rwkv7_fast.cu) is left UNCHANGED as an independent golden.
  `bench/test_mega_rkv.py` does `torch.equal(grouped, G× gemv_m1)` → proves the
  rewrite against a reference that did NOT change with it (a bug in the rewrite
  cannot hide by mutating both sides). Propagation to gemv_m1 itself is a
  SEPARATE later round, re-gated against a numpy oracle.
- **STATUS**: implemented on branch `mega-gemv-bw` (commit below). NOT gated,
  NOT measured. Bit-exact gate + SASS load-width confirmation are the next step
  (can run co-resident — correctness/SASS don't need a clean card). **Speed
  measurement is HELD for a clean single-tenant 5090 window** — no dirty GEMV
  BW numbers will be published (the whole point is peak-BW achievement, which a
  co-resident tenant destroys).

## 1. The projection (to be confirmed, not asserted) — **REFUTED by §10**

If V1+V2 move GEMV from ~81% → ~92% of peak BW, GEMV busy drops
~5260 × (1 − 81/92) ≈ **−630 us/step**, taking BUSY/step 7410 → ~6780, i.e.
136.4 → ~149 tok/s ≈ **96% of Bo's 155.2**. That would close most of the
remaining gap on the single dominant kernel class alone; the add_ln / wkv /
sparse-path residuals (F0063 §6c) are the follow-on levers. **This is a
projection from the F0063 per-kernel table, flagged as such — the clean-window
framing-2 re-measure is the actual gate on the number.**

## 2. Bit-exactness argument (the audit trail)

For one k-chunk, original: `x0=(h2f(x[k]),h2f(x[k+1]))` from `*(half2*)(x+k)`,
`x1=(h2f(x[k+2]),h2f(x[k+3]))` from `*(half2*)(x+k+2)`. V1: `xp=*(int2*)(x+k)`
reads the same 8 bytes = halves x[k..k+3]; `xp.x` = first 4 bytes = bits of
(x[k],x[k+1]) → `__half22float2(*(half2*)&xp.x)` = the same `x0`; `xp.y` →
the same `x1`. Identical for the weight row. FMA sequence per `acc[j]` is
unchanged. V2 reorders only LOADS (issues chunk k and chunk k+kstride loads
before their FMAs); the FMA sequence into each `acc[j]` remains chunk(k) 4
terms then chunk(k+kstride) 4 terms — the same order the scalar loop produces.
Alignment: k ≡ 0 (mod 4 halves) and kstride ≡ 0 (mod 4 halves) with K%4==0 and
16-byte tensor base ⇒ every `int2` address is 8-byte aligned. ∴ byte-identical.

## 3. Gate plan (next step — co-resident OK, no clean card needed)

1. **SASS diff** (cuobjdump `gemv_grouped_m1` on main vs branch): confirm V1 emits
   one `LDG.E.64` per weight row where main had two `LDG.E.32` (evidence the load
   actually widened — if main already emitted `.64`/`.128`, V1 is neutral and the
   lever is V2's MLP; report either way).
2. **Bit-exact gate**: `bench/test_mega_rkv.py` (torch.equal vs G× gemv_m1) for
   V1 AND V2 (`-DRWKV7_GEMV_KUNROLL2`) — must be zero differing bytes both.
3. **Greedy e2e**: `verify_m1d` full stack + MEGA + WKV_CUDA + PDL under CUDA
   graph, 1.5B 24/24 + 7.2B 8/8 EXACT, V1 and V2.
4. **HELD**: framing-2 kernel-loop re-measure (A / V1 / V2) + serving bsz1 —
   ONLY in a confirmed clean single-tenant window (0 compute-apps, 0
   tracking/Pending pods), per the sky-yield rule. Log held; do not measure dirty.

## 4. Gate results — CORRECTNESS DONE (2026-07-18, tower, co-resident with hb_compile)

Validated on the 5090 tower while `hb_compile` (root, non-sky, ~98% util) was
co-resident — correctness/SASS are contamination-invariant, so this ran without
a clean window. Build: `mega.py`'s JIT `cpp_extension.load` (sm_120, `-O3
-Xptxas -O3`); V2 via `NVCC_APPEND_FLAGS=-DRWKV7_GEMV_KUNROLL2` + cache clear.

**SASS — the load genuinely widened (not a compiler no-op):** for the (256,4)
7.2B config, gemv_grouped_m1's weight/x loads went from paired 32-bit
`LDG.E.CONSTANT R5,[R18]` + `LDG.E.CONSTANT R2,[R18+0x4]` (two strided 4-byte
loads) → one **`LDG.E.64.CONSTANT`** (single contiguous 8-byte load): 30→15
load instrs per config, same bytes, half the instructions, double the width.
V2 additionally hoists **10 loads before the first FMA** (2 chunks × 5) vs V1's
5 — the 2× memory-level parallelism materialized in codegen, confirmed across
two independent build paths (SASS saved `tmp/f0064_sass/{before,v1,v2}.sass`).

**Bit-exact gates — all PASS, both variants (zero failures):**

| gate | V1 (64-bit load) | V2 (+KUNROLL2) |
|---|---|---|
| `test_mega_rkv.py` torch.equal (rkv G=3 / o G=1 / rkvo G=4, 5 shapes × 2 families × 3 scales) | PASS, zero differing bytes | PASS, zero differing bytes |
| `verify_m1d.py` 1.5B greedy, cuda-graph, full stack+MEGA+WKV_CUDA+PDL | 24/24 EXACT | 24/24 EXACT |
| `verify_m1d.py` 7.2B same config (mem-fraction 0.5, fit alongside hb_compile) | 8/8 EXACT | 8/8 EXACT |

So the rewrite is proven byte-identical to the UNCHANGED gemv_m1 reference AND
SASS-confirmed to actually widen the load. The value claim (faster) remains
unproven — the framing-2 A/V1/V2 kernel-loop re-measure is HELD for a clean
window (dirty BW numbers are meaningless on the very axis we're optimizing).

**Ops discovery (flagged, real reproducibility risk):** the live `rwkvmain`
serving container had NEVER had the megakernel deployed this session —
`mega.py`/`rwkv7_mega.cu` were absent (last deploy predates them;
`rwkv510`/`vllmrwkv` containers same stale state). F0063's "verified in
repo-mega" did NOT survive what looks like a container recreate between
sessions. Fixed by re-running `scripts/deploy.sh` (idempotent; byte-verified
post-deploy). Lesson: the clean-window measure MUST re-confirm the overlay is
actually deployed in the target container before trusting any number — a
recreated container silently reverts to no-megakernel. (deploy.sh is idempotent
so the fix is cheap; the trap is assuming it persisted.)

## 6. Clean per-kernel ranking → round-2 is essential (not optional)

The F0063 clean D-config framing-2 trace (`bench/results/mega_framing2_D_7.2b_5090.txt`,
BUSY/step 7409.7 us) ranks the levers precisely:

| kernel | count/step | us/step | % BUSY | lever |
|---|---|---|---|---|
| gemv_grouped_m1<256,2> (r/k/v/o) | 64.5 | 2660.2 | 35.9% | **F0064 round-1** (done, gated) |
| gemv_m1<256,2> (ffn.key + resid proj) | 32.3 | 2600.1 | 35.1% | **F0064 round-2** (this section) |
| sparse_cmix_f32acc | 32.3 | 430.8 | 5.8% | next |
| add_ln<16> | 64.5 | 425.9 | 5.7% | next |
| cuBLAS gemvx (**lm_head**, 1 call) | 1.0 | 324.3 | 4.4% | **NO lever** — already ~98% BW (see below) |
| lora_stage2<8> / stage1<128> | 32.3 | 256.2 / 207.3 | 6.3% | next |
| gn_gatecorr / wkv_decode | 32.3 | 95.0 / 84.4 | 2.4% | small |
| float16_copy + FillFunctor (sparse zeros+cast pair) | 32.3 | 29.9 + 27.2 | 0.8% | F0060 Stage-B fusion target |

**The two M==1 GEMV kernels are ~71% of BUSY/step and nearly equal (2660 + 2600).**
F0064 round-1 touched only `gemv_grouped_m1` (2660). To actually reach Bo's 92.4%,
the identical rewrite MUST also land on `gemv_m1` (2600) — round-2 is half the GEMV
budget, not a nice-to-have.

**Revised projection (still to be measured, both need the clean window):**
- round-1 alone (grouped, −12% BW): 2660→~2340, saves ~320 us/step → BUSY ~7090
  → ~142.6 tok/s ≈ **91.9% of Bo**.
- round-1 + round-2 (both GEMVs): saves ~630 us/step → BUSY ~6780 → ~149 tok/s
  ≈ **96% of Bo** (crossing 92.4%).

**Round-2 implemented** (this branch): the same V1 + V2 rewrite applied to
`gemv_m1_kernel` AND `gemv_m1_sqrelu_kernel` in rwkv7_fast.cu (both M==1, loop
byte-identical to grouped; sqrelu's post-loop activation untouched). `gemv_mb_kernel`
(the multi-batch / concurrency path, line ~289) has a different M-loop body and is
deferred to its own careful round (round-2c, the high-concurrency axis).

**Round-2 gate strategy** (gemv_m1 is grouped's reference, so changing it needs an
independent golden): (a) TRANSITIVE — round-1 pinned `grouped_v1 == gemv_m1_old`
(zero differing bytes, §4); round-2 leaves grouped_v1 untouched, so if
`test_mega_rkv` still passes (`grouped_v1 == gemv_m1_v1`) then
`gemv_m1_v1 == grouped_v1 == gemv_m1_old` transitively; (b) DIRECT — capture
`gemv_m1_old(x,w)` golden tensors before the change, `torch.equal` after; (c)
sqrelu via `test_sqrelu_gate.py`; (d) greedy `verify_m1d` 1.5B + 7.2B. All
co-resident-safe (correctness is contamination-invariant) — pre-gated so the clean
window is pure measurement.

**Round-2 gate results — ALL GREEN (2026-07-19, tower, co-resident):** deployed
via deploy.sh into the serving container (post-deploy md5 byte-identical);
scope-check clean (exactly 2 hunks: the two M==1 kernel bodies, header and
gemv_mb byte-identical). Both V1 and V2: `test_mega_rkv` transitive gate PASS
zero differing bytes (incl. the Stage-A2 o/rkvo bonus gate); **DIRECT golden
90/90 PASS** (5 shapes × up to 9 configs × 3 scales × 2 seeds, captured from an
isolated build of the preserved `.pre-r2.bak` — the strong proof, not just
transitive); `test_sqrelu_gate` 427/427; `verify_m1d` full-stack cuda-graph
1.5B **24/24** + 7.2B **8/8** EXACT. SASS: gemv_m1<256,4> went 30× 32-bit
`LDG.E.CONSTANT` (paired +0x0/+0x4) → **15× `LDG.E.64.CONSTANT`** (V1, matching
grouped's own count) → 43 loads hoisted across chunk-pair strides before the
FMA block (V2). Zero failures anywhere. Method notes for reproducers:
`TORCH_LIBRARY(rwkv7_fast,...)` is hardcoded, so a golden-capture build cannot
coexist in-process with the live extension — capture from an isolated scratch
build; this torch install ignores `NVCC_APPEND_FLAGS`, so V2 was injected via a
scoped `sitecustomize.py` shim on `cpp_extension.load` (inert by default,
verified via build.ninja to touch only `rwkv7_fast`). Audit artifacts in
`tmp/` on the tower (golden .pt, SASS dumps, capture/compare scripts).

**lm_head candidate KILLED by arithmetic** (correcting the table's earlier
"swap for our GEMV?" note): 7.2B lm_head = vocab 65536 × hidden 4096 × fp16 =
536.9 MB/step; the physical floor at 1691.7 GB/s is ~317 us. cuBLAS gemvx
measures 324.3 us = **~98% of peak BW — no headroom**. Do not spend a round
here. (General lesson, same as F0063's honest-ledger style: compute the
byte-floor BEFORE nominating a kernel as a lever.) The realistic post-GEMV
levers are the latency-bound small-kernel chain — add_ln 6.6 us/call × 64.5 +
gn_gatecorr + shifts + lora glue + the sparse zeros/cast pair — where the tool
is FUSION (fewer kernels), i.e. F0060's Stage-B, not wider loads.

## 7. Clean-window measurement runbook (dispatch verbatim when the waiter fires)

Waiter: `tmp/f0064_waiter.sh` on the tower (decoupled setsid/nohup, PID logged
in `tmp/f0064_waiter.log`) — writes `tmp/f0064_window_open` when 0 compute-apps
+ 0 Pending pods + util<10%. On fire, dispatch with THIS leg design — chosen so
NO .bak swapping is needed mid-window (the r1+r2 tree stays deployed
throughout; the flag matrix itself provides attribution):

| leg | config | isolates | compare against |
|---|---|---|---|
| A′ | flags OFF (MEGA=0 WKV_CUDA=0 PDL=0), r2 fast.cu | **pure V1 wide-load at max exposure** — flags-off routes ALL projections through gemv_m1 (161.65 calls/step, F0063 framing-2 A) | F0063 A = 136.9 serving / 134.96 kernel-loop |
| D′V1 | full stack (MEGA+WKV_CUDA+PDL), r1+r2, V1 | **the headline** | F0063 D = 137.8 / 136.43; Bo 155.2 / same-session 155.75 |
| D′V2 | same, rebuilt `-DRWKV7_GEMV_KUNROLL2` | V2's 2× load-hoist on top of V1 | D′V1 |

Per leg: greedy-smoke hard gate (8/8) → serving bsz1 sweep (~5-6 min) →
framing-2 trace on D′V1 (and D′V2 if window holds): the per-kernel table must
show grouped 2660→? and gemv_m1 2600→? us/step — the DIRECT BW-achievement
evidence, stronger than end-to-end tok/s. Priority if the window is short:
D′V1 serving+trace > A′ > D′V2 > 1.5B bonus. Budget ~25 min total (F0063 §6b
precedent). MANDATORY pre-checks: (1) overlay actually deployed in the target
container (the §4 container-recreate trap — grep the deployed fast.cu/mega.cu
for `RWKV7_GEMV_KUNROLL2`); (2) JIT cache state matches the leg (V1 = default
build, V2 = NVCC_APPEND_FLAGS rebuild + cache clear); (3) sky-yield sentinel
armed (any Pending pod → stop within a minute, log the yield). Anchor
provenance: F0063's clean A/D are reused as the baseline (same sglang base
754524d, same card, measured clean 2026-07-18); A′ additionally re-anchors
in-window — if A′ deviates >2% from BOTH 136.9 and the r2-predicted uplift
band, suspect environment drift and say so rather than publishing.

## 9. 3090 (sm86) same-session A/B — mechanism test: PREDICTION CONFIRMED

Run design: clean 3090 (our box, 0% util), Leg0 (pre-F0064 baseline) → V1 →
V2, identical configs per model across legs, with the prediction PRE-REGISTERED
before any Leg1/2 numbers existed: *the 3090's 936 GB/s bus is already
saturated by the narrow-load kernels, so the 3090 gain should be small (+0~3%);
a large gain would falsify the F0060 fast-card thesis.*

**Second-arch bit-exact re-gate: ALL GREEN** (V1+V2: test_mega_rkv +
test_mega_a2 + test_mega_o_model zero differing bytes, sqrelu oracle, DIRECT
golden vs pre-F0064 capture byte-identical, greedy 1.5B 24/24 + 7.2B 8/8).
PDL intrinsics compile out on sm86 by guard, as designed.

**Result — every reproducible delta inside the predicted band:**

| axis | Leg0 → V1 → V2 |
|---|---|
| gemv_m1 4096² (graphed us/call) | 40.62 → 39.66 → 39.57 (**−2.4~2.6%**) |
| sep 3× gemv_m1 7.2B | 121.1 → 118.4 → 118.1 (**−2.3~2.5%**) |
| gemv_m1 ffn-k 16384×4096 (biggest bytes) | 152.3 → 152.4 → 152.4 (**dead flat**) |
| grouped rkv 7.2B | 114.9 → 114.9 → 114.8 (flat) |
| e2e bsz1 1.5B | 261.3 → 261.9 → 261.7 (+0.2%) |
| e2e bsz1 7.2B | 72.3 → 72.3 → 72.9 (+0.0~0.8%) |

**Verdict: the fast-card thesis HOLDS.** Gains appear only where latency-hiding
is imperfect (square shapes, ~−2.5%); the bus-saturating fat shape gains
nothing — the 3090 was already at its achievable BW with narrow loads. V2-vs-V1
is noise everywhere on sm86 (bus-limited, not latency-limited — extra loads in
flight buy nothing). Nothing here contradicts the 5090 ~+9% projection; the
5090 sits at 81% of peak (the latency-limited regime the square shapes sample),
which is precisely where the rewrite pays. The 5090 number remains the open
question and is NOT claimed until measured clean.

Environment note (second data point for the §4 drift trap): the 3090 host
repo's overlay is stale and deploy.sh does not live on boxes (it is Mac-side
orchestration); the container's editable sglang install is the live tree.
Verify the live tree's identity before gating on ANY box.

Raw: `bench/results/f0064_ab_{leg0,leg1,leg2}_{15b,72b}_3090.json`,
`f0064_ab_kernel_microbench_3090.json` (full 3-leg tables + verdict),
`f0064_ab_blocker_3090.json` (the pdl.cuh staging omission, RESOLVED, kept for
the record). All leak-scanned clean.

## 10. FLAGSHIP RESULT (2026-07-19, clean 5090 window): NULL — and the corrected attribution

The window opened naturally (the co-resident D-Robotics compile exited on its
own; nothing was killed). Measurement: same container lineage, same harness,
same configs as F0063's clean session — the ONLY changed variable is the F0064
kernels. All greedy gates EXACT; GPU brackets matched F0063's footprint to the
MiB (19740→19748); zero yields; both instruments (serving wall-clock AND
43-step kernel-loop trace) agree.

**Serving bsz1 7.2B**: A′=136.9 (F0063 A: 136.9), D′V1=137.9 (F0063 D: 137.8),
B′=138.3, C′=137.9, **D′V2=136.5 (−0.94%)**. 1.5B: A′=477.9 / D′V1=492.7 /
B′=490.6 / C′=492.4 (F0063: 479.2/493.0) — all flat.
**Kernel-loop 7.2B**: A′=134.95 (F0063: 134.96), D′V1=136.57 (F0063: 136.43,
+0.10%), D′V2=135.32 (−0.81%). **Per-kernel (the direct evidence)**: grouped
2660.15→2656.14 (−0.15%) / gemv_m1 2600.09→2599.81 (−0.01%) under V1; under V2
grouped 2696.85 (+1.38%) / gemv_m1 2624.46 (+0.94%) — a real regression, in
the same direction on both instruments. Every OTHER kernel in the 533-row
table is flat, isolating the change. Overlap 79.0→79.1%, gap −79.9→−80.0 us
(PDL state untouched).

**Verdict: V1 neutral (keep — see below), V2 REJECTED (consistent ~1%
regression on sm120; noise on sm86). The projected ~149 tok/s did not happen.
We remain at 88.0% of Bo's 155.2 (82.1% of the 168.0 ceiling).**

### Why the projection was wrong — the corrected per-kernel byte-floor accounting

The §1 projection derived "GEMV at ~81% of peak BW" from END-TO-END tok/s ÷
ceiling (136.43/168.0) and attributed the whole gap to GEMV streaming
efficiency. Doing the arithmetic PER KERNEL (which §6's own table already
enabled — the same discipline §7 applied to kill the lm_head candidate, applied
inconsistently) refutes that:

| kernel | bytes/step | floor @1691.7 GB/s | measured | achieved BW |
|---|---|---|---|---|
| gemv_grouped_m1 (r/k/v/o, 4×4096²×2B×32L) | 4.295 GB | 2539 us | 2656 us | 1617 GB/s = **95.6%** |
| gemv_m1 (ffn.key, 16384×4096×2B×32L) | 4.295 GB | 2539 us | 2600 us | 1652 GB/s = **97.7%** |

**The GEMVs were already at 95.6-97.7% of achievable BW.** Total genuine GEMV
headroom ≈ 178 us/step, not the projected 630. And the SASS-level explanation
for V1's flatness: the "two strided 32-bit loads" sit at +0x0/+0x4 —
CONSECUTIVE addresses. At sector (32B) granularity the warp's request stream is
IDENTICAL to the 64-bit version: both halves of every sector are requested
either way, and the coalescer merges them. V1 halves the LDG instruction count
but moves the same bytes in the same pattern — on a bandwidth-wall kernel that
changes nothing (the 3090's −2.4% on square shapes was the instruction-issue
side effect, visible only where bytes/instruction ratio is worse). V2's
regression: the +kstride hoisted load (2 KB away) costs registers and disrupts
access locality for zero benefit when the memory system is already saturated.

**Where the gap to Bo ACTUALLY lives** (completing the accounting): BUSY/step
7410 vs floor 5950 → 1460 us excess. GEMV excess ≈ 178. lm_head ≈ +7 over
floor. wkv ≈ +44. The remainder — **~1.2-1.3 ms/step — is the latency-bound
small-kernel chain** (add_ln 426, lora_stage1/2 463, sparse glue/casts/zeros
~120, gn_gatecorr 95, shifts 105, kk/lora_gates 112): kernels that move almost
no bytes but pay per-launch/per-kernel overhead 32-64× each. Bo's step = 6440
us = the same 5950 floor + only ~490 us of overhead — because albatross runs
13 kernels/step vs our 533. **The megakernel thesis was the right read all
along; F0064 attacked the wrong term.** Lever redirected: Stage-B fusion
(fold add_ln into GEMV epilogues, fuse the lora chain, fuse gn+glue, kill the
zeros/cast pair) targeting −800~1000 us of the ~1300 → step ~6.4-6.6 ms →
**148-156 tok/s ≈ parity with Bo's 155.2** — now with arithmetic that has
per-kernel floors behind it, not a ratio guess.

**Disposition**: V1 stays as the default source (bit-exact proven on sm120+
sm86 with direct goldens, SASS-verified, strictly-no-worse, halves LDG count,
tiny wins on 3090 square shapes). V2 stays in-source under the macro,
documented DO-NOT-ENABLE (this measured regression is the reproducible record).
The §1 projection text is kept above, marked refuted — the falsification is
published, not buried. Harness fix shipped: mega_flag_matrix.sh tag matching
(exact-string → prefix family + fail-loud; the F0064 run found decorated tags
silently selected the wrong fixture).

Raw (all leak-scrubbed): `bench/results/f0064/c1_72b_{A,B,C,D}.json`,
`c1_15b_f0064_{A,B,C,D}.json`, `framing2_{A,D,Dv2}_7.2b.txt`; full logs +
traces + the preserved wrong-fixture failed run on the tower under
`repo-mega/bench/results/f0064*/` and `logs/mega/f0064/`.

## 8. Artifacts

- Kernel: `sglang_overlay/.../rwkv7_kernels/cuda/rwkv7_mega.cu` (gemv_grouped_m1
  V1 default + V2 `#ifdef RWKV7_GEMV_KUNROLL2`); reference `rwkv7_fast.cu`
  gemv_m1_kernel UNCHANGED.
- Branch `mega-gemv-bw` (NOT main — held for review until gated + measured clean,
  per the F0063 hold-on-separate-branch lesson).
- [[F0063]] (the flagship measurement + §6c decomposition this attacks) ·
  [[F0060]] [[F0061]] (the mega prefabs) · ADR-0008 (megakernel feasibility).
</content>
</invoke>
