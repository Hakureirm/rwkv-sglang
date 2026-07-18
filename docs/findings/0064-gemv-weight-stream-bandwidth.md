---
doc_kind: finding
finding_id: F0064
title: "GEMV weight-stream bandwidth (#50 follow-on): the flagship bsz1 gap to Bo (D=87.9% of 155.2) lives in BUSY/step, not launch overlap — GEMV is ~74% of busy at ~81% of peak BW vs Bo's ~92%. Two bit-exact load-path rewrites of gemv_grouped_m1 (V1: one 64-bit int2 coalesced load per 4-half chunk replacing two strided 32-bit __half2 loads; V2 opt-in RWKV7_GEMV_KUNROLL2: K-unroll x2 load-hoist for 2x memory-level parallelism), both keeping the EXACT Threads*4 partition + FMA order so torch.equal vs the UNCHANGED gemv_m1 reference still holds. STATUS: implemented on branch mega-gemv-bw; bit-exact gate + SASS load-width confirm PENDING; speed measurement HELD for a clean single-tenant 5090 window (never measured dirty)."
status: open — CORRECTNESS DONE (bit-exact V1+V2 all green, SASS confirms the load widened 32→64-bit + V2's 2× hoist); speed measurement HELD for a clean single-tenant 5090 window
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

## 1. The projection (to be confirmed, not asserted)

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
