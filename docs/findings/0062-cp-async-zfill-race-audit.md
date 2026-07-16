---
doc_kind: finding
finding_id: F0062
title: "cp.async race-class audit vs Albatross ff144b6b ('bvec' zero-fill race, 2026-07-14): ALL our cp.async sites CLEAN — the bug class is structurally absent (we predicate by control flow, never by the src-size operand; every issued copy owns a thread-distinct destination; the one inline-PTX site has no src-size operand at all), and the verdict is evidenced, not vibed: a positive-control repro of their class on the 3090 corrupts 22.1% of outputs and racecheck flags it (131072 WAW hazards/launch), while our kernels take 0 hazards over 39 branch-decisive launches, 0 synccheck errors on the named-barrier pair handoff, and a SASS census shows 552 sm86 + 24 sm120 LDGSTS with ZERO predicated (zero-fill-capable) forms"
last_verified_commit: "(this commit) — kernels audited at 3a67404 state, byte-identical on the box (sha256-matched)"
discovered_by: Fable 5 (agent), 2026-07-16
severity: info
status: closed — audit complete, no fix needed; wkv on-device racecheck deferred to the next free sm100+ window (compile-level evidence already closes the class for it)
related: [F0055, F0058]
---

# Finding F0062: does our cp.async usage carry the race class Albatross just hotfixed? (No — with sanitizer + SASS evidence)

## 0. TL;DR

- **Their bug** (BlinkDL/Albatross `ff144b6b`, "!!! bug fix !!! `bvec` cp.async
  race", 2026-07-14): their `cp_async(dst, src, pred)` wrapper "predicates" by
  setting the PTX **src-size operand** to 0. Per PTX semantics, `cp.async`
  **always writes cp-size bytes to shared memory**; src-size < cp-size
  zero-fills the tail — so a "predicated-off" copy is really an in-flight
  async **zero-fill of the destination**. Their warp 1 (pred=false) computed
  the *same* `bvec + lane` smem addresses as warp 0's real copy: two async
  writes to the same words, real data racing zeros, consumer reads whichever
  landed last. Fix (theirs): route pred-off lanes to a scratch `bvec_dummy`.
  vllm-rwkv, which ships Albatross kernels wholesale, inherited the bug and
  adopted the fix within ~14 h (their PR#10).
- **Our verdict: every cp.async site CLEAN — the class is structurally
  impossible in our kernels**, for three independent reasons (any one
  suffices): (1) we predicate **by control flow** (loop bounds / block-uniform
  `if`) — a not-issued copy writes nothing; we never use the src-size operand
  as a predicate; (2) every *issued* copy's destination is a thread-distinct
  16 B chunk (t-indexed partition — no two threads, on or off, ever share a
  destination); (3) the one inline-PTX site (`rwkv7_wkv.cu`) doesn't even
  *have* a src-size operand — its 4th operand is the L2 cache-policy
  descriptor (64-bit `"l"` binding; the `.L2::cache_hint` grammar requires a
  trailing policy operand), so it is an unconditional full-16 B copy.
- **Evidence, not reasoning alone** (3090 / sm86, CUDA 12.9 sanitizer,
  sources sha256-matched to the audited tree):
  - *Positive control* (our minimal repro of THEIR class,
    `bench/probes/cp_async_zfill_control.cu`): functionally wipes
    **2,895,184 / 13,107,200 outputs (22.1%)** on the 3090 — the class is
    real and vicious on sm86 — and `racecheck` flags it: WAW race at the
    predicated `LDGSTS`, **131072 hazards/launch, 3/3 launches**. The fixed
    (dummy-destination) variant: 0 wiped, 0 hazards — the tool neither
    misses the class nor cries wolf.
  - *Our kernels*: `compute-sanitizer --tool racecheck` over
    `bench/probes/race_audit_driver.py` — **39 launches covering every
    cp.async pipeline and its branch-decisive shapes** (ragged M/N, odd+even
    k-tile parity, split-K WritePartial, both w4a8/w8a8 algos, bias/no-bias,
    bf16) — **0 hazards, 0 errors, 0 warnings**. `synccheck` (validates the
    `bar.sync 1+wn, 64` named-barrier pair handoff): **0 errors**.
  - *Machine level*: SASS census of the built extensions — sm86: 240 (w4) +
    216 (w8) + 96 (w8a8) LDGSTS, all `LDGSTS.E.BYPASS.128 [R], [R.64]`,
    **zero with a trailing predicate** (the zero-fill selector — their bug's
    machinery prints as `LDGSTS.E [R], [R.64], !P0`, observed only in our
    buggy control). sm120 compile of `rwkv7_wkv`: 24 LDGSTS, zero
    predicated, all carrying the `desc[UR4]` cache-policy descriptor.
- **Gates re-run green** on the same box/build (see §5).
- **Scope note**: `rwkv7_wkv.cu` is the sm120 kernel (its
  `st.global.L2::evict_last.v4.b64` 256-bit stores need sm100+; ptxas
  rejects it on sm86 and the loader falls back to Triton — the site is
  unreachable on the 3090 fleet by construction). Static + SASS evidence
  closes the class for it; an on-device racecheck is owed opportunistically
  when an sm100+ card is next free (the 5090 is pinned by RL training).

## 1. Their bug, precisely

Wrapper (all three shipped trees, `rwkv7_wkv_fp16_v2.cu`):

```cuda
template <int Bytes>
__device__ __forceinline__ void cp_async(void* smem, const void* global, bool pred) {
  int bytes = pred ? Bytes : 0;                       // <-- src-size, NOT a guard
  ...
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;"
               ::"r"(addr), "l"(global), "n"(Bytes), "r"(bytes));
}
```

PTX ISA, `cp.async`: *cp-size bytes are always written to dst; "if src-size is
smaller than cp-size, then the remaining bytes in destination dst are filled
with zeros."* With src-size = 0 the instruction is a full-width async
**zero-fill**, not a no-op.

The racing call (pre-fix; five kernels across faster3/3a/4 had the pattern):

```cuda
cp_async<4>((half2*)bvec + lane, (half2*)(b_ptr + t) + lane, i < 32);
```

Threads 32–63 (`i >= 32`, pred=false) share `lane = i & 31` with threads 0–31,
so both warps target the same `bvec[lane]` — warp 0's real copy vs warp 1's
zero-fill, unordered within the same commit-group window. Their fix
(ff144b6b) adds a `bvec_dummy[HALF2_N]` scratch buffer and routes pred-off
lanes to it:

```cuda
cp_async<4>((i < 32 ? bvec : bvec_dummy) + lane, (half2*)(b_ptr + t) + lane, i < 32);
```

Same commit also fixes an unrelated signed-int-overflow UB in faster3a's
`rotator1` (int → uint32_t wraparound); we never adopted the phase-rotator
trick, so that class is N/A for us (grep: no hits).

At SASS level the machinery is visible: the predicated form compiles to
`LDGSTS.E [Rdst], [Rsrc.64], !P0` — a **trailing** predicate operand selecting
src-size (false ⇒ zero-fill write still happens). This is distinct from a
**leading** `@P0 LDGSTS ...` instruction guard (which skips the instruction
entirely and is benign). Our audit checked for exactly this signature.

Why a passing bit-exact gate proves nothing here: the hazard is
timing-resolved per launch. Our control measured the buggy pattern wiping
22.1% of outputs on an idle 3090 — but the rate is contention- and
shape-dependent; a colder configuration can pass a gate N times and corrupt
under serving load.

## 2. Audit scope

Repo-wide grep (`cp.async|__pipeline|memcpy_async|mbarrier` over
`*.cu/.cuh/.h`): exactly four files use async copies, all under
`sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda/`:
`rwkv7_w4.cu`, `rwkv7_w8.cu`, `rwkv7_w8a8.cu`, `rwkv7_wkv.cu`.
`rwkv7_mega.cu` (Stage-A/A2), `rwkv7_fast.cu`, `rwkv7_glue.cu`,
`rwkv7_lora.cu`, `rwkv7_ln.cu`, `rwkv7_sparse_cmix.cu`: no async copies at
all (mega's only inline asm is `griddepcontrol.launch_dependents` — PDL, not
a copy); plain loads are immune to this class by construction.

## 3. Per-site verdicts (all CLEAN)

| Site (kernel, cp.async lines) | Copy form | Predication | Dst sharing | wait → barrier → first read | Double-buffer WAR |
|---|---|---|---|---|---|
| `rwkv7_w4.cu` `gemm_w4_tc` (L324/L330) | `__pipeline_memcpy_async(dst,src,16)` 3-arg — **no zfill arg** | loop bounds (`t < a_chunks`) — instruction not issued | `t`-partitioned 16 B chunks, disjoint | `commit` → `wait_prior(1)` → `__syncthreads` (L344) → dequant/MMA reads | end-of-iter `__syncthreads` (L375) precedes restage of the buffer consumed 2 steps ago; unconditional commit keeps group accounting exact on the last iter |
| `rwkv7_w4.cu` `gemm_w4a8_tc` (L679/L685) | 3-arg | loop bounds; dead rows zeroed ONCE pre-loop behind `__syncthreads` (L671) and never restaged | disjoint | `wait_prior(1)`/`(0)` → `__syncthreads` (L713) → unpack reads `s_bq[buf]` | closing `__syncthreads` (L779). Named pair handoff `bar.sync 1+wn, 64` (L736): ids 1–4 (0 = `__syncthreads`), the 2 warps of a wn-pair = 64 uniform threads, CTA-scope membar for participants; the per-iteration block barrier makes cross-iteration barrier skew impossible. Per-warp `s_epi` staging fenced by `__syncwarp` on both sides |
| `rwkv7_w8.cu` `gemm_w8_tc` (L285/L291) | 3-arg | loop bounds | disjoint | `wait_prior(1)` → `__syncthreads` (L305) | L334 (same proof as `gemm_w4_tc`) |
| `rwkv7_w8.cu` `gemm_w8_tc_large` (L603/L610) | 3-arg | loop bounds (live rows only) | disjoint | `wait_prior(1)` → `__syncthreads` (L624) | L655. Dead-row zeroing (L586) has no *immediate* barrier, but its first cross-thread read is behind L624 and cp.async targets are row-disjoint from the zeroed region |
| `rwkv7_w8a8.cu` V1 (L115/L121), V2 (L234/L240) | 3-arg | loop bounds; dead rows pre-zeroed behind L107/L228 | disjoint | `wait_prior(1)`/`(0)` → `__syncthreads` (L142/L263) | L157/L281; V2 per-warp `s_epi` + `__syncwarp` both sides |
| `rwkv7_wkv.cu` `wkv_decode` (helper L104–115, call L194) | inline PTX `cp.async.cg.shared.global.L2::cache_hint [dst],[src],16,%2` — **src-size omitted**; `%2` is the 64-bit `createpolicy` descriptor (grammar: `.level::cache_hint` requires trailing cache-policy; a `.b64` register cannot bind the 32-bit src-size slot) | block-uniform `if (live)`; the dead-slot branch uses plain zero stores and issues no copies (empty commit group is harmless) | `t`-partitioned, disjoint | `commit + wait_group 0` (L216) → `__syncthreads` (L217) → first read (L229) | single stage per launch (no rotation); pass-2 in-place state update is column-private per thread; pre-store `__syncthreads` (L298) |

The three structural invariants that make the Albatross class impossible here
(each independently sufficient):

1. **Predication is control flow, never the src-size operand.** A copy that
   shouldn't happen is *not issued* — there is no "off but still writing"
   state. (Their bug lives entirely in src-size-as-predicate.)
2. **Issued destinations are thread-owned.** Every `__pipeline_memcpy_async`
   destination is a distinct 16 B chunk derived from the loop index `t`
   (`t`-partition of rows×chunks); no two threads of any predicate state
   share a destination in the same window.
3. **Wait-then-block-barrier before first consumer read**, and a block
   barrier between last read of a buffer and its restage (the WAR edge), on
   every pipeline — checked line-by-line above.

## 4. Adversarial validation (3090 / sm86 dev box, CUDA 12.9 compute-sanitizer, torch 2.11.0+cu129; kernel sources sha256-verified identical to the audited tree)

Positive/negative control (`bench/probes/cp_async_zfill_control.cu` — our own
minimal construction of the class; probe only, never linked into extensions):

```
[buggy] launches=200 blocks=1024  wiped=2895184 / 13107200 outputs  (cuda: no error)
[fixed] launches=200 blocks=1024  wiped=0 / 13107200 outputs  (cuda: no error)

========= Error: Race reported between Write access at void control_kernel<(bool)0>(...)+0xb0
=========     and Write access at void control_kernel<(bool)0>(...)+0xb0 [131072 hazards]
   (x3 launches)
========= RACECHECK SUMMARY: 3 hazards displayed (3 errors, 0 warnings)     <- buggy
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)     <- fixed
```

`cuobjdump -sass` of the control confirms the flagged instruction at `+0xb0`
is `LDGSTS.E [R11], [R4.64], !P0` — the trailing-predicate (zero-fill
selector) form.

Our kernels (`bench/probes/race_audit_driver.py`, 39 launches: w4
m1/small/tc/dequant + split-K long-K shape, w4a8 both algos × {33,65,256} ×
{2048,1088} K × ragged N=2112, w8 m1/small/tc/tc_large + ragged M=33/129 +
odd nk, w8a8 V1+V2 × {31,65,257} × bias/no-bias × fp16/bf16):

```
========= COMPUTE-SANITIZER (racecheck)
race_audit_driver: 39 audited-kernel launches completed OK (sm86)
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)

========= COMPUTE-SANITIZER (synccheck)
race_audit_driver: 39 audited-kernel launches completed OK (sm86)
========= ERROR SUMMARY: 0 errors
```

SASS census (predicated-LDGSTS = the bug machinery; expect zero):

```
rwkv7_w4.so   (sm86):  240x LDGSTS.E.BYPASS.128 [R], [R.64]   — 0 predicated
rwkv7_w8.so   (sm86):  216x LDGSTS.E.BYPASS.128 [R], [R.64]   — 0 predicated
rwkv7_w8a8.so (sm86):   96x LDGSTS.E.BYPASS.128 [R], [R.64]   — 0 predicated
rwkv7_wkv.so  (sm120, compile-only on the 3090 box): 24x LDGSTS.E.BYPASS.128
              [R...], desc[UR4][R.64...] — 0 predicated; stores 12x STG.E.ELL2.256
```

## 5. Gate re-runs on the audited build (un-instrumented, same box)

verify_w4.py / verify_w8.py / verify_w8a8.py / verify_w4a8.py: all green
(w4: kernel-numerics + small-M bit-exact vs M1 + TC OK; w8: ALL OK incl.
tc-large ragged; w8a8: V1+V2 exact on fp16+bf16 incl. M=33/N=4160 edges and
bias path, batch-invariance exact; w4a8: both algos exact vs the bit-mimic
reference across M∈{65..512} × 4 shapes, K-pad + ragged-N + batch-invariance
+ cross-algo bit-identity PASS). No code changed in this audit, so this is a
re-confirmation on the exact bytes audited, not a fix gate.

## 6. Repro

```
# on the box (sources rsync'd flat or via the overlay-relative fallback):
nvcc -arch=sm_86 -O3 -o cp_async_zfill_control bench/probes/cp_async_zfill_control.cu
./cp_async_zfill_control buggy 200 && ./cp_async_zfill_control fixed 200
compute-sanitizer --tool racecheck ./cp_async_zfill_control buggy 3
TORCH_EXTENSIONS_DIR=... python3 bench/probes/race_audit_driver.py
compute-sanitizer --tool racecheck  --error-exitcode 3 python3 bench/probes/race_audit_driver.py
compute-sanitizer --tool synccheck --error-exitcode 3 python3 bench/probes/race_audit_driver.py
```

## 7. Standing guardrails (for future cp.async work, incl. the W1 sm120 kernels)

- Never predicate a cp.async via the src-size operand unless the destination
  is provably private to the issuing thread; prefer control-flow predication
  (don't issue), or give pred-off lanes a scratch destination.
- Any new cp.async site must keep the three invariants of §3 and be added to
  `bench/probes/race_audit_driver.py` (a shape hitting each branch), then
  racecheck + synccheck re-run — sanitizer evidence, not gate passes, is the
  race-tier bar.
- `rwkv7_wkv` on-device racecheck: run the wkv section of the driver on the
  next free sm100+ window (it self-gates on capability; everything is
  already staged).
