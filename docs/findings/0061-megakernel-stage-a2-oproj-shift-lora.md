---
doc_kind: finding
finding_id: F0061
title: "Megakernel Stage-A2 (#50): the next fused-block components on the 3090 — (1) o_proj folded into a role-generic grouped GEMV + the 4-role whole-block r/k/v/o prefab (gemv_rkvo_m1), bit-exact (kernel torch.equal + real 1.5B/7.2B model byte-identical), launch-neutral on sm86 (o is post-WKV) but the sm120 grid's r/k/v/o stage; (2) token-shift prologue (shift_lerp6) confirmed already-fused (read-state+lerp+write, no HBM round-trip of the shifted intermediate), byte-exact re-gated on the 3090 (ALL EXACT); (3) LoRA-into-grid hits the anticipated bit-exactness obstacle — a single grouped launch has one block dim, but 7.2B r/k/v/o is Threads=256 and lora_stage1 is hard-tuned Threads=128 (4-warp serial sum), so folding lora-down changes its reduction tree (4->8 partials) and breaks byte-identity vs the shipping lora4_m1; reported next-stage. Default OFF byte-identical (zero regression); grouped-GEMV perf recovered after catching a by-value-pointer-pack regression (~1.5x slower -> reverted to __restrict__ scalar params: grouped graphed +3.09/+6.47us again)"
last_verified_commit: "HEAD (rwkv7_mega.cu role-generic + gemv_o_m1/gemv_rkvo_m1 + rwkv7.py o_proj wiring + test_mega_rkv.py/test_mega_o_model.py/mega_insitu_launches.py)"
discovered_by: Opus 4.8 (agent), 2026-07-16
severity: info
status: open - Stage-A2 components landed (o_proj role + whole-block r/k/v/o prefab gated; shift prologue re-verified; LoRA obstacle documented). sm120 PDL chaining + flagship perf staged (do NOT touch the 5090 during RL training).
related: [F0060, F0058, F0056, F0051, F0023]
machine: 3090 (sm86), rwkvmain container, CUDA 12.9 / torch 2.11.0+cu129
---

# Finding F0061: RWKV-7 megakernel Stage-A2 — o_proj role + whole-block r/k/v/o prefab, shift prologue re-verified, LoRA-into-grid obstacle

Stacks on F0060 (Stage-A1: `gemv_rkv_m1`, the r/k/v grouped GEMV). Builds the next
fused-block components that are bit-exact-gateable on sm86 WITHOUT PDL — the prefab
pieces the sm120 (5090) megakernel assembly chains later (PDL is sm90+ only, F0060 §0).
All env-gated `RWKV_MEGA` (default OFF); the 3090 gates STRUCTURE + CORRECTNESS, the
flagship overlap number is an sm120 run (F0060's fast-card re-framing stands).

## 0. TL;DR (per component, in the F0060 §7.5 staged order)

| # | component | built? | kernel gate | model gate | 3090 launches | 5090 role |
|---|---|---|---|---|---|---|
| 1 | **o_proj as a grouped role** + 4-role `gemv_rkvo_m1` | **YES** | PASS (0 bytes) | PASS (0 bytes) | neutral (o post-WKV) | the whole-block r/k/v/o stage prefab |
| 3 | **token-shift prologue** (`shift_lerp6`) | already built (F0056) | PASS (ALL EXACT, re-gated) | — (element-wise) | 1 (unchanged) | the block prologue prefab |
| 2 | **LoRA down-proj into the grid** | **obstacle** (next-stage) | — | — | — | needs a per-role-config grid (sm120) |

Cumulative launches/layer (pure CUDA kernels, DtoD memcpy excluded): **18 → 16 (−2)**,
unchanged from Stage-A1 — Stage-A2's o_proj routing is **launch-neutral on the 3090**
(o_proj is post-WKV, so it cannot share the r/k/v launch without the 5090's persistent
grid; F0060 §7.1-7.2). The Stage-A2 value is the **bit-exact 4-role whole-block
`gemv_rkvo_m1`** the 5090 grid folds into ONE launch, plus the re-verified prologue.

## 1. Component 1 — o_proj folded into a role-generic grouped GEMV (BUILT, two gates PASS)

**What.** Generalized `gemv_rkv_m1_kernel` (3-role) into `gemv_grouped_m1_kernel`
(G∈{1..4}, `blockIdx.y` = role) and layered three public ops on it:
- `gemv_rkv_m1` (G=3) — the r/k/v stage (re-pointed to the general kernel; byte-identical).
- `gemv_o_m1` (G=1) — o_proj as a role (F0060 §7.5 "add o_proj").
- `gemv_rkvo_m1` (G=4) — the whole-block r/k/v/o stage the sm120 grid chains (r/k/v roles
  0-2, WKV, then o_proj role 3, with PDL between). **This is the Stage-A2 prefab.**

o_proj is another M==1 `[N,K]·[1,K]^T` GEMV with **(N,K)=(H,H)** exactly like r/k/v, so it
takes the **identical** `_select_config`; each role's fp32 accumulation is `gemv_m1_kernel`
verbatim → byte-identical by construction. `models/rwkv7.py` routes the deployed o_proj
through `mega.gemv_o_m1` under `RWKV_MEGA` (eligibility mirrors the r/k/v mega block).

**Perf-regression caught + fixed (house discipline).** The first cut passed a by-value
pointer pack (`struct{const dtype* x[4]; w[4];}`) with a runtime-indexed `pack.x[proj]`.
Bit-exact but **~1.5× SLOWER** (1.5B grouped graphed 30.05→47.16 µs, 7.2B 114.9→182.5) —
the dynamic-indexed pack defeats the `__restrict__` aliasing / register analysis. Reverted
to **explicit `__restrict__` scalar params + a 4-way ternary on `blockIdx.y`** (the uniform
per-block select the original 3-pointer kernel used); grouped graphed recovered to
**+3.09 µs (1.5B) / +6.47 µs (7.2B)** vs separate — matching F0060's +3.17/+6.19. Lesson
recorded in the .cu header.

**Kernel gate** (`bench/test_mega_rkv.py`, `torch.equal`): **PASS — zero differing bytes**
for rkv (G=3, regression), o (G=1), and rkvo (G=4) across shapes {1.5B, 7.2B, 0.1B, odd-N
→ OutTile=1, small-K} × {uniform, heavy-tailed} × scale {0.5, 2, 8}.

**Model gate** (`bench/test_mega_o_model.py`, real `Rwkv7Attention` at the deployed 1.5B +
7.2B configs, its real-shaped r/k/v/o projection weights + deployed `_select_config`):
**PASS — byte-identical** on both — `gemv_o_m1 == gemv_m1(xo, w_o)`,
`gemv_rkvo_m1 == stack(gemv_m1 ×4)`, `gemv_rkv_m1 == stack(gemv_m1 ×3)`. (Same rigor as the
shipping mega_model_gate: bit-exactness is value-independent, so the real module's weights
at the real config are sufficient; F0060 §6.)

**In-situ** (`bench/mega_insitu_launches.py`, deployed `Rwkv7DecoderLayer`, shipping flags,
torch.profiler; DtoD memcpy excluded):
- **RWKV_MEGA=0** (default): 18 launches/layer — `gemv_m1<T,ot> ×4` for r/k/v/o (**unchanged**
  from the pre-Stage-A2 stack → zero regression on the default path).
- **RWKV_MEGA=1**: 16 launches/layer — `gemv_grouped_m1<T,ot> ×2` = r/k/v (G=3, one launch) +
  o_proj (G=1, one launch). o_proj **routed through the mega grouped kernel, byte-identical,
  but still 1 launch** (post-WKV). Even eager the 2 grouped launches beat the 4 separate
  (1.5B 42.87 vs 47.94 µs; 7.2B 156.4 vs the 4×gemv_m1). 1.5B cfg `<128,2>`, 7.2B `<256,1>`
  (autotune) — o shares r/k/v's config, so grouping stays bit-exact.

**Honest read.** o_proj on the 3090 is **launch-neutral** — the −2 is entirely the
Stage-A1 r/k/v grouping; Stage-A2 makes o_proj a *mega role* (byte-identical) and delivers
the **4-role `gemv_rkvo_m1`** that lets the 5090 persistent grid collapse r/k/v/o to ONE
launch (the launch win o_proj structurally cannot get on the 3090). Prefab built + gated.

## 2. Component 3 — token-shift prologue already fused (shift_lerp6), re-verified byte-exact

The coordinator's goal ("one prologue kernel producing the shifted input the grouped
projections consume, no HBM round-trip of the shifted activation") is **already met by
`shift_lerp6`** (F0056 W1' glue). `rwkv7_glue.cu::shift_lerp6_kernel` does, in ONE kernel:
read the paged token-shift state `conv` (the previous token) → `sh = round_fp16(conv[ci])`
→ 6-way lerp `out[j] = round_fp16(x + round_fp16(mix[j]·round_fp16(sh−x)))` → write the 6
lerped activations **and** the state update `conv[ci] ← x`. The shifted (prev-token)
intermediate is consumed **in registers**, never materialized to a separate HBM tensor.

- **In-situ:** `shift_lerp6_kernel<256> ×1` is active in the deployed stack at both 1.5B
  and 7.2B, RWKV_MEGA on and off (it is the prologue the grouped r/k/v GEMV consumes).
- **Byte-exact re-gate on the 3090** (`bench/test_glue.py` vs the torch token_shift+lerp
  reference): **ALL EXACT** — T∈{1,2,4,8,32}, padded-replay slots (rows zeroed, conv
  untouched), and out-of-range indices. Pure element-wise reordering → greedy 24/24 EXACT
  holds (it is under the shipping production gate, F0056).

The deeper "lerp never leaves smem" fold (`shift_lerp6` INTO the r/k/v GEMV prologue,
F0060 §7.5) is a **5090 whole-block-grid item**, NOT a 3090 win: `xr..xv` are also consumed
by the LoRA down-projections + `kk_kmix`, so on the 3090's per-stage-launch structure they
must materialize for those consumers; folding only the r/k/v lerp into the GEMV would
duplicate it. In the 5090 persistent grid every stage is in one launch, so the lerp is
produced once and stays on-chip for all consumers — that is where this fusion pays.

## 3. Component 2 — LoRA down-proj into the r/k/v grid: the anticipated bit-exactness obstacle

The albatross `rkv_lowrank_pre` packs the big r/k/v projections + the small LoRA GEMVs into
ONE grid via interleaved block-role dispatch. Replicating it bit-exactly against OUR shipping
`lora4_m1` (ADR-0004: reproduce our own kernel's bytes) hits a hard wall on sm86:

1. **One launch has one block dim; the roles need different reduction configs.** Measured
   in-situ: the 7.2B r/k/v/o grouped GEMV runs `gemv_grouped_m1<256,1>` (**Threads=256**),
   while `lora_stage1_kernel<128>` (the LoRA down-proj) is hard-tuned to **Threads=128** with
   a 4-warp cross-warp **serial** sum (`partial[Threads/32] = partial[4]`). Folding lora-down
   into the r/k/v grid forces it into a 256-thread block → the serial sum becomes 8 partials
   in a different order → **NOT byte-identical** to the shipping `lora4_m1`. (At 1.5B the
   threads happen to coincide at 128, but that is an autotune coincidence — it breaks the
   moment `_select_config` picks a different Threads class — and the r/k/v `out_tile` still
   differs from lora-down's 1-output-per-block.)
2. **The chain is two dependent stages.** Only `lora_stage1` (down-proj, shares K=H with
   r/k/v) is co-launchable; `lora_stage2` (up-proj) needs stage1's output **after** the
   tanh/sigmoid activation — a producer→consumer barrier that cannot live in the same
   non-persistent launch. So the best case is −1 launch (fold down-proj only), and even that
   only bit-exactly where the Threads class coincides.

Per the coordinator's pre-authorization ("if the interleaved-role packing can't hold
bit-exactness … deliver o_proj + shift and report LoRA as next-stage with the specific
obstacle"), Component 2 is **reported next-stage**. It becomes tractable on the **5090**,
where the whole-block persistent grid can host per-role block configs (each role its own
warp count) under PDL — i.e. the LoRA-into-grid fold is a sm120 assembly step, not a 3090
one. albatross gets away with it because it has no bit-exact-vs-existing-path constraint;
we do (the greedy-EXACT gate + ADR-0004).

## 4. Cumulative state + updated sm120 (5090) assembly plan

**3090 launches/layer:** 18 → **16** (−2, Stage-A1 r/k/v grouping; Stage-A2 o_proj neutral).
Default OFF path byte-identical (zero regression), verified in-situ (kernel histogram
unchanged) + by the rkv model/kernel gates.

**Now PREFAB (bit-exact-gated on sm86, ready for the 5090 to chain):**
- `shift_lerp6` — the block prologue (read-state + lerp + write, on-chip shifted). [F0056]
- `gemv_rkv_m1` (G=3) — the r/k/v projection stage. [F0060]
- `gemv_o_m1` (G=1) — o_proj as a mega role. [F0061]
- **`gemv_rkvo_m1` (G=4)** — the whole-block r/k/v/o projection stage. [F0061] ← new
- `wkv_decode` CUDA kernel — the WKV recurrence stage. [F0058]

**Still needs the 5090 (PDL / persistent grid), unchanged from F0060 §7 + the new items:**
1. Build `rwkv7_mega.cu` for `sm_120`; the guarded `griddepcontrol` assembles on sm90+; set
   `cudaLaunchAttributeProgrammaticStreamSerialization` on the grouped launches.
2. Chain the block: `shift_lerp6` → `gemv_rkv_m1`/`gemv_rkvo_m1` (roles 0-2) → `wkv_decode`
   → o_proj (role 3 of `gemv_rkvo_m1`) → LoRA/kk, via `launch_dependents`/`wait`, captured in
   one CUDA graph. **This is what folds o_proj's separate launch into the whole-block grid**
   (the 3090 launch-neutral o_proj becomes a real −1 on the 5090) AND recovers the sub-peak
   M==1 GEMV the saturated 3090 hides.
3. Component-2 LoRA-into-grid becomes tractable here (per-role block configs under the
   persistent grid) — the sm120 continuation of the interleaved-role pack.
4. Gate on sm120: `verify_m1d` greedy-EXACT 24/24 (1.5B) + fixture 8/8 (7.2B); re-run
   `test_mega_rkv.py` + `test_mega_o_model.py` (arch-independent, re-confirm on the silicon).
5. Measure the flagship: bsz1 fp16 before/after, 7.2B + 1.5B, vs the 1691.7 GB/s ceiling +
   the 79.4% → 92.4% frame (F0060 §7.4).

## 5. Honest ledger

- **Delivered (bit-exact, two-gated, zero-regression):** the role-generic grouped GEMV;
  o_proj as a mega role (`gemv_o_m1`, wired + gated); the whole-block **`gemv_rkvo_m1`**
  4-role prefab (kernel + real-weight model gates, zero differing bytes); the caught+fixed
  by-value-pack perf regression; the re-verified `shift_lerp6` prologue (ALL EXACT on the
  3090); the in-situ launch histogram (18→16, o routed, default unchanged).
- **Obstacle (reported, not a failure):** LoRA-into-grid — block-dim uniformity vs per-role
  reduction config breaks byte-identity vs `lora4_m1` on sm86; sm120 persistent-grid item.
- **Staged (sm120-gated, do NOT touch the 5090 during RL training):** the PDL chain wiring
  (§4.1-4.2) that turns the prefabs + inert scaffolding into the continuous-stream win and
  folds o_proj's launch; the flagship before/after; the LoRA sm120 fold. None blocked by a
  correctness/feasibility finding.

## 6. Artifacts + cross-references

- Kernel/loader/wiring: `sglang_overlay/.../rwkv7_kernels/cuda/rwkv7_mega.cu`
  (`gemv_grouped_m1_kernel` + `gemv_rkv_m1`/`gemv_o_m1`/`gemv_rkvo_m1`), `.../mega.py`,
  `models/rwkv7.py` (o_proj `RWKV_MEGA` block).
- Gates/probes: `bench/test_mega_rkv.py` (kernel, +o/rkvo), `bench/test_mega_o_model.py`
  (real-weight model), `bench/mega_insitu_launches.py` (in-situ histogram), `bench/test_glue.py`
  (shift_lerp6 byte-exact).
- [[F0060]] (Stage-A1 r/k/v grouping + the 3090 bsz1 profile + fast-card re-framing + the PDL-
  on-sm86 impossibility this builds on) · [[F0058]] (WKV CUDA kernel = the WKV stage prefab) ·
  [[F0056]] (shift_lerp6 = the prologue this re-verifies) · [[F0051]] (launch-count profile) ·
  ADR-0008 (megakernel feasibility) · ADR-0004 (zero-FLA; every role transcribes our `gemv_m1`).
