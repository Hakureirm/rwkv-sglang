---
doc_kind: finding
finding_id: F0063
title: "Megakernel sm120 assembly (#50): the PDL chain is LIVE — griddepcontrol wait/launch_dependents wired across the whole bsz1 decode block (8 CUDA files + 2 triton glue kernels, RWKV_PDL default OFF), bit-exact armed AND unarmed (kernel battery zero differing bytes; greedy 24/24 @1.5B + 8/8 @7.2B under CUDA graph with the full stack + MEGA + WKV_CUDA + PDL), SASS-verified (ACQBULK/PREEXIT = sm_120 lowering); CLEAN flagship numbers landed 2026-07-18: 7.2B D=137.8 tok/s serving bsz1 (82.0% of the 168.0 ceiling) / 136.43 tok/s matched kernel-loop framing (79.0% overlapped transitions, up from 60.8% dirty) = 87.9% of Bo's 155.2 (87.6% of our own same-session albatross v3b re-measurement, 155.75); 1.5B D=493.0 tok/s; Option-B deploy.sh validated on the tower (exact base 754524d, --check clean, byte-verified); F0062 compute-sanitizer racecheck+synccheck closed (63/63 launches, 0 hazards/errors)"
last_verified_commit: "(this series) on bbdddfd; clean flagship measurement session added no code changes"
discovered_by: Fable 5 (agent), 2026-07-17
severity: info
status: closed — chain built + gated, clean flagship numbers landed for 7.2B + 1.5B (both framings), same-session albatross v3b corroboration done, F0062 racecheck+synccheck closed
related: [F0060, F0061, F0062, F0058, F0056]
machine: 5090 tower (sm120, driver 595-open/CUDA 13.2 host, dev-cu12 container CUDA 12.9.1, torch 2.11.0+cu129, sglang main 754524d + RWKV-7 overlay)
---

# Finding F0063: sm120 PDL-chain assembly + flagship measurement (#50)

## 0. TL;DR

- **The PDL chain is BUILT, LIVE and bit-exact-gated on sm120** (commits
  7863f12 + 0099c5a): griddepcontrol wait/launch_dependents wired across the
  whole bsz1 decode block — 8 CUDA files (shift_lerp6/1, grouped r/k/v + o
  GEMV, gemv_m1(+sqrelu), lora_stage1/2, add_ln, gn_gatecorr, wkv_decode,
  sparse_cmix) PLUS the two triton glue kernels (kk_kmix, lora_gates via
  gdc intrinsics + launch_pdl). Env `RWKV_PDL` (default OFF), per-stage
  `RWKV_PDL_SCOPE`. Every wired kernel waits before its first
  producer-dependent read; arming changes scheduling only.
- **Gates all green, armed AND unarmed**: kernel battery zero differing bytes
  (mega 10/10, glue ALL EXACT, wkv OVERALL PASS both state dtypes +
  batch-invariance, lora/ln/sparse/sqrelu/lora_gates PASS, kk_kmix
  sha256-identical); e2e greedy under CUDA graph 1.5B **24/24** + 7.2B
  **8/8** EXACT with full stack + MEGA + WKV_CUDA + PDL. SASS confirms
  ACQBULK/PREEXIT (sm_120 lowering of griddepcontrol).
- **PDL measured ACTIVE in production** (dirty-window trace, D-config server):
  **60.8% of same-stream consecutive kernel transitions overlap** — negative
  inter-kernel gaps are impossible without programmatic launch — netting the
  step's kernel-gap total down to +10.9 us/step (89.3 us positive vs 78.4 us
  overlap-gained). Also: the c=1 GPU timeline is CONTIGUOUS (overlap
  scheduler hides all host prep) -> the bsz1 gap to albatross lives INSIDE
  the graph, not in serving overhead.
- **Option-B deploy.sh validated on sm120** (exact base 754524d, --check
  clean, byte-verified) — the #55 second-platform proof.
- **Flagship timing legs: CLEAN, LANDED (2026-07-18)** — the window opened
  (SkyPilot API server down, card idle, zero tracking pods) and the §6b
  runbook ran end-to-end with zero yields needed. 7.2B: A (anchor) 136.9 /
  D (flagship) 137.8 tok/s serving bsz1, both greedy 8/8 EXACT; matched
  kernel-loop framing (the fair comparison to Bo, whose 155.2 is
  kernel-harness event-timing) A=134.96 / D=136.43 tok/s, D = 82.0% of the
  168.0 ceiling (serving) / 81.2% (kernel-loop) = **87.9% of Bo's 155.2**;
  clean-window PDL overlap is 79.0% of same-stream kernel transitions (up
  from 60.8% dirty), net gap/step now NEGATIVE (-79.9 us — overlap-gained
  exceeds remaining positive gaps). Same-session albatross v3b (g1g
  checkpoint) independently reproduced Bo's number: 155.75 tok/s (D is
  87.6% of that). 1.5B: A=479.2 / D=493.0 / B=491.6 / C=492.9 tok/s, all
  EXACT. F0062's owed compute-sanitizer racecheck+synccheck also closed
  this window: 63/63 launches, 0 hazards, 0 errors, both tools. Full
  numbers, per-kernel tables and raw files: §6, §6c, `bench/results/
  mega_*_5090.*`.

## 1. Environment + deploy validation (#55 Option-B on sm120)

- Fresh serving container `rwkv-serve` (host docker per the 2026-07-17 platform
  regime: 持续服务 = 宿主 docker; `--restart=no`; yields to sky scheduling).
  Image: cached `lmsysorg/sglang:dev-cu12` (no registry pull). sglang checkout
  fetched to the EXACT patch base `754524d8de95be98cc2fd55cb02ba6822cf98ee2`
  (the image ships b28bc10; GitHub direct fetch from the tower).
- `scripts/deploy.sh` (BOX= local, VENV_PY=python3): additive overlay copied,
  `upstream_edits.patch` applied with `--check` CLEAN (no drift), RWKV_CHAIN
  registered; the is_hybrid_ssm WARN is the documented benign case on main.
  Deployed files byte-verified (cmp) against the synced tree: zero diffs.
  **Option-B is validated on sm120** — same path as the 3090 proof.
- Models: `/data/rwkv/rwkv-sglang/models/rwkv7-{7.2b,1.5b}-fla`.

## 2. Phase 1 — server boot + smoke + anchor

- 7.2B fp16 server (serve.sh full W1' stack + RWKV_STATE_FP16=1 = the
  legFinal_B anchor config; MEMFRAC=0.85, CGMAXBS=32 + max-running 32 — pool
  shrunk for RL co-residency, does not touch the bs=1 decode path; the F0056
  anchor ran CGMAXBS=512).
  Boot note: mem-fraction 0.75 FAILS on this card with the RL co-resident —
  `handle_max_mamba_cache` needs the state pool inside rest_memory; 512 slots
  x ~17 MB (fp16 state) = 8.7 GB did not fit next to 13.4 GB weights.
- Greedy smoke: fixture `oracle_rwkv7_72b_eiffel` **8/8 EXACT** via /generate.
- bsz1 anchor (64-in/256-out c=1): **100.7 tok/s DIRTY** — genesis_sim2sim
  (motion cxk) co-resident at 90-97% GPU util (~4.4 GB). The brief's "idle sim
  ~3%" state did not hold this session; −25% vs the 133.4 clean anchor. All
  headline timing legs deferred to quiescence (monitors armed: RL-quiet +
  sky-pod-Pending yield sentinel per the platform rules).

## 3. Phase 2 — prefab re-gates on sm120 (all PASS)

| gate | result |
|---|---|
| `test_mega_rkv.py` kernel (rkv G=3 / o G=1 / rkvo G=4, 5 shapes x 2 families x 3 scales) | PASS — zero differing bytes |
| `test_mega_o_model.py` real 1.5B + 7.2B weights (models symlinked to /models) | PASS — byte-identical |
| `test_glue.py` shift_lerp6/1 (pads, out-of-range) | ALL EXACT |
| `test_wkv_cuda.py` (fp32+fp16 state, pads, 64-step chain, batch-invariance) | OVERALL PASS — first gate on sm100+ silicon (3090 falls back by construction, F0062 §0) |
| e2e `verify_m1d` full stack + MEGA + WKV_CUDA, cuda-graph ON | 1.5B **24/24 EXACT**, 7.2B **8/8 EXACT** |

- verify_m1d needed a version-adaptive fix on main >= ~754524d:
  `disable_piecewise_cuda_graph` became per-phase backends
  (`cuda_graph_backend_prefill="disabled"` is the legacy flag's mapping) —
  the F0060 §6 "ServerArgs version skew" item, now closed.
- 5090 microbench observation (dirty, but the exactness gates are the point):
  7.2B r/k/v grouped graphed 21.23 -> 17.96 us (+3.27 us saved/block) vs the
  3090's 121.07 -> 114.88 (+6.19) — the launch-gap recovery is ~3x larger as a
  FRACTION of block time on the fast card (15% vs 5%), consistent with F0060's
  fast-card thesis. Caveat: 96 MB L2 makes isolated microbench BW meaningless
  here (weights go L2-resident; production streams 10.07 GB/step); only
  in-situ/full-model numbers are honest on this card.

## 4. Phase 3 — the PDL chain (the new work; commit 7863f12)

- `cuda/rwkv7_pdl.cuh`: device `rwkv7_pdl_wait()` / `rwkv7_pdl_launch_dependents()`
  (`__CUDA_ARCH__ >= 900` guarded; PTX-documented NO-OPS on plain launches) +
  host `rwkv7_launch_maybe_pdl(...)` (cudaLaunchKernelEx +
  PROGRAMMATIC_STREAM_SERIALIZATION when armed, plain `<<<>>>` otherwise).
  Env: `RWKV_PDL=1` master (cc>=9.0 runtime-checked), `RWKV_PDL_SCOPE=a,b,..`
  per-stage arming (glue,mega,lora,ln,wkv,fast,sparse) for incremental
  attribution without rebuilds.
- Wired sites (every armed kernel waits before its first producer-dependent
  read; launch_dependents at the tail unless noted):
  shift_lerp6/1 (glue) -> gemv_grouped r/k/v + o roles (mega) -> lora_stage1/2
  (lora) -> wkv_decode (wkv; the state cp.async stays PRE-wait = a real
  producer-independent prologue, and the trigger sits right after the `o`
  store, before the bulk state write-back — dependents schedule early while
  the consumer's own wait still spans our full grid) -> gn_gatecorr + add_ln
  (ln) -> gemv_m1(+sqrelu) (fast) -> sparse_cmix (sparse).
- Chain breaks (documented, not wired): the 2 Triton glue kernels
  (lora_gates, kk_kmix — triton 3.6 gdc intrinsics exist as a follow-up), the
  stock vectorized_layer_norm boundary, lm_head (stock ParallelLMHead /
  LogitsProcessor), and the sparse path's 2 native side-kernels (at::zeros
  fill + fp32->fp16 cast — identified as F0060's "2x tiny"; a Stage-B fusion
  target).
- Gates: the full kernel battery (mega/glue/wkv/lora/ln/sparse/sqrelu) run
  ARMED and UNARMED — identical PASS, zero differing bytes (arming changes
  scheduling only). e2e greedy under CUDA graph with RWKV_PDL=1: 1.5B 24/24 +
  7.2B 8/8 EXACT — PDL launches capture into and replay from CUDA graphs on
  this stack (corroborating ADR-0008 A0.1 + the Albatross v3b precedent).
- SASS: `griddepcontrol.wait/.launch_dependents` lower to **ACQBULK/PREEXIT**
  on sm_120 (the earlier "0 GRIDDEPCONTROL instrs" grep was the sm90 mnemonic
  — worth remembering for future SASS censuses). 18 instrs in rwkv7_mega.so
  alone; cubins are sm_120.

## 5. Phase 5 — F0062's deferred wkv racecheck on sm120

- `race_audit_driver.py` plain run: **63 audited-kernel launches completed OK
  on sm120** (the wkv section now runs on-device; 39 on sm86).
- The compute-sanitizer racecheck run was **ABORTED by us mid-run**: with the
  RL pipeline (genesis chunked + hb_compile) co-resident, host load hit
  368-496 and sshd stopped answering (banner-exchange timeouts) — the
  sanitizer was contributing to crushing a box 19-21 users share. Killed
  (SIGTERM/KILL), load recovered 496 -> 61 within ~2 min. Re-run owed in a
  quiet window, AFTER the flagship legs (it serializes the GPU; lowest
  priority insert). Lesson recorded: long container jobs go through
  `docker exec -d` + log file, never tied to the ssh channel.
- **CLOSED, 2026-07-18 (clean window, last item in the runbook)**: both
  `compute-sanitizer --tool racecheck --error-exitcode 3` and
  `--tool synccheck --error-exitcode 3` over `race_audit_driver.py` ran to
  completion in the same-tenant (us-only) window — **63/63 audited-kernel
  launches, 0 hazards displayed (0 errors, 0 warnings) under racecheck, 0
  errors under synccheck**, exit code 0 both times. Total wall time for the
  whole battery (both tools) was ~100s — confirming the earlier abort was
  the co-resident host load, not the instrumented workload itself being
  fundamentally slow. Raw logs: `bench/results/mega_racecheck_7.2b_5090.log`,
  `bench/results/mega_synccheck_7.2b_5090.log`. This closes F0062's owed
  item; no further racecheck debt outstanding.

## 6. Flagship timing (Phase 4) — CLOSED 2026-07-18, clean numbers landed

Matrix (bench/mega_flag_matrix.sh, 64-in/256-out c=1, greedy-smoke hard gate
per leg, headline pair A/D run first): A anchor / B +MEGA / C +MEGA+WKV /
D +MEGA+WKV+PDL. Framings: (1) serving bsz1 vs the 133.4 anchor; (2) matched
kernel-loop event timing vs albatross 155.2 = 92.4% of the 168.0 tok/s
sparse-byte ceiling (1691.7 GB/s / 10.07 GB/step). Both framings are now
populated for A and D — see §6c for the clean results; this closes the
"Clean results: TBD" gap left open below.

## 6a. Dirty-window shakeout (SUPERSEDED by §6c — kept for the record)

Validated under DIRTY conditions (genesis sim2sim motion-cxk co-resident at
90-100% util the whole session — numbers below are pipeline shakeouts, NOT
headlines, and are superseded by the clean measurements in §6c):

- **D-config e2e works**: boot + smoke 8/8 EXACT with MEGA+WKV_CUDA+PDL armed
  (serving log shows `[rwkv7_pdl] PDL chain ARMED`); dirty c=1 97.9 tok/s vs
  dirty anchor 100.7 (within co-residency noise; sim at 93-97%).
- **Framing 2 = profiler-trace parser** (`bench/step_span_from_trace.py`;
  `sglang.bench_one_batch` is NOT RWKV-viable — it bypasses the scheduler's
  mamba-pool setup and index-OOBs). Key shakeout-trace findings (48-step
  window, D-config, dirty):
  * the c=1 GPU timeline is CONTIGUOUS — no inter-step idle (p99 inter-kernel
    gap 0.29 us): the overlap scheduler fully hides host prep, so the
    133.4-vs-155.2 gap lives INSIDE the graph, not in serving overhead;
  * **PDL measured ACTIVE in production: 60.8% of same-stream consecutive
    kernel transitions OVERLAP** (negative gaps — impossible without
    programmatic launch); positive gaps 89.3 us/step vs 78.4 us/step
    overlap-gained -> net gap only +10.9 us/step;
  * step composition (533 kernels/step): grouped r/k/v+o GEMV 64.5x +
    ffn.key GEMV 32.2x + lm_head (stock cuBLAS gemvx!) = ~74% of busy;
    fattest non-GEMV: add_ln 64.5x (~670 us dirty), sparse_cmix 32x,
    lora_stage1/2 32x each; hand-CUDA WKV only ~2.7 us/layer; the sparse
    path's zeros-fill + fp16-cast pair ~65 us/step (fusion target).
- **Same-session albatross v3b staged**: harness + our g1g-7.2b checkpoint
  (Bo used g1f — same dims, identical byte traffic; disclosed) build + run
  fine in the rwkv-serve container (the albatross5090 container lost its GPU
  binding — stale docker device access). Dirty shakeout 86.8 tok/s (their
  code pays the same co-residency tax; clean re-run in the window). Note
  v3b builds with --use_fast_math (their numerics posture, unchanged).

## 6c. Clean results (2026-07-18, single-tenant window, zero yields needed)

The SkyPilot API server was unreachable this window (why the card was free);
the §6b runbook ran start-to-finish with no co-resident processes and no
tracking pods at any point (checked before every leg + a ~20-25s sentinel
throughout; see §7). GPU memory was flat at 19740-19748 MiB across all four
7.2B legs (pre/post `nvidia-smi` snapshots bracketing every leg) — no other
tenant ever touched the card during measurement.

**7.2B serving bsz1 (framing 1)** — `bench/results/mega_flagmatrix_c1_{A,D,B,C}_7.2b_5090.json`,
run log `mega_flagmatrix_run_7.2b_5090.log`:

| leg | config | tok/s | gate |
|---|---|---|---|
| A | anchor (MEGA=0 WKV_CUDA=0 PDL=0) | 136.9 | greedy 8/8 EXACT |
| D | flagship (MEGA+WKV_CUDA+PDL) | 137.8 | greedy 8/8 EXACT |
| B | +MEGA only | 138.1 | greedy 8/8 EXACT |
| C | +MEGA+WKV | 137.8 | greedy 8/8 EXACT |

All four legs land in a 136.9-138.1 tok/s band (~0.9% spread) — at bsz1
serving wall-clock granularity the MEGA/WKV_CUDA/PDL kernel-level savings
are not cleanly resolved above noise (B is nominally the highest of the
four). D = 82.0% of the 168.0 ceiling. This is expected and is exactly why
framing 2 exists (below): the doc's own dirty-window finding that "the c=1
GPU timeline is CONTIGUOUS" means serving overhead is near-zero at bsz1, so
serving-level wall clock and kernel-loop timing should — and do — agree
closely (A: 136.9 serving vs 134.96 kernel-loop, 1.4% apart).

**A-leg deviation, investigated**: A=136.9 runs 2.2-2.9% above the historical
133-134 range (F0056's 133.4 anchor), tripping the runbook's own >2% guard.
Ruled out: co-residency (GPU memory flat at 19740-19748 MiB throughout,
confirmed single-tenant); CGMAXBS 32-vs-512 (already established neutral in
this doc's own Phase 2 notes). Leading explanation: the documented sglang
base bump (b28bc10 -> 754524d, ~5 days of upstream main commits between the
F0056 measurement and this session) — a genuine small upstream speedup,
consistent with A and D moving up together. Not independently bisected to a
specific upstream commit/PR (disproportionate to a 2.6% delta on a live
shared-GPU window); flagged here rather than silently published.

**7.2B matched kernel-loop framing (framing 2)** — the apples-to-apples
comparison to Bo, whose 155.2 is kernel-harness event-timing, not serving.
Captured via `/start_profile` (num_steps=48) on a live c=1 decode stream,
parsed with `bench/step_span_from_trace.py`; full per-kernel tables in
`bench/results/mega_framing2_{A,D}_7.2b_5090.txt`:

| leg | span/step | tok/s | kernels/step | overlapped transitions | net gap/step |
|---|---|---|---|---|---|
| A (anchor) | 7409.6 us | 134.96 | 599.4 | 11.7% | +33.6 us |
| D (flagship) | 7329.8 us | 136.43 | 533.5 | **79.0%** | **-79.9 us** |

D = 81.2% of the 168.0 ceiling, and **87.9% of Bo's 155.2** (136.43/155.2).
The clean-window PDL overlap (79.0%) is markedly higher than the dirty-window
reading (60.8%) — a cleaner confirmation that PDL is genuinely active: gaps
go net NEGATIVE in D (overlap-gained 104.5 us/step exceeds the remaining
positive gaps of 24.6 us/step), vs A's net-positive +33.6 us/step with only
baseline-level overlap. Kernel count/step for D (533.5) exactly reproduces
the dirty shakeout's "533 kernels/step" figure — a useful cross-check that
kernel *count* is contamination-invariant even though kernel *timing* isn't.
Per-kernel notable: D's fused `gemv_grouped_m1_kernel` (64.51/step, 2660.15
us/step) plus the residual unfused `gemv_m1_kernel` (32.26/step, 2600.09
us/step) sum to 5260.24 us/step vs A's single unfused path (161.65/step,
5316.72 us/step) — GEMV busy is down ~56.5 us/step, but D's total BUSY/step
(7409.7) is still slightly above A's (7376.0), because a few other kernels
(notably `wkv_decode_kernel` at 84.42 us/step vs A's `_wkv_recurrent_kernel`
at 75.50 us/step) run marginally hotter in this sample — within the noise
of a 43-step trimmed window, not independently re-verified with a larger
sample in this session. Net effect: the large launch-overlap win is mostly
absorbed by BUSY/step being flat-to-slightly-up, so the end-to-end SPAN/step
gain is modest (~80 us/step, ~1.1%) despite the striking overlap-percentage
jump.

**Same-session albatross v3b** (`bench/results/mega_albatross_v3b_g1g_7.2b_5090.log`):
built + ran in the `rwkv-serve` container (the dedicated `albatross5090`
container's GPU binding was stale, per the dirty-session note); checkpoint
`rwkv7-g1g-7.2b-20260523-ctx8192.pth` (Bo's own number used g1f — same
architecture/dims, disclosed; v3b's own CUDA graph + `cuda::Event`
kernel-loop harness, `--use_fast_math` is their posture, unchanged). Result:
**tok_s_p50 = 155.75** (p10=6.411 ms, p50=6.421 ms, p90=6.432 ms) — this
independently reproduces Bo's published 155.2 to within 0.35% on our own
card, which is itself a useful corroboration. D vs this same-session number:
136.43/155.75 = **87.6%** (essentially the same ratio as the 87.9% vs Bo's
published 155.2).

**1.5B chain** (bonus leg; `bench/results/mega_flagmatrix_c1_{A,D,B,C}_1.5b_5090.json`,
run log `mega_flagmatrix_run_1.5b_5090.log`), same matrix/gates, model
`rwkv7-1.5b-fla` (24 layer, 2048 embd, 32 head):

| leg | config | tok/s | gate |
|---|---|---|---|
| A | anchor | 479.2 | greedy EXACT |
| D | flagship | 493.0 | greedy EXACT |
| B | +MEGA only | 491.6 | greedy EXACT |
| C | +MEGA+WKV | 492.9 | greedy EXACT |

D/A = +2.9% — a larger relative gain than 7.2B's +0.7%, consistent with
fixed per-kernel savings being a bigger fraction of the much smaller 1.5B
per-token step time. Context cited at task hand-off (not independently
re-derived in this session, reported as given): ceiling ~636 tok/s, a prior
"ours" reading of 409 (64% of that ceiling), albatross at 554 (87%). Against
those: D=493.0 is 77.5% of the 636 ceiling and 89.0% of the cited 554
albatross figure; if the 409 reading is the correct prior comparison point
for this same anchor-vs-flagship measurement, D is +20.5% over it. This
provenance is flagged as unverified-this-session rather than asserted as a
clean delta.

## 6b. The clean-window runbook — EXECUTED 2026-07-18, see §6c for results

Ran end-to-end exactly as staged below, zero modifications needed. Total
elapsed from window re-confirmation to full close-out (7.2B matrix + A/D
traces + same-session albatross v3b + 1.5B matrix + racecheck/synccheck
battery): under 20 minutes, no yields. `docker stop rwkv-serve` issued at
the end to release the card back to the (currently-down) sky scheduler.

```bash
# 0) guest checks (MANDATORY): no Pending pods, util < 10%
ssh $TOWER 'sudo k3s kubectl get pods -A --field-selector=status.phase=Pending --no-headers; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
ssh $TOWER docker start rwkv-serve
# 1) 7.2B matrix (A anchor -> D flagship -> B/C attribution), ~25 min:
ssh $TOWER 'docker exec -d rwkv-serve bash -c "cd /data/rwkv/rwkv-sglang/repo-mega && bash bench/mega_flag_matrix.sh /models/rwkv7-7.2b-fla 30070 /data/rwkv/rwkv-sglang/logs/mega/clean 72b > /data/rwkv/rwkv-sglang/logs/mega/matrix72b.log 2>&1"'
#    expected: A ~133-134 (>2% deviation -> investigate: CGMAXBS=32-vs-512 is
#    documented-neutral; the sglang base bump b28bc10->754524d is the first
#    suspect); D = the flagship number vs 168.0 ceiling (=100%) and 155.2 (Bo).
# 2) traces for framing-2 + PDL attribution (A and D boots + /start_profile 48
#    steps + bench/step_span_from_trace.py) — span/step p50, busy, gap, overlap%.
# 3) same-session albatross v3b (~3 min, in rwkv-serve, competitor code stays in
#    scratch): cd /data/rwkv/rwkv-sglang/scratch/albatross_v3b/faster3b_2607
#    && python3 rwkv7_fast_v3b_b1t1_260713.py --model /data/rwkv/rwkv-sglang/models/rwkv7-g1/rwkv7-g1g-7.2b-20260523-ctx8192.pth
#    (g1g vs Bo's g1f: same dims, disclosed; --use_fast_math is THEIR posture)
# 4) 1.5B matrix (bonus): same script, /models/rwkv7-1.5b-fla, tag 15b.
# 5) racecheck re-run (LAST, it serializes the GPU): compute-sanitizer racecheck
#    + synccheck over bench/probes/race_audit_driver.py — docker exec -d + log.
# every leg brackets nvidia-smi snapshots (matrix does this); any Pending pod
# mid-leg -> abort leg, docker stop rwkv-serve, log the yield here.
```

## 7. Honest ledger

- Delivered (2026-07-17 session): the assembled + two-tier-gated PDL chain
  (C++ 8 files + triton 2 kernels), the sm120 Option-B deploy validation, the
  sm120 prefab re-gates (incl. the FIRST on-silicon wkv-CUDA gate), the
  ServerArgs skew fix, the framing-2 trace parser with direct PDL-overlap
  attribution, the staged albatross v3b same-session harness, and the
  dirty-window evidence that PDL is live in production (60.8% overlapped
  transitions).
- Delivered (2026-07-18 clean-window session): the clean 7.2B A/D/B/C
  flagship numbers (both framings), the %-of-168.0 headline (D = 82.0%
  serving / 81.2% kernel-loop), the same-session 155.2 comparison (albatross
  v3b re-measured at 155.75, D = 87.6-87.9% of it either way), the 1.5B bonus
  matrix (A/D/B/C all EXACT), and the F0062 racecheck+synccheck re-run
  (63/63 launches, 0 hazards/errors, both tools) — see §6c and §5.
- NOT delivered / open follow-ups: independent bisection of the A-leg's
  2.2-2.9% deviation to a specific upstream sglang commit (flagged, not
  chased — disproportionate to the delta size on a shared window); a
  larger-than-43-step re-sample of the D-leg's BUSY/step nuance (the
  `wkv_decode_kernel` vs `_wkv_recurrent_kernel` ~9 us/step gap noted in
  §6c is inside plausible sampling noise, not independently confirmed with
  a bigger window); independent verification of the 1.5B "409/554/636"
  comparison-point provenance cited at task hand-off.
- Co-residency/yield events (per the sky-priority rule — every yield logged
  with what it interrupted and whether re-run):
  * 2026-07-17 ~16:55 (session clock): **SKY-YIELD** — managed-job pod
    `tracking-dance2b-26-*-head` went Pending wanting the GPU. We were in the
    WAITING state (no measurement running; servers already down); yielded by
    `docker stop rwkv-serve` (zero footprint) within ~1 min of the Pending
    event. No leg interrupted, nothing to re-run. Resume = `docker start
    rwkv-serve` after the job clears.
  * Whole-session co-resident: genesis sim2sim (motion cxk) at 90-100% util
    until ~15:5x, then hb_compile (model-10-seconds.yaml) at ~98% — all
    timing legs held back; only pipeline shakeouts (disclosed dirty) ran.
  * **2026-07-18 (clean-window session): ZERO yields.** Window opened
    because the SkyPilot API server itself was unreachable (the training
    scheduler could not refill the queue while down) — confirmed at Step 0
    (0 compute processes, 40 MiB/0% util, zero tracking/Pending pods) and
    re-checked before every leg plus a ~20-25s background sentinel for the
    full session duration (7.2B matrix -> A/D traces -> albatross v3b ->
    1.5B matrix -> racecheck/synccheck battery). No tracking pod ever
    appeared; nothing was interrupted. `docker stop rwkv-serve` issued
    proactively at the end (not a yield — routine release once all planned
    work was complete) to leave zero footprint for whenever the sky control
    plane recovers.

## 8. Artifacts + cross-references

- Kernels: `sglang_overlay/.../rwkv7_kernels/cuda/rwkv7_pdl.cuh` + the 7 wired
  .cu files (commit 7863f12); harness `bench/mega_flag_matrix.sh`;
  `bench/verify_m1d.py` (ServerArgs adaptivity); trace parser
  `bench/step_span_from_trace.py`; racecheck driver
  `bench/probes/race_audit_driver.py`.
- Clean-window (2026-07-18) raw results, all under `bench/results/`:
  `mega_flagmatrix_c1_{A,D,B,C}_7.2b_5090.json` and `_1.5b_5090.json` (per-leg
  serving bsz1 sweeps), `mega_flagmatrix_run_{7.2b,1.5b}_5090.log` (full
  matrix run logs incl. every gate result), `mega_framing2_{A,D}_7.2b_5090.txt`
  (kernel-loop parser output + per-kernel tables), `mega_albatross_v3b_g1g_
  7.2b_5090.log` (same-session competitor re-measurement),
  `mega_racecheck_7.2b_5090.log` + `mega_synccheck_7.2b_5090.log`
  (compute-sanitizer closure). Raw `.trace.json.gz` chrome traces (~2 MB
  each) intentionally NOT landed — no unique information beyond the parsed
  aggregate already in the `.txt` files (same convention as
  `w1prime_step_attribution_7.2b_5090.json`'s trace_source note).
- [[F0060]] [[F0061]] (the prefabs + plan this executes) · [[F0062]] (racecheck
  debt, now closed — §5) · [[F0058]] (WKV stage) · ADR-0008 (feasibility + A0
  constants) · `rwkv-competitors/albatross-megakernel-study-2026-07-13.md`
  (structure).
