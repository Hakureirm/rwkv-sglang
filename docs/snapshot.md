---
doc_kind: snapshot
project: rwkv-sglang
title: "RWKV-7 × sglang adaptation — canonical state"
date: 2026-07-02
status: active
last_verified_commit: 13eef84
schema_invariant: |
  - Every ADR referenced anywhere MUST appear once in §"ADR roster".
  - Every finding referenced anywhere MUST appear once in §"Findings ledger".
  - §"Environment" is the single source of truth for the dev box.
  - The scope decision (RWKV-7 on sglang) appears EXACTLY ONCE (under §"Scope").
  - When a section is rewritten, delete superseded prose in the SAME edit (anti-F1).
---

# rwkv-sglang — Snapshot (canonical state)

> Canonical state document (ADSD Part 3). README/runbooks are projections.
> (The working dir was historically `rwkv-vllm`; this project targets **sglang** —
> ADR-0001.)

## Scope

**Scope: RWKV-7 on sglang** (ADR-0001, accepted 2026-06-30, on a verified
latest-upstream re-analysis — see [[F0004]]).

**Wedge**: *the first production-grade RWKV-7 serving in sglang — dynamic
batching + chunked prefill + recurrent state cache + 8/4-bit quant, on consumer +
datacenter GPUs, matching `rwkv-lm` accuracy.* Goals: match rwkv-lm accuracy +
albatross speed/VRAM across batch sizes; sglang-native dynamic batching + chunked
prefill + constant-size state cache; 8/4-bit quant no slower than 16-bit; broad GPU support.
**Delivered standing (honest, F0014/F0015):** accuracy TIES rwkv-lm (1.5B lambada 0.6728 vs
ref 0.6711, MMLU 0.5235 vs 0.5110; 7.2B greedy-EXACT + full lambada 0.7425 — ref number still
to be measured); we WIN int8 speed / VRAM / real serving; **albatross wins same-precision raw-kernel latency**
(a static-batch, no-serving micro-bench) and we do NOT claim to match it — the
original "match albatross raw speed" aspiration proved to require whole-time-mix
mega-kernel fusion, deliberately declined for elegance.

## Current phase

**Correctness DONE + verified (exact 0.1B/1.5B/7.2B, dynamic batching safe-by-default, cuda-graph;
RWKV-7 path 100% FLA-free); accuracy = PARITY with rwkv-lm (lm-eval [[F0014]]: 1.5B lambada
0.6728 vs 0.6711, MMLU 0.5235 vs 0.5110).** Honest same-precision
(fp16) decode standing: **default config 0.46-0.85× albatross**; **with the three opt-in hand-written
kernels (in-place WKV + sparse FFN + fused GEMV) 0.49-0.90×** (7.2B bsz1 0.83×, 1.5B bsz8 0.90×;
`bench/results/comparison_clean.md`). albatross still leads raw decode (monolithic mega-kernel ~92% BW);
we WIN on VRAM, int8 (7.2B ≥ albatross-fp16 cross-precision), and real serving (albatross has none).
**Serving wedge now MEASURED [[F0016]]**: decode throughput scales ~50× with concurrency
(166→8298 tok/s bsz 1→128) at flat VRAM (256 concurrent seqs = +202 MiB; 64× context = +4 MiB) —
`bench/results/serving_scale/`. README reframed to lead with the won axes (concurrency/VRAM/int8/
accuracy), same-precision single-stream chart demoted to an honest "one axis albatross leads" section. ALL
milestones done: Phase-0 + M0 + M1 + M2 + M3(comparison+lm-eval) + M3b(de-FLA) + M4(int8) + M5(multi-GPU)
+ M-rigor + M6(3 CUDA kernels) + **M7(int4: 3 hand-written kernels + GPTQ, [[F0017]])** + ShareGPT
serving bench + the **10-GPU all-card sweep, Turing→Blackwell** (`bench/results/multigpu.md`) +
7.2B full lambada 0.742 + 7.2B serving-scale. **v0.1.0 tagged + released.** Remaining = int4 bsz64+
tiling, 7.2B GPTQ (streamed calibration), fp8, TP/PP, upstream PR.
- ✅ Recon/arch/baselines/re-analysis → sglang chosen. [[F0001]][[F0002]][[F0003]][[F0004]]
- ✅ ADR-0001 (scope/wedge), ADR-0002 (integration), ADR-0003 (M1 scope & slicing).
- ✅ M1 plan (`docs/design/m1-implementation-plan.md`) + correctness gate `bench/oracle_numpy.py`.
- ✅ **M0 DONE**: `rwkv-sgl` (sglang 0.5.10.post1, torch2.9.1+cu128, CUDA True) +
  `rwkv-ref` on `gpu-box`. RWKV-7 0.1B from ModelScope; numpy oracle + fixture.
- ✅ **M1 DONE** — RWKV-7 0.1B runs in sglang, **EXACT greedy-match vs oracle**
  (lead-verified `bench/verify_m1d.py`; HEAD 700e554). [[F0005]]
  - M1a kernels (gate pass) · M1b converter (399 tensors) · M1c boot · M1d exact match.
  - Deliverable: `sglang_overlay/` (model+backend+config+wiring) + converter; deploy
    via `scripts/deploy.sh`. scale=1.0, 2 conv token-shift states, fp32 state.
- ✅ **M2** scale + perf:
  - ✅ **M2-baseline** [[F0006]]: bf16 0.1B + **1.5B EXACT**, fits 3090; throughput profiled.
  - ✅ **M2b cuda-graph** [[F0008]]: decode **7.5-21× faster than eager**, still EXACT (lead-verified);
    no code change (inherited capture hooks); launch w/o `--disable-cuda-graph`. (vs-albatross ratios:
    see the clean fp16 table in [[F0014]], NOT the old co-tenant numbers.)
  - ✅ **radix-cache auto-off** [[F0009]]: `server_args.py` registers RWKV7 →
    `_handle_mamba_radix_cache(support_mamba_cache=False)` (mirrors KimiLinear); dynamic-batch
    correctness EXACT (identical/shared-prefix/mixed), lead-verified. Safe-by-default.
  - ✅ **7.2B EXACT 8/8** (32L/4096d, ~17.5GB, fits 3090); 0.1B+1.5B+7.2B all exact.
    (⬜ proper state-aware MambaRadixCache for prefix reuse is a later optimization; 2.9B n/a on box.)
- ✅ **M3 clean comparison + lm-eval** [[F0014]] (`bench/results/{comparison_clean,lm_eval}.md`, exclusive
  GPU, ≥7 medianed): same-precision **fp16 decode ours/alb 0.46-0.85× (7.2B 0.58-0.77×) — albatross
  faster**; accuracy PARITY (lambada 0.673 vs 0.671, MMLU 0.524 vs 0.511; albatross-fp16 also greedy-exact
  → no accuracy win either way). VRAM win = flat-in-batch (materializes at large bsz) + int8 weight floor.
- ✅ **M3b de-FLA DONE** [[F0010]] (ADR-0004 satisfied): decode + prefill both run our own
  `rwkv7_kernels/wkv_recurrent.py`; our vendored FLA files deleted; **the RWKV-7 execution path is
  100% FLA-free** (lead-verified); greedy still EXACT; **zero speed cost**. Chose self-written over
  albatross-vendor (cleanest IP). NB (precise, anti-grep-gotcha): the overlay's edited *upstream*
  sglang files still carry two top-level `sglang…fla…` imports (`server_args.py`,
  `attention_registry.py`) — that is sglang's OWN mamba/gated-delta code, **never on the RWKV path**;
  no `flash-linear-attention` PyPI dependency exists. (The `-fla` in model dir / converter names means
  the *fla-format checkpoint layout*, not a code dep.)
- ✅ **M4 w8a8-int8** [[F0011]]: int8 vs bf16 decode **+46-59% @1.5B/7.2B** (clean; −10% @0.1B bsz1,
  launch-bound) + weight bytes −41-46% (7.2B −46%, safetensors-derived in `comparison_clean.md`), 7.2B EXACT.
- ✅ **elementwise fusion** [[F0013]]: 78→54 kernels/layer, +5-11% decode (EXACT, lead-verified).
  **KEY: bit-exact-greedy caps fusion at the elementwise subset** (matmul fusion perturbs values →
  breaks strict batch gate → needs the lm-eval-parity gate).
- ✅ **M4 int8 (w8a8)** [[F0011]]: 7.2B ours-int8 matches/beats albatross-fp16 (decode 0.90/1.21/0.88× —
  CROSS-precision bonus, not same-precision); weight bytes −46%; 1.5B accuracy drift small (lm-eval).
- ✅ **M5 multi-GPU** [[F0012]]: greedy-EXACT T4/L4/A10G/A100/H100, no per-arch change.
- ✅ **M-rigor** [[F0014]] + **adversarial audit #1** (5-dim): clean numbers + lm-eval done; audit found
  NO code correctness bug; fixed the secret-leak + doc-sediment it flagged.
- ✅ **M6 CUDA endgame** — two hand-written, greedy-EXACT, batch-invariant, FLA-free, opt-in kernels:
  - phase-1 [[F0015]]: fused fp16 GEMV (`rwkv7_fast.cu`, `RWKV_FAST_LINEAR=1`) for r/k/v/o+key proj;
    +5-9% bsz1 (cuda-graph amortizes the eager 1.1-1.6×).
  - phase-2 (`docs/design/m6-sparse-ffn.md`): **sparse sqrelu FFN value-proj** (`rwkv7_sparse_cmix.cu`,
    `RWKV_SPARSE_FFN=1`) — `relu(k)²` is **86-90% zero** (measured) → hand fp32-accum SpMV skips ~9/10
    value-weight reads (TRUE bandwidth win past the dense ceiling). **+28.8% bsz1 @7.2B**; verify_m1d
    + verify_batch PASS (0.1B/1.5B/7.2B).
  - **Combined: 7.2B bsz1 45.9→64.3 tok/s (+40%) = 0.81× albatross-fp16** (was 0.58×); 1.5B →0.63×.
- ✅ **M6 phase-3 — in-place indexed WKV state I/O** (`docs/design/m6-sparse-ffn.md`): the WKV
  recurrence (only batch-scaling decode component, profiled) reads/writes the paged state pool
  directly (no gather/scatter); greedy-EXACT + verify_batch (pad-slot guard `0<=cidx<n_slots`).
  Lifts batched decode: **7.2B bsz32 0.61→0.72×, 1.5B bsz32 0.57→0.70× albatross (~+24%)**.
  **Combined standing now 0.49-0.90× across all sizes/bsz** (7.2B bsz1 0.83×, 1.5B bsz8 0.90×;
  was 0.46-0.85×) — `bench/results/{comparison_clean.md,best2}`. (Qwen3.5 comparison out of scope.)
- ✅ **M7 int4 (weight-only w4)** [[F0017]]: hand-written `rwkv7_w4.cu` — `gemv_w4_m1` (M=1) +
  **`gemm_w4_small` (2≤M≤8, one weight read feeds all M rows; every row BIT-identical to the M=1
  kernel, torch.equal-verified)** + `dequant_w4` for M>8; GROUP=64 sym, fp32 accum, cuda-graph
  safe, FakeTensor regs. Offline RTN (`bench/quant_w4.py`) **and GPTQ** (`bench/{calib_run,
  gptq_w4}.py`, RWKV_CALIB Hessian hook, wikitext). **1.5B: faster than (or ties) fp16 at EVERY
  bsz≤32** (1.56×/1.45×/1.35×/1.04×/1.17×/**1.03×** at bsz 1/2/4/8/16/32; bsz64 0.77× — M=64
  long-K ffn shapes, tiling work remains) via 3-kernel dispatch: gemv_w4_m1 + gemm_w4_small
  (2≤M≤8, rows bit-identical to M=1) + **gemm_w4_tc (8<M≤64: wmma tensor cores, in-smem int4
  dequant, one weight-dequant per block, deterministic split-K — no atomics)**;
  checkpoint 2.9→1.2 GB, serve VRAM −950 MiB; lambada **GPTQ −3.34pt** (RTN −4.95; int8 −2.15).
  **7.2B (RTN)**: bsz1 **102.8 tok/s = 1.29× albatross-fp16 (79.6, cross-precision), 1.56× ours-fp16
  best (65.7)**; fixture greedy **EXACT 8/8**; lambada **0.7161 vs 0.7425 bf16 (−2.64pt)**;
  **9.8 GB total serve VRAM** (checkpoint 4.8 GB, 3.0×) → fits a 16 GB card. 7.2B GPTQ deferred
  (value-proj Hessian 16384²=1GB/layer ×32 — needs streamed accumulation). Default path
  regression-clean (24/24). Opt-in `RWKV_W4=1`.
- ✅ **ShareGPT serving bench** (`bench/results/serving_scale/`, standard `bench_serving`, 1.5B,
  500 reqs): peak 1275 out-tok/s / 3361 total-tok/s; @16 req/s median TTFT **273 ms**.
- ✅ **10-GPU all-card sweep (Turing→Blackwell incl. B200 + RTX PRO 6000; int8 = sm80–90 only, sgl-kernel limit)** (`bench/results/multigpu.md` + `allcards.json`): bf16 greedy-EXACT
  on ALL 10 (T4/L4/A10G/A100-40/-80/L40S/H100/H200/B200/RTX-PRO-6000); **int4 runs + bsz1-faster
  on all 10, Turing→Blackwell** (2.04× L4 … 1.41× RTX PRO 6000 … 1.09× H200); int8 sm80–90 only
  (T4: cutlass Error Internal; Blackwell: explicit NotImplementedError — upstream sgl-kernel).
  7.2B full lambada **0.742** (`out/lmeval_72b-lambada`). Chunked-prefill gate 48/48 exact
  (`bench/verify_chunked_prefill.py`).
- ✅ **published**: single clean commit `9a24433` → github.com/Hakureirm/rwkv-sglang (PUBLIC);
  docs carry a human track (`docs/human/`, 中文+mermaid) + this dense agent track.
- ✅ (2026-07-02, post-audit fixes): small-M int4 kernel `gemm_w4_small` (bsz≤8 all faster than
  fp16) · 7.2B int4 measured (102.8 tok/s bsz1, EXACT 8/8, lambada 0.7161, 9.8 GB — and verified
  live on a real 16 GB T4) · T4-int8 diagnosed (cutlass int8 needs sm80+, "Error Internal" @sm75)
  · T4 fp16 baseline (24/24 EXACT; fp16≈bf16 speed on T4) · CONTRIBUTING.md.
- 🔄 **remaining**: v0.1.0 tag + release notes (drafted, `scratchpad/release_notes_v0.1.0.md`) ·
  int4 fused tensor-core GEMM for M>8 · per-arch small-M cutover (T4 crossover earlier than 3090) ·
  fp8 · (stretch) time-mix mega-fusion · TP/PP · upstream PR.

> Dev model: `sglang_overlay/` (new+edited files) → `scripts/deploy.sh` rsyncs into the
> box's wheel sglang site-packages (no editable build). Head config = 12×64 (from r_k).

> ✅ **BENCHMARK RIGOR (DONE)**: the old shared-GPU numbers (F0006/F0008/F0009 +
> `bench/results/{comparison,albatross_3090,profile}.md`, measured with a ~1.3GB isaaclab
> co-tenant) are **superseded**. All headline ours-vs-albatross numbers were RE-RUN on the
> **exclusive** 3090 in one controlled session (both engines, same GPU/prompts/bsz, ≥7 medianed,
> one process at a time) + **lm-eval vs rwkv-lm** → `bench/results/{comparison_clean,lm_eval}.md`
> ([[F0014]]). The M6 fast-path numbers likewise ([[F0015]], `bench/results/fast_linear/`). Cite
> only the clean files. See memory `feedback-benchmark-rigor`.

> Env constraint discovered (M0): sglang **main pins cu13/torch2.11** → won't run on the
> box driver 575 (max CUDA 12.9). Pinned to **v0.5.10.post1** (torch 2.9.1/cu128, has full
> Mamba/GDN substrate). uv installs run detached (nohup) to survive ssh drops + `UV_HTTP_TIMEOUT=600`.

## ADR roster

| ADR | Title | Status | Date |
|---|---|---|---|
| 0001 | Scope & rationale — RWKV-7 on sglang | accepted | 2026-06-30 |
| 0002 | sglang integration approach for RWKV-7 | accepted | 2026-06-30 |
| 0003 | M1 scope & slicing into gated increments | accepted | 2026-06-30 |
| 0004 | No FLA dependency in deliverable (kernel endgame = albatross/own) | accepted | 2026-06-30 |

## Findings ledger

| ID | Title | Severity | Status |
|---|---|---|---|
| F0001 | Dev box & environment recon | info | open |
| F0002 | RWKV-7 architecture & serving-framework mapping | info | open |
| F0003 | Parity baselines, oracle & acceptance test | info | open |
| F0004 | Verified latest-upstream re-analysis (vLLM/sglang/HF) | info | open |
| F0005 | M1 complete — RWKV-7 0.1B exact greedy-match in sglang | info | closed_by_M1 |
| F0006 | M2-baseline — bf16+1.5B exact; throughput; decode eager-bound | info | open |
| F0007 | Albatross 3090 baseline + gap (decode ~30-57× eager) + vendor-kernel path | info | open |
| F0008 | M2b cuda-graph — decode 7.5-21× faster, exact, gap vs albatross ~2-3× | info | open |
| F0009 | 7.2B exact + dynamic-batch correctness (radix auto-off) + comparison (gap shrinks→~1.2-1.8× @7.2B) | info | open |
| F0010 | M3b de-FLA complete — own WKV kernel (decode+prefill), 100% FLA-free, zero speed cost | info | closed_by_M3b |
| F0011 | M4 w8a8-int8 — decode FASTER than bf16 (+15-53%) + weight bytes -41-46% (safetensors), 7.2B EXACT | info | open |
| F0012 | Multi-GPU coverage — greedy-EXACT on 10 GPU types / 7 SM gens (Turing→Blackwell incl. B200 + RTX PRO 6000); int4 on all 10; int8 = sm80–90 only (sgl-kernel) | info | open |
| F0013 | Fusion +5-11% decode (EXACT); bit-exact caps fusion at the elementwise subset | info | open |
| F0014 | Clean same-precision standing — raw speed loses, accuracy TIES, VRAM/int8/serving win; CUDA endgame chosen | info | open |
| F0015 | CUDA endgame result — fused fp16 GEMV greedy-EXACT, +5-9% bsz1 decode @1.5B/7.2B; cuda-graph amortizes the eager win; mega-kernel to match albatross DECLINED | info | open |
| F0016 | Serving-scale measured — ~50× concurrency throughput at flat VRAM; context-invariant memory (O(1)-state wedge) | info | open |
| F0017 | Hand-written weight-only int4 — faster than fp16 at every bsz≤8 (1.04–1.56×); 7.2B: 102.8 tok/s bsz1 (1.29× albatross-fp16), fixture-EXACT, lambada −2.64pt, 9.8GB total; GPTQ 1.5B −3.34pt; M>8 dequant ~0.5× (fused GEMM = endgame) | info | open |

## Environment (single source of truth)

See [[F0001]]. Dev box (ssh alias, key auth) = Ubuntu 22.04, 1× RTX 3090, 40 cores,
31 GB RAM. Driver = **CUDA 12.9** → the deliverable targets **sglang v0.5.10.post1**
(the newest release that driver runs; `main` needs CUDA 13). Network: ✅ pypi/aliyun/
**modelscope**; ❌ github/HF → clone-on-Mac→rsync; models via ModelScope (token in an
untracked `~/.rwkv_secrets.sh`, never committed). References on Mac under `refs/`
(gitignored): `sglang`, `RWKV-LM`, `Albatross`, the two RWKV-7 vLLM PR diffs.

## sglang integration map (confirmed paths, HEAD f920a37)

- Template model: `python/sglang/srt/models/qwen3_next.py` (Gated DeltaNet).
- State backends: `python/sglang/srt/layers/attention/linear/{gdn_backend,
  kda_backend,lightning_backend}.py` → add `rwkv7_backend.py`.
- Vendored fla ops: `python/sglang/srt/layers/attention/fla/` (gated-delta subset;
  **no rwkv7** → port `chunk`/`fused_recurrent` + deps).
- State cache: `python/sglang/srt/mem_cache/mamba_radix_cache.py` (+
  `mamba_checkpoint_pool.py`).
- Token-shift: `python/sglang/srt/layers/attention/mamba/causal_conv1d.py`.

## Next actions (all core milestones done — remaining = ship + optional polish)

1. **PUBLISH**: push the self-contained overlay deliverable (targets the verified
   **v0.5.10.post1**) to `github.com/Hakureirm/rwkv-sglang` (fresh repo I create; PRIVATE
   first, then flipped public). Pre-public adversarial audit done +
   fixes applied. ⚠️ user must ROTATE the ModelScope token (exposed before the history purge).
2. (optional) **fp8** (needs an fp8 weight-scale converter) — extend the int8 path.
3. (optional) **7.2B full scored lm-eval** (currently greedy-EXACT on an 8-token fixture;
   1.5B has the full scored lambada/MMLU parity run).
4. (optional) **serving polish**: World-tokenizer + OpenAI-API surface; extreme-context
   prefill (T≥4096); state-aware MambaRadixCache for RWKV prefix reuse (re-enable radix safely).
5. (stretch) **upstream PR** to sgl-project — needs a re-port onto sglang `main` + a
   CUDA-13 box (or a cloud GPU) to runtime-verify against main (box driver = CUDA 12.9).
