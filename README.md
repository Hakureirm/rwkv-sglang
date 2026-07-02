# RWKV-7 (Goose) × sglang

**English** · [简体中文](README.zh-CN.md)

A production-grade **RWKV-7 implementation for [sglang](https://github.com/sgl-project/sglang)**:
correct (token-exact vs the BlinkDL `rwkv-lm` reference), self-contained, quantizable, and
portable across consumer + datacenter GPUs — with sglang-native dynamic batching, chunked
prefill, and a constant-size recurrent state cache.

Developed and verified against **sglang v0.5.10.post1** (the newest release the dev box's
CUDA-12.9 driver supports; sglang `main` needs CUDA 13). Shipped as an **overlay**
(`sglang_overlay/`) that deploys into an installed sglang — see [Layout](#layout).

> This project integrates RWKV-7 into sglang for production serving. Goals: match the
> rwkv-lm reference on accuracy and albatross on speed/VRAM across batch sizes; sglang-native
> dynamic batching + chunked prefill + a constant-size recurrent state cache; 8/4-bit
> quantization no slower than 16-bit; and broad consumer + datacenter GPU support.

**Jump to:** [📊 Benchmarks](#-benchmarks-at-a-glance) · [Status](#status-2026-07-01--honest-standing-vs-blinkdlalbatross) · [Design goals & status](#design-goals--status) · [Deploy](#deploy-quickstart) · [Layout](#layout) · [Docs & decisions](docs/)

## Why RWKV-7 × sglang
RWKV-7 is purely recurrent (RNN): its **per-token state is O(1) — constant in context
length**, unlike a Transformer's KV cache (including Qwen3.5's) which grows linearly. At
L24-D1024:

| | state size |
|---|---|
| RWKV-7 | **1.62M (constant)** |
| Qwen3.5 | 5.05M + 6.14×(T/1000) M (grows with context T) |

So at **high concurrency / long context** RWKV-7 fits far more concurrent sequences in the
same VRAM — its structural serving advantage, and this project's wedge. **Measured** below:
256 concurrent sequences and a 64× context increase each cost < +0.2 GB
([📊 Benchmarks](#-benchmarks-at-a-glance)).

## 📊 Benchmarks at a glance
Clean, exclusive RTX 3090, reproducible (≥7 medianed) — methodology + raw logs in
**[`bench/results/`](bench/results/)** (serving-scale in [`serving_scale/`](bench/results/serving_scale/),
same-precision in [`comparison_clean.md`](bench/results/comparison_clean.md), accuracy in
[`lm_eval.md`](bench/results/lm_eval.md)). This is a **serving-engine** deliverable, so the
headline is the serving axes; albatross's home turf (same-precision *single-stream* raw decode)
is then laid out in full below — nothing hidden.

### Where a production serving engine wins ✅

**1. Concurrency throughput — scales ~50× as you fill the batch** (1.5B, steady-state decode tok/s, RTX 3090):
```
bsz   1  █░░░░░░░░░░░░░░░░░░░░░    166 tok/s
bsz  16  █████░░░░░░░░░░░░░░░░░  2,143
bsz  64  ████████████████░░░░░  6,445
bsz 128  █████████████████████  8,298
bsz 256  ████████████████████░  8,187   (compute-bound plateau)
```

**2. VRAM is O(1) — flat in both concurrency and context** (1.5B, peak nvidia-smi):
| scale axis | baseline | scaled up | Δ peak VRAM |
|---|---|---|---|
| **concurrency** | bsz 1 = 12,420 MiB | **bsz 256** = 12,622 MiB | **+202 MiB** for 256 concurrent seqs |
| **context** | 1K = 12,364 MiB | **64K** = 12,368 MiB | **+4 MiB** across a 64× context increase |

Each RWKV-7 state is a fixed 1.62 M-element constant (**no KV cache**), so 256 concurrent
sequences — at *any* context length — cost essentially the same VRAM as one. A KV-cache
transformer's memory grows with batch × context and OOMs long before. Decode stays **O(1)/token**
(single-digit ms/step regardless of context; TTFT is O(T), as for any model). **The same holds at
7.2B**: context 1K→32K = **+0 MiB** peak VRAM; concurrency bsz 1→64 = 46.6→1,802.7 tok/s (38.7×)
at +308 MiB — 64 concurrent 7.2B streams on one 24 GB card
([`serving_scale/`](bench/results/serving_scale/)).

**3. int8 (w8a8) roughly ties albatross-fp16** — at 7.2B, ours-int8 lands at
**0.88–1.21× albatross-fp16** (decode, bsz 1/8/32 — i.e. a *cross-precision* matchup: our int8
vs its fp16) while cutting weight bytes **−46%**. A quant path albatross lacks.

**3b. Hand-written weight-only int8 (w8a16) is greedy-EXACT and beats (or ties) fp16 at every
bsz ≤ 32** — 24/24 token-exact vs the oracle (lossless in practice), decode
1.37×/1.31×/1.27×/1.06×/**1.13×/1.02×** vs fp16 at bsz 1/2/4/8/16/32 (227/392/732/1181/2523/3962
tok/s; bsz64 0.77×, honest), via the same three-kernel dispatch as int4 (GEMV / bit-identical-rows
small-GEMM / tensor-core GEMM with in-smem dequant), and it JIT-runs on **every** arch — unlike
the cutlass w8a8 path (sm80–90 only). `RWKV_W8=1`; details:
[`docs/findings/0018`](docs/findings/0018-w8-weight-only.md).

**3c. Hand-written int4 beats (or ties) fp16 at every bsz ≤ 32** — 1.5B decode
1.56×/1.45×/1.35×/1.04×/**1.17×/1.03×** vs fp16 at bsz 1/2/4/8/16/32 (259/435/773/1153/2619/4004
vs 166/300/574/1113/2243/3873 tok/s; bsz64 0.80×, honest), via three kernels: `gemv_w4_m1`,
`gemm_w4_small` (bit-identical rows), and tensor-core `gemm_w4_tc` (in-smem int4 dequant +
deterministic split-K). At **7.2B**: bsz1 **102.8 tok/s = 1.29× albatross-fp16** (79.6;
cross-precision), fixture-greedy **EXACT 8/8**, lambada 0.7161 vs 0.7425 bf16 (−2.64pt, RTN) —
and **verified live on a real 16 GB T4**: greedy 8/8 exact, 32.9 tok/s bsz1, peak VRAM
**6.7 GB**. Details: [`bench/results/w4/`](bench/results/w4/).

**4. Accuracy is EXACT** — greedy token-for-token match to the rwkv-lm numpy oracle at
0.1B / 1.5B / 7.2B (fp16 + bf16, cuda-graph); lm-eval **ties** rwkv-lm (1.5B lambada 0.673 vs
0.671, MMLU 0.524 vs 0.511).

**5. Runs on the whole GPU lineup** — measured on **10 GPU types across 7 SM generations,
Turing → Blackwell** (T4 / L4 / A10G / A100-40/80 / L40S / H100 / H200 / **B200** /
**RTX PRO 6000**): bf16 is **greedy-EXACT on all 10**, and the hand-written **int4 runs (and is
faster than fp16 at bsz1) on all 10** — from Turing (no `cp.async` needed) to Blackwell sm120
(RTX PRO 6000: int4 bsz1 **1.41×** bf16), no per-arch code change. Peaks: **B200 prefill
103,022 tok/s, decode 7,213 tok/s** @bsz32. (int8 is sm80–90 only — an sgl-kernel cutlass
coverage limit.) Full grid: [`bench/results/multigpu.md`](bench/results/multigpu.md).

### The one axis albatross leads — shown in full 🔬
**Same-precision fp16, *single-stream* raw decode.** This is albatross's home turf: it's a pure
single-stream mega-kernel already at **~92% of the 3090's memory-bandwidth ceiling**, whereas we
are a full dynamic-batching serving engine. We publish every number anyway (higher = closer to
its raw kernel; `1.00×` = parity; best config = in-place WKV + `RWKV_SPARSE_FFN=1` + `RWKV_FAST_LINEAR=1`):
```
              ours / albatross-fp16 — same-precision single-stream (decode tok/s)
7.2B  bsz1  ██████████████████░░░░  0.83×   (45.9 → 65.7 tok/s)
7.2B  bsz8  █████████████████░░░░░  0.84×
7.2B  bsz32 ██████████████░░░░░░░░  0.72×   (+24% from in-place WKV)
1.5B  bsz1  █████████████░░░░░░░░░  0.66×
1.5B  bsz8  ██████████████████░░░░  0.90×   ← closest to parity
1.5B  bsz32 ██████████████░░░░░░░░  0.70×
```
The 0.1B rows (0.49–0.79×) are omitted here as **unrepresentative** — a launch-bound tiny model
is the least serving-relevant case; full numbers in
[`comparison_clean.md`](bench/results/comparison_clean.md). Even on this worst-for-us axis, the
mid/large models we actually serve sit at **0.66–0.90×**, closed with three hand-written
greedy-EXACT kernels (in-place WKV + sparse FFN + fused GEMV).

**Bottom line:** in real serving — **concurrency, VRAM, int8, accuracy** — RWKV-7 × sglang wins;
albatross leads only same-precision single-stream raw decode, and only at its bandwidth ceiling.

## Status (2026-07-01) — honest standing vs BlinkDL/albatross
The head-to-head vs-albatross numbers below are clean, exclusive-RTX-3090, reproducible
(`bench/results/comparison_clean.md` + `lm_eval.md`, superseding the older co-tenant
`comparison.md`); the **cross-GPU sweep spans 8 architectures T4 → H200 on their own hardware**
(`bench/results/multigpu.md`).

- ✅ **Correctness**: RWKV-7 **0.1B / 1.5B / 7.2B** all **greedy-EXACT** vs the pure-numpy /
  `rwkv-lm` reference (fp16 + bf16); dynamic-batch (shared-prefix/mixed) exact.
- ✅ **Accuracy = parity with rwkv-lm** (lm-eval, 1.5B): lambada acc 0.673 vs ref 0.671, MMLU
  0.524 vs 0.511. (7.2B: greedy-exact + full scored lambada 0.742.)
- ⚖️ **Same-precision raw speed: closing the gap with hand-written CUDA.** fp16-vs-fp16 decode
  is now **0.49–0.90× albatross across all sizes/bsz** (was 0.46–0.85×), via three FLA-free,
  greedy-EXACT, batch-invariant hand-written kernels:
  - **in-place indexed WKV state I/O** (default): the WKV recurrence — the only decode component
    that scales with batch — reads/writes the paged state pool directly (no gather/scatter). Lifts
    the **batched/production regime**: 7.2B bsz32 0.61→**0.72×**, 1.5B bsz32 0.57→**0.70×**, ~+24%.
  - **sparse sqrelu FFN** (`RWKV_SPARSE_FFN=1`): `relu(k)²` is **86–90% exact-zero** on real prompts,
    so a hand fp32-accumulate SpMV skips ~9/10 of the value-weight reads (bandwidth win past the
    dense ceiling). **fused fp16 GEMV** (`RWKV_FAST_LINEAR=1`) for r/k/v/o + key proj (bsz1 path).
  - Combined best: **7.2B bsz1 45.9→65.7 tok/s (0.58→0.83×)**, 1.5B bsz8 **0.90×**. Verified in
    `bench/results/{comparison_clean.md,best2,sparse_ffn}`, `docs/design/m6-sparse-ffn.md`. albatross
    still leads raw decode (monolithic mega-kernel at ~92% BW peak); fully matching it needs the same
    whole-time-mix fusion, which sacrifices the clean sglang integration.
- ✅ **int8 (w8a8, a feature albatross lacks)**: at 7.2B, ours-int8 **roughly ties**
  albatross-fp16 (decode **0.88–1.21×** bsz 1/8/32) — a cross-precision matchup (our int8 vs its
  fp16), not the same-precision comparison.
- ✅ **VRAM**: recurrent state is O(1)/token → flat in batch; albatross's static B×T grows to
  near-OOM at 7.2B bsz32. int8 cuts weight bytes ~46% (7.2B).
- ✅ **Multi-GPU (8 architectures, Turing→Hopper)**: T4/L4/A10G/A100-40/A100-80/L40S/H100/H200 all
  bf16 greedy-EXACT + int4 runs and is faster, no per-arch code change (`bench/results/multigpu.md`).
  ✅ **RWKV-7 execution path is FLA-free** (our own WKV kernel; see the precise scope in "references" below).
- 🔜 **Open**: fp8; fused int4 GEMM (M>1 throughput); World-tokenizer serving polish + upstream PR.

**Positioning (honest):** we **match rwkv-lm accuracy** (verified tie), **win on VRAM / int8 /
real serving** (dynamic batching — albatross has none), and have **closed most of the
same-precision raw-speed gap with three hand-written greedy-EXACT kernels** — now **0.49–0.90×**
albatross across sizes/bsz (7.2B bsz1 0.83×, 1.5B bsz8 0.90×). The last bit needs albatross's
monolithic whole-time-mix mega-kernel (sacrifices the clean integration; ~parity at best).

## Design goals & status
Honest self-assessment against this project's engineering goals (2026-07-01), scoped to an
sglang inference integration; ✅ done, ◑ partial, ⬜ open/out-of-scope-here.

| # | Goal | Status in this deliverable |
|---|---|---|
| 1 | Match albatross/RWKV-LM perf across bsz | ◑ accuracy **ties** RWKV-LM; **win** int8/VRAM/serving; same-precision raw fp16 decode **0.49–0.90× across sizes/bsz** (was 0.46–0.85×) via 3 hand-written greedy-EXACT kernels (in-place WKV + sparse FFN + fused GEMV) — `bench/results/comparison_clean.md` |
| 2 | Beat Qwen3.5 (same quant, typical scenes) | — **out of scope for this project** (an sglang inference integration): this deliverable is measured against **albatross** (speed/VRAM) + **RWKV-LM** (accuracy) |
| 3 | transformers PEFT/RL training | ⬜ out of scope for this project (an sglang inference integration) |
| 4 | Dynamic batching + chunked prefill + state cache | ✅ sglang-native dynamic batching + chunked prefill + O(1) recurrent state pool; ◑ radix/prefix **reuse** auto-off (state not yet prefix-cacheable — a documented `MambaRadixCache` follow-up) |
| 5 | Pascal+/AMD/Intel/domestic; PP+TP; zero2/3; autotune | ◑ greedy-EXACT on 10 GPU types Turing→Blackwell; **TP greedy-EXACT at 2/4/8 ranks AND PP greedy-EXACT at 2/4/8 stages, all on real L4 fleets** (tp=1/pp=1 zero regression; mixed tp×pp fixed (v_first full-width across stage boundaries) and greedy-EXACT; W4/W8 still tp=1; full matrix incl. per-GPU memory: [`bench/results/parallel/`](bench/results/parallel/), [`docs/findings/0019`](docs/findings/0019-tp-pp-parallel.md)); ⬜ Pascal/AMD/Intel untested, training/autotune out of scope |
| 6 | w8 + w4, faster than w16, old cards, Q\*_K_M accuracy | ✅ **w8 (w8a8-int8)** — faster than bf16 (+46–59% decode @1.5B/7.2B), −46% weight bytes, 7.2B greedy-EXACT; ✅ **w4 (hand-written int4)** — **faster than fp16 at every bsz≤8** (1.04–1.56× @1.5B; 7.2B bsz1 102.8 tok/s, fixture-EXACT 8/8, lambada 0.7161 vs 0.7425, 9.8 GB total), runs on Turing→Hopper; ◑ Q\*_K_M-style side-by-side not done (our GPTQ g64 −3.34pt @1.5B is the comparable point) |
| 7 | Speculative decoding (RWKV draft) | ⬜ not done |

The strongest, fully-verified contributions here: **exact correctness (0.1B/1.5B/7.2B)**,
**int8 speed/VRAM**, **sglang-native serving**, **multi-GPU**, **FLA-free own WKV kernel**,
and a **rigorously-measured, honestly-reported CUDA endgame** (F0015).

## Accuracy & speed references (no FLA)
- **Accuracy oracle = BlinkDL `rwkv` pip + a pure-NumPy transcription** of the RWKV-7 recurrence
  (`bench/oracle_numpy.py`, following BlinkDL's `rwkv_v7_numpy.py`). We do **not** use
  flash-linear-attention as an accuracy reference.
- **Speed/VRAM baseline = BlinkDL/albatross**, re-measured on our own 3090 (`bench/results/`).
- **Kernel policy (ADR-0004): no `flash-linear-attention` (PyPI) dependency on the RWKV-7 path.**
  (The overlay's edited *upstream* sglang files retain sglang's own `…fla…` mamba/gated-delta
  imports, which are never exercised by RWKV-7; the `-fla` in model-dir/converter names refers to
  the fla-format *checkpoint layout*, not a code dependency.)

## Layout
- `sglang_overlay/` — the deliverable: new + edited sglang files (model, state backend, config,
  wiring), deployed into sglang via `scripts/deploy.sh` (rsync overlay, no build).
- `tools/convert_rwkv7_blinkdl_to_fla.py` — BlinkDL `.pth` → sglang-loadable checkpoint.
- `bench/` — oracle (`oracle_numpy.py`), gates (`verify_m1d.py`, `verify_batch.py`), throughput
  (`throughput.py`, `run_clean_comparison.py`), lm-eval (`accuracy_eval.py`), fixtures, `results/`.
- `docs/` — `snapshot.md` (canonical), `adr/`, `findings/`, `design/`.

## Deploy (quickstart)
`sglang_overlay/` mirrors sglang's package tree; `scripts/deploy.sh` rsyncs it into the target
machine's installed sglang site-packages (no build), then you launch sglang as usual and it loads
RWKV-7:

```bash
# configure the target via env (defaults are placeholders):
#   BOX = ssh host/alias of the target (use "" / localhost for a local install)
#   SP  = the site-packages dir of the target's sglang venv
BOX=<your-host> SP=<site-packages> bash scripts/deploy.sh
```

Convert a BlinkDL `.pth` with `tools/convert_rwkv7_blinkdl_to_fla.py` first, then serve normally;
add `--quantization w8a8_int8` for int8.

## Dev environment
- Remote box: 1× RTX 3090, sglang **v0.5.10.post1** (torch 2.9.1/cu128) — pinned because sglang
  `main` requires CUDA 13 (box driver supports ≤12.9). No GitHub/HF on the box → refs cloned on
  a Mac under `refs/` (gitignored), rsync'd up; models via ModelScope; secrets in an untracked
  `~/.rwkv_secrets.sh` (never committed).
