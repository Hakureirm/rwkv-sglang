# Contributions — RWKV-7 on SGLang

This repo delivers the production-grade **RWKV-7 serving adaptation for SGLang**
(dynamic batching + chunked prefill + O(1) recurrent-state pool, greedy
token-exact vs the BlinkDL rwkv-lm numpy reference), with **zero FLA
dependency** and a family of hand-written CUDA kernels (WKV recurrence /
weight-only int4 / int8 / fused-LoRA), measured on 10 GPU types across 7 SM
generations. Entry points: [README.md](README.md) ·
[docs/snapshot.md](docs/snapshot.md) · [docs/](docs/) (ADRs + findings ledger).

## §1 Requirement scorecard (status honest as of 2026-07-07)

Numbered to match BlinkDL's own reposted bounty text exactly (his 2026-07-06 Zhihu repost) —
an earlier version of this table used a different, informal numbering under which requirement
#2 (beat Qwen3.5) had no row at all. Fixed here; see `ROADMAP.md` for the forward-looking view.

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Match new-Albatross/RWKV-LM accuracy, speed, VRAM (across bsz), all 4 frameworks + transformers training | ✅ 1.5B full / ◑ 7.2B, **sglang only** (see ADR-0001 for scope) | [`bench/results/comparison_clean.md`](bench/results/comparison_clean.md), [`lm_eval.md`](bench/results/lm_eval.md) (lambada 0.673 vs ref 0.671, MMLU 0.524 vs 0.511); vs-Albatross ratio tracks memory bandwidth (L4 0.90× down to B200 0.51×, docs/BENCHMARKS.md §7) — **ongoing kernel-fusion work is closing this, not yet done** (H100 0.59×→0.646× post-F0051; long-term, task #5, Albatross itself isn't at its own ceiling either per BlinkDL directly) |
| 2 | Beat Qwen3.5 at matched size+quant, across cloud/desktop/mobile/embedded | ✅ cloud+desktop tiers (speed) / ◑ Apple Silicon (speed done, accuracy partial) / — mobile+embedded (no hardware) | **RWKV-7 wins same-precision peak concurrency at both size tiers on both owned GPUs** (5090: +21.9%/+43.7%; 3090: +11.7%/≥+27.0%) — the decisive full-spectrum metric per this project's own doctrine; single-stream bf16 currently favors Qwen3.5 (RWKV's hand kernels are fp16-only), fp16-optimized RWKV reclaims single-stream too. Compression rate: **RWKV-7 1.5B beats Qwen3.5-2B on two independent platforms** (cloud 0.6085 vs 0.6729; MLX 0.5926 vs 0.6719). Qwen3.5-2B correctness independently oracle-gated against BlinkDL's own numpy reference (F0050). MATH500 avg@64 in progress. Full chapter: `docs/BENCHMARKS.md` §13, findings F0044–F0050 |
| 3 | HF PEFT/RL trainability | n/a | training-track scope; this repo is the inference/serving adaptation (README design-goals table) |
| 4 | Dynamic batching + chunked prefill + state cache | ✅ | batching/chunked-prefill greedy-EXACT (`bench/verify_batch.py`, `bench/verify_chunked_prefill.py`); **state prefix cache via MambaRadixCache now ENABLED** (`server_args` support_mamba_cache=True + `scripts/deploy.sh` is_hybrid_ssm patch) — `verify_batch.py --radix-on` greedy EXACT at 0.1B+1.5B (shared-prefix 5/5, mixed 6/6), where the plain token radix corrupted ([`radix_correctness.md`](bench/results/radix_correctness.md)/F0008); **~98% cache hit rate on a realistic high-reuse load (2048-tok shared prefix), TTFT 784→200ms** (the earlier 30% was a cold-start-worst-case test artifact); the only RWKV serving stack with a state prefix cache at all — F0022 |
| 5 | Pascal+/AMD/domestic cards; PP+TP inference; zero2/3 training; autotune | ◑ | TP 2/4/8 + PP 2/4/8 + mixed, all greedy-EXACT (F0019); **verified on sglang main under cuda-graph ON (2×L4): TP=2 and PP=2 both greedy 24/24 == single-GPU + deterministic; tp1 peak 2,582.6 → tp2 3,026.2 (1.17×), pp2 2,288.8 (0.89×) — F0036**. Autotune: **done**, 11-card matrix (F0023/25/27). Fixed a PP+cuda-graph crash on main (folded into PR #30115). Pascal routing guard `42fd6fa`; **real Pascal/AMD hardware validation not yet done (task #9, named gap)**; domestic cards only via the yuueang Ascend NPU community channel; zero2/3 training out of scope (inference-only repo) |
| 6 | w8/w4 quant: VRAM ↓, ≥w16 speed on common cards (old cards too), near-Q4_K_M accuracy | ✅ w8 lossless / ◑ w4 real progress, not at the bar yet | [`bench/results/w4/`](bench/results/w4/), F0017/F0018/F0034/F0035/**F0043**; w4 ≥fp16 at every bsz≤32 (3090, 1.5B); w8 uncheatable-lossless (24/24 greedy). **Asymmetric GPTQ (F0043, zero kernel change) closes 27-35% of the fp16 gap depending on metric, but 1.5B int4 MATH500 avg@64 is still a large reasoning-chain collapse (0.40→0.22) even improved** — Q4_K_M-level accuracy is not yet reached; 7.2B's version of this question is being measured now, decides whether a full K-quant rewrite is worth building. sm120: our own s8-wmma w8a8 is the only int8 path (upstream cutlass absent); 7.2B/one 5090 int8 serves **1.86× the concurrency + 13.1% higher peak** than fp16 can reach (corrected 2026-07-07 from an undertested grid, F0047). Published: ModelScope `Hakureirm/rwkv7-g1-{1.5b-w8g64,1.5b-w4gptq,7.2b-w4gptq}` |
| 7 | Preliminary speculative decoding; DFlash etc. as follow-up | ◑ correctness done / speed partial | **`bench/spec_gate.py` 10/10 token-identical (F0046)**, Strategy B built on sglang main's spec-V2 plugin architecture, two real RWKV-specific bugs found and fixed. Hand-rolled draft-decode CUDA graph: real **1.5–1.6× speedup on the draft step**, net still 2.6–4.5× slower than spec-off — honest partial win. DFlash itself researched and explicitly not pursued (ADR-0007) — **consistent with BlinkDL's own "as follow-up" framing**, not a shortfall. |
| Decreed | uncheatable compression (+position curve) | ✅ | [`bench/uncheatable_eval.py`](bench/uncheatable_eval.py): 1.5B full-corpus POOLED compression fp16 **0.6085** / w8 **0.6086** (lossless) / w4-GPTQ 0.6514 (old N) / asymmetric 0.6186 (clean N=300, F0043); position curve CSVs. Current-HEAD re-run **bit-identical**; **w8a8 0.6161, == cutlass bit-for-bit on the sm120 kernel** |
| Decreed | MATH500 avg@64 + best-bsz speed | ✅ | [`bench/math500_avg64.py`](bench/math500_avg64.py); 1.5B **avg@64 = 40.60%** (32k rollouts) — F0024; drift-gate CLOSED; **quantization on the reasoning ruler: fp16 0.4042 → w8a8 0.3812 (−2.3pt) → int4 GPTQ 0.1498 sym/0.2199 asym (−25.6pt/−18.6pt collapse)** — the low-variance ruler resolves what compression rate hides |

## §2 Original contributions (the scoring core)

Each line: what + where + key number (with baseline) + verification gate + commit.

- **FLA-free WKV recurrence kernel** (decode + varlen prefill, in-place indexed
  state I/O) — `rwkv7_kernels/wkv_recurrent.py`; greedy-EXACT at 0.1B/1.5B/7.2B;
  initial release `b3e1c86`, in-place variant in the M6 line.
- **Weight-only int4 family** `gemv_w4_m1` / `gemm_w4_small` (rows bit-identical
  to M=1) / `gemm_w4_tc` (wmma, in-smem dequant, deterministic split-K, sm80+
  cp.async pipeline) — `cuda/rwkv7_w4.cu`, F0017; 1.5B bsz1 **259.1 vs fp16
  166.5 (1.56×)**, ≥fp16 at every bsz≤32 (3090); gate `bench/verify_w4.py`;
  `0687e8c`, `bf553de`, `cbf3c07`.
- **Weight-only int8 (w8a16) family** — `cuda/rwkv7_w8.cu`, F0018; greedy
  **24/24 EXACT** (lossless in practice), **227.4 vs 166.5** at bsz1, ≥fp16 at
  every bsz≤32; gate `bench/verify_w8.py`; `f00d1aa`, `9f100a7`.
- **Fused 4-chain LoRA kernel** (~12 launches → 2) — `cuda/rwkv7_lora.cu`,
  F0020; fp16 bsz1 **203.0 → 226.5 (+11.6%)**, greedy 24/24 EXACT; raw
  transcript `bench/results/headline/raw/`; `edcd8a3`.
- **Head-parallel TP + layer-partition PP** with `v_first` cross-stage plumbing,
  incl. root-causing an upstream pitfall (PP tensor transfer chunk-send is
  lossless only for tp-replicated tensors) — F0019, fix `cbf3c07`; full matrix
  greedy-EXACT (tp 2/4/8, pp 2/4/8, mixed), full matrix in
  `bench/results/parallel/` + `bench/results/multigpu.md`; `9f100a7`, `1f775c6`.
  **Upstream impact:** reported as sglang issue
  [#30015](https://github.com/sgl-project/sglang/issues/30015) (2026-07-03,
  model-independent tp2pp2 repro + root cause + two fix options,
  `docs/upstream_pp_allgather_issue.md`), and fixed by our PR
  [#30095](https://github.com/sgl-project/sglang/pull/30095): carries the
  TP-sharded flag in the proxy-tensor **metadata** (sender-only `all_gather_exclude`;
  the receiver reads it off the wire, so the two sides can't disagree) and wires
  it through `scheduler_pp_mixin` from a one-line model opt-in attribute, with a
  registered `tp2×pp2` gloo test (sharded tensor round-trips exact, replicated
  still all-gathered, corruption reproduced without the fix). This is a more
  robust alternative to the concurrent community PR
  [#30058](https://github.com/sgl-project/sglang/pull/30058), which took our
  option-1 primitive but requires the exclude set on both send and recv and left
  the model wiring as a follow-up.
- **GPTQ for RWKV-7** (activation-aware, Hessian capture hook + streamed/
  sharded accumulation for models whose Hessian set exceeds GPU+RAM) —
  `bench/{calib_run,gptq_w4}.py`; 1.5B lambada −3.34pt (vs RTN −4.95);
  `197c051`, `380cfb5`, `ad59947`.
- **Hand-written sparse sqrelu FFN + fused fp16 GEMV + fused elementwise**
  (M6 line) — `cuda/rwkv7_sparse_cmix.cu` (adapted, see §3), `fused.py`;
  fp16 single-stream 0.66→0.73× of the albatross mega-kernel at 1.5B bsz1.
- **SGLang `main` port** — verified greedy-EXACT (0.1B+1.5B) on main @a3f6680;
  patch + apply guide in [`sglang_main_port/`](sglang_main_port/); `85b45b5`.
- **Official-metric harnesses** (uncheatable compression, MATH500 avg@64) —
  `f19a953`.

## §3 Adapted code (full disclosure)

Two CUDA kernels are adapted from BlinkDL/Albatross (Apache-2.0):
`cuda/rwkv7_fast.cu` (fp16 GEMV) and the sparse channel-mix starting point —
see [`cuda/ALBATROSS_LICENSE`](sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda/ALBATROSS_LICENSE)
and [`cuda/NOTICE`](sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda/NOTICE).
Everything else in the kernel family (w4/w8/lora/wkv/TP/PP/GPTQ) is original
work of this repo.

## §4 Measurement discipline

Exclusive GPU for head-to-head numbers (ours ≥7-run median, albatross 3×);
per-card and serving-scale sweeps are single-run with **raw logs committed**
(`bench/results/**/raw/`, `allcards.json`); accuracy via greedy token-exact
gates + scored lm-eval; every README claim carries its number and its baseline.

## §5 Release model

Milestones M0–M6 were developed on a private dev box and published as a clean
initial release (`b3e1c86`) followed by incremental commits; each pre-release
milestone's evidence triple = initial-release content + its finding doc
(F0001–F0016) + its `bench/results/` artifact.

## §6 Reproduce (five commands)

```bash
python tools/convert_rwkv7_blinkdl_to_fla.py --pth <ckpt.pth> --out <fla-dir>  # or use fla-hub checkpoints
bash scripts/deploy.sh                                                     # overlay onto sglang v0.5.10.post1
python bench/greedy_check.py --model <fla-dir> --fixture bench/fixtures/oracle_rwkv7_15b_eiffel.json
python bench/throughput.py --model <fla-dir> --dtype float16 --cuda-graph --disable-radix-cache --batch-sizes 1,8,32
# accuracy: server + lm_eval local-completions (bench/results/lm_eval.md) / bench/uncheatable_eval.py
```
