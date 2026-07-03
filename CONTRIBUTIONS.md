# Contributions — RWKV-7 on SGLang

This repo delivers the production-grade **RWKV-7 serving adaptation for SGLang**
(dynamic batching + chunked prefill + O(1) recurrent-state pool, greedy
token-exact vs the BlinkDL rwkv-lm numpy reference), with **zero FLA
dependency** and a family of hand-written CUDA kernels (WKV recurrence /
weight-only int4 / int8 / fused-LoRA), measured on 10 GPU types across 7 SM
generations. Entry points: [README.md](README.md) ·
[docs/snapshot.md](docs/snapshot.md) · [docs/](docs/) (ADRs + findings ledger).

## §1 Requirement scorecard (status honest as of 2026-07-03)

| Requirement | Status | Evidence |
|---|---|---|
| 1. Match RWKV-LM / Albatross accuracy, speed, VRAM (across bsz) | ✅ 1.5B full / ◑ 7.2B | [`bench/results/comparison_clean.md`](bench/results/comparison_clean.md) + `clean/` raw, [`lm_eval.md`](bench/results/lm_eval.md) (lambada 0.673 vs ref 0.671, MMLU 0.524 vs 0.511) |
| 2. HF PEFT/RL trainability | n/a | training-track scope; this repo is the inference/serving adaptation (README design-goals table) |
| 3. Dynamic batching + chunked prefill + state cache | ◑ | batching/chunked-prefill greedy-EXACT: `bench/verify_batch.py`, `bench/verify_chunked_prefill.py`; radix-off is a **correctness decision** ([`radix_correctness.md`](bench/results/radix_correctness.md)); state-aware prefix cache = designed follow-up |
| 4. Pascal+ / AMD; PP + TP inference | ◑ | TP 2/4/8 + PP 2/4/8 + mixed, all greedy-EXACT: [`bench/results/parallel/`](bench/results/parallel/) (+`raw/` transcripts), F0019; 10-GPU grid [`multigpu.md`](bench/results/multigpu.md) + `bench/results/allcards.json`; Pascal routing guard `42fd6fa`; Pascal/AMD hardware runs pending |
| 5. w8/w4 quant: VRAM ↓, ≥ w16 speed, near Q*_K_M accuracy | ✅ speed+VRAM+acc / ◑ Q*_K_M cmp | [`bench/results/w4/`](bench/results/w4/) (+`raw/`), F0017/F0018; w4 ≥fp16 at every bsz≤32 (3090, 1.5B); **7.2B GPTQ lambada 0.7297 vs bf16 0.7425 (−1.28pt), 192/192 GPTQ, 4.6 GB, on a real 16 GB T4**; w8 uncheatable-lossless. Published: ModelScope `Hakureirm/rwkv7-g1-{1.5b-w8g64,1.5b-w4gptq,7.2b-w4gptq}` |
| 6. Speculative decoding (preliminary) | ⬜ | designed (state checkpoint/rollback plan), not yet implemented |
| Decreed: uncheatable compression (+position curve) | ✅ | [`bench/uncheatable_eval.py`](bench/uncheatable_eval.py) + [`bench/results/uncheatable/`](bench/results/uncheatable/): 1.5B full-corpus POOLED compression fp16 **0.6085** / w8 **0.6086 (lossless, +0.0001)** / w4-GPTQ 0.6514; position curve CSVs |
| Decreed: MATH500 avg@64 + best-bsz speed | harness ✅ / numbers pending | [`bench/math500_avg64.py`](bench/math500_avg64.py) (prompt/sampling/grader ported verbatim from the reference) |

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
  greedy-EXACT (tp 2/4/8, pp 2/4/8, mixed), transcripts in
  `bench/results/parallel/raw/`; `9f100a7`, `1f775c6`.
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
