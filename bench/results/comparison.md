# RWKV-7 × sglang vs Albatross — full comparison (RTX 3090)

> ⚠️ **SUPERSEDED by `comparison_clean.md`.** The numbers in THIS file were measured
> while another job (isaaclab) shared the GPU (nvidia-smi baseline **1304 MiB**, contended
> SMs), which depressed OUR end-to-end serving throughput (albatross's kernel-only timing
> was unaffected). They have been re-measured on the now-**exclusive** 3090 (baseline
> ~1 MiB), ≥5 repeats medianed, one GPU process at a time — see **`comparison_clean.md`**
> (which also adds the int8 rows). Keep this file only for provenance; cite the clean one.

This benchmark compares **our** sglang RWKV-7 serving impl against **Albatross**
(`faster3a_2605/rwkv7_fast_v3a.py`, BlinkDL's hand-tuned fp16 CUDA engine), both
measured on the **same RTX 3090** (`gpu-box`, GPU0, baseline 1304 MiB), across
{0.1B, 1.5B, 7.2B} × bsz {1, 8, 32}.

## Configuration (read this before the numbers — the comparison is NOT apples-to-apples)

| | OURS (sglang) | Albatross |
|---|---|---|
| what it is | full serving engine: scheduler, continuous/dynamic batching, paged state pool, tokenizer-capable | static-shape **kernel+CUDAGraph micro-bench**: one fixed `(B,T)` forward, no scheduler, no batching, no queueing |
| precision | **bf16** weights, **fp32** recurrent state | **fp16** weights, fp16 WKV state (+dither) |
| kernels | fla **Triton** (per-op) WKV + torch linears | hand-tuned **CUDA**: WMMA tensor-core / cublasLt GEMMs, fused WKV, sparse-FFN |
| graph | sglang CUDA graph (decode only; extend/prefill not graphed) | whole forward in one CUDAGraph |
| timing | end-to-end `Engine.generate` incl. scheduler overhead | CUDA-event around graph replay only |
| radix cache | **OFF** (required for RWKV correctness — see radix_correctness.md) | n/a |

So Albatross is a near-upper-bound "kernel + graph" number; ours is a real server. The
residual gap = **kernel quality** (Triton vs hand-tuned fp16 tensor-core CUDA) + sglang's
**serving overhead**. Both are exact RWKV-7 (greedy-matched to the numpy oracle).

Ours: `throughput.py --dtype bfloat16 --cuda-graph --cuda-graph-max-bs 32 --disable-radix-cache`.
Decode = steady-state (prefill-subtracted), prefill = bsz×1024 / TTFT. Albatross:
`bench_v3a.sh <pth> <label> 1,8,32` (decode = `Bx1`, prefill = `Bx1024`, p50 over 20 iters).

## Decode throughput (tok/s) — higher is better

| model | bsz | ours | albatross | ratio (ours/alb) | gap (alb/ours) |
|---|---|---|---|---|---|
| 0.1B | 1  | 453.7   | 1173.1  | 0.39 | **2.59×** |
| 0.1B | 8  | 3230.2  | 5453.9  | 0.59 | 1.69× |
| 0.1B | 32 | 10040.2 | 24567.6 | 0.41 | 2.45× |
| 1.5B | 1  | 141.9   | 309.2   | 0.46 | 2.18× |
| 1.5B | 8  | 863.3   | 1222.9  | 0.71 | 1.42× |
| 1.5B | 32 | 2760.6  | 5297.5  | 0.52 | 1.92× |
| **7.2B** | 1  | 43.6  | 77.0  | 0.57 | **1.77×** |
| **7.2B** | 8  | 268.8 | 399.3 | 0.67 | **1.49×** |
| **7.2B** | 32 | 923.1 | 1476.2 | 0.62 | **1.60×** |

## Prefill throughput (tok/s) — higher is better

| model | bsz | ours | albatross | ratio (ours/alb) | gap (alb/ours) |
|---|---|---|---|---|---|
| 0.1B | 1  | 18751.8 | 63734.9  | 0.29 | 3.40× |
| 0.1B | 8  | 39231.5 | 190732.8 | 0.21 | 4.86× |
| 0.1B | 32 | 38157.2 | 235657.2 | 0.16 | **6.18×** |
| 1.5B | 1  | 9498.4  | 15200.3  | 0.62 | 1.60× |
| 1.5B | 8  | 12825.7 | 21499.8  | 0.60 | 1.68× |
| 1.5B | 32 | 12915.7 | 20721.0  | 0.62 | 1.60× |
| **7.2B** | 1  | 2989.0 | 4071.7 | 0.73 | **1.36×** |
| **7.2B** | 8  | 3505.5 | 4290.4 | 0.82 | **1.22×** |
| **7.2B** | 32 | 3345.5 | 4017.3 | 0.83 | **1.20×** |

## Peak VRAM (whole-GPU nvidia-smi MiB, incl. 1304 baseline) — lower is better

| model | bsz | ours | albatross | note |
|---|---|---|---|---|
| 0.1B | 1  | 3712  | 2101  | ours = mem-fraction 0.15 reservation; weights only 0.4 GB |
| 0.1B | 8  | 3776  | 2393  | |
| 0.1B | 32 | 3794  | 4737  | ours flat; albatross grows with batch |
| 1.5B | 1  | 6310  | 4707  | ours = mem-fraction 0.25; weights 3.0 GB |
| 1.5B | 8  | 6522  | 6309  | ~par |
| 1.5B | 32 | 6522  | **11627** | ours **1.8× lower** (flat vs batch-growth) |
| **7.2B** | 1  | 17178 | 15887 | ours = mem-fraction 0.72; weights 14.4 GB |
| **7.2B** | 8  | 17484 | 19009 | ours lower |
| **7.2B** | 32 | **17484** | **23987** | albatross at 97.6% of the 24 GB card; **ours 17.5 GB — fits with ~6.9 GB headroom** |

(Speed rows for 0.1B/1.5B use mem-fraction 0.5 for un-chunked prefill; VRAM rows use the
minimum mem-fraction that serves bsz=32. Decode is mem-fraction-insensitive — it matches
across runs. sglang reserves `mem_fraction_static` eagerly, so ours is a *reserved budget*,
tunable down toward the bf16-weights floor; albatross reports *actual* allocation that grows
with B×T activations.)

## Reading the results

- **The gap shrinks dramatically with model size.** Decode gap falls from ~2.5× (0.1B) to
  **~1.5–1.8× at 7.2B**; prefill from up to 6.2× (0.1B) to **~1.2–1.4× at 7.2B**. Small
  models are launch/per-op-overhead bound, where Albatross's whole-forward CUDAGraph and
  fused kernels dominate and sglang's scheduler overhead is proportionally large. At 7.2B
  the work is compute-bound, so kernel quality matters less and we close most of the gap —
  **on the production-relevant size we are within ~1.2–1.8× of a hand-tuned fp16 tensor-core
  engine while doing real dynamic-batch serving.**
- **The residual is kernel quality, not architecture.** cuda-graph already removed the
  ~30–57× eager-mode launch gap (F0008). What's left is fla-Triton per-op vs Albatross's
  WMMA/cublasLt fused fp16 CUDA — exactly the M3b kernel-vendoring target. (Albatross's
  kernels compile + run on the 3090, so vendoring is viable; see albatross_3090.md.)
- **VRAM is a genuine win at scale.** RWKV-7 decode is O(1) state per token, so our footprint
  is **flat in batch size**, whereas Albatross's static B×T forward grows and nearly OOMs the
  card at 7.2B bsz32 (23987/24576 MiB). Our 7.2B serves bsz32 in 17.5 GB.
- **Fairness caveats (against us being flattered):** Albatross is fp16 (slightly faster
  kernels than bf16) and times *kernels only*; ours is a full server timed end-to-end. The
  0.1B prefill gap is inflated by Albatross's astronomical small-model prefill (235k tok/s)
  meeting our per-request scheduler overhead — not representative of production (7.2B).

## Correctness (both exact)
All three sizes greedy-match the pure-numpy fp32 oracle token-for-token (bf16, cuda-graph
ON): 0.1B & 1.5B (F0006/F0008), and **7.2B 8/8** (this milestone; `verify_m1d.py` +
`oracle_rwkv7_72b_eiffel.json`). Dynamic-batch correctness verified exact with
`disable_radix_cache=True` (radix_correctness.md).
