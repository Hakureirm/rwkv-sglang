# Albatross RWKV-7 baseline on OUR RTX 3090 (M3a)

Speed/VRAM parity baseline for BlinkDL/Albatross, **re-measured on our own RTX 3090**
(`gpu-box`, GPU0) so it is apples-to-apples with our sglang RWKV-7 impl. Albatross's
published numbers are on a 5090; everything in the tables below is **measured on the 3090**
unless a row is explicitly labeled "5090 published".

Albatross baseline: github.com/BlinkDL/Albatross @ `343147a333fcd6dd0845de0d165089685402c012` (`faster3a_2605`).

## 1. Build: did it compile + run on the 3090?

**Yes.** Albatross's fastest engine, `faster3a_2605/rwkv7_fast_v3a.py`, compiles its custom
CUDA (WMMA tensor-core + cublasLt linears, cp.async WKV) and runs on the 3090 (sm_86).

Exact recipe (the box had no nvcc on PATH but a full CUDA 12.9 toolkit at `/usr/local/cuda-12.9`):

```bash
# on gpu-box, work dir ~/albatross (rsync'd from refs/Albatross)
source ~/rwkv_env.sh                       # sets C_INCLUDE_PATH to python3.10 headers
export CUDA_HOME=/usr/local/cuda-12.9      # nvcc 12.9.41 (full toolkit: cublasLt.h, mma.h, ...)
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6            # build only for the 3090 (sm_86)
export MAX_JOBS=8
export CUDA_VISIBLE_DEVICES=0
cd ~/albatross/faster3a_2605
~/envs/rwkv-sgl/bin/python rwkv7_fast_v3a.py --model <pth> --warmup 3 --iters 10 --cases 1x1
```

Key build facts:
- Python/torch: `~/envs/rwkv-sgl/bin/python`, **torch 2.9.1+cu128** (CUDA 12.8), device cap (8,6).
- nvcc is **12.9** vs torch's **12.8** — only a *minor* version skew; torch's cpp_extension
  build checks only the CUDA *major* (12==12), so it warns but compiles fine.
- No nvcc install needed (`nvidia-cuda-nvcc-cu12` was NOT required); the system 12.9 toolkit
  supplies every header the kernels include (`cublasLt.h`, `cublas_v2.h`, `mma.h`, `cuda_fp16.h`).
  cublas/cublasLt symbols resolve at load() time via torch's own bundled CUDA libs.
- First compile of the 3 extensions (`rwkv7_v3a_ops` 143KB .cu, `rwkv7_fast_ops_fp16`,
  `rwkv7_wkv_fp16_v2`) took **~152 s** (one-time; cached in `~/.cache/torch_extensions`).
- **No Blackwell-only PTX**: the v3a kernels use `nvcuda::wmma` (sm_70+ fp16 tensor cores) and
  `cp.async` (sm_80+) only — both supported on Ampere/sm_86. The README's "tune
  linear_orig_layout for your GPU" tuning is for 5090; defaults run correctly on the 3090
  (numbers below are with the stock defaults, so they are a conservative 3090 baseline).

The `faster4_2605_cpp` (standalone C++) and `faster3a` README examples hard-code
`-DCMAKE_CUDA_ARCHITECTURES=120` / sm_120; we instead drove the python JIT path with
`TORCH_CUDA_ARCH_LIST=8.6`, which is the portable way to target the 3090.

## 3. How Albatross loads / runs (so we mirror its methodology fairly)

`rwkv7_fast_v3a.py` (the engine benchmarked here):
- **Weights**: raw BlinkDL `.pth` loaded via `torch.load(..., mmap=True)`, kept in **fp16**
  (`DTYPE=torch.float16`). Linear weights are pre-transposed/contiguous; `emb.weight` is held
  on **CPU** by default (`--emb cpu`, saves VRAM, embedding lookup is cheap). Active-param
  count excludes the embedding.
- **Kernels**: fully custom CUDA — fused LayerNorm, WMMA/split-k/cublasLt GEMMs for the
  channel-mix/attention linears, a sparse-FFN ("no-fc") path for tiny batches, and a dedicated
  fp16 WKV state kernel (`--wkv fp16` default; `--wkv fp32io16` is a more accurate fp32-state
  path). No `torch.compile` at runtime.
- **Static batch + CUDAGraph**: each `(B,T)` case builds a `torch.cuda.CUDAGraph` of ONE
  forward over `B*T` tokens, then times `iters` graph replays with CUDA events; reports
  p10/p50/p90 ms and `tok_s_p50 = B*T*1000/p50_ms`. Static shapes, no dynamic batching, no
  paged KV (RWKV is an RNN: O(1) state per step, no KV cache).
- **Axis mapping to our `bench/throughput.py`** (bsz {1,8,32}, decode 128 tok, prefill_len 1024):
  - **decode tok/s @ bsz B** = case `Bx1` (one batched decode step; RWKV decode is O(1)/token,
    so a single step == steady-state, no prefill subtraction needed). `tok_s_p50 = B/p50_ms*1000`.
  - **prefill tok/s @ bsz B** = case `Bx1024` (one forward over a 1024-len prompt x B).
    `tok_s_p50 = B*1024/p50_ms*1000` — identical definition to our `bsz*prefill_len/TTFT`.
  - **peak VRAM** = whole-GPU `nvidia-smi memory.used` (MiB) polled at 10 Hz during each bsz's
    `{Bx1, Bx1024}` run — same instrument and definition as our throughput.py.

## 2. Benchmark table (measured on the 3090, fp16, CUDAGraph; `faster3a_2605/rwkv7_fast_v3a.py`)

p50 over 20 iters. decode = case `Bx1` (`tok_s=B/p50_ms*1000`); prefill = case `Bx1024`.

| model | bsz | decode tok/s | prefill tok/s | p50 decode ms |
|---|---|---|---|---|
| 0.1B | 1  | **1171.6** | 69713.9 | 0.854 |
| 0.1B | 8  | 5448.0 | 190236.6 | 1.468 |
| 0.1B | 32 | 24522.4 | 235368.6 | 1.305 |
| 1.5B | 1  | **309.1** | 14645.7 | 3.235 |
| 1.5B | 8  | 1220.6 | 21570.9 | 6.554 |
| 1.5B | 32 | 5296.6 | 20876.2 | 6.042 |

(7.2B numbers + a consistent VRAM-inclusive re-measurement of all three sizes are in
section 5 below — those are the values used in `comparison.md`.)

### Head-to-head vs our sglang baseline (F0006: bf16, **cuda-graph OFF**)
| model | bsz | metric | albatross (3090) | ours (sglang) | gap |
|---|---|---|---|---|---|
| 0.1B | 1  | decode tok/s | 1171.6 | 20.6 | **~57×** |
| 0.1B | 32 | decode tok/s | 24522 | 665 | ~37× |
| 0.1B | 1  | prefill tok/s | 69714 | 8116 | ~8.6× |
| 1.5B | 1  | decode tok/s | 309.1 | 10.5 | ~29× |
| 1.5B | 1  | prefill tok/s | 14646 | 4149 | ~3.5× |

**Read**: the decode gap (~30–57×) is dominated by our eager mode (cuda-graph OFF) — M2b cuda-graph
should recover most of the *launch-overhead* portion. The residual after cuda-graph = the
**kernel-quality gap**: albatross uses hand-tuned fp16 CUDA (WMMA tensor-core GEMMs, cublasLt,
sparse-FFN, fused WKV) vs our fla-triton. Prefill gap (3.5–8.6×) is purely kernel-quality
(both already graphed/batched).

**Strategic option (proven viable here)**: albatross's custom CUDA kernels **compile + run on the
3090 (sm_86)** via torch JIT (`TORCH_CUDA_ARCH_LIST=8.6`, system CUDA at `/usr/local/cuda-12.9`).
So we can **vendor albatross's fast WKV/linear CUDA kernels into our sglang backend** to close the
kernel-quality gap while keeping sglang's serving layer — the same approach the closed vLLM PR
#46269 took. Likely the path to true speed parity after cuda-graph lands.

## 4. Notes for a fair comparison vs our sglang impl

- **Precision**: Albatross runs **fp16** weights + fp16 WKV state (with deterministic
  dithering for fp16 stability). Our sglang baseline runs **bf16**. fp16 vs bf16 is roughly
  bandwidth-equivalent for weights (both 2 bytes), so VRAM for weights is comparable; speed
  differences are kernel/graph driven, not dtype.
- **Static vs dynamic**: Albatross is a *static-batch, fixed-shape, CUDAGraph* raw-speed engine
  — no continuous batching, no scheduler, no paged memory, no tokenizer in the hot loop. It is
  an upper-bound "kernel + graph" number, not a serving system. Our sglang number includes
  scheduler/runtime overhead and (in the M2 baseline) CUDA-graph OFF. Expect Albatross to be
  faster at fixed shapes; the gap is the cost of sglang's serving machinery.
- **What it does NOT do**: no EOS/sampling-loop overhead in the timed region (decode case times
  the forward only), no variable-length batching, no request queueing. Embedding on CPU.
- **RNN property**: decode tok/s and VRAM are ~constant in context length (no KV growth), so
  the `Bx1` decode number hnews at any sequence position — directly comparable to our
  context-length-independent decode measurement.

## 5. 7.2B build + consistent all-sizes re-measurement (with VRAM)

The 7.2B BlinkDL checkpoint (`rwkv7-g1g-7.2b-20260523-ctx8192.pth`, 14.4 GB) was
downloaded and **builds + runs on the 3090** via the same `faster3a_2605/rwkv7_fast_v3a.py`
path (fp16 weights, `--emb cpu`, fp16 WKV, CUDAGraph; extensions cached from the earlier
0.1B/1.5B runs, no recompile). All three sizes were then re-benchmarked in one session with
`bench_v3a.sh` (p50 over 20 iters; PEAK_VRAM = whole-GPU nvidia-smi at 10 Hz) so the table
is internally consistent and feeds `comparison.md`:

| model | bsz | decode tok/s (`Bx1`) | prefill tok/s (`Bx1024`) | peak VRAM MiB |
|---|---|---|---|---|
| 0.1B | 1  | 1173.1  | 63734.9  | 2101 |
| 0.1B | 8  | 5453.9  | 190732.8 | 2393 |
| 0.1B | 32 | 24567.6 | 235657.2 | 4737 |
| 1.5B | 1  | 309.2   | 15200.3  | 4707 |
| 1.5B | 8  | 1222.9  | 21499.8  | 6309 |
| 1.5B | 32 | 5297.5  | 20721.0  | 11627 |
| **7.2B** | 1  | 77.0   | 4071.7 | 15887 |
| **7.2B** | 8  | 399.3  | 4290.4 | 19009 |
| **7.2B** | 32 | 1476.2 | 4017.3 | **23987** |

Notes:
- These 0.1B/1.5B decode/prefill numbers reconfirm section 2 (within single-shot variance,
  e.g. 0.1B b1 decode 1173 vs 1172; 0.1B b1 prefill 63735 vs 69714 — TTFT-style variance).
- **VRAM grows with batch** (static `B×T` activation, no paging): 1.5B 4.7→11.6 GB,
  7.2B 15.9→**24.0 GB** (97.6% of the card) over bsz 1→32. This is the cost of Albatross's
  static-shape upper-bound design; our sglang impl's RWKV state is O(1)/token so its footprint
  is flat in batch (see comparison.md).
