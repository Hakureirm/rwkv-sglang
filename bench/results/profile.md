# RWKV-7 × sglang — decode/prefill GPU-time profile (RTX 3090)

> ⚠️ **SUPERSEDED / illustrative** — this is a 2026-06-30 **co-tenant, bsz1-only** profile
> that predates the M6 kernels. Cite `comparison_clean.md` for the clean same-precision
> standing and `docs/design/m6-sparse-ffn.md` for the bsz1-vs-bsz32 component breakdown that
> motivated the in-place WKV kernel. Kept for the qualitative where-does-time-go picture only.

Read-only profiling to locate where our RWKV-7 decode + prefill GPU time goes, vs
the albatross (BlinkDL) baseline. Box (RTX 3090, sm_86), sglang
v0.5.10.post1, torch 2.9.1+cu128, bf16 compute / fp32 state, tp=1, cuda-graph ON,
radix off. All numbers measured on **2026-06-30** (co-tenant GPU).

> Methodology, in short. Two independent instruments, cross-validated against the
> end-to-end engine number:
> 1. **End-to-end** `bench/throughput.py` (sglang Engine, cuda-graph ON) — the
>    decode/prefill tok/s used as the roofline denominator.
> 2. **Per-component CUDA-event timing** (`bench/profile_components.py`) — builds the
>    *real deployed* `Rwkv7Attention`/`Rwkv7FeedForward`/`Rwkv7DecoderLayer` modules
>    + a stub backend that replicates `rwkv7_backend.py`'s decode/extend hot path
>    verbatim (so kernels/math are identical to production; random weights → matmul
>    time is value-independent). Each component is timed two ways: **eager** (N
>    back-to-back launches, includes launch overhead) and **graphed** (same N launches
>    captured in a `CUDAGraph` + replayed = pure GPU-busy, what production pays under
>    cuda-graph). Graphed is the primary attribution.
> 3. **`torch.profiler` (CUDA activities)** on the same modules for kernel **count**
>    and per-kernel names.
>
> Validation: summing the graphed per-component times × num_layers (+ lm_head + emb)
> gives a synthesized GPU-busy decode step of **6.86 ms (1.5B)** and **22.58 ms
> (7.2B)** → **145.8 / 44.3 tok/s**, vs the measured **142.1 / 43.5 tok/s** — within
> **2.6% / 1.8%**. The attribution therefore accounts for essentially 100% of the
> decode step; numbers below are trustworthy at the component level.
>
> NB: the *deployed* model is the M4 quant-aware build (projections are sglang
> `ReplicatedLinear`, not `nn.Linear`). With `quant_config=None` (our config) these
> dispatch to unquantized `F.linear` — bit- and perf-identical to `nn.Linear`. All
> timings here are the dense bf16 path.

---

## 0. Measured end-to-end (this run, cuda-graph ON, radix OFF, bsz1)

| model | decode tok/s | per-token | prefill tok/s (T=1024) | TTFT |
|---|---|---|---|---|
| 1.5B | **142.1** | 7.04 ms | 9284 | 110.3 ms |
| 7.2B | **43.5**  | 22.99 ms | 2964 | 345.5 ms |

(Reconfirms the comparison.md headline: ours 142 / 43.6 vs albatross 309 / 77.)

Param counts (from the real modules):

| model | H | L | heads×hd | total params | **read/token (≠emb)** | bytes/token (bf16) |
|---|---|---|---|---|---|---|
| 1.5B | 2048 | 24 | 32×64 | 1.527 B | **1.393 B** | 2.786 GB |
| 7.2B | 4096 | 32 | 64×64 | 7.199 B | **6.931 B** | 13.861 GB |

The embedding table (134M / 268M params) is **not** the relevant decode read — decode
gathers one row. The lm_head **is** read in full. So bytes/token = (total − emb) × 2.

---

## 1. Bandwidth roofline (decode, bsz1). This is the ceiling.

3090 peak HBM ≈ **936 GB/s**. Effective BW = bytes/token ÷ per-token latency.

| model | engine | tok/s | latency | eff. GB/s | **% of 936 peak** |
|---|---|---|---|---|---|
| 1.5B | **ours**       | 142.1 | 7.04 ms | 396 | **42.3%** |
| 1.5B | albatross      | 309.1 | 3.24 ms | 861 | **92.0%** |
| 1.5B | *dense ceiling*| 336.0 | 2.98 ms | 936 | 100% |
| 7.2B | **ours**       | 43.5  | 22.99 ms| 603 | **64.4%** |
| 7.2B | albatross      | 77.0  | 12.99 ms| 1067| **114%** ⚠ |
| 7.2B | *dense ceiling*| 67.5  | 14.81 ms| 936 | 100% |

**Reading it:**
- **1.5B: we run at 42% of peak; albatross at 92%.** Albatross is essentially
  bandwidth-saturated; we leave **~2.2×** of bandwidth on the floor — that gap is
  *kernel fragmentation + sub-peak GEMVs*, not fundamental.
- **7.2B: we're at 64% of peak — much healthier** (bigger GEMMs amortize better;
  this is why the size-7.2B gap in comparison.md is only ~1.5–1.8×).
- **⚠ Albatross's 7.2B implies 114% of peak — physically impossible on dense reads.**
  Therefore albatross reads **fewer than param×2 bytes/token at 7.2B bsz1.** The
  documented cause is its **sparse-FFN ("no-fc") path**: at tiny batch the sqrelu
  activation is mostly zero, so it skips a large fraction of the FFN value-proj
  weight reads. **A dense-GEMM impl like ours is hard-capped at the dense ceiling
  (67.5 tok/s on this card); we're at 43.5 = 64% of it. Beating ~67 tok/s at 7.2B
  bsz1 requires activation-sparse FFN, which no amount of kernel tuning gives.**

---

## 2. Decode per-component breakdown (graphed GPU-busy = what cuda-graph pays)

Ranked by share of the full decode step. Per-layer time × L, plus lm_head/emb ×1.

### 1.5B (step = 6.86 ms; 24 layers)
| component | per-layer µs | × L µs | **% step** |
|---|---|---|---|
| **ffn** (key+value GEMM, H×inter) | 109.2 | 2621 | **38.2%** |
| **8 LoRA matmuls** + gate-math    | 43.6  | 1047 | **15.3%** |
| **r/k/v proj** (3 GEMM H×H)       | 38.4  | 921  | **13.4%** |
| token-shift lerp (6×)             | 14.8  | 356  | 5.2% |
| o_proj (1 GEMM) + gate-mul        | 14.1  | 338  | 4.9% |
| **lm_head** (×1)                  | 325.8 | 326  | 4.7% |
| kk/k-mix + L2-norm                | 12.0  | 287  | 4.2% |
| token-shift (state gather/scatter)| 10.7  | 257  | 3.7% |
| **WKV recurrence**                | 10.0  | 240  | **3.5%** |
| gate-correction                   | 7.6   | 181  | 2.6% |
| layernorms (attn+ffn)             | 7.3   | 175  | 2.6% |
| g_norm (GroupNorm)                | 4.4   | 106  | 1.5% |
| emb + final_norm (×1)             | 5.3   | 5    | 0.08% |

### 7.2B (step = 22.58 ms; 32 layers)
| component | per-layer µs | × L µs | **% step** |
|---|---|---|---|
| **ffn**                           | 363.0 | 11617| **51.4%** |
| **r/k/v proj**                    | 155.0 | 4960 | **22.0%** |
| **8 LoRA matmuls** + gate-math    | 52.6  | 1682 | 7.4% |
| o_proj + gate-mul                 | 47.5  | 1521 | 6.7% |
| lm_head (×1)                      | 690.6 | 691  | 3.1% |
| token-shift lerp (6×)             | 14.2  | 454  | 2.0% |
| layernorms                        | 11.5  | 367  | 1.6% |
| kk/k-mix + L2-norm                | 10.3  | 331  | 1.5% |
| **WKV recurrence**                | 9.8   | 314  | **1.4%** |
| token-shift (gather/scatter)      | 9.4   | 301  | 1.3% |
| gate-correction                   | 6.6   | 210  | 0.9% |
| g_norm                            | 3.9   | 126  | 0.6% |
| emb + final_norm (×1)             | 7.3   | 7    | 0.03% |

### Roll-up
| bucket | 1.5B | 7.2B |
|---|---|---|
| **all matmuls** (ffn+rkv+lora+o_proj+lm_head) | **76.6%** | **90.7%** |
| **elementwise "glue"** (lerp+kk+token-shift+gate-corr+norms) | **19.9%** | **7.9%** |
| **WKV recurrence** | 3.5% | 1.4% |

**Findings:**
- **The matmuls are where the time is** (77% / 91%). At 1.5B the FFN alone is 38%;
  at 7.2B it's 51% (the FFN value+key are the two largest weight reads).
- **The WKV kernel is NOT a decode bottleneck** — 3.5% / 1.4%. The state is tiny
  (K×V=64×64/head); our triton scan kernel is only **2.8 µs/layer**. A custom CUDA
  WKV buys almost nothing at decode.
- **The 8 LoRA matmuls are pathologically inefficient.** Graphed 43.6 µs/layer (1.5B)
  for matrices whose *ideal* read time is ~0.4 µs → **~1% of peak BW**. They are 8
  tiny separate GEMVs (M=1, rank 96–256) that each underutilize the GPU. 15.3% of
  the 1.5B step is almost pure overhead.
- **GEMV efficiency vs the roofline (1.5B):** ffn ~66% of peak, r/k/v ~70%, o_proj
  ~64%, lm_head ~88% (the one big enough to saturate). The small GEMVs drag the
  average to 42%.
- **~20% of the 1.5B step is elementwise glue** spread across ~40 tiny kernels —
  pure memory-bound activation churn that albatross fuses into GEMM epilogues.

### Kernel count (why cuda-graph is mandatory)
`torch.profiler` over one decoder layer + lm_head (1.5B, eager): **78 CUDA kernel
launches**, **29 distinct kernels**. ≈ **77 kernels / layer / token**, so a full 1.5B
decode step launches **~1850 kernels**. Of the 78: ~12 GEMM/GEMV (the 11 `gemvx`
kernels = exactly 3 r/k/v + 8 LoRA), ~40 `vectorized_elementwise`, 6 `index`
(token-shift gather/scatter), 2 layernorm + 2 reduce (L2/gate-corr), 1
`_wkv_recurrent_kernel` (2.8 µs), 2 DtoD memcpy (state contiguous copies). Eager
launch overhead is enormous (e.g. LoRAs: eager 990 µs vs graphed 44 µs) — cuda-graph
removes essentially all of it, which is why it already recovered the ~30–57× eager gap.

---

## 3. Prefill (T=1024, bsz1): is the sequential-scan WKV the bottleneck at long T?

Per-layer graphed GPU-busy.

### 1.5B (per-layer 2.87 ms; 24 layers ≈ 68.9 ms GPU)
| component | µs/layer | % layer |
|---|---|---|
| ffn (2 GEMM) | 1197 | 41.7% |
| **WKV scan** | **899** | **31.3%** |
| r/k/v proj (3 GEMM) | 486 | 16.9% |
| o_proj | 140 | 4.9% |
| 8 LoRA | 122 | 4.2% |
| token-shift | 28 | 1.0% |

→ **WKV / (WKV + linear-GEMM) = 31.6%.**

### 7.2B (per-layer 8.48 ms; 32 layers ≈ 271 ms GPU)
| component | µs/layer | % layer |
|---|---|---|
| ffn | 4513 | 53.3% |
| r/k/v proj | 1709 | 20.2% |
| **WKV scan** | **1339** | **15.8%** |
| o_proj | 569 | 6.7% |
| 8 LoRA | 305 | 3.6% |
| token-shift | 40 | 0.5% |

→ **WKV / (WKV + linear-GEMM) = 15.9%.**

**Findings:**
- **Yes at 1.5B, partly: the sequential-scan WKV is the #2 prefill cost (31.6%)** —
  a real secondary bottleneck. It is O(T) sequential steps with no tensor-core use,
  so it scales linearly with T while the GEMMs are batched/efficient.
- The prefill **GEMMs are already efficient** (1.5B ffn = 57 TFLOPS, r/k/v = 53
  TFLOPS — near the 3090's practical bf16 peak ~60 TFLOPS). The FFN, not the scan,
  is #1.
- **At 7.2B the scan drops to 15.9%** (the larger GEMMs dominate). This matches the
  small measured prefill gap at 7.2B (1.2–1.4×).
- GPU-busy is 62% (1.5B) / 78% (7.2B) of TTFT — prefill is **not** cuda-graphed in
  sglang, so the remainder is launch/scheduler overhead.

---

## 4. fp16 vs bf16 (decode, 1.5B)

| dtype | synth GPU-busy step | tok/s ceiling |
|---|---|---|
| bf16 | 6.86 ms | 145.8 |
| fp16 | 6.91 ms | 144.8 |

**No difference (<1%).** On Ampere/sm_86, fp16 and bf16 share the same tensor-core
and memory throughput, and the workload is memory-bound. **Albatross's fp16 is NOT
its speed advantage** — its advantage is kernel fusion + sparse-FFN. Don't switch to
fp16 for speed (and it would cost the fp32-state accuracy guarantee).

---

## 5. Ranked optimization recommendations (expected payoff vs the roofline)

Payoff = measured share of the decode step that is recoverable. The unifying theme:
**we lose to albatross by running ~78 fragmented kernels/layer at 42–64% of peak BW;
albatross runs a handful of fused, near-saturated kernels.**

| # | optimization | regime | measured target | expected decode payoff |
|---|---|---|---|---|
| **1** | **Fuse the elementwise glue into GEMM epilogues/prologues** (token-shift lerp, kk/k-mix, gate-correction, g_norm) — collapse ~40 tiny kernels | 1.5B (glue-bound) | 19.9% of step (1.5B), 7.9% (7.2B) | 1.5B: ~6.86→~5.5 ms → **~182 tok/s** |
| **2** | **Batch the 8 LoRA matmuls** into 2 grouped GEMMs (stack the 4 down-projs, 4 up-projs) | both | 1047 µs (15.3%) at ~1% peak → ideal ~50 µs | 1.5B: recovers ~14% → stacks with #1 toward **~210 tok/s** |
| **3** | **Vendor albatross's WMMA/cublasLt linear kernels** for r/k/v/o/ffn (lifts GEMV from ~60% to ~90% of peak) | 7.2B esp. | r/k/v+o+ffn = 56.5% (1.5B) / 80% (7.2B) | 7.2B: 64%→~90% peak → **43.5→~61 tok/s** |
| 4 | **Fuse r/k/v into one grouped GEMM** (3 GEMVs → 1) | both | subset of #3 | modest beyond #3 (launches already graphed) |
| 5 | **Chunked / tensor-core WKV for prefill** (decode WKV is only 1–3%, skip it there) | prefill 1.5B | 31.6% of 1.5B prefill layer | recovers up to ~31% of 1.5B prefill |
| 6 | **Activation-sparse FFN ("no-fc")** — the only way past the 7.2B dense ceiling | 7.2B bsz1 | FFN = 51% of 7.2B step; dense cap = 67.5 tok/s | needed to reach albatross's 77 tok/s |
| — | **fp16 instead of bf16** | — | measured 0% | **don't** (no gain, loses fp32-state accuracy) |

**Single highest-payoff move:** **vendor albatross's fused fp16 CUDA kernels** — the
WMMA/cublasLt linears with **fused token-shift/lerp/gate epilogues + batched LoRAs**
(items 1+2+3 in one drop-in). The data says ~96% of the decode step is matmuls +
elementwise glue, and we run them at 42% (1.5B) / 64% (7.2B) of peak through ~78
kernels/layer. One fused kernel set collapses those launches and lifts BW
utilization toward albatross's ~90%, projected to roughly **double 1.5B decode
(142 → ~280 tok/s, ~90% of albatross's 309)** and push **7.2B to ~61 tok/s** (the
dense ceiling). Beating 7.2B's 67-tok/s dense ceiling additionally requires the
sparse-FFN (#6). These kernels are already proven to compile + run on the 3090
(albatross_3090.md), so this is the concrete M3b path to speed parity.
