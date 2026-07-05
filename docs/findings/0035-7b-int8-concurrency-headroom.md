# F0035 — 7.2B on a single 32 GB 5090: int8 unlocks 2.32× concurrency and a 16.8% higher peak fp16 cannot reach

**Date:** 2026-07-06 · **Status:** MEASURED (both configs booted + swept on the RTX 5090) · **Prior:** F0034 (w8a8 V2 + the 1.5B strategic pivot)

## The claim, and why 1.5B was the wrong place to make it

F0034 showed that on **1.5B** the w8a8 int8 GEMM beats fp16 cuBLAS (1.03–1.55× at
M≥512) yet e2e sits at 0.9466× fp16 — because 1.5B is not VRAM-limited, so int8's
weight-halving buys nothing in capacity, and the battle is pure compute where our
own fp16 stack is already excellent. The decisive, **fp16-unreachable** win for int8
is where fp16 runs out of memory: **7.2B on a single 32 GB card.**

## Measured (RTX 5090, 32 GB, sglang main, identical launch: mem-fraction 0.93, cuda-graph ON, 64-in/256-out)

RWKV-7 state is **constant-size** (no KV cache; a fixed per-request recurrent state),
so the state-pool slot count *is* the max concurrency. Per-request state ≈ 33 MB
(fp32, identical for both — it is model state, independent of weight quantization).

| | fp16 7.2B | w8a8 7.2B |
|---|---|---|
| weights | 14.4 GB | 7.75 GB |
| **max concurrency (state-pool slots)** | **221** | **512 (2.32×)** |
| state pool allocated | 6.94 GB | 16.03 GB |
| free after startup | 5.26 GB (near the card's limit) | 6.14 GB |
| **peak output throughput** | **5,983 tok/s @ c192** | **6,987 tok/s @ c512 (1.168×)** |

fp16 physically caps at 221 concurrent (14.4 GB weights leave the pool room for only
221 × 33 MB); above that it OOMs. w8a8 (7.75 GB weights) serves 512 concurrent and
its throughput is **still climbing at 512** (6,275 @320 → 6,660 @448 → 6,987 @512),
so 6,987 is a floor on its advantage, not a ceiling.

## The honest mechanism (what int8 does and does NOT do here)

At **matched concurrency ≤ 221**, fp16 is *faster per step* (e.g. c192: fp16 5,983 vs
w8a8 5,008; at these moderate batch sizes the GEMM M is in w8a8's weaker regime and
it also pays the activation-quant tax). w8a8 wins **only** by having the VRAM headroom
to run at concurrency fp16 cannot reach. So the result is not "int8 makes each step
faster at 7.2B" — it is:

> On a single 32 GB 5090, w8a8 serves RWKV-7 7.2B at **2.32× the concurrency** and a
> **16.8% higher peak throughput** than fp16 can — because fp16 is pinned against the
> card's memory limit at 221 concurrent, and only int8 frees enough room for the
> recurrent-state pool to go further.

This is a structural, VRAM-driven fact, not a throughput-saturation artifact: it holds
regardless of kernel tuning, and it is exactly the axis on which int8 is unreachable
by fp16 on consumer Blackwell.

## Combined with F0033/F0034 — the full sm120 int8 story, honestly scoped

Upstream cutlass `int8_scaled_mm` does not exist on sm120 at all, so no other RWKV
serving stack has *any* int8 on consumer Blackwell. rwkv-sglang's hand-written s8-wmma
kernel (V2): (1) beats fp16 cuBLAS on the GEMM itself at M≥512 (1.03–1.55×, F0034);
(2) is greedy lambada-certified (0.6486 vs 0.6509 cutlass); (3) halves weight VRAM;
and now (4) on 7.2B turns that VRAM into **2.32× concurrency + 16.8% higher peak** than
fp16 can reach on a 32 GB 5090.

## Cross-references

`bench/results/72b/sweep_72b_{fp16,w8a8,w8a8_max}.json` (raw sweeps) · F0033 (s8 probe)
· F0034 (V2 GEMM + quant tax) · BENCHMARKS §4.
