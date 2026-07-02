---
doc_kind: finding
finding_id: F0012
title: "Multi-GPU coverage — greedy-EXACT on T4/L4/A10G/A100/L40S/H100/H200, no per-arch change; int4 runs on all incl. Turing"
last_verified_commit: "HEAD"
discovered_by: multi-GPU sweep + lead independent T4 re-run, 2026-06-30 (int4 all-card added 2026-07-02)
severity: info
status: open
related: [F0005, F0011, F0017]
---

# Finding F0012: Multi-GPU coverage

## Hypothesis
Goal: broad GPU coverage (consumer + datacenter) — 支持全部常见专业和消费卡. Our triton-kernel + sglang deliverable should
run correctly on all common consumer + datacenter GPU architectures with no per-arch code.

## Method
Cross-GPU sweep on a real instance of each card: image = CUDA 12.4 devel + sglang 0.5.10.post1 +
our overlay (== `deploy.sh`); model = the same BlinkDL `.pth` the fixtures came from
(`BlinkDL/rwkv7-g1`), converted to fla. Correctness gate per GPU = `bench/verify_m1d.py` greedy
**EXACT** vs the numpy-oracle fixture (bf16 + cuda-graph, radix off = production config); speed =
`bench/throughput.py`. Lead independently re-ran T4. Full tables + raw JSON:
[`../../bench/results/multigpu.md`](../../bench/results/multigpu.md) + `allcards.json`.

## Result — greedy-EXACT on EVERY architecture (no per-arch code change)
1.5B, bf16 + cuda-graph; decode tok/s bsz1/8/32; int4 (our hand-written GEMV) decode bsz1:

| GPU | arch (sm) | bf16 greedy | bf16 decode 1/8/32 | int4 bsz1 (vs bf16) |
|---|---|---|---|---|
| T4 | Turing 7.5 | **24/24** (lead-verified) | 65 / 447 / 592 | 115 (**1.77×**) |
| L4 | Ada 8.9 | **24/24** | 76 / 521 / 737 | 155 (**2.04×**) |
| A10G | Ampere 8.6 | **24/24** | 105 / 767 / 986 | 198 (**1.88×**) |
| A100-40GB | Ampere 8.0 | **24/24** | 162 / 1223 / 4370 | 205 (1.27×) |
| A100-80GB | Ampere 8.0 | **24/24** | 166 / 1341 / 4417 | 205 (1.23×) |
| L40S | Ada 8.9 | **24/24** | 171 / 1090 / 4150 | 288 (**1.68×**) |
| H100 | Hopper 9.0 | **24/24** | 230 / 1788 / 6569 | 261 (1.14×) |
| H200 | Hopper 9.0 | **24/24** | 242 / 1875 / 6938 | 263 (1.09×) |

- **bf16 correctness held on Turing / Ampere(80,86) / Ada / Hopper** — the WKV + fused-glue triton
  kernels JIT-compiled + ran on sm75/80/86/89/90 with NO per-arch change (only image deps:
  `libnuma1` for sgl_kernel + `CPATH`→headers for triton JIT). ⇒ **broad-GPU-coverage goal met**.
- **int4 runs on ALL 8 incl. Turing (sm7.5)** — the kernel has no `cp.async` (not limited to sm80+);
  bsz1 faster than bf16 everywhere, biggest on bandwidth-starved cards (see F0017 + multigpu.md §1).
- **int8** runs on Ampere/Ada/Hopper; ~neutral vs bf16 at 1.5B on Hopper (bf16 saturates), decode
  win is on Ampere consumer (F0011). **fp8 (H100): BLOCKED** — the strict `load_weights` rejects
  sglang's runtime fp8 `weight_scale` params (int8 works via its offline scale-baking converter; no
  fp8 converter exists yet).

## Conclusion
The deliverable is **portable across all common consumer + pro NVIDIA GPUs** (Turing→Hopper),
greedy-EXACT everywhere, and the hand-written int4 path runs (and speeds up) on every one of them —
the broad-GPU-coverage goal is satisfied + reproducible. fp8 is the one gap (needs an fp8 weight-scale converter).

## Cross-references
[[F0005]] (correctness) · [[F0011]] (int8) · [[F0017]] (int4) · `bench/results/multigpu.md`.
