# F0032 — Equal-conditions comparison with vllm-rwkv, and the synthetic-vs-real-load reversal

**Date:** 2026-07-06 · **Status:** COMPLETE (both GPUs, all raw committed) · **Prior:** F0007 (albatross scope), F0031 (spec-decode)

## What was measured

rwkv-sglang (sglang main + full kernel stack) vs rwkv-rs/vllm-rwkv (their fork, commit
`4bf0239a1`, Albatross-ported kernels + vLLM V2 runner + decode-wave scheduler), under
conditions built to survive hostile review:

- Same GPUs (RTX 3090, RTX 5090), each engine measured on both.
- Same weights: `rwkv7-g1g-1.5b-20260526-ctx8192.pth`, verified **tensor-bitwise equal** to
  our fla-format directory (converter committed; vllm-rwkv only loads raw BlinkDL .pth,
  with the model config parsed from the filename).
- Same client per protocol: the synthetic sweep used one client ported to their OpenAI
  endpoint (identical 64-in/256-out, temp 0, ignore_eos, same request-count math); the
  real-load run used one neutral `sglang.bench_serving` invocation for both engines —
  **all 8 real-load runs processed exactly 168,913 input / 109,861 output tokens.**
- Each engine at its documented best config. Disclosure: vllm-rwkv's tip crashes on the
  first decode as shipped (interface drift from its automated daily upstream rebase, no
  test coverage); a documented 2-line compat shim was required to measure it at all.

## Correctness

vllm-rwkv reproduces our numpy fp32 oracle fixture **24/24 token-exact on both GPUs**,
byte-identical across sm86/sm120. Two independent engines converging on one reference is
mutual validation — recorded plainly to their credit.

## Results (full tables in BENCHMARKS §7b/7c; raw in bench/results/{vllmrwkv,realload}/)

**Synthetic fixed-shape sweep (64-in/256-out):** vllm-rwkv leads bsz1 on the 5090 (1.1352×)
and high concurrency on both cards (up to 1.2621× at 5090 c512); rwkv-sglang leads bsz1 on
the 3090 (0.8114) and the c8–64 middle. A disputed first 3090 run was re-measured clean
(max_num_seqs explicitly sized to 24 GB, clocks logged 210→1980 MHz): **the re-run
reproduced the original within ~2% (slightly lower)** — the original was valid; the
3090/5090 asymmetry is a hardware effect (bandwidth favors fused-layer kernels), not a
config artifact.

**Real-load ShareGPT (variable-length, the realistic test):** the conclusion REVERSES.
At peak load rwkv-sglang leads output throughput on BOTH cards — **9,602 vs 8,865 tok/s
(1.0832×) on the 5090; 3,974 vs 2,805 (1.4168×) on the 3090** — with lower median TTFT
(2,503 vs 3,458 ms; 7,297 vs 12,750 ms). At light steady load (5090 @16 req/s) the two
converge within a few percent and trade latency tails. (3090 "16 req/s" is mild overload
for both engines — ~11–12 req/s achieved — flagged in the tables.)

## The finding that matters beyond this comparison

**A fixed-shape synthetic sweep and a variable-length real workload ranked the two engines
in opposite order at high load.** Mechanism: vllm-rwkv's decode-wave scheduler + Albatross
batch kernels excel when every request has the same shape (the synthetic case); sglang's
continuous dynamic batching packs uneven, staggered requests with fewer bubbles (the real
case). Neither number is wrong — they answer different questions. Methodology rule adopted:
**a serving comparison is incomplete without a variable-length workload**, and any claim of
high-concurrency superiority must state which regime it was measured in.

Secondary notes for the record: our int8 w8a8 peak on the 3090 (9,851) exceeds vllm-rwkv's
fp16 peak (8,583) — quantization is a lever they currently lack; on the 5090 that lever
waits on an int8×int8 MMA path for sm120 (upstream cutlass gap; our current w8 TC kernel
does fp16 MMAs, no FLOP win at large M). vllm-rwkv repo facts relevant to reproduction:
daily force-push rebase (pin commits), fp16 compute hardcoded, no committed performance
numbers of its own as of this date.

## Cross-references

BENCHMARKS.md §7b/7c · `bench/results/vllmrwkv/` (sweeps + correctness incl. token ids;
`*_3090_v2.json` = the clean re-measure) · `bench/results/realload/` (8 runs + summaries)
· competitive notes in the private survey repo.
