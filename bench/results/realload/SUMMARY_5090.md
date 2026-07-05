# RWKV-7 1.5B — ShareGPT real-workload head-to-head: rwkv-sglang vs vllm-rwkv

Equal-conditions serving benchmark on ShareGPT (variable-length real conversations),
measured fresh on both GPUs 2026-07-06. Complements the synthetic 64-in/256-out sweep.

## Method (identical for both engines)
- Neutral client: `python -m sglang.bench_serving` (stock sglang, run from the vllmrwkv
  container base python on both machines), driving each server over /v1/completions
  (--backend sglang-oai for rwkv-sglang, --backend vllm for vllm-rwkv).
- Identical invocation both engines: --dataset-name sharegpt (same sharegpt.json, md5
  8d2f1dcd711aaa227cf46aecfbcfb262 on all machines), --num-prompts 500,
  --sharegpt-context-len 8192, --seed 42, default ignore_eos ON,
  --tokenizer <rwkv7-1.5b-fla dir> (ONE client tokenizer for both engines).
- Rates: --request-rate inf (peak) and 16 (steady). Two runs per engine per machine.
- Same weights: rwkv-sglang = fla dir rwkv7-1.5b-fla; vllm-rwkv =
  rwkv7-g1g-1.5b-20260526-ctx8192.pth (tensor-verified equal).
- Best config each: rwkv-sglang = scripts/serve.sh throughput (full hand-written kernel
  stack, fp16, --cuda-graph-max-bs 512, --mem-fraction-static 0.85,
  --max-running-requests 512, radix off). vllm-rwkv = RUNBOOK recipe
  (VLLM_USE_V2_MODEL_RUNNER=1, --gpu-memory-utilization 0.70, --max-num-batched-tokens 8192)
  with --max-num-seqs 384 (sized to VRAM, NOT the RUNBOOK's example 64).
- Both vllm-rwkv trees carry the 2-line postprocess_state compat shim (verified present in
  vllm/v1/worker/gpu/model_states/rwkv.py on both machines; pristine 4bf0239a1 TypeErrors
  on first decode without it).

### Equal-conditions proof
All 8 runs (2 engines x 2 machines x 2 rates) processed EXACTLY 168,913 input tokens and
generated EXACTLY 109,861 output tokens. Same prompts in, same tokens out (identical
weights + ignore_eos + greedy temp=0). All deltas below are pure engine performance.
Tokenizer parity: one client tokenizer (rwkv trie, fla dir) used for both; vllm-rwkv
server-side retokenized output (109,741) agrees with the fla count to 99.9%.

## 5090 (tower) — output/total tok/s; TTFT & TPOT in ms
| Engine       | Load      | Output tok/s | Total tok/s | Med TTFT | P99 TTFT | Med TPOT | P99 TPOT |
|--------------|-----------|-------------:|------------:|---------:|---------:|---------:|---------:|
| rwkv-sglang  | peak(inf) |      9602.17 |    24365.67 |  2502.63 |  4525.32 |    21.28 |   493.49 |
| rwkv-sglang  | 16 req/s  |      3300.08 |     8374.01 |    31.60 |   166.25 |     5.49 |     9.86 |
| vllm-rwkv    | peak(inf) |      8865.42 |    22496.14 |  3457.83 |  6257.51 |    25.89 |   355.61 |
| vllm-rwkv    | 16 req/s  |      3350.93 |     8503.05 |    24.10 |   155.99 |     4.99 |     8.80 |
(request throughput req/s: sglang 43.70 / 15.02 ; vllm 40.35 / 15.25)

## 3090 (box) — output/total tok/s; TTFT & TPOT in ms
| Engine       | Load      | Output tok/s | Total tok/s | Med TTFT  | P99 TTFT  | Med TPOT | P99 TPOT |
|--------------|-----------|-------------:|------------:|----------:|----------:|---------:|---------:|
| rwkv-sglang  | peak(inf) |      3974.28 |    10084.79 |   7297.18 |  12696.94 |    68.99 |  1400.49 |
| rwkv-sglang  | 16 req/s  |      2476.67 |     6284.60 |    315.72 |   2360.69 |    91.37 |   465.91 |
| vllm-rwkv    | peak(inf) |      2804.82 |     7117.28 |  12750.48 |  23320.89 |    86.51 |  1521.17 |
| vllm-rwkv    | 16 req/s  |      2599.95 |     6597.42 |    374.98 |   1208.20 |    61.81 |   219.24 |
(request throughput req/s: sglang 18.09 / 11.27 ; vllm 12.77 / 11.83)
NOTE: 16 req/s OVERLOADS the 3090 for both engines (achieved 11.3 / 11.8 req/s < 16), so
the 3090 "16 req/s" row is a mild-overload regime, not true steady-state. On the 5090,
16 req/s is comfortably steady (both ~15 req/s, low latency).

## Plain-language read
- Throughput leader: rwkv-sglang. Peak output tok/s: 5090 9602 vs 8865 (+8.3%);
  3090 3974 vs 2805 (+41.7%). The lead is much larger on the 3090.
- Latency: split by regime.
  * PEAK (both machines): rwkv-sglang leads on TTFT (median & p99) and median TPOT.
    vllm-rwkv has the lower p99 TPOT tail on the 5090 (356 vs 493) but not on the 3090.
  * STEADY 16 req/s on the 5090 (true steady): near-tie; vllm-rwkv marginally lower TTFT
    (24 vs 32 med) and TPOT (5.0 vs 5.5 med). rwkv-sglang lower ITL. Throughput ~equal
    (offered-load-bound).
  * 16 req/s on the 3090 (overload): vllm-rwkv better TPOT (62 vs 91 med, 219 vs 466 p99)
    and p99 TTFT (1208 vs 2361); rwkv-sglang better median TTFT (316 vs 375). Mixed.
- Peak vs steady: at peak both engines saturate; rwkv-sglang wins throughput and cold-start
  latency. Under light steady load (5090 @16) the engines converge and vllm-rwkv holds a
  slim latency edge.

## Corrected synthetic sweep on the 3090 (vllm-rwkv, in64/out256)
Clean re-run on the properly-sized max-num-seqs-384 server (single clean server, all
concurrencies), vs the old box numbers.
| Concurrency | v2 out_tok/s | old out_tok/s | delta   |
|-------------|-------------:|--------------:|:--------|
| 1           |        187.2 |         190.1 | -1.5%   |
| 8           |        906.1 |         918.0 | -1.3%   |
| 32          |       3370.7 |        3361.7 | +0.3%   |
| 64          |       5163.1 |        5365.0 | -3.8%   |
| 128         |       6628.3 |        6921.3 | -4.2%   |
| 256         |       7841.1 |        8130.6 | -3.6%   |
| 384 (peak)  |       8492.8 |        8583.2 | -1.1%   |
FINDING: the corrected sweep REPRODUCES the old numbers within run-to-run noise (v2 is
1-4% LOWER, not higher). The old box numbers were valid, NOT wedge-degraded — the old
large phase already used max-num-seqs 384. Files: their_sweep_{small,large}_3090_v2.json.

## 3090 health note
Idle SM clock 210 MHz (normal downclock). Under load it sustains 1695 MHz and peaks
1980 MHz (verified from a continuous nvidia-smi log across the whole run) — confirms no
wedge residue. vllm-rwkv built 43 FULL decode CUDA graphs up to size 512 (0.73 GiB) and
the engine ran up to 381 concurrent requests at peak (not under-configured).

## Config asymmetries accepted (each engine at its own recommended best)
- rwkv-sglang mem-fraction 0.85 vs vllm-rwkv gpu-mem-util 0.70 (each engine's recipe default).
- rwkv-sglang concurrency ceiling 512 (serve.sh throughput, fixed) vs vllm-rwkv 384 (sized
  to VRAM / sweep top). Both far exceed the 16-req/s need; at peak both saturate on the
  500-prompt burst (sglang ran ~290-337 avg concurrency, vllm ~315-352).
- Client is stock sglang bench_serving for BOTH engines; a benign non-fatal warning
  ("rwkv7-g1g-1.5b is not a local folder") is a secondary tokenizer-name probe that does
  not affect measurement (the --tokenizer fla dir is what tokenizes; token counts match).

## Raw files
Tower /data/bench5090/realload/ ; Box ~/vllmrwkv-data/realload/ (+ results/ for sweeps):
{sglang,vllm}_{5090,3090}_{inf,r16}.{json,stdout.txt}, {sglang,vllm}_server_*.log,
driver_*.log, clocks_3090.csv, their_sweep_{small,large}_3090_v2.json.
