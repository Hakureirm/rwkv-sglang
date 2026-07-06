# F0036 — PP + cuda-graph was broken on main (v_first proxy); fix VERIFIED + first TP/PP production throughput

**Date:** 2026-07-06 · **Status:** FIXED + VERIFIED (2×L4, cuda-graph ON, greedy 24/24) + merged into PR #30115 · **Prior:** F0019 (TP/PP correctness, v0.5.10, cuda-graph OFF)

## What the TP/PP audit uncovered

The user asked whether TP/PP was stale. It was: F0019 verified TP 2/4/8 + PP 2/4/8 +
tp2×pp2 **greedy-exact on real multi-GPU, but with cuda-graph OFF** (functional gate
config — the tok/s there are gate numbers, not production throughput). Nobody had ever
run **PP with cuda-graph ON**. First attempt on 2×L4 (pp=2, production config)
crashed during decode cuda-graph capture:

```
rwkv7.py:558  v_first = pp_proxy_tensors["v_first"]
→ KeyError: 'v_first'   (in decode_cuda_graph_runner capture)
```

## Root cause

RWKV-7 hands **two** tensors across the PP stage boundary in `PPProxyTensors`:
`hidden_states` and `v_first` (layer-0's value projection, which every later layer's
v-residual mix consumes). But the decode cuda-graph buffer allocator hardcoded the proxy
keys to `{hidden_states, residual}` (+ `topk_indices` via the `get_pp_proxy_topk_size`
hook). So on a non-first PP rank the captured graph read `pp_proxy_tensors["v_first"]`
from a buffer that had no such slot → KeyError at capture. cuda-graph requires a stable
input pointer, so the model cannot self-provide the slot — the runner must allocate it.
F0019's PP verification ran cuda-graph OFF, which is exactly why this never surfaced.

**Impact:** PR #30115's PP support was cuda-graph-OFF only; production (graph ON) PP
crashed. TP-only was unaffected (no PP proxy).

## Fix (mirrors the existing topk_indices mechanism, ~4 small core edits)

- `model_runner.get_pp_proxy_v_first_size()` — returns `hidden_size` for RWKV-7 on
  non-first PP ranks, `None` otherwise (parallel to `get_pp_proxy_topk_size`).
- `runner_utils/buffers.py` — allocate a persistent `v_first` slot `(max_bs, hidden_size)`
  in the decode buffer when that size is set, so capture/replay share a stable pointer.
- Both proxy fill paths (`populate_from_forward_batch` and the cuda-graph buffer
  registry's `source_fn`) now source a buffer key the running model does not send this
  step (e.g. the default `residual` slot for a model that hands `v_first` instead) to
  `None` and skip it, instead of `KeyError`.

Existing PP models (llama-style `{hidden_states, residual}`, DeepSeek `topk_indices`) are
unaffected: the guard is a no-op when every buffer key is present in the real proxy, and
`get_pp_proxy_v_first_size` returns `None` for them.

## Verification

All four edited files compile. **Single-GPU non-regression confirmed on the 5090 tower**
(fix branch, tp=1, cuda-graph ON: boots + greedy 24/24 — the 4 edits are inert for
non-PP paths, all gated on pp_size>1). Multi-GPU pp=2 correctness/throughput (the fix's
actual target) is the remaining gate; it runs on 2×L4.

Infra note (cost/repro): the multi-GPU verify was blocked for several rounds by a drifted
`dev-cu12` base image whose scheduler 503s at startup **even at tp=1** (unrelated to this
fix — reproduced with the inert-for-tp1 code). Pinning the base to the digest verified
working on the tower (`sha256:49627efd…`) resolved it. Lesson: pin the serving base image
for multi-GPU CI, and capture results to a local file (the progress-spinner ANSI mangles
piped stdout).

## Verified result (2×L4, 1.5B bf16, cuda-graph ON, wall-clock tok/s)

| config | greedy vs tp=1 | deterministic | c1 | c8 | c32 | c64 (peak) |
|---|---|---|---|---|---|---|
| tp=1 (1 GPU) | (reference) | yes | 72.6 | 482.3 | 1,612.9 | 2,582.6 |
| **tp=2 (2 GPU)** | **24/24 exact** | yes | 105.3 | 655.9 | 2,008.6 | **3,026.2** (1.17×) |
| **pp=2 (2 GPU)** | **24/24 exact** | yes | 65.4 | 367.7 | 1,365.5 | **2,288.8** (0.89×) |

**pp=2 now boots and runs correctly with cuda-graph ON** (previously KeyError'd at capture)
— the fix works. Both TP=2 and PP=2 are greedy-exact vs single-GPU and deterministic:
multi-GPU does not change the output, now under the production cuda-graph path. Honest
throughput read (1.5B is small and L4 interconnect is PCIe, not NVLink): TP=2 buys ~1.17×
at c64; PP=2 is 0.89× (pipeline bubbles dominate at this model size — PP's role is fitting
models larger than one card, not per-token speedup at 1.5B). These are the first TP/PP
numbers on main under cuda-graph ON; F0019's were cuda-graph OFF gate configs. Raw:
`bench/results/tppp_l4_main.json`.

## Cross-references

`python/sglang/srt/model_executor/{model_runner,runner_utils/buffers,runner/decode_cuda_graph_runner,cuda_graph_buffer_registry}.py`
(on the PR fork) · F0019 (TP/PP correctness) · project-upstream-model-pr memory.
