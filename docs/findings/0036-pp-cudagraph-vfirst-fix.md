# F0036 — PP + cuda-graph was broken on main (v_first proxy); fix + first TP/PP production throughput

**Date:** 2026-07-06 · **Status:** FIX WRITTEN + compiles; 2×L4 re-verify in flight · **Prior:** F0019 (TP/PP correctness, v0.5.10, cuda-graph OFF)

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

All four edited files compile. Correctness/throughput on 2×L4 (pp=2, cuda-graph ON,
greedy vs tp=1) is the gate — running on branch `rwkv7-pp-cudagraph-fix`; folds into
PR #30115 and unblocks the first TP/PP **production** throughput numbers (req#4, which had
correctness but zero cuda-graph-ON serving data).

## Cross-references

`python/sglang/srt/model_executor/{model_runner,runner_utils/buffers,runner/decode_cuda_graph_runner,cuda_graph_buffer_registry}.py`
(on the PR fork) · F0019 (TP/PP correctness) · project-upstream-model-pr memory.
