---
doc_kind: finding
finding_id: F0008
title: "M2b — CUDA graph for RWKV-7 decode: 7.5-21× speedup, exact, gap vs albatross → ~2-3×"
last_verified_commit: "6ebbe0a"
discovered_by: M2b agent + lead independent verify, 2026-06-30
severity: info
status: open
related: [F0006, F0007]
---

# Finding F0008: M2b CUDA-graph decode

## Hypothesis
The ~30-57× decode gap vs albatross (F0007) is mostly eager-mode launch overhead;
CUDA graph should recover most of it without touching the kernels.

## Method
Boot sglang without `--disable-cuda-graph` (+ `--cuda-graph-max-bs` for bsz>24).
Verify greedy still exact (`bench/verify_m1d.py --cuda-graph`), measure decode tok/s
(`bench/throughput.py --cuda-graph`). Lead independently re-ran the correctness gate.

## Result (RTX 3090, bf16, steady-state decode)
| model | bsz | decode OFF | decode ON | speedup | albatross | residual gap |
|---|---|---|---|---|---|---|
| 0.1B | 1 | 20.5 | **436.6** | 21.3× | 1171.6 | ~2.7× |
| 0.1B | 8 | 169.6 | 3213.0 | 18.9× | – | – |
| 0.1B | 32 | 663.1 | 9869.5 | 14.9× | 24522 | ~2.5× |
| 1.5B | 1 | 11.4 | **142.0** | 12.5× | 309.1 | ~2.2× |
| 1.5B | 32 | 350.3 | 2644.7 | 7.5× | – | – |
- Correctness with cuda-graph ON: **EXACT 24/24** for 0.1B AND 1.5B bf16
  (lead-verified independently). VRAM cost of graphs ~124-200 MiB.

## Key facts / decisions
- **Zero code changes needed** — `Rwkv7AttnBackend(MambaAttnBackendBase)` inherits
  `init_forward_metadata_capture/replay_cuda_graph` → metadata buffers
  (`query_start_loc`, `mamba_cache_indices`) are already address-stable; our T==1
  decode path (token_shift + fused_mul_recurrent_rwkv7) has no host syncs. (The
  `AttentionCGSupport` enum the plan referenced doesn't exist in 0.5.10.post1;
  capture is gated purely by the inherited hooks.) "Enabling" = a launch flag.
- **Launch recipe**: do NOT pass `--disable-cuda-graph`; set `--cuda-graph-max-bs
  >= peak concurrency` (default auto-clamps to ~24 on a 24GB GPU → bsz>24 silently
  runs eager).
- ⚠️ **RADIX CACHE CORRECTNESS (production blocker)**: B≥3 identical-prompt requests
  can diverge — RWKV's per-request recurrent state is NOT prefix-cacheable, but the
  token radix cache shares identical prefixes. Reproduces with cuda-graph OFF too
  (orthogonal to M2b); intermittent; **gone with `disable_radix_cache=True`**. ⇒
  **RWKV serving must force `disable_radix_cache` until MambaRadixCache is wired**
  (the M2 prefix-cache-state-fit open question). cuda-graph introduces NO regression.

## Conclusion
cuda-graph collapses the decode gap vs albatross from ~30-57× to **~2-3×**. The
residual is now pure **kernel-quality** (albatross fused fp16 WMMA/cublasLt vs our
per-op triton) — the F0007 kernel-vendoring route (M3b). Prefill gap (3.5-8.6×)
unchanged (extend isn't graphed) — also kernel-quality.

## Next
1. **Production correctness**: force `disable_radix_cache` for RWKV (safe default) +
   later wire MambaRadixCache for proper state-aware prefix reuse.
2. **M3b kernel vendoring**: albatross fp16 WKV/linear kernels under sglang → close
   the residual ~2-3× decode + 3.5-8.6× prefill toward true parity.
3. 7.2B correctness + full ours-vs-albatross table (production config) + lm-eval.

## Cross-references
[[F0006]] [[F0007]] · `bench/verify_m1d.py --cuda-graph`, `bench/throughput.py --cuda-graph`.
