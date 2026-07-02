---
doc_kind: finding
finding_id: F0005
title: "M1 complete — RWKV-7 0.1B runs in sglang, exact greedy-match vs oracle"
last_verified_commit: "700e554"
discovered_by: M1a/M1b/M1c+M1d implementer agents + lead independent verify, 2026-06-30
severity: info
status: closed_by_M1
related: [F0002, F0003, F0004]
---

# Finding F0005: M1 complete (RWKV-7 0.1B × sglang correctness)

## Hypothesis
RWKV-7 can be served correctly in sglang v0.5.10.post1 by reusing the Mamba/
hybrid-linear plumbing + vendored fla rwkv7 triton kernels, matching the
rwkv-lm/numpy reference exactly (ADR-0001/0002/0003).

## Method
Sliced per ADR-0003: M1a vendor kernels → M1b BlinkDL→fla converter + numpy oracle
→ M1c model+backend+config+wiring (boot) → M1d load+greedy-match. Lead-independent
verification via `bench/verify_m1d.py` (separate from the implementer's harness).

## Result (verified, HEAD 700e554)
- **Boot**: `sgl.Engine(rwkv7-0.1b-fla, skip_tokenizer_init, disable_cuda_graph,
  disable_piecewise_cuda_graph, dtype=float32, tp_size=1)` → BOOTED. MambaPool
  allocates `temporal (12,64,64)` + two `(768,1)` conv, all fp32; 399 weights
  loaded; routed through `Rwkv7AttnBackend` (HybridLinearAttnBackend, full_attn=[]).
- **Greedy match (M1d gate)**: sglang greedy temp=0 output == numpy/BlinkDL oracle
  **token-for-token** (24/24, EXACT_MATCH True, no divergence) for the Eiffel
  fixture. Holds for BOTH recurrent and chunk(tf32) prefill kernels. Batch-2
  identical-request slot isolation: both exact.
- **Kernel gate (M1a)**: decode kernel ~1.3e-6 vs naive; chunk ≤8.1e-3 (tf32).
- **Converter (M1b)**: 0.1B .pth → 399 fla tensors, shapes identical to fla-hub.

## Key integration decisions (non-obvious)
- **`scale=1.0`** in the recurrence (NOT `K**-0.5`): the reference does `o=S@r`
  with no r-scaling, and GroupNorm eps (64e-5) breaks scale-invariance → 1.0 needed
  for exact match.
- **Token-shift = two `(768,1)` MambaPool conv states** (attn/ffn), width-2 causal
  (prev-token); fp32 (bf16 default corrupts shift → breaks greedy). `memory_pool.py`
  already supports a multi-entry conv list (no edit needed).
- **Module names mirror fla keys** → `load_weights` uses default_weight_loader,
  no remap (399/399 consumed). Class `RWKV7ForCausalLM` (matches config.json
  `architectures`) + `Rwkv7ForCausalLM` both in EntryClass.
- **Two unavoidable sglang-core edits for the all-linear (zero full-attn) case**:
  `model_runner_kv_cache_mixin.profile_max_num_token` div-by-zero guard
  (`cell_size==0`), and a `Rwkv7NoOpFullAttnBackend` (real full backends probe the
  empty full-KV pool / reject fp32 at construction). No upstream all-attention-free
  model exists, so sglang core never handled this.
- **Boot flags**: BOTH `disable_cuda_graph=True` AND `disable_piecewise_cuda_graph
  =True` required for M1.

## Conclusion
rwkv-lm accuracy parity achieved for 0.1B (exact). M1 closed. Remaining goals:
scale to larger sizes + bf16; serving features under load (dynamic batching,
chunked prefill, state cache) validated; **albatross speed/VRAM parity** (the hard
part); quant 8/4-bit; broad GPU coverage (consumer + datacenter). See snapshot §"Next".

## Cross-references
ADR-0001/0002/0003 · [[F0002]] [[F0003]] [[F0004]] · gates `bench/oracle_numpy.py`,
`bench/verify_m1d.py` · deliverable `sglang_overlay/` + `tools/convert_rwkv7_blinkdl_to_fla.py`.
