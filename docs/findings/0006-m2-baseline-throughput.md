---
doc_kind: finding
finding_id: F0006
title: "M2-baseline — bf16 + 1.5B exact greedy-match; throughput baseline; decode is eager-bound"
last_verified_commit: "b92fd10"
discovered_by: M2-baseline agent + lead, 2026-06-30
severity: info
status: open
related: [F0005, F0003]
---

# Finding F0006: M2-baseline (perf dtype, scale, throughput)

## Hypothesis
M1 correctness (0.1B fp32) should hold at bf16 and at larger sizes; throughput
profiling will show where the albatross speed gap is.

## Method
sglang Engine on converted fla checkpoints (cuda-graph OFF). Greedy vs the pure-
numpy fp32 oracle (`bench/oracle_numpy.py`). `bench/throughput.py` (new) measures
decode tok/s (128-tok steady-state, aggregate over batch), prefill tok/s (1024-tok
prompt, TTFT), peak VRAM (via nvidia-smi — Engine runs the model in a subprocess
so in-proc `max_memory_allocated` reads ~0).

## Result (RTX 3090)
### Correctness — all EXACT 24/24 vs oracle
- 0.1B: fp32 EXACT (regression), **bf16 EXACT** (no token flip).
- 1.5B (`rwkv7-g1g-1.5b`, 24L/2048d/32heads/64hd, 795 tensors): fp32 EXACT, bf16
  EXACT. VRAM ~10.1GB (fp32) / ~8.6GB (bf16) — fits 3090 with headroom.
- Converter + kernels + model needed NO changes to scale 0.1B→1.5B (dims auto-derived;
  note head count DOES vary by size — derive from r_k, never hardcode).

### Throughput (bf16, cuda-graph OFF; decode tok/s aggregate over batch)
| model | bsz | decode tok/s | prefill tok/s (chunk) | peak VRAM (alloc policy) |
|---|---|---|---|---|
| 0.1B | 1 | 20.6 | 8116 | ~7.5GB |
| 0.1B | 8 | 168 | 58288 | ~7.5GB |
| 0.1B | 32 | 665 | 190931 | ~7.5GB |
| 1.5B | 1 | 10.5 | 4149 | ~9.0GB |
| 1.5B | 8 | 78.6 | 31455 | ~9.0GB |
| 1.5B | 32 | 338 | 102490 | ~9.0GB |
(`recurrent` prefill: similar decode; lower low-batch prefill — chunk wins TTFT at
low bsz, recurrent edges ahead at bsz32. Decode identical across both ±2%.)

## Conclusion
**Correctness is solid (exact at fp32+bf16, 0.1B+1.5B).** The speed story: prefill
is healthy; **decode is launch/overhead-bound in eager mode** (bsz1 ~20 tok/s vs
albatross ~145). Near-linear batch scaling (bsz1→32 ≈30×) confirms per-step Python/
launch overhead dominates single-stream decode. ⇒ **cuda-graph (M2b) is the single
highest-leverage decode-speed fix** before any albatross comparison is meaningful.
VRAM is flat across bsz (state pool pre-allocated; constant-size recurrent state —
the RWKV property) — meeting the "constant VRAM" design goal.

## Next
- **M2b cuda-graph** for decode (UNIFORM_SINGLE_TOKEN_DECODE; make token_shift conv
  update + recurrence graph-capturable; stable index buffers). Top priority for speed.
- Then M3: run albatross on the same 3090, compare apples-to-apples.

## Cross-references
[[F0005]] M1 · [[F0003]] acceptance grid · `bench/throughput.py`, `bench/verify_m1d.py`.
