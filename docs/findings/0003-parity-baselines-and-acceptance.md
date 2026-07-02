---
doc_kind: finding
finding_id: F0003
title: "Parity baselines (rwkv-lm, albatross) & acceptance test definition"
last_verified_commit: (initial)
discovered_by: recon (P10, 2026-06-30)
severity: info
status: open
related: [F0001, F0002]
---

# Finding F0003: Parity baselines & acceptance test definition

## Hypothesis
The project goal "达到 rwkv-lm 和 albatross 的 性能/速度/精度/显存" (match rwkv-lm and
albatross on performance / speed / accuracy / VRAM) needs to be turned into a
concrete, measurable acceptance test on our single 3090.

## Method
Identified the two baselines and their reported numbers (web + READMEs). Defined
a falsifiable acceptance grid. (Albatross/rwkv-lm deep numbers to be re-confirmed
by the running `rwkv7-latest-upstream-recon` workflow.)

## Result

### Baselines
- **rwkv-lm** = `github.com/BlinkDL/RWKV-LM` (`rwkv` pip pkg / ChatRWKV same
  lineage) → the **accuracy / numerical-correctness** reference. Concretely the
  oracle = `rwkv` pip in **cuda fp16, `RWKV_V7_ON=1`** + the pure-numpy forward
  `RWKV-v7/rwkv_v7_numpy.py` (bit-level logits). **⚠️ `fla` is NOT aligned with
  this reference (README: "performance is quite worse") → do NOT use fla as the
  accuracy oracle** (corrected by [[F0004]]).
- **albatross** = `github.com/BlinkDL/Albatross`, BlinkDL's efficient RWKV
  inference engine (custom CUDA kernels + fast sampling) → the **speed / VRAM**
  reference. Reported (RWKV-7 7.2B fp16, single **RTX 5090**):
  - batched decode ~10250 tok/s; bsz1 decode ~145 tok/s; bsz1 prefill ~11289
    tok/s; bsz320 decode ~5848 tok/s with **constant speed & VRAM**.
  - NOTE: those are **5090** numbers. For a fair bar we must run albatross on the
    **same 3090** and compare against our vLLM impl on that 3090.

### Tokenizer
RWKV "World" tokenizer is a **custom trie tokenizer** (not HF BPE). vLLM needs a
compatible tokenizer wrapper (confirm whether an HF-format `rwkv_tokenizer`/
`fast` variant exists; else wrap the rwkv-world trie). Tracked as open question.

### Acceptance test grid (run all on the SAME 3090)
| Axis | Metric | Pass bar |
|---|---|---|
| Accuracy | greedy-token match vs rwkv-lm over N≥1000 prompts; + lm-eval (lambada/piqa) delta | exact-greedy match (fp32 ref) ; lm-eval Δ within noise |
| Decode speed | tok/s @ bsz ∈ {1,16,64,320} | ≥ albatross-on-3090 (target), report ratio |
| Prefill speed | tok/s (long-ctx chunked prefill) | ≥ albatross-on-3090 (target) |
| VRAM | peak GB @ each bsz; constant-vs-ctx-len check | ≤ albatross + constant w.r.t. ctx (RWKV property) |
| Quant | 8/4-bit VRAM ↓ ; speed ≥ 16-bit | VRAM down, not slower than fp16 |
| Sizes | RWKV-7 World/G1 0.1B → as large as 3090 fits (≈7.2B fp16, larger quantized) | correctness + speed at each |

### Models (ModelScope, to confirm exact repo ids)
RWKV-7 "World" / "G1" series: 0.1B / 0.4B / 1.5B / 2.9B / 7.2B. Start at 0.1B for
the correctness loop (fast iteration), scale up for perf.

## Conclusion
M1 gate = **greedy-match vs rwkv-lm on RWKV-7 0.1B** (correctness first). Perf
parity is a later milestone benchmarked head-to-head against albatross **on the
3090**. Build a `bench/` harness that drives both engines with identical
prompts/bsz and emits the grid above.

## Cross-references
- [[F0001]] env, [[F0002]] architecture. Snapshot §"Next actions".
