# F0048 — Qwen3.5 has no viable same-tier "int8" comparison point; FP8 is the closest sglang-native substitute, and it's slower at bsz1

**Date:** 2026-07-07 · **Status:** MEASURED (RTX 5090, sglang native `--quantization fp8`) +
a definitively-answered negative result (true int8) · **Context:** the cross-project
same-quantization-tier comparison (`memory/project-qwen35-benchmark.md`) had only covered
bf16 (stock) and RWKV's fp16 (optimized, not directly comparable) so far. This finding
answers whether a genuine int8-tier comparison is possible.

## The ask, and why it doesn't resolve cleanly

RWKV-7 already ships hand-written w8a8 (true int8, tensor-core) and w8g64 kernels with
published 7.2B numbers (bsz1 60.2 tok/s, peak 7,587 tok/s @ c640 — §4, F0035/F0047). The task
was to find an equivalent **int8** path for Qwen3.5 using sglang's own native quantization
support — explicitly not a hand-written kernel — and if none exists, report the gap honestly
rather than force a substitute.

## What sglang actually supports (source-verified, not assumed)

Checked `sglang/srt/layers/quantization/w8a8_int8.py` (the genuine-integer-int8 method,
`--quantization w8a8_int8`) directly:

- `W8A8Int8Config.get_config_filenames()` returns `[]` and `W8A8Int8LinearMethod.create_weights()`
  unconditionally allocates the `weight` parameter as `torch.int8` for the checkpoint's own
  weight loader to fill, with a paired `weight_scale` (float32) parameter. There is no
  `is_checkpoint_int8_serialized`-style branch anywhere in the file (checked all 387 lines) —
  no fallback path that quantizes bf16/fp16 weights on the fly. **The checkpoint itself must
  already contain int8 weights + a scale tensor.** Pointing `--quantization w8a8_int8` at a
  plain bf16 Qwen3.5 checkpoint would not quantize it — it would fail to load correctly (the
  `weight_scale` tensor the loader expects to find isn't present) — this is not a workable
  path.
- `--quantize-and-serve` (which sounds like the generic "quantize any checkpoint on serve"
  escape hatch) is gated to `{modelopt, modelopt_fp8, modelopt_fp4, nvfp4_online,
  modelopt_mixed}` only in `ModelConfig._validate_quantize_and_serve_config()`, and even for
  those methods it unconditionally raises `NotImplementedError` right now ("currently disabled
  due to compatibility issues"). It is dead code for every quantization method today,
  including its own intended scope.
- The only sglang-native method that takes a **plain, unquantized bf16 checkpoint** and
  produces a genuinely quantized, working model via a CLI flag alone — no calibration, no
  separate checkpoint — is **`fp8`** (not int8): `Fp8Config` defaults
  `is_checkpoint_fp8_serialized=False`, and `Fp8LinearMethod.process_weights_after_loading`
  branches on exactly that flag to quantize bf16 weights to fp8 with computed per-tensor/
  per-channel scales at load time when it's `False`. This is a real, working dynamic-quant
  path.
- No pre-quantized genuine-int8 (`w8a8_int8`-format) checkpoint for Qwen3.5-2B or -9B exists
  publicly (checked HF/ModelScope). What does exist: AWQ int4 (`QuantTrio/Qwen3.5-9B-AWQ`,
  `cyankiwi/Qwen3.5-2B-AWQ-4bit`), GPTQ-Int4 (community `mssfj/Qwen3.5-9B-GPTQ-INT4`; official
  Qwen GPTQ-Int4 only ships for the unrelated 27B/35B-A3B/397B-A17B MoE tiers), and a
  disputed-quality community FP8 repo for 9B (`lovedheart/Qwen3.5-9B-FP8`) — no official
  small-dense FP8 release. None of these are int8.

**Conclusion: there is no sglang-native, no-hand-written-kernel path to genuine int8 for
Qwen3.5 today.** This is the honest gap the task asked to surface if it existed. AWQ/GPTQ
would let a *4-bit* comparison happen (against RWKV's separate int4/GPTQ tier, not int8 — a
different comparison, not attempted here since it wasn't the ask). FP8 is the only same-CLI,
zero-extra-work quantization sglang has for this architecture — reported below as a clearly
labeled bonus, not a same-tier substitute (FP8 is an 8-bit *floating-point* format; RWKV's
w8a8 is 8-bit *integer* — same bit width, different numerics, not the same tier).

## FP8 (bonus, not int8): does it even work on the GDN hybrid layers?

Yes, cleanly, on both sizes. `Qwen3_5GatedDeltaNet`'s `in_proj_qkvz`/`in_proj_ba`/`out_proj`
and the interleaved full-attention layers' qkv/o_proj all pass the fp8 quant_config through
normally (the model code only special-cases `modelopt_fp4`, not `fp8`); `conv1d` stays
unquantized regardless (hardcoded, architecture-wide, not fp8-specific). Booted both sizes
with `--quantization fp8` pointed directly at the existing plain-bf16 checkpoint dirs, no
conversion step:

| model | weight mem (bf16 → fp8) | boot | GDN kernel dispatcher | sanity generation |
|---|---|---|---|---|
| Qwen3.5-2B | ~4.55 GB → 3.16 GB | clean | unaffected (Triton, as normal) | coherent (" Paris." + correct) |
| Qwen3.5-9B | ~17.6 GB → 11.51 GB | clean | unaffected | coherent (" Paris." repeated, greedy artifact not a quant artifact) |

## Speed: FP8 loses at bsz1, same pattern RWKV's own w8a8 shows at 1.5B

| model | precision | bsz1 out tok/s | vs bf16 |
|---|---|---:|---:|
| Qwen3.5-2B | bf16 | 336.0 (prior round) | — |
| Qwen3.5-2B | **fp8** | **206.6** | **−38.5%** |
| Qwen3.5-9B | bf16 | 96.0 (prior round) | — |
| Qwen3.5-9B | **fp8** | **71.7** | **−25.3%** |

This is the same "quantization tax dominates at bsz1" pattern already documented for RWKV's
own w8a8 (§4: "1.5B e2e is 0.9466× fp16" — a *regression* at small scale from the per-token
activation-quant launch cost) — not a Qwen3.5-specific or FP8-specific weakness, a
cross-architecture, cross-quantization-format phenomenon at this batch size.

## Concurrency: real (not KV-starved) reading through c=256, still climbing — true peak not fully bisected

Qwen3.5-9B FP8, `--mem-fraction-static 0.85`, `--cuda-graph-max-bs 256`
(`max_total_num_tokens=73,310` at boot):

| c | 1 | 8 | 32 | 64 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|---|
| out tok/s | 71.7 | 564.8 | 2,148.0 | 3,571.7 | 5,457.9 | 5,936.0 | **6,213.0 (still climbing)** |

**A genuine false-plateau was caught and discarded here**, not just checked for: pushing
`--cuda-graph-max-bs` to 288 to chase the still-climbing curve dropped
`max_total_num_tokens` from 73,310 to just **23,006** — because `max_running_requests`
governs the mamba-cache reservation, and that reservation shares the same memory pool as the
*real* per-token KV-cache this hybrid architecture needs for its 6 full-attention layers
(exactly the mechanism documented for bf16-9B in the sibling project's round 4). At the
288-config, the scheduler could only keep ~71 requests actually decoding at once (the rest
queued), and every re-measured concurrency point came back 34–39% *lower* than the same
nominal concurrency at the 256-config (e.g. c=128: 3,620.7 vs 5,457.9; c=256: 3,819.7 vs
6,213.0, now *declining* instead of climbing) — a clean demonstration of the same artifact
flagged elsewhere, not a real throughput reading. **These 288-config numbers are discarded as
an admission-control artifact**, not reported as FP8's behavior.

The 256-config curve is the reliable one, but it is itself only mildly provisioned
(73,310 tokens ÷ 320 tokens/request ≈ 229 request-equivalents, against 256 requested) — so
even its c=256 point may be a slight underestimate. Finding FP8's true unconstrained peak
would need the same `--context-length`/`--mem-fraction-static` co-tuning dance round 4 used
to separate the mamba-cache and real-KV-cache budgets for bf16-9B. **Not done here** — this
is a bonus data point attached to an already-answered gap question, not a full peak-finding
exercise, and chasing it further wasn't judged worth the GPU time this round. Flagged as an
open follow-up if a future session wants a clean FP8 peak number.

## Bottom line

- **True int8, same tier as RWKV's w8a8: not achievable for Qwen3.5 today** via sglang-native
  means — no pre-quantized checkpoint exists publicly, and sglang's `w8a8_int8` method has no
  dynamic/on-the-fly quantization path (confirmed by reading the implementation, not
  assumed). This is the honest gap.
- FP8 (bonus, not same-tier) works cleanly end-to-end on Qwen3.5's hybrid GDN architecture on
  both sizes tested, with zero hand-written code, but is **25–39% slower than bf16 at bsz1**
  and its true concurrency peak is bracketed at **≥6,213 tok/s @ c256 for the 9B** (climbing,
  not yet found) — not directly comparable to RWKV's int8 numbers given the tier and
  model-size (7.2B vs 9B) mismatches; no same-tier head-to-head table is produced because one
  cannot honestly be produced yet.

## Cross-references

Sanity + speed JSONs on the 5090 tower (not copied into this repo, per the existing
qwen35-benchmark-project convention of keeping comparison JSONs on the shared box):
`bench/results/qwen35/qwen35_{2b,9b}_fp8_bsz1_5090.json`,
`qwen35_9b_fp8_sweep_5090.json` (the reliable 256-config run),
`qwen35_9b_fp8_sweep_5090_v2.json` (the 288-config false-plateau, kept as a documented
negative example). RWKV int8 reference numbers: §4 / [F0035](0035-7b-int8-concurrency-headroom.md)
/ [F0047](0047-fp16-72b-concurrency-correction.md). Sibling bf16-9B false-plateau precedent:
`memory/project-qwen35-benchmark.md` round 4 (not in this repo).
