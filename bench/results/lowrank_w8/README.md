# Low-rank (LoRA) weights at int8: three metrics, three arms, two models

Bo's guidance in the adaptation group (2026-07-31): the low-rank matrices can be
w8-quantized, and it applies to all of them. This artifact is the measurement.

## Method

Fake-quant round-trip (quantize → dequantize in place, stock model serves the
rounded weights), so the scheme's accuracy is measured end-to-end with zero kernel
surgery — the same de-risk pattern as `bench/make_fakequant_w4.py`. Symmetric int8,
group 64 along the contraction axis; rank-96 up-projections take one scale per row
(96 is not divisible by 64). Vectors and norms untouched in every arm.

Arms: `baseline` (bf16), `big-w8` (r/k/v/o + ffn key/value, 144 tensors — the
w8g64 recipe), `big+lora-w8` (the same plus every LoRA down/up: +192 tensors,
144 at g64 + 48 per-row). The audit line in each run log lists exactly what was
rounded, so an arm that silently failed to apply would be visible.

Harness: the in-tree HF `rwkv7` implementation (transformers-rwkv#2), bf16 on an
RTX 5090. Teacher-forced scoring for lambada (sabotage self-check: shifting the
scored position by one collapses accuracy to 0.0000). Uncheatable-eval replicates
`bench/uncheatable_eval.py`'s replication of the official methodology (ctx 4000,
[0]-prefixed chunks, bpb against utf-8 bytes; all 15 local subsets, 7500 docs).
MATH500 greedy@1, `<think>` prompt convention of `bench/math500_avg64.py`,
math_verify scoring. Same protocol across arms; deltas are the claim, absolute
numbers are protocol-specific.

## Results

`rwkv7-g1h-1.5b` (reasoning line — the model where MATH500 has discriminating
power; base-line world models sit at a ~6% floor under this prompt):

| arm | uncheatable pooled_bpb | MATH500 greedy@1 |
|---|---|---|
| baseline | 0.586906 | 0.3880 |
| big-w8 | 0.587088 | 0.3940 |
| big+lora-w8 | 0.587160 | 0.4080 |

`RWKV-x070-World-1.5B-v3`:

| arm | uncheatable pooled_bpb | lambada acc (teacher-forced) |
|---|---|---|
| baseline | 0.627276 | 0.6872 |
| big-w8 | 0.627334 | 0.6878 |
| big+lora-w8 | 0.627384 | 0.6883 |

## Reading

Adding the low-rank matrices to the int8 set costs +7.2e-5 bpb (+0.012%) on the
reasoning model and +5.0e-5 on the base model; every task metric moves inside its
noise band (MATH500 ±1σ ≈ 2.2 pt at n=500; lambada ±1σ ≈ 0.65 pt at n=5153).
No metric shows damage. Consistent with the w8a8 evidence already in
`docs/BENCHMARKS.md` (that checkpoint quantizes low-rank too, but confounds it
with activation quantization; this experiment isolates the weights).

Files: per-arm uncheatable jsons (world arms unprefixed, `g1h15b_` prefix for the
reasoning model) and the two harnesses (`lambada_lora_w8.py` carries the arms and
the quantizer, `hf_bo_evals.py` the two Bo-metric evals).
