# ROCm gfx1100 all-size quantization evidence

This directory contains the raw output from the BF16 W8G64/W4G64 acceptance
run on a 48 GiB `gfx1100` GPU with ROCm 7.2.1. The model revisions are the
0.4B, 1.5B, 2.9B, 7.2B, and 13.3B entries in
[`bench/rocm_model_matrix.json`](../../rocm_model_matrix.json). The separate
0.1B bring-up result is in
[`rocm_gfx1100_quant_e2e.json`](../rocm_gfx1100_quant_e2e.json).

## Protocol

- dense, W8G64, and W4G64 checkpoint-directory bytes;
- dense/W8/W4 decode and prefill throughput at batches 1 and 8;
- HIP graphs enabled, radix cache disabled, 64 decode tokens, prefill length 256;
- W8: 24-token dense NumPy oracle plus identical/shared-prefix/mixed dynamic
  batches;
- W8 and W4: 512-token single-shot versus 2x256 chunked-prefill continuation.

All selected gates completed for all five sizes. See [`summary.json`](summary.json)
for machine-readable timings and ratios. The `*_throughput.json` files are the
direct benchmark output and the `.log` files retain the complete process output.

## Deliberate W4 exclusion

This run used `--skip-w4-batch`; therefore its pass status does **not** certify
RTN W4 quality or full-model batch invariance. The standalone W4 HIP projection
is numerically gated and batch-exact, and W4 chunked prefill passed at all five
sizes. However, the lossy 0.4B RTN checkpoint produced a different greedy
continuation for eight identical prompts than for batch 1. That complete
negative result is retained in
[`04b_w4g64_batch_known_failure.log`](04b_w4g64_batch_known_failure.log).

The W4 model-quality gate remains open pending a quality-preserving GPTQ/K-quant
or mixed-precision checkpoint. The later fused-dequant M=9..256 implementation
replaced the large-M dequantize-then-rocBLAS path; its operator and end-to-end
evidence is recorded in [F0084](../../../docs/findings/0084-rocm-quant-prefill.md).
