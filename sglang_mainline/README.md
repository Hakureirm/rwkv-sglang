# The mainline RWKV-7 implementation

**This is the code that runs.** Every performance number in `docs/BENCHMARKS.md`
measured after 2026-07-05, the same-card comparison in
`bench/results/samecard_btl/`, and the correctness gates are produced by this
tree, applied over sglang `main`.

It is committed here because it was not committed anywhere before. The repository
carried two other copies, and neither was what we were running:

- `sglang_overlay/` — the v0.5.10 line. A different mechanism for the same job
  (its chunk-boundary state reset keys on `extend_prefix_lens`; this tree zeroes
  fresh slots upstream instead). Retired.
- `sglang_main_port/new_files.tgz` — a snapshot of this line, but a stale one:
  it carries 5 of the 12 kernel modules and is missing the megakernel, the fused
  glue, the fused LoRA, w8a8, w4 and the WKV kernels, i.e. most of what the
  benchmark flags turn on.

That gap was found while reviewing an external PR that patched
`sglang_overlay/`: the patch targeted code we publish but do not run, and a
correctness gate run against the deployed tree could not have tested it either
way. Measuring one artifact while publishing another is the defect; this
directory closes it.

## Layout

Paths mirror their destination under `python/sglang/` in an sglang checkout:

```
srt/configs/rwkv7.py                            model config
srt/models/rwkv7.py                             the model
srt/layers/attention/linear/rwkv7_backend.py    linear-attention backend
srt/layers/attention/rwkv7_kernels/             12 modules + CUDA sources
srt/speculative/rwkv_spec_worker.py             speculative decode worker
```

Wiring edits to sglang's own files stay in
`sglang_main_port/upstream_edits.patch`.

## Provenance

`srt/layers/attention/rwkv7_kernels/cuda/` carries `NOTICE` and
`ALBATROSS_LICENSE`: one GEMV kernel is a byte-for-byte port from BlinkDL's
Albatross (Apache-2.0) and is attributed there. Everything else is this
project's own.
