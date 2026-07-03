# Upstream PR draft — `[Model] Support RWKV-7 (Goose)`

Status: DRAFT (2026-07-02). Base verified: sglang `main` @ a3f6680 (greedy
token-exact 0.1B + 1.5B, official `dev-cu12` container). Artifacts:
[`../sglang_main_port/`](../sglang_main_port/).

## PR title
`[Model] Support RWKV-7 (Goose) — all-linear recurrent model with O(1) state`

## PR body (draft)

### Motivation
RWKV-7 ("Goose", BlinkDL) is a purely recurrent architecture: per-token state is
O(1) in context length (L24-D1024: a constant 1.62M elements vs a KV cache's
linear growth). That makes it structurally strong for high-concurrency / long-
context serving — measured on this implementation: 256 concurrent sequences cost
+202 MiB peak VRAM; a 64× context increase costs +4 MiB (RTX 3090, 1.5B).
sglang currently has no RWKV support; this PR adds the RWKV-7 model family
(0.1B–7.2B public checkpoints, fla config format `model_type="rwkv7"`).

### What's added
- `configs/rwkv7.py` — Rwkv7Config (+ registry entries in
  `utils/hf_transformers/common.py`, `configs/__init__.py`)
- `models/rwkv7.py` — the model: token-shift + time-mix (WKV-7 recurrence) +
  channel-mix (sqrelu), LoRA-style low-rank gates, greedy token-exact vs the
  BlinkDL rwkv-lm numpy reference at 0.1B/1.5B/7.2B (fp16+bf16, cuda-graph,
  dynamic batching, chunked prefill). Head-parallel TP + llama-pattern PP —
  full matrix greedy-exact on real multi-GPU (tp 2/4/8, pp 2/4/8, mixed
  tp2×pp2).
- `layers/attention/linear/rwkv7_backend.py` — linear-attention backend on the
  mamba state pool: two width-2 token-shift conv states + the (H,64,64) fp32 WKV
  state; triton WKV kernel (decode + varlen prefill), FLA-free.
- `layers/attention/rwkv7_kernels/` — self-contained triton/CUDA kernels
  (WKV recurrence; optional opt-in CUDA extras).
- Wiring: attention_registry (all-linear → NoOp full-attn stub),
  pool_configurator cell_size==0 guard (all-linear token budget),
  server_args radix-off for RWKV-7 (recurrent state is not prefix-cacheable
  until a state-aware MambaRadixCache exists).

### Correctness evidence
- Greedy token-exact vs the pure-numpy rwkv-lm oracle: 0.1B / 1.5B / 7.2B.
- lm-eval (1.5B): lambada 0.673 vs reference 0.671; MMLU 0.524 vs 0.511.
- Dynamic batching / chunked prefill / cuda-graph: exact (gates in `bench/`).
- 10 GPU types, 7 SM generations (Turing→Blackwell), bf16 exact on all.

### Performance evidence (RTX 3090, 1.5B, decode tok/s)
bsz 1→256: 166 → 8,298 (plateau ~8.2k); VRAM flat (+202 MiB @bsz256).
H100 ShareGPT (bench_serving): peak 14,245 total tok/s, TTFT 69 ms @16 req/s.

### Open questions for maintainers
0. Heads-up (found while validating mixed tp×pp): `send_tensor_dict`'s
   chunk-send optimization (`reshape(tp,-1)[tp_rank]` + rank-wise reassembly)
   silently corrupts NON-tp-replicated PPProxyTensors entries. RWKV-7's
   `v_first` (a per-rank head slice) hit this; we work around it by
   all-gathering to full width before the stage boundary. Other hybrid models
   shipping sliced proxy tensors would hit the same — worth an assert or a
   per-tensor replicated flag upstream.
1. Preferred placement for the RWKV kernels (in-tree triton vs sgl-kernel).
2. Radix: we force-disable for RWKV-7; a state-aware MambaRadixCache is the
   follow-up — same shape as the GDN/hybrid discussion.
3. Optional opt-in CUDA kernels (fused GEMV / sparse FFN / weight-only int4+int8)
   are NOT in this PR (kept in our overlay); can follow up if wanted.

## Mechanics (when executing)
1. Fork sgl-project/sglang on the Hakureirm account; branch `rwkv7-support`.
2. `git apply sglang_main_port/upstream_edits.patch` + untar new files; drop the
   opt-in extras (RWKV_W4/W8/FAST/SPARSE paths) to keep the PR minimal — strip
   to: config, model (plain path), backend, triton WKV kernel, wiring.
3. Run upstream lint/format (pre-commit), unit tests, add a model test mirroring
   `test_generation_models.py` conventions.
4. Re-run greedy gates inside `dev-cu12` against the branch.
5. PR text from this draft; link accuracy/bench evidence from the public repo.

## Blockers / notes
- None technical.
