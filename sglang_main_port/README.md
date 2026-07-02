# RWKV-7 port to sglang `main`

The overlay (developed against **v0.5.10.post1**) verified on **sglang main**
(`base_commit.txt`, currently a3f6680) inside the official CUDA-12.9 dev
container (`lmsysorg/sglang:dev-cu12`) — **greedy 24/24 token-EXACT vs the
numpy oracle at 0.1B and 1.5B (bf16)**, same result as on v0.5.10.post1.

Contents:
- `upstream_edits.patch` — the diff against sglang main for the 7 upstream files
  (config registry now in `utils/hf_transformers/common.py`; radix-off for
  RWKV-7 under the new `_handle_mamba_radix_cache` semantics; all-linear
  `cell_size==0` guard now in `model_executor/pool_configurator.py`).
- `new_files.txt` / `new_files.tgz` — the RWKV-7-only additive files, as
  verified in the container. Canonical source: `../sglang_overlay/` (the model,
  backend, and kernels are byte-identical; `_linear_backend()` in
  `models/rwkv7.py` selects `forward_batch.attn_backend` (v0.5.10) vs the
  global forward context (`main`) at runtime, so ONE code base serves both).

Apply to a sglang checkout at (or near) the base commit:
```bash
cd sglang
git apply upstream_edits.patch
tar xzf new_files.tgz          # or copy the same paths from ../sglang_overlay
```

GeForce note: sglang-main containers need CUDA 12.x on consumer cards (CUDA-13
forward compat excludes GeForce); the official `dev-cu12` image works — clear
`LD_LIBRARY_PATH` if the bundled `/usr/local/cuda/compat` libcuda shadows the
host driver.
