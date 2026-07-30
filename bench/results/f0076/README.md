# F0076 — the HF port at non-square width, vs BlinkDL's own runtime

`nonsquare_vs_official_runtime.json` is the result; the three `harness_*.py` files are
what produced and diagnosed it, committed because a result without its harness is a
claim, not a reproduction. They run inside any CUDA-less container with `rwkv` (pip)
installed and the transformers-rwkv PR#2 tree on `PYTHONPATH` (mounted at `/tf` here);
`/cache` is any scratch directory.

- `harness_e2e.py` — downloads the reference 0.1B `.pth`, converts it with the PR's own
  converter, and compares logits + 32-token free-running greedy against the `rwkv`
  package at `cpu fp32` (`RWKV_V7_ON=1`). Writes the JSON. Note in-file: its first
  version fed the reference the last prompt token twice and reported 23/32 — that
  number was a harness artifact, not an implementation difference.
- `harness_generation_probe.py` — the smoke that first exposed the `_init_weights`
  regression: the port generated "civil civil civil" where the reference generated
  " Paris, France".
- `harness_zeroed_weights_probe.py` — the diagnosis: 249 of 402 tensors zeroed after
  loading, by an `_init_weights` that wrote in place and bypassed `_is_hf_initialized`.
