# sglang_overlay — additive RWKV-7 files for sglang

The RWKV-7-**only** files, mirroring the sglang package tree (`sglang/srt/...`):

- `configs/rwkv7.py` — the `Rwkv7Config`
- `models/rwkv7.py` — `RWKV7ForCausalLM`
- `layers/attention/linear/rwkv7_backend.py` — the linear-attention backend (+ the
  `Rwkv7NoOpFullAttnBackend` all-linear stub)
- `layers/attention/rwkv7_kernels/**` — the hand-written CUDA/Triton kernels
- `speculative/rwkv_chain_worker.py` — the RWKV_CHAIN speculative worker

These are purely additive: they do not exist upstream, so they never drift.

## The ~129 lines of genuine edits to *upstream* files are NOT here

Earlier this overlay also shipped 6–7 churny **upstream** files as full-file
copies (~11k lines, frozen at v0.5.10.post1). Rsyncing those over a newer sglang
clobbered a year of upstream fixes and broke imports on every image bump (F0059).
Those copies are gone. The genuine edits — `Rwkv7Config` registration (now in the
upstream-moved `utils/hf_transformers/common.py`), the all-linear `cell_size==0`
guard (`model_executor/pool_configurator.py`), `Rwkv7NoOpFullAttnBackend` +
`rwkv7_config` (`model_runner.py`), RWKV-7 radix-off (`server_args.py`), and the
F0036 `v_first` PP + cuda-graph fix — are delivered as a patch:
**`../sglang_main_port/upstream_edits.patch`** (10 files, 129 ins / 4 del), plus two
anchored idempotent injections for the two huge churn-prone files (`spec_info.py`,
`scheduler.py`).

## Deploy

`scripts/deploy.sh` drives all of it — rsync the additive files, `git apply` the
patch (idempotent, `--check`ed so a base drift fails loudly), run the two
injections, clear bytecode:

```bash
# wheel install:
BOX=my-gpu-host SP=/opt/venv/lib/python3.10/site-packages bash scripts/deploy.sh
# local / editable dev-container (lmsysorg/sglang:dev-cu12):
BOX= VENV_PY=python3 bash scripts/deploy.sh
```

`sglang_main_port/` packages the same delta (patch + `new_files.tgz`) for applying
to a plain upstream checkout without this repo; the additive files there are
byte-identical to the ones in this overlay.
