# RWKV-7 port to sglang

What a stock sglang checkout needs to serve RWKV-7: a patch for sglang's own files,
plus the RWKV-7-only files that are simply added.

**Base: `v0.5.17`** — see `base_commit.txt`. Earlier revisions of this directory
targeted a `main` commit; upstream has since grown extension points that removed
four of the hunks (below), so the base is now a release tag rather than a moving
commit.

## Apply

```bash
cd sglang                  # at v0.5.17
git apply upstream_edits.patch
tar xzf new_files.tgz
```

## Contents

- `upstream_edits.patch` — the diff against sglang's own files: **10 files, 15 hunks.**
- `new_files.txt` / `new_files.tgz` — the RWKV-7-only additive files (31 files),
  **generated from `../sglang_mainline/`, which is the tree that actually runs.**
  They used to be a stale snapshot of it; that gap is closed and the tarball is now
  regenerated from that directory rather than maintained beside it.

## Verified on this base

Clean room: a fresh `v0.5.17` worktree, the two commands above, nothing else.

| gate | result |
|---|---|
| RWKV-7 modules import | 17/17 |
| `bench/greedy_check.py`, 0.1B vs the numpy oracle | 24/24 tokens, exact |
| `bench/verify_batch.py`, 1.5B, bf16, cuda-graph ON | PASS — all three batches exact |
| serving a request end to end | radix cache resolved off, triton linear-attn backend, coherent output |

`verify_batch.py` at **0.1B** fails its SHARED-PREFIX and MIXED batches. That is not
this port: it diverges byte-for-byte the same way on the older base, with the same
`models/rwkv7.py`, so it tracks model size rather than the serving stack. Run that
gate at 1.5B or larger; the note in the script says so now.

Caveat on the container used: it is provisioned for an older sglang (sgl-kernel
0.4.4, flashinfer 0.6.12) while v0.5.17 asserts newer minimums, so the runs above set
`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`. They exercise this port against v0.5.17's
Python, not v0.5.17 on its own supported kernel versions.

## What the patch no longer carries

Upstream grew registries that let an out-of-tree model declare itself instead of
editing core files. Four hunks are gone for that reason — the behaviour did not
change, only where it is expressed:

| used to patch | now |
|---|---|
| `attention_registry.py` — an `elif runner.rwkv7_config is not None` arm building the backend | `register_linear_attn_model(...)`, called from our own `configs/rwkv7.py`; upstream's `else` arm resolves the backend out of that registry (`attention_registry.py:467`) |
| `model_runner.py` — a `Rwkv7Config` import, a `rwkv7_config` property, and one more term in the `existing` chain | the same registry, reached through `mambaish_config()` (`configs/hybrid_arch.py:125`) |
| `pool_configurator.py` — an all-linear `cell_size == 0` branch sizing the token pool | upstream's own zero-KV branch, **plus `--max-total-tokens` at launch** (see below) |

`server_args.py` stays a core edit, and it is worth writing down why rather than
rediscovering it: the natural home for it is `register_model_override`, but that
registry validates every declared field against ServerArgs' `resolvable=True`
whitelist — 36 fields at v0.5.17 — and `disable_radix_cache` is not one of them.
Declaring it raises at the registration slot. (`uses_mamba_radix_cache` *is* on the
whitelist, but that is the mamba radix cache, a different switch.)

One more trap in the same area: the registry's `uses_mamba_radix_cache` field is
overloaded. `mem_cache/kv_cache_builder.py` reads it to decide whether the model gets
a mamba/linear state pool **at all**, so setting it `False` — which reads like the
right thing for a model whose state is not prefix-cacheable — would leave RWKV-7
without its state pool. It stays `True`; the radix cache is switched off by the
`server_args.py` hunk, which the mamba-radix pass then short-circuits on.

### `--max-total-tokens` is now required for a serving batch

RWKV-7 has no full-attention KV cache, so sglang cannot size the token pool from
per-token KV bytes — there are none. Upstream's fallback for a zero-KV model is one
context length, which a serving batch outgrows. Measured at v0.5.17, 128 requests x
200 tokens against the 0.1B model:

| | token pool | retractions | peak running reqs |
|---|---|---|---|
| no flag | 8192 (= `context_len`) | repeated | 31 of 64 |
| `--max-total-tokens 262144` | 262144 | none | 64 of 64 |

The removed `pool_configurator.py` hunk computed this inside sglang and capped it at
`1 << 20`. `scripts/serve.sh` now passes that same cap by default (`MAXTOTALTOK`), and
`bench/throughput.py` / `bench/serving_scale.py` take a matching `--max-total-tokens`.
Anything else that builds an Engine for a serving batch has to pass it too, or it
measures a pool nobody chose.

## Note for GeForce cards

sglang containers need CUDA 12.x on consumer cards (CUDA-13 forward compat excludes
GeForce); the official `dev-cu12` image works — clear `LD_LIBRARY_PATH` if the
bundled `/usr/local/cuda/compat` libcuda shadows the host driver.
