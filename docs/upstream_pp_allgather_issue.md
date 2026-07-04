# Upstream issue draft — PP proxy-tensor transfer corrupts non-TP-replicated tensors

Target: sgl-project/sglang. Verified against `main` (parallel_state.py, scheduler_pp_mixin.py)
and v0.5.10.post1 — identical code path on both. Written to submit as a GitHub issue
(issue-first, because the fix touches a core distributed primitive + a model-facing contract,
so the maintainers should pick the API).

---

## Title
PP + TP: `send_tensor_dict` all-gather optimization silently corrupts non-TP-replicated tensors in `PPProxyTensors`

## What happens
When a model runs with both pipeline parallel (pp>1) and tensor parallel (tp>1), any tensor a
model hands across a pipeline stage boundary via `PPProxyTensors` is silently corrupted **unless
it is identical across the attention-TP group**. Replicated tensors (`hidden_states`, `residual`)
are fine — that's every in-tree transformer today, which is why nobody has hit this — but a model
that carries a *TP-sharded* tensor across stages (e.g. a linear/recurrent model whose per-rank
state or a per-rank activation slice must travel with the hidden state) gets a wrong-but-
plausible tensor on the receiving stage. No error is raised; the output is just wrong.

## Root cause
`scheduler_pp_mixin.py` sends the proxy dict with the attention-TP all-gather optimization on by
default:

```python
# scheduler_pp_mixin.py
self.require_attn_tp_allgather = not self.server_args.enable_dsa_prefill_context_parallel  # True by default
...
self.pp_group.send_tensor_dict(
    tensor_dict=tensor_dict,
    all_gather_group=self.attn_tp_group if self.require_attn_tp_allgather else None,
)
```

`send_tensor_dict` then applies, to **every** tensor in the dict, a "send only my 1/tp slice,
all-gather on receive" bandwidth trick:

```python
# parallel_state.py  send_tensor_dict
if all_gather_group is not None and tensor.numel() % all_gather_size == 0:
    tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]   # send 1/tp
# ... recv_tensor_dict reassembles with all_gather_group.all_gather(...)
```

This is only lossless if the tensor is **identical on every rank of `all_gather_group`** (each
rank's 1/tp slice is then a genuine piece of the whole). For a TP-sharded tensor each rank holds
*different* data, so reassembly interleaves slices from different ranks → a franken-tensor.

Concretely (measured on a 2-way TP × 2-way PP run of an RWKV-7 model whose layer-0 value
projection `v_first` is a per-rank head slice, dtype bf16): both receiving ranks got byte-
identical wrong tensors whose checksum matched neither sender; greedy output diverged
deterministically at a fixed token (12/24 at 1.5B, 5/24 at 0.1B), dtype-independent (same at
fp32) — the signature of a data-layout corruption, not fp noise.

## Minimal repro (conceptually)
Run any model with `tp=2 pp=2` where the model puts a tp-sharded tensor into the returned
`PPProxyTensors` (not just `hidden_states`). The sharded entry comes out corrupted on stage 1.
(Happy to provide a tiny standalone repro model if useful.)

## Fix options (maintainers' call on the API)
The transfer needs to know which proxy entries are TP-replicated (safe to all-gather-split) and
which are sharded (must be sent whole). Two backward-compatible options:

1. **Per-key opt-out.** Add `all_gather_exclude: Optional[Set[str]] = None` to
   `send_tensor_dict`/`recv_tensor_dict`; skip the reshape/all-gather for those keys. The model
   declares its sharded proxy keys (e.g. a `PPProxyTensors.tensor_parallel_sharded_keys` set),
   the scheduler forwards it. ~15 lines; zero behavior change for existing models.
2. **Opt-in instead of opt-on.** Make the all-gather trick apply only to an explicit allowlist of
   keys (`{"hidden_states", "residual"}`) rather than every tensor. Safer default, but changes
   the code path for current models (needs a benchmark to confirm no regression).

I've been running option (1)'s effect as a model-side workaround (all-gather the sharded tensor
to full width before returning it from the model, slice it back after recv) — it restores exact
correctness (tp2×pp2 greedy 24/24 EXACT at 0.1B and 1.5B). Happy to send a PR for whichever
option you prefer.

## Note
Not urgent for current in-tree models (they only ship replicated proxy tensors). Filing it
because it's a latent, silent-corruption footgun for the linear/hybrid models that are landing
(GDN/Mamba/RWKV-style), and the failure mode (wrong output, no error) is nasty to debug.
