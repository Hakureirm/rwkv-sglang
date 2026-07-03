# RWKV-7 production correctness: dynamic-batching + radix cache (Task 1)

RWKV-7 is an RNN: each request carries its **own** recurrent state (`S`, token-shift).
sglang's default **token radix cache** shares identical *token prefixes* across
requests and lets a later request inherit an earlier one's cached prefix. That is
correct for attention (KV is a pure function of the token prefix) but **wrong for an
RNN** — the recurrent state is per-request and is not reconstructable from the shared
prefix nodes. Result (F0008): with radix cache ON, batches with shared/identical
prefixes diverge. Fix: **`disable_radix_cache=True`** (server: `--disable-radix-cache`).

## 1a. Batch-correctness — VERIFIED EXACT (0.1B, bf16, cuda-graph ON, radix OFF)

`bench/verify_batch.py` boots the 0.1B Engine with `disable_radix_cache=True`,
`disable_cuda_graph=False` (cuda-graph ON), `dtype=bfloat16`, and runs three batches.
Per-prompt B=1 outputs are the references (B=1 has no cross-request prefix to share);
identical-prompt copies are *also* checked against the numpy-oracle fixture (bit-level
ground truth).

```
source ~/rwkv_env.sh && CUDA_VISIBLE_DEVICES=0 ~/envs/rwkv-sgl/bin/python \
  bench/verify_batch.py --model /home/user/rwkv_models/rwkv7-0.1b-fla \
  --fixture bench/fixtures/oracle_rwkv7_01b_eiffel.json --dtype bfloat16 --cuda-graph
```

```
VERIFY_BATCH  model=/home/user/rwkv_models/rwkv7-0.1b-fla
  dtype=bfloat16  cuda_graph=ON  disable_radix_cache=True  n=24
ORACLE          [37138, 45, 44312, 47, 20996, 304, 25740, 109, 37480, 4600, 332, 39990, 4596, 37138, 45, 44312, 45, 32227, 22748, 37924, 4596, 3491, 709, 47]
eiffel B=1 == numpy-oracle : True
[1] IDENTICAL  bsz=4  every-output==oracle : True  (4/4)
[2] SHARED-PREFIX  bsz=5  every-output==B1-ref : True  (5/5)  tags=['eiffel', 'sp1', 'eiffel', 'sp2', 'eiffel']
[3] MIXED  bsz=6  every-output==B1-ref : True  (6/6)  eiffel==oracle : True  tags=['eiffel', 'eiffel', 'sp1', 'distinct', 'eiffel', 'sp2']
OVERALL: PASS (all batches exact)
```

**Result: EXACT.** Every request in every batch (>=4 identical, shared-prefix with
divergent tails, and mixed identical+distinct) matches its single-request output, and
identical-prompt copies match the numpy oracle token-for-token.

### Bug reproduction (control: radix ON) — confirms the failure is real & severe
Same script with `--radix-on --identical-bsz 8` (radix cache left ON):

```
  dtype=bfloat16  cuda_graph=ON  disable_radix_cache=False  n=24
eiffel B=1 == numpy-oracle : True          # first request (cnew cache) is fine
[1] IDENTICAL  bsz=8  every-output==oracle : False  (0/8)   # ALL 8 corrupted
      req#0 DIVERGED: [22590, 38499, 22638, 39920, 47, 11, 46, 3448, 24192, ...]   # garbage, != oracle
      ... (req#1..7 identical garbage)
[2] SHARED-PREFIX ... False (0/5)
[3] MIXED ... False (0/6)  eiffel==oracle : False
OVERALL: FAIL: IDENTICAL,SHARED-PREFIX,MIXED
```

The very first cnew-cache request is correct; every subsequent request that hits a
shared prefix node inherits stale/empty recurrent state and produces garbage. This is
exactly F0008 and it is **not intermittent at B>=4 with identical prompts** — it is a
hard, reproducible corruption. (cuda-graph is ON in both runs, so it is orthogonal:
the corruption is the radix cache, not the graph.)

## 1b. The safe mechanism: launch flag (no clean in-scope config hook exists)

I traced exactly how sglang decides the cache type (sglang 0.5.x on box):

1. `ServerArgs._handle_model_specific_adjustments()` (server_args.py) reads
   `hf_config.architectures[0]` and dispatches on a **hard-coded architecture list**.
   Mamba/linear models that must avoid the plain radix cache are registered there and
   call `_handle_mamba_radix_cache(..., support_mamba_cache=False)` which sets
   `disable_radix_cache=True` (e.g. `KimiLinearForCausalLM`, `BailingMoeV2_5ForCausalLM`).
   **RWKV7 (`RWKV7ForCausalLM`/`Rwkv7ForCausalLM`) is NOT in this list**, so
   `disable_radix_cache` keeps its default `False`.
2. `Scheduler.init_cache_with_memory_pool()` then chooses the cache:
   - `is_hybrid_ssm = (model_runner.hybrid_gdn_config is not None or
     model_runner.mamba2_config is not None)`.
   - RWKV7 is exposed via the overlay's **own** `rwkv7_config` property (fnewed into
     `mambaish_config` only for *state-pool* allocation), and deliberately is **not**
     `mamba2_config`/`hybrid_gdn_config` (wiring MambaRadixCache is a later milestone).
   - So `is_hybrid_ssm=False` and, with radix ON, the scheduler builds a **plain
     `RadixCache`** → the buggy prefix-sharing path. (With radix OFF + chunked prefill
     it builds `ChunkCache`, which never shares prefixes → correct.)

**Is there a clean model/config attribute sglang respects?** No. The radix decision is
made entirely from (a) the hard-coded arch list in `server_args.py` and (b)
`server_args.disable_radix_cache`, both *before* and *independent of* `models/rwkv7.py`
and `configs/rwkv7.py`. There is no generic `getattr(hf_config, "disable_radix_cache")`
or equivalent hook that `Rwkv7Config` could set. The two files I am permitted to edit
for this task are never consulted by the radix-decision code, so **no clean in-scope
config-level enforcement exists**.

**Required safe mechanism (production):**
- Offline `sgl.Engine(...)`: pass **`disable_radix_cache=True`**.
- Server: launch with **`--disable-radix-cache`**.

This is mandatory for RWKV-7 serving until MambaRadixCache (state-aware prefix reuse)
is wired (separate milestone). It costs prefix-cache reuse but RWKV decode is O(1)/token
and context-length-independent, so the serving impact is small.

**Recommended upstream fix (out of this task's edit scope — `server_args.py`):** register
RWKV-7 in the same ladder the other linear models use, so the flag is automatic:

```python
elif model_arch in ["RWKV7ForCausalLM", "Rwkv7ForCausalLM"]:
    self._handle_mamba_radix_cache(model_arch=model_arch, support_mamba_cache=False)
```

This one branch makes `--disable-radix-cache` unnecessary (sglang forces it, with a
logged warning) and mirrors `KimiLinearForCausalLM` exactly. It edits a core file
(`server_args.py`) that is not part of the model overlay, so it is left for the lead to
land. Until then, the launch flag is the documented, verified-safe path.

## 1a (robustness): also verified EXACT on 1.5B
Same `verify_batch.py` on `rwkv7-1.5b-fla` (bf16, cuda-graph ON, radix OFF):
IDENTICAL 4/4, SHARED-PREFIX 5/5, MIXED 6/6, eiffel B=1 == numpy-oracle == all batch
copies. `OVERALL: PASS (all batches exact)`.
