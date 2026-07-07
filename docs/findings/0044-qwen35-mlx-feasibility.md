---
doc_kind: finding
finding_id: F0044
title: "Qwen3.5 runs on MLX today via mlx-lm 0.31.3 out of the box: native hybrid Gated-DeltaNet + interleaved full-attention support, bf16 generate needs zero conversion step, 4-bit quant via mlx_lm.convert also verified end-to-end — the Apple-Silicon tier does not have to be RWKV-only"
last_verified_commit: "HEAD"
discovered_by: Sonnet 5 (agent-assisted), 2026-07-07
severity: info
status: open
related: [F0037, F0038, F0042]
---

# Finding F0044: Qwen3.5 on MLX — feasibility probe, verdict YES

## Context / question being asked

`mlx_port/` (F0037–F0041) is a complete, hand-written, oracle-gated RWKV-7 MLX port with zero
dependency on `mlx-lm` or any other model library. The benchmark plan compares RWKV-7 against
Qwen3.5 (Alibaba's hybrid Gated-DeltaNet linear-attention model — same DPLR mathematical family as
RWKV-7: 18 linear-attention layers + 6 full-attention layers per 24-layer block) at matched sizes
across hardware tiers. The GPU/cloud tier already has real Qwen3.5-2B/9B numbers via its own native
serving-framework support. This probe answers the one open question for the Apple-Silicon tier:
**can MLX run Qwen3.5 at all, or does this tier have to be reported as RWKV-only** because a
from-scratch port (mirroring the RWKV-7 effort) would be required first?

Pre-registered order of operations (cheapest-check-first, matching the CoreML/ANE probe's
discipline of not building/downloading anything before a config-level check rules it in or out):
check the library before downloading any weights; only download+run if the library check is
promising or inconclusive; regardless of outcome, ground a port-size estimate in real numbers.

## Method

1. Checked whether `mlx-lm` (pip) recognizes Qwen3.5's `model_type` / architecture before
   downloading anything.
2. Since it did (see below), verified an already-present local checkpoint's integrity, then ran
   `mlx_lm.generate` directly against it, and `mlx_lm.convert -q` to check the quantization path
   too (relevant since this project's benchmark methodology weighs quantized comparisons heavily).
3. Sized a from-scratch-port counterfactual against `mlx_port/`'s actual line counts and git
   history, per the task brief, even though the positive result in steps 1–2 makes this moot for
   the current cycle.

## Result 1 — mlx-lm already ships a real (non-stub) implementation

Installed version: `mlx-lm` **0.31.3** — confirmed via `pip index versions mlx-lm` to be the
current PyPI latest (not a nightly/dev/self-built copy), running on `mlx` (core) **0.31.2**, the
same MLX core version `mlx_port/README.md` documents its own numbers against. `mlx_lm/models/`
contains `qwen3_5.py` (531 lines) and `qwen3_5_moe.py` (larger MoE variants; not exercised here).

Reading `qwen3_5.py` confirms this is a genuine architecture implementation, not a placeholder:

- `GatedDeltaNet` (the linear-attention layer): real conv1d + delta-rule recurrent state update,
  short-conv gating, sigmoid beta / softplus-based decay (`compute_g`), delegating the actual
  per-step recurrence to `mlx_lm/models/gated_delta.py` (283 lines) — which contains a **hand-written
  `mx.fast.metal_kernel`** (`gated_delta_step`, SIMD-reduction per-token state update over
  `[B,T,Hv,Dv,Dk]`), not a naive/slow Python fallback. This is structurally the same kind of
  hand-rolled Metal recurrence kernel this project's own `mlx_port/rwkv7_mlx.py` writes for RWKV-7's
  WKV scan — i.e., real per-architecture kernel engineering exists here, this isn't a thin wrapper.
- `DecoderLayer` interleaves `GatedDeltaNet` and a standard `Attention` (imported from
  `qwen3_next.py`, 491 lines) based on `(layer_idx + 1) % full_attention_interval != 0` — the exact
  hybrid interleaving this project already established for Qwen3.5 on the GPU tier. `qwen3_5.py`
  reuses `qwen3_next.py`'s `Attention`/`MLP`/`RMSNormGated`/`SparseMoeBlock` wholesale, meaning
  Qwen3.5 support in `mlx-lm` sits on top of already-exercised Qwen3-Next hybrid-architecture
  infrastructure rather than being an isolated one-off.
- `make_cache()` returns a **dual cache-type list** per layer — `ArraysCache` (recurrent state) for
  linear-attention layers, real `KVCache` for full-attention layers — and `Qwen3_5TextModel.__call__`
  builds two different masks (`create_attention_mask` for the KV-cache path,
  `create_ssm_mask` for the recurrent path) and routes each layer to the right one.

## Result 2 — checkpoint used, and its integrity

A `Qwen/Qwen3.5-2B` checkpoint was already present locally at (ephemeral scratch path)
`/private/tmp/qwen35_mlx_test/Qwen3.5-2B` (inherited from an earlier, unrelated interrupted attempt;
not re-downloaded this pass). Verified before trusting it:

- `model.safetensors-00001-of-00001.safetensors`: 4,548,221,488 bytes on disk; index.json declares
  `total_size: 4548144832` (the ~77KB delta is exactly safetensors header overhead) — sizes match.
- `safe_open(...).keys()` → all **632/632** tensor keys enumerate and are readable.
- No `.lock`/`.incomplete` files anywhere in the checkpoint dir; the `.cache/huggingface/download/*.metadata`
  sidecars are the normal post-completion bookkeeping `huggingface_hub` leaves behind, not
  in-progress markers.
- `config.json`: `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"`,
  and a literal `layer_types` list confirming **18 `linear_attention` + 6 `full_attention` across 24
  layers** (`full_attention_interval: 4`) — this matches this project's already-established GPU-tier
  finding for Qwen3.5's hybrid ratio exactly; it is a cross-machine confirmation of the same number,
  not an independent new one. The checkpoint also carries a populated `vision_config` and MTP weight
  keys (`mtp.*` present among the 632 tensors) — it is the VL/omni checkpoint even in "text" form,
  consistent with the same nuance already on record from the GPU-tier bring-up. `mlx-lm`'s
  `Model.sanitize()` strips `vision_tower`/`model.visual`/`mtp.*` keys automatically; no manual
  intervention was needed to get a text-only load.

## Result 3 — it actually runs: bf16 direct load, and 4-bit quant via convert

**bf16, zero conversion step** (loaded directly from the raw HF-layout directory — no
`mlx_lm.convert` run first):

```
$ python3 -m mlx_lm.generate --model /private/tmp/qwen35_mlx_test/Qwen3.5-2B \
    --prompt "The capital of France is" --max-tokens 40
==========
Thinking Process:

1.  **Analyze the Request:** The user is asking a simple factual question: "The capital of France is".

2.  **Retrieve Knowledge:** Access knowledge about
==========
Prompt: 15 tokens, 9.257 tokens-per-sec
Generation: 40 tokens, 28.024 tokens-per-sec
Peak memory: 3.811 GB
```

`mx.default_device()` → `Device(gpu, 0)`; `mx.device_info()` → `{'device_name': 'Apple M5', ...}` —
confirmed running on Metal GPU, not a CPU fallback (a CPU fallback for a 632-tensor, 2B-parameter
bf16 model would be far slower than 28 tok/s).

**4-bit quantization via `mlx_lm.convert`** (relevant because this project's benchmark methodology
weighs quantized comparisons heavily — this checks the full toolchain, not just raw-precision load):

```
$ python3 -m mlx_lm.convert --hf-path /private/tmp/qwen35_mlx_test/Qwen3.5-2B \
    --mlx-path /private/tmp/qwen35_mlx_test/Qwen3.5-2B-mlx-4bit -q --q-bits 4 --q-group-size 64
[INFO] Loading
[INFO] Using dtype: bfloat16
[INFO] Quantizing
[INFO] Quantized model with 4.503 bits per weight.
```

Output: 1.0 GiB (down from 4.3 GiB, ≈4.3× compression, consistent with the reported 4.503 bits/weight).
Generation from the quantized copy also produced coherent output, faster and lighter:

```
$ python3 -m mlx_lm.generate --model /private/tmp/qwen35_mlx_test/Qwen3.5-2B-mlx-4bit \
    --prompt "The capital of France is" --max-tokens 40
==========
Thinking Process:

1.  **Analyze the Request:** The user is asking "The capital of France is". This is a straightforward factual question.

2.  **Retrieve Knowledge:** Access
==========
Prompt: 15 tokens, 69.415 tokens-per-sec
Generation: 40 tokens, 91.493 tokens-per-sec
Peak memory: 1.131 GB
```

Hardware/software context matches `mlx_port/README.md` exactly (same machine, same MLX core
version): Apple M5, 32 GiB unified memory, macOS 27.0, MLX core 0.31.2. Any future head-to-head
numbers on this tier will be genuinely same-machine comparable to the existing RWKV-7 MLX numbers,
with no cross-machine normalization caveat needed.

## Why this is credible, not a hollow "it loaded" claim

Three independent signals rule out "it technically imported but is silently broken or running a
degenerate/CPU path":

1. **Output is coherent, structured, on-topic reasoning prose** in both the bf16 and 4-bit runs (not
   repetition, not garbage tokens, not off-topic) — the same bar this project's own GPU-tier
   bring-up notes use for a sanity check at this stage of investigation.
2. **Throughput is in a believable, non-degenerate range**: 28.0 tok/s (bf16) / 91.5 tok/s (4-bit)
   bsz1 decode for a 2B-class hybrid model is in the same ballpark as this project's own RWKV-7 1.5B
   MLX numbers (32–36 tok/s bf16-weight decode, F0037) — not 10× slower (which would suggest an
   accidental CPU fallback) and not suspiciously faster (which would suggest a step got skipped).
   Quantization roughly tripling decode throughput while cutting memory ~3.4× is directionally
   consistent with this project's own RWKV-7 w8/w4 finding (F0039: bsz1 decode is bandwidth-bound,
   so smaller weights directly buy throughput).
3. **The kernel is real engineering, not a stub**: `gated_delta.py`'s hand-written Metal kernel
   (§Result 1) is exactly the kind of per-architecture optimization work a silently-broken or
   thin-wrapper implementation would not bother with.

## Honest limits of this probe (what was *not* checked)

Per this project's rigor discipline, this is a **feasibility smoke test**, not a certified
correctness or performance result, and should not be cited as either without follow-up work:

- **No numerical correctness check was done.** "Coherent, not garbled" is the bring-up-stage bar
  this project's own GPU-tier notes use at this stage, not the oracle-exact bar `mlx_port/` holds
  RWKV-7 to (24/24 greedy-token match against `bench/oracle_numpy.py`, F0037's gate). Before citing
  a Qwen3.5-MLX number next to an oracle-gated RWKV-7-MLX number, a real check (e.g. greedy
  continuation or logit comparison against a reference implementation for a handful of tokens) is
  owed — it wasn't attempted here.
- **Throughput numbers above are single runs**, not `bench_mlx.py`-style gated multi-run medians.
  They are evidence of "it works, in a sane range," not a benchmark result.
- **Only the dense 2B checkpoint was tested**, per the task's explicit scope (no 9B download this
  pass). Whether the 9B checkpoint is dense or routes through `qwen3_5_moe.py` was not re-verified
  on this machine.
- Text-only path; the checkpoint's vision tower was never exercised (stripped by `sanitize()`),
  which is fine for this project's text-only benchmark scope but worth noting explicitly.

## Port-size counterfactual (moot this cycle, included because the task asked for it regardless of outcome)

Since `mlx-lm` already covers Qwen3.5, no from-scratch port is needed and none was built. For the
record, sizing the hypothetical anyway, grounded in real numbers rather than a guess:

`mlx_port/`'s first oracle-gated RWKV-7 commit (`1215c98`) landed **845 lines total**
(`rwkv7_mlx.py` 454, `gate_oracle.py` 153, `bench_mlx.py` 123, `README.md` 115); the model file alone
has since grown to 571 lines (quant support, etc.). It needs exactly **one** state type (the WKV
recurrent state) and **zero** KV-cache or attention-mask concepts, because RWKV-7 is pure-recurrent
end to end.

`mlx-lm`'s own shipped Qwen3.5 implementation — a reasonable proxy for "what a from-scratch port
would have to contain," since it's a real, independent, from-scratch implementation of this exact
architecture — is measurably larger and structurally different: `qwen3_5.py` itself (531 lines) +
its dedicated hand-written Metal delta-rule kernel `gated_delta.py` (283 lines) + the standard
attention/MLP/MoE machinery it reuses from `qwen3_next.py` (491 lines) that a from-scratch port would
otherwise have to write itself + a **two-cache-type** abstraction (`ArraysCache` for the recurrent
layers, real `KVCache` for the attention layers) + **two different mask constructions**
(`create_attention_mask` and `create_ssm_mask`) that have to be routed correctly per layer inside one
decoder loop. Even conservatively, that is comfortably **1.3–2× the RWKV-7 port's core-model line
count**, concentrated exactly where the task's hypothesis said it would be: a hybrid architecture
needs a real KV-cache/attention path *in addition to* a recurrent-state path, which RWKV-7's
pure-recurrent design never has to reconcile. This is a **medium-to-large** counterfactual, not a
small one — the task's structural-complexity hypothesis is confirmed by these numbers, not merely
asserted.

## Decision

**The Apple-Silicon (MLX) tier does not have to be reported as RWKV-only.** Qwen3.5-2B is includable
via `mlx-lm`'s own native, actively-maintained support — no fork, patch, or from-scratch port
required. This mirrors the GPU tier's own methodology (using the opponent's native
serving-framework implementation as the comparison baseline, not hand-rolling a mirror port for
symmetry): the "zero external model deps" discipline in `mlx_port/README.md` and the
zero-FLA policy governing this project's *own* RWKV-7 deliverable were never a requirement to also
hand-write the competitor's architecture — they govern what this project ships as its own RWKV-7
implementation, not the yardstick used to run the competitor.

**Before this becomes a cited benchmark number**, three follow-ups (out of scope for this
feasibility pass):
1. ~~A `bench_mlx.py`-equivalent multi-run median bsz1/prefill sweep for Qwen3.5 on MLX.~~ **Done —
   see [F0045](0045-qwen35-mlx-matched-benchmark.md)**: multi-run bf16/int4 decode+prefill vs RWKV-7
   1.5B, same machine, same protocol. Split result (RWKV-7 wins decode, Qwen3.5 wins prefill, int4
   decode is a near-tie) — not a sweep for either side.
2. A real correctness check beyond "coherent, not garbled" — does not need to clear RWKV-7's
   24/24 oracle-exact bar, but "reads fine" alone is not enough to sit next to an oracle-gated number.
   **Still open** — F0045's int4 coherence sample degenerated into repetition, which is exactly the
   kind of thing this stronger check would quantify properly.
3. Confirm whether the 9B checkpoint is dense or MoE on this stack before scoping the second
   matched-size comparison tier (1.5B↔2B is fully unblocked already; 7.2B↔9B is not yet checked here).
   **Still open.**

## Cross-references

`mlx_port/README.md` (the RWKV-7 MLX port this probe compares against) · F0037 (MLX fused-Metal WKV
default, same hardware/MLX-version baseline used here) · F0038 (M5 hardware ceilings — the bandwidth-bound
decode framing used to sanity-check the throughput numbers above) · F0042 (the other Apple-Silicon
extension probe — CoreML/ANE — which reached a clean negative; this one reaches a clean positive) ·
`mlx_lm/models/qwen3_5.py`, `qwen3_5_moe.py`, `gated_delta.py`, `qwen3_next.py`, `cache.py`, `base.py`
(installed `mlx-lm` 0.31.3, read as reference for both the feasibility check and the port-size
counterfactual).
