# M4 — RWKV-7 × sglang 8/4-bit quantization

> ⚠️ **The int8 SPEED tables below were measured on the earlier co-tenant GPU (superseded).**
> The clean, exclusive-GPU int8 numbers are in `comparison_clean.md` (int8-vs-albatross-fp16 +
> int8-vs-bf16 bonus rows): int8-vs-bf16 decode is **+46-59% at 1.5B/7.2B** but **−10% at 0.1B
> bsz1** (small model, launch-bound). The weight-byte savings (−41/−46%) here are deterministic
> (safetensors-summed) and stand.

Box: `gpu-box` (1× RTX 3090, sm_86 Ampere, INT8 tensor cores; **no fp8** — needs
Hopper sm_89+). sglang **0.5.10.post1** (torch 2.9.1/cu128). Compute dtype bf16,
cuda-graph ON, radix-cache OFF (RWKV production config). VRAM via nvidia-smi.

## Path used

1. **Model refactor (regression-gated):** the linear projections in
   `models/rwkv7.py` (attn `r/k/v/o_proj`, ffn `key/value`) and the LoRA down/up
   linears (`{w,a,g,v}_lora.lora.0/.2`) were switched from `nn.Linear` to sglang's
   quant-aware **`ReplicatedLinear`** (tp=1), threaded with `quant_config`. With
   `quant_config=None` they use sglang `UnquantizedLinearMethod` = `F.linear`,
   bit-identical to `nn.Linear`.
   - **Regression gate PASSED — greedy still EXACT at bf16** (no quant):
     0.1B 24/24, 1.5B 24/24 (and 0.1B fp32 24/24). The refactor is safe.
2. **8-bit = native sglang `w8a8_int8`** (per-channel symmetric int8 **weight**,
   per-token dynamic int8 **activation**, sgl_kernel `int8_scaled_mm` on the
   Ampere INT8 tensor cores). Weight scales are a closed-form per-output-channel
   max — **no calibration data needed**. Offline converter
   `tools/quantize_w8a8_int8.py` emits `<lin>.weight` (int8) + `<lin>.weight_scale`
   (fp32) and writes `quantization_config={"quant_method":"w8a8_int8"}` into
   config.json so `sgl.Engine(model_path=…)` auto-loads it (no CLI flag needed).
3. **WKV recurrence/state and the small per-channel params** (x_*, k_k, k_a, r_k,
   g_norm, all norms, embeddings, lm_head, biases) are **NOT quantized** — bf16/fp32.
   FLA-free property preserved (no fla reintroduced).

Model-weight VRAM (deterministic, sum of tensor bytes):

| model | bf16 weights | w8a8-int8 weights | saved | quantized-linears only |
|---|---|---|---|---|
| 1.5B | 3054.8 MB | 1799.4 MB | **−1255 MB (−41%)** | 2517.9 → 1262.5 MB (**2.0×**) |
| 7.2B | ~13856 MB | ~7203 MB | **~−6653 MB (~−48%)** | 13319.5 → 6666.7 MB (**2.0×**) |

(embeddings+lm_head are unquantized and equal in both — 536.9 MB at 1.5B.)

> NOTE: the 7.2B row is an earlier rough estimate. The **authoritative** per-dtype
> weight footprint (summed directly from the safetensors headers) is in
> `comparison_clean.md`: 7.2B bf16 13731.3 → int8 7386.6 MiB = **−46%** (1.5B −41%
> matches). Cite the −46% figure; this ~48% predates that measurement.

## 8-bit (w8a8-int8) — 1.5B

Decode tok/s (steady-state, prefill-subtracted), cuda-graph captured to bs32:

| bsz | bf16 | w8a8-int8 | int8 vs bf16 |
|---|---|---|---|
| 1  | 142.1 | 163.4 | **+15.0%** |
| 8  | 863.4 | 1156.7 | **+34.0%** |
| 32 | 2753.4 | 3275.4 | **+19.0%** |

Prefill tok/s (prompt 1024):

| bsz | bf16 | w8a8-int8 |
|---|---|---|
| 1  | 9434.9 | 6635.1 (−30%, per-token quant overhead dominates tiny token count) |
| 8  | 12813.4 | 13605.0 (+6%) |
| 32 | 12916.3 | 13787.0 (+7%) |

Peak VRAM (nvidia-smi footprint = peak − 1304 MiB baseline; mem_fraction=0.5):

| bsz | bf16 | w8a8-int8 |
|---|---|---|
| 1 | 7210 | 7014 |
| 8 | 7454 | 7258 |

> NB sglang's `mem_fraction_static` makes total VRAM ≈ a fixed fraction of the GPU
> regardless of model size — the elastic state-cache pool **reinvests** the freed
> weight bytes into more cache headroom, so peak-VRAM at fixed mem_fraction
> *understates* the win. The honest VRAM↓ is the model-weight table above (−41%).

**Accuracy (greedy free-running vs the bf16/numpy oracle fixture):**
- 1.5B w8a8-int8: **12/24** tokens match (diverges at token 12). Free-running is
  cascade-sensitive — the first 12 tokens are bit-for-bit the bf16 trajectory,
  then one near-tie logit flips and the suffix cascades. Quantizing the LoRAs vs
  keeping them bf16 made no real difference (10/24 vs 12/24 — same noise band), so
  the LoRAs are not the accuracy bottleneck; full-quant is kept (per spec).

## 8-bit (w8a8-int8) — 7.2B

- **Accuracy: greedy 8/8 EXACT vs the oracle fixture** — at 7.2B the model absorbs
  int8 weight noise with zero greedy drift on the fixture. (Quant accuracy improves
  with scale, the opposite of the small-model drift above.)
- Weights: 13.9 GB bf16 → 7.2 GB int8 (**−6.65 GB, −48%**).

Decode tok/s (cuda-graph, mem_fraction 0.85, prefill 512):

| bsz | bf16 | w8a8-int8 | int8 vs bf16 |
|---|---|---|---|
| 1 | 43.6 | 66.9 | **+53.4%** |
| 8 | 315.0 | 464.4 | **+47.4%** |

Prefill tok/s:

| bsz | bf16 | w8a8-int8 |
|---|---|---|
| 1 | 2744.5 | 2372.1 (−14%) |
| 8 | 3551.4 | 6365.2 (**+79%**) |

Peak VRAM (nvidia-smi footprint, mem_fraction 0.85):

| bsz | bf16 | w8a8-int8 | saved |
|---|---|---|---|
| 1 | 16954 MiB | 13678 MiB | **−3276 MiB** |
| 8 | 17488 MiB | 14210 MiB | **−3278 MiB** |

> At 7.2B the win is unambiguous on **every** axis even through sglang's elastic
> cache: decode **+47–53%**, prefill +79% at batch, peak VRAM **−3.3 GB**, greedy
> **EXACT**. Quant pays off more at scale (weight bandwidth dominates decode, and
> the GEMMs are large enough for the INT8 tensor cores to win on prefill too).

## 4-bit — blocked on this box/version (honest)

Attempted **bnb nf4** (the only *on-the-fly* 4-bit sglang exposes; AWQ/GPTQ need
offline calibration). The model is bnb-ready (`bitsandbytes_stacked_params_mapping`,
`default_bitsandbytes_target_modules` added; `bitsandbytes==0.49.2` installed via
uv). Probe: `bench/quant_4bit_bnb.py`. Result — **boots into a loader bug in the
pinned sglang 0.5.10.post1**:

```
KeyError: [rwkv7.load_weights] unexpected checkpoint key:
          model.layers.0.attn.a_lora.lora.0.qweight
```

Root cause: `BitsAndBytesModelLoader._unquantized_generator` renames every target
`*.weight` → `*.qweight` when yielding (loader.py:1779), but
`BitsAndBytesLinearMethod.create_weights` registers the param as `weight`
(bitsandbytes.py:267), and the loader's post-load step then asserts the `.qweight`
name is in `named_parameters()` (loader.py:1915). The two halves disagree, so the
non-stacked on-the-fly bnb path doesn't load for *any* `ReplicatedLinear`-based
model in this version — not specific to RWKV. Making it work needs patching
sglang's `loader.py` (outside the clean model overlay), not just our `load_weights`.

Even if integrated, bnb itself prints *"bitsandbytes quantization is not fully
optimized yet. The speed can be slower than non-quantized models."* — i.e. it would
likely **fail the "not slower than bf16" gate** (it's the same class as the upstream
rwkv-pip int8 we already beat). AWQ/GPTQ 4-bit need a calibration dataset + an
offline quantizer (autoawq/auto-gptq), infeasible on this air-gapped box (no HF /
dataset access). **fp8** (w8a8_fp8) needs Hopper sm_89+ — unavailable on the sm_86
3090.

Deterministic 4-bit VRAM projection (had it integrated, int4 ≈ 4× on the linears):
1.5B linears 2517.9 MB → ~630 MB; total weights 3054.8 → ~1167 MB (−62%). Real
but unverified — not claimed as a result.

## Verdict

- **w8a8-int8 meets AND beats our 8-bit goal on the 3090 (quant no slower than
  16-bit): VRAM↓ AND decode *faster* than bf16** — not merely "not slower".
  - 1.5B decode: +15% / +34% / +19% (bsz 1/8/32); weights −41%.
  - 7.2B decode: **+53% / +47%** (bsz 1/8); peak VRAM **−3.3 GB**; prefill +79% at
    bsz8; greedy **8/8 EXACT**. The win grows with model size.
  - This beats upstream rwkv-pip int8 (which is *slower* than fp16). Differentiator:
    sglang's INT8 tensor-core `int8_scaled_mm` + per-token dynamic activation quant,
    vs bnb-style LLM.int8 mixed-precision decomposition.
- **Best method on Ampere/3090 = w8a8-int8 (8-bit).** fp8 needs Hopper; 4-bit is
  blocked (bnb loader bug in pinned sglang + offline-calibration infeasibility).
- Accuracy: greedy drift is negligible at 7.2B (EXACT) and modest at 1.5B (12/24
  free-running, cascade-sensitive). Larger models absorb int8 weight noise better.
- Decode gains are below the naive "halve the bytes ⇒ 2×" because RWKV decode also
  runs the **unquantized** WKV recurrence + token-shift + GroupNorm and pays a
  per-token activation-quant cost; only the linear weight reads halve. Still a clear
  win, larger at scale (7.2B) and batch (more GEMM-bound).

## Reproduce

```bash
# 1) deploy the quant-aware model overlay
bash scripts/deploy.sh
# 2) produce an int8 checkpoint (offline, no calibration data)
~/envs/rwkv-sgl/bin/python tools/quantize_w8a8_int8.py \
    --src /home/user/rwkv_models/rwkv7-1.5b-fla \
    --dst /home/user/rwkv_models/rwkv7-1.5b-w8a8
# 3) greedy accuracy (auto-detects quant_method from config.json)
~/envs/rwkv-sgl/bin/python bench/verify_m1d.py \
    --model /home/user/rwkv_models/rwkv7-1.5b-w8a8 \
    --fixture bench/fixtures/oracle_rwkv7_15b_eiffel.json --dtype bfloat16
# 4) throughput + VRAM, bf16 vs int8
~/envs/rwkv-sgl/bin/python bench/throughput.py --model <dir> --dtype bfloat16 \
    --batch-sizes 1,8,32 --cuda-graph --cuda-graph-max-bs 32 --disable-radix-cache
```
