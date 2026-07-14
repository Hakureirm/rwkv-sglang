---
doc_kind: finding
finding_id: F0057
title: "RWKV_STATE_FP16 long-context positional compression gate (7.2B fp16, 3090, full N=7500 UncheatableEval corpus, positions to ~3.3k tokens): GREEN — per-bucket Δ(state16−state32) stays inside the ~1e-5-bit same-flag rerun band at every well-sampled bucket including the tail ([2560-3072): +8.9e-6 bits at 457k tokens), signs alternate (6+/4−, no positional slope), pooled Δ −5.4e-8 bpb; the fp16-state per-token perturbation measures ~6e-3 bits RMS, zero-mean, non-compounding — 1,659× smaller than the w4a8 activation tax F0055 resolved on the same instrument; honest scope: the official corpus's longest document is 3,351 tokens, so this gate certifies to ~3.3k positions (within the checkpoint's ctx8192 training), not 8k+"
last_verified_commit: "Mac repo HEAD 7f20446 (RWKV new-files incl. 55e12b7 state-fp16 + 0bf9e27 glue-fusion defaults); box container = sglang main 754524d + upstream_edits.patch 8d2fda3a + these new-files"
discovered_by: Fable 5 (agent), 2026-07-14
severity: info
status: closed — GREEN; RWKV_STATE_FP16 positioning unchanged (documented opt-in, F0056 §6), now with the positional axis certified
related: [F0056, F0055, F0040]
---

# Finding F0057: `RWKV_STATE_FP16` — the positional compression curve (the missing long-context gate)

## 0. Why this gate exists

F0056 landed `RWKV_STATE_FP16` (halved temporal-state storage) behind three green rulers —
lambada Δ+0.0002 @1.5B, pooled compression Δ+1e-6 bpb @1.5B ctx≤4000, MATH500 avg@64 −0.32pt
@7.2B — but all three are **short-to-mid context** instruments (MATH500 generations average
~480 tokens). State-precision error is precisely the class of error that could compound with
sequence position: the fp16 state is re-read and re-written every step, so a per-step rounding
tax would show up as a positional *slope* even where pooled numbers stay flat. The
compression-vs-position curve is the instrument that resolves exactly that axis (it is also the
axis the UncheatableEval project itself surfaces, via byte-wise tracking + its Compression-Lens
visualization). This finding closes that gap for the 7.2B tier — and scopes honestly what "long"
can mean on this corpus.

## 1. Setup

- **Box**: RTX 3090 24 GB (sm86), long-lived `rwkvmain` container (sglang **main** lineage,
  git checkout `754524d`), same container class as F0055's cert. GPU idle-verified before,
  between, and after legs (1 MiB / 0%).
- **Code**: sglang main + `sglang_main_port/upstream_edits.patch` (10 upstream files, md5
  `8d2fda3a…`, byte-identical Mac↔box) + the RWKV new-files at F0056 HEAD (`RWKV_STATE_FP16`
  in `configs/rwkv7.py` / `rwkv7_backend.py` / `wkv_recurrent.py`; all five glue fusions
  default in `models/rwkv7.py` + kernels, incl. `rwkv7_ln.cu`/`ln_fused.py` new since the
  box's last deploy). Deploy-repair note, for the paper trail: an initial whole-overlay copy
  onto the main-lineage tree broke 6 infra files (v0.5.10↔main skew — `server_args.py`,
  `model_runner.py`, `model_runner_kv_cache_mixin.py`, `attention_registry.py`,
  `configs/__init__.py`, `configs/mamba_utils.py`, plus a `hf_transformers_utils.py` that was
  never main-compatible); recovered via `git checkout` to stock + patch re-application
  (`model_runner.py` hunks landed with fuzz≤2/offset≤35), import- and boot-verified. The
  container is left in this state deliberately: it is now F0056-current (state-fp16 +
  fusions), verified by two full server boots and three eval runs.
- **Model**: `rwkv7-7.2b-fla` fp16 (**not** a quant checkpoint — isolates STATE precision).
  Checkpoint provenance: `rwkv7-g1g-7.2b-20260523-ctx8192.pth` (ModelScope `.msc` metadata) —
  **ctx8192-trained**, so every evaluated position in this run is comfortably inside the
  trained context; no beyond-train-ctx caveat applies to any number below.
- **Serving**: canonical `scripts/serve.sh` throughput mode (fast-path stack, 11 RWKV_* envs
  incl. the five F0056 glue fusions — verified from the live server's `/proc/<pid>/environ`),
  `--dtype float16 --attention-backend triton --page-size 1 --disable-piecewise-cuda-graph
  --disable-radix-cache --chunked-prefill-size 4096`, `CGMAXBS=32` sized to the harness
  concurrency (32), `MEMFRAC=0.90 --max-running-requests 48` (24 GB card: serve.sh's default
  512 state slots × 33 MB/req fp32 does not fit beside 14.4 GB of weights; 48 slots covers
  concurrency 32 with headroom).
- **Two legs, identical boots except the env**: leg A default (fp32 state), leg B
  `RWKV_STATE_FP16=1`. One at a time. **Flag-engagement evidence (leg B)**: the boot log's
  pool line halves exactly — `ssm_state size: 1.53GB` (A) → `0.77GB` (B) at the same 48
  slots — and `RWKV_STATE_FP16=1` present in the scheduler process's `/proc/environ`.

## 2. Instrument

`bench/uncheatable_eval.py` (faithful port of Jellyfish042 uncheatable_eval; REF chunking
`[0]+chunk`, CE over every real token, bpb/compression formulas at REF L761-763), scored via
our server's `/generate` `input_token_logprobs` (smoke-verified on this lineage: leading
`None`, token-id alignment asserts). Corpus: the **full** on-box UncheatableEval-2026-04 set —
15 categories × 500 docs = 7,500 documents (`bench/data/uncheatable_full/`), the same corpus
as the landed pooled anchor. `--ctx-len 4000`, concurrency 32.

**Position resolution (new this finding):** the position curve's `[1024,∞)` catch-all was
subdivided into 512-token bins up to 3584 (`[1024,1536) … [3072,3584) [3584,∞)`); the first
five bucket edges are unchanged, so historical `*_curve.csv` baselines remain bucket-for-bucket
comparable on `[0,1024)`. Corpus length reality (measured with the model's own tokenizer over
all 7,500 docs): max document = **3,351 tokens**, p99 = 3,075, p90 = 2,853; zero documents
exceed 4,096. Two consequences, stated plainly:

1. **The honest max evaluated position is ~3.3k tokens.** The official corpus is a
   short-document corpus (fresh monthly news/wiki/papers/code); there is no 8k/16k data to
   score without leaving the official protocol, and no longer historical corpus exists on this
   box (searched). A true 8k+ positional gate needs a non-Uncheatable long-doc corpus — the
   residual gap, flagged, not silently papered over.
2. Because max doc (3,351) < chunk size (4,000), **no document is ever chunked**: every doc is
   scored in one pass with continuous recurrent state, so token positions are true absolute
   positions with no state resets — exactly the regime a state-error gate wants (and de-facto
   equivalent to the official byte-wise tracker's `enable_chunking=False` requirement).

**Noise instrument**: ao3_english (the tail-heaviest category, p50 = 2,870 tokens) rerun
**twice** on the same leg-A server, same flag — per-bucket |Δ| between identical runs is
≤ 6.6e-6 bits (most buckets ~1e-6), pooled |Δ| = 1.0e-8 bpb. The scoring path is nearly
deterministic under concurrency-32 batching; **~1e-5 bits/bucket** is a conservative rerun
band at these token counts. Raws: `*_noiserep{1,2}.json`.

## 3. Results

**Pooled anchors (cross-validation):**

| leg | pooled bpb | pooled compression % | wall |
|---|---|---|---|
| landed anchor (`uncheatable_7.2b_fp16_HEAD_full.json`, same box+corpus, HEAD-era code) | 0.5413283 | 6.766603 | 4570 s |
| leg A (fp32 state) | **0.5413279** | 6.766599 | 4217 s |
| leg B (fp16 state) | **0.5413279** | 6.766598 | 4046 s (−4.0%) |
| **Δ (B − A)** | **−5.4e-8** | −1e-6 | — |

Leg A reproduces the landed anchor to **−3.4e-7 bpb** across a different code lineage
(HEAD-era vs main@754524d + F0056 kernels) and a different day — the instrument is solid.
(Side observation: leg B's wall is 4% shorter; halving state bytes speeds up even this
prefill-dominated workload.)

**The key table — per-position-bucket Δ (mean −log₂p per token, B − A):**

| bucket | tokens | leg A | leg B | Δ (B−A) | vs ~1e-5 band |
|---|---|---|---|---|---|
| [0-64) | 480,000 | 3.362115 | 3.362117 | +2.1e-6 | 0.21× |
| [64-128) | 480,000 | 2.509084 | 2.509084 | +3.5e-7 | 0.04× |
| [128-256) | 958,794 | 2.373577 | 2.373577 | +1.8e-8 | 0.00× |
| [256-512) | 1,797,091 | 2.187125 | 2.187122 | −2.4e-6 | 0.24× |
| [512-1024) | 3,045,215 | 2.105511 | 2.105512 | +1.2e-6 | 0.12× |
| [1024-1536) | 2,590,013 | 2.016042 | 2.016041 | −6.2e-7 | 0.06× |
| [1536-2048) | 2,020,353 | 1.899429 | 1.899426 | −3.7e-6 | 0.37× |
| [2048-2560) | 1,336,324 | 1.979415 | 1.979417 | +1.3e-6 | 0.13× |
| **[2560-3072)** | **457,182** | 2.226247 | 2.226255 | **+8.9e-6** | **0.89×** |
| [3072-3584) | 3,443 | 2.172960 | 2.172912 | −4.8e-5 | see below |
| [3584+) | 0 | — | — | (empty by corpus construction) | — |

- **Every well-sampled bucket (≥457k tokens) is inside the rerun band**, including the last
  well-sampled tail bucket [2560-3072) at 0.89×. Signs alternate (`+++-+--++-`, 6+/4−): a
  zero-mean wobble, **no positional slope** — the state error does not compound with position.
- **The [3072-3584) bin (−4.8e-5) is statistical dust, not a tail signal**, on two grounds:
  (a) it holds only 3,443 tokens — treating the per-token fp16-state perturbation as i.i.d.,
  the [2560-3072) delta implies σ_token ≈ |8.9e-6|·√457,182 ≈ **6.0e-3 bits/token RMS**, which
  predicts a mean wobble of ~1.0e-4 at n=3,443; the observed −4.8e-5 is **0.47×** that — fully
  expected magnitude; (b) its **sign is negative** (fp16 state scoring *better*) — the opposite
  of an error-accumulation signature.
- Per-dataset pooled deltas: all 15 corpora within ±1.5e-6 bpb, no outlier category.
- Calibration against the sibling instrument use: F0055's w4a8 activation-quant cert measured
  a positional tax of +0.0147…+0.0248 bits per bucket on this same curve — the largest
  well-sampled delta here is **1,659× smaller**. Where a8 was plainly visible, state-fp16 is
  indistinguishable from rerun noise.

**Verdict (pre-registered rule — GREEN iff Δ inside the noise band at every bucket including
the tail): GREEN.** Positions certified to ~3.3k tokens (the corpus max), all within the
checkpoint's ctx8192 training. The F0056 positioning (documented opt-in throughput switch,
default OFF to keep the zero-flag bitwise-oracle tier) is unchanged — this finding removes the
"unverified beyond short context" asterisk up to 3.3k and sharpens the F0055/F0056 cross-cert:
RWKV-7's precision-sensitive axis is the *activations*, not the state's storage width, and
that now holds **per-position**, not just pooled.

## 4. Harness ↔ official-protocol alignment note (bonus check, 2026-07-14)

- Upstream repo is `Jellyfish042/uncheatable_eval` (GitHub; the "UncheatableEval" spelling is
  the HF space/dataset name). Our cached REF snapshot
  (`scratchpad/official_evals/uncheatable_evaluator.py`, fetched 2026-07-03) still matches the
  current upstream entry points (`eval_single.py` + `EvaluationConfig`, `chunk_size=4000`
  default, `load_data_smart` formats, bos/eod handling) — no drift that touches our port.
- The official positional instrument is `track_byte_wise_data` (requires
  `enable_chunking=False`) plus the **Compression-Lens** HF space, which plots per-BYTE-position
  loss for a single pasted document (input cap `MAX_TEXT_LENGTH = 16384`). Ours bins per-TOKEN
  position pooled over the corpus. Same axis in substance; two honest differences — byte vs
  token x-axis, single-document vs corpus aggregation — and both cancel in this finding's A/B
  delta (identical binning on both legs). Not reworked here.
- Corpus vintage: box corpus is the 2026-04 monthly snapshot (fetched 2026-07-03); newer
  monthly snapshots likely exist upstream. Noted, not blocking: both legs share the corpus and
  the pooled anchor is same-corpus.

## 5. Artifacts

`bench/results/uncheatable_positional_7.2b_fp16_state{32,16}_3090.json` (+ `_curve.csv`),
`bench/results/uncheatable_positional_7.2b_fp16_state32_3090_noiserep{1,2}.json`
(+ `_curve.csv`) — landed in this repo with the model field normalized to the checkpoint
basename (house convention), infra-identifier grep clean. Box copies remain in the box repo's
`bench/results/`. Server boot/eval logs on the box container at `/tmp/leg{A,B}_boot.log`,
`/tmp/leg{A,B}_eval.log`, `/tmp/noiserep{1,2}.log`.

## Cross-references

[[F0056]] (the flag under test; §4's gate ladder is what this extends to the positional axis) ·
[[F0055]] (same curve instrument resolving the w4a8 activation tax 3 orders of magnitude above
this measurement — the contrast that makes GREEN meaningful) · `docs/BENCHMARKS.md` §2
(compression ruler discipline) · [[F0040]] (the 1.5B/MLX compression-curve precedent).
