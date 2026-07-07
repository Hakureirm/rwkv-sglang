---
doc_kind: finding
finding_id: F0050
title: "Qwen3.5-2B correctness gate against Bo Peng's independent numpy fp32 reference: PASSES on both live serving paths this project publishes numbers from (mlx-lm bf16 on Apple Silicon, sglang bf16 on the 5090 tower) — top-1 exact match and identical top-10 token sets on all three independent implementations; this project's Qwen3.5-2B numbers can now be cited as genuinely Qwen3.5, not merely 'reads coherently'"
last_verified_commit: "HEAD"
discovered_by: Sonnet 5 (agent-assisted), 2026-07-07
severity: info
status: open
related: [F0044, F0045, F0048, F0049]
---

# Finding F0050: Qwen3.5-2B numpy-oracle correctness gate

## Context / question being asked

Every RWKV-7 number this project publishes passes a bit-exact oracle gate first:
`bench/oracle_numpy.py` / `mlx_port/gate_oracle.py` check 24/24 greedy tokens against a pure-numpy
fp32 reference before any speed number is trusted. Qwen3.5 — this project's comparison target —
has never been checked the same way. F0044 and F0045 both explicitly flagged this gap in their own
follow-up lists ("no oracle gate — none exists for Qwen3.5 in this repo," "correctness verification
currently stops at coherent/non-garbled, not the 24/24 oracle-exact bar RWKV-7 requires"). This
finding closes that gap for the 2B tier: does the Qwen3.5-2B this project actually serves (via
mlx-lm and via sglang) agree with an independent, from-scratch numerics reference, or could the
published Qwen3.5 numbers be quietly describing a subtly-misconfigured variant?

Bo Peng (RWKV's creator) maintains exactly the missing reference:
[`run_rwkv7_qwen35.py`](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py) —
an independent pure-numpy fp32 implementation of both RWKV-7 and Qwen3.5's forward pass (both are
DPLR/gated-linear-attention architectures, which is why one script covers both), plus a companion
[`run_qwen35_make_pth.py`](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_qwen35_make_pth.py)
that flattens an HF Qwen3.5 checkpoint into the `.pth` state-dict format the reference expects. Both
were written and pinned against `Qwen/Qwen3.5-0.8B`; this project's actual comparison tier is
Qwen3.5-2B, so the first job was verifying — not assuming — that both scripts generalize across the
Qwen3.5 dense family.

## Method

1. Fetched both scripts fresh (2026-07-07); vendored unmodified copies at `qwen35_gate/vendor/` for
   diffing against any future upstream changes.
2. Reused an already-downloaded, integrity-verified `Qwen/Qwen3.5-2B` HF checkpoint at
   `/private/tmp/qwen35_mlx_test/Qwen3.5-2B` (bf16, the same checkpoint F0044/F0045 already
   confirmed byte-complete: 632/632 safetensors keys, sizes matching `index.json`) — no re-download
   needed.
3. Ran `run_qwen35_make_pth.py` against it unmodified (see generalization findings below) to
   produce a flat text-only `.pth`.
4. Adapted the `Qwen35` class from `run_rwkv7_qwen35.py` (see generalization findings) into
   `qwen35_gate/numpy_reference.py`, and ran it on the probe text `" Eiffel"` (the upstream script's
   own default probe) to get top-10 next-token logits/probabilities.
5. Got the *same* probe's top-10 next-token distribution from two actually-running serving paths —
   the two this project's Qwen3.5 numbers are actually published from:
   - **mlx-lm 0.31.3** (Apple Silicon tier, F0044/F0045's serving path), via `generate_step`'s
     public per-step logprobs API.
   - **sglang** (cloud tier, this project's headline serving path), via an ephemeral container on
     the 5090 tower running the same `lmsysorg/sglang:dev-cu12` image and `--dtype bfloat16
     --trust-remote-code` flags this project's own cloud-tier benchmarks use, queried through
     sglang's native `/generate` endpoint with `return_logprob`/`top_logprobs_num`.
   - Both were fed the **identical token IDs** the numpy reference used (via the same
     `transformers.AutoTokenizer.encode(text, add_special_tokens=False)` call, bypassing each
     runtime's own tokenizer wrapper) so any disagreement found is isolated to model math, not
     tokenization drift between harnesses.
6. Compared all three pairwise: top-1 token match, top-5 token-*set* match, max absolute
   probability delta on the tokens shared across both top-10 lists.

## Generalization findings (0.8B → 2B) — verified, not assumed

`run_qwen35_make_pth.py` **needed zero changes**. It's driven entirely by key-name prefixes
(`model.language_model.*` stripped, `visual.*`/`vision_tower.*` dropped, `mtp.*` dropped), not by
any hardcoded shape — confirmed by running it against the 2B checkpoint and checking its own
self-reported metadata: `kept_tensors: 320`, `skipped_vision_tensors: 297`,
`skipped_mtp_tensors: 15` (320+297+15 = 632, the full checkpoint, with no overlap between
categories), `numel: 1,881,825,088`.

`run_rwkv7_qwen35.py`'s `Qwen35` class **needed one real fix**: it hardcodes `self.C = 1024`
(hidden size) — correct for 0.8B, wrong for 2B (`config.json` reports `hidden_size: 2048`). Running
it unmodified against the 2B `.pth` would have silently produced wrong matrix-multiply shapes and
either crashed or (worse) run with truncated/garbage weight slicing. Fixed by deriving it from the
checkpoint itself: `self.C = W["embed_tokens.weight"].shape[1]`. `self.n_layer` (hardcoded `24`)
was also switched to a derived value for robustness, though it happens to still be 24 for 2B.

Every *other* hardcoded architecture constant in the same `__init__` — `H=16, N=128` (linear-attn
heads/head-dim), `conv_len=4`, `aH=8, aKV=2, aN=256` (full-attention heads/kv-heads/head-dim), the
`i % 4 != 3` hybrid-layer pattern, and the GQA rope constants (base `10000000.0`, `rope_dim = N //
4`) — was cross-checked **by hand** against 2B's `config.json` (`linear_num_key_heads`,
`linear_key_head_dim`, `linear_conv_kernel_dim`, `num_attention_heads`, `num_key_value_heads`,
`head_dim`, `full_attention_interval`, `rope_parameters.rope_theta=10000000`,
`rope_parameters.partial_rotary_factor=0.25` ⇒ `rope_dim = head_dim // 4 = 64`) and found to match
exactly. They were **left hardcoded** because they were confirmed correct for 2B, not because
they're assumed to generalize further — a future 4B/9B run through this same script must re-verify
each one against that size's own `config.json` before trusting it (this project's own 9B tier has
**not** been checked yet; see Scope below).

**Incidental cross-validation**: `mlx-lm`'s own from-scratch HF-checkpoint sanitizer
(`qwen3_5.TextModel.sanitize()`) independently applies the same `+1.0` shift to RMSNorm weights
that Bo's script hardcodes throughout (`W[...] + 1`) whenever it detects an unsanitized raw
checkpoint (conv1d weight not yet axis-permuted, or MTP weights still present — both true of the
raw HF download). Two independently-written implementations converging on the same non-obvious
checkpoint convention (Qwen3.5 stores RMSNorm weights as a delta from identity, not the weight
itself) is good evidence this quirk is being handled correctly in both, rather than the two
agreeing because they share a bug.

## Result

Probe `" Eiffel"` tokenizes to `[242476, 300]` (`" Eiff"` + `"el"` — not a single token; there is
no preceding context, so a generic-punctuation top-1 like `"\n"` is an unsurprising completion of
an isolated two-token fragment, not a sign of misconfiguration).

| rank | numpy fp32 (reference) | mlx-lm bf16 (live) | sglang bf16 (live) |
|---|---|---|---|
| 1 | tok=198 `'\n'` p=0.10807 | tok=198 `'\n'` p=0.10540 | tok=198 `'\n'` p=0.11157 |
| 2 | tok=11 `','` p=0.05592 | tok=11 `','` p=0.05642 | tok=11 `','` p=0.05617 |
| 3 | tok=271 `'\n\n'` p=0.04851 | tok=271 `'\n\n'` p=0.04677 | tok=271 `'\n\n'` p=0.04956 |
| 4 | tok=369 `' is'` p=0.04265 | tok=369 `' is'` p=0.04394 | tok=369 `' is'` p=0.04366 |
| 5 | tok=318 `' ('` p=0.04160 | tok=318 `' ('` p=0.04127 | tok=318 `' ('` p=0.04106 |
| 6 | tok=220 `' '` p=0.03203 | tok=220 `' '` p=0.03214 | tok=220 `' '` p=0.03202 |
| 7 | tok=25 `':'` p=0.02344 | tok=13 `'.'` p=0.02209 | tok=13 `'.'` p=0.02338 |
| 8 | tok=13 `'.'` p=0.02215 | tok=25 `':'` p=0.02209 | tok=25 `':'` p=0.02195 |
| 9 | tok=579 `"'s"` p=0.02133 | tok=579 `"'s"` p=0.02209 | tok=579 `"'s"` p=0.02114 |
| 10 | tok=321 `' and'` p=0.00908 | tok=321 `' and'` p=0.00921 | tok=321 `' and'` p=0.00915 |

Top-1 through top-6 are rank-identical and token-identical across **all three** independent
implementations (pure-numpy fp32 on CPU, mlx-lm bf16 on Metal, sglang bf16 on CUDA/flashinfer).
Ranks 7–9 show a harmless reshuffle among three tokens whose probabilities are within ~0.002 of
each other even in the fp32 reference — exactly the kind of noise expected from bf16 rounding, not
a disagreement. The top-10 **token set is identical, 10/10, across all three.**

| comparison | top-1 | top-5 set | shared tokens | max abs prob diff | verdict |
|---|---|---|---|---|---|
| numpy fp32 vs mlx-lm bf16 | match | match | 10/10 | 0.00267 | **PASS** |
| numpy fp32 vs sglang bf16 | match | match | 10/10 | 0.00358 | **PASS** |

Both live serving paths this project actually publishes Qwen3.5-2B numbers from agree with the
independent reference well inside the noise expected from a genuine fp32-vs-bf16 precision gap
(sub-1-percentage-point on every shared token, no token appearing in one top-10 and absent from
another). **Verdict: the Qwen3.5-2B this project has been benchmarking is genuinely Qwen3.5-2B —
not a wrong-config, wrong-checkpoint, or silently-broken variant.** This directly retires the risk
flagged as an open gap in F0044/F0045: the project's Qwen3.5 accuracy/speed numbers no longer rest
on "reads coherently" alone.

This is also a meaningful check specifically for the sglang path: this project's own cloud-tier
bring-up notes (`memory/project-qwen35-benchmark.md`, first 2026-07-07 round) already found one
real dtype bug in sglang's hybrid-SSM kernels for this exact model family (the `--dtype float16`
causal-conv1d triton crash, worked around with `--dtype bfloat16`). A numerically-silent kernel bug
— wrong-but-not-crashing — is exactly the failure mode a gate like this is for for, and mlx-lm
agreeing with numpy does not, by itself, rule that out for sglang's independent kernel
implementation. Running the sglang leg specifically (rather than treating the mlx-lm match as
sufficient) is what actually retires that risk.

## Scope / what this does NOT establish

- **2B only.** This project also publishes Qwen3.5-9B numbers (F0048 and others) at similar
  weight/effort. The 9B tier has not been run through this gate. The generalization findings above
  ("everything but hidden_size/n_layer was hand-verified against 2B's config, not assumed") apply
  *only* to 2B — 9B's own constants would need the same by-hand check against its own
  `config.json` before this gate could be trusted there (e.g. its `hidden_size`/`n_layer`/head
  counts could easily differ again, the same way 2B's `hidden_size` differed from 0.8B's).
- **Single probe position, not a decode oracle.** This checks one forward pass's next-token
  distribution, matching the upstream script's own scope — not a 24-step greedy-decode oracle the
  way `bench/oracle_numpy.py` gates RWKV-7. It would not catch a bug that only manifests after
  several autoregressive steps (e.g. a subtly wrong recurrent-state update that only drifts
  visibly after N steps). Extending to multi-step greedy-decode agreement is a natural next
  hardening step if this gate is promoted to a pre-publish requirement rather than a one-time
  check.
- **Tolerance, not bit-exactness.** The `PASS` bar here is top-1 match + top-5 set match + max
  prob delta under an 0.02 absolute tolerance — deliberately looser than RWKV-7's 24/24 bit-exact
  gate, because fp32-CPU vs bf16-Metal vs bf16-CUDA are legitimately different numeric paths (this
  mirrors this project's own fp16-vs-bf16 discipline elsewhere, e.g. why RWKV-7's own hand kernels
  are fp16-gated and silently no-op under bf16). A tighter tolerance would start rejecting
  legitimate precision noise, not real bugs.

## Process / guardrails

The sglang leg used a fifth, ephemeral, single-purpose container on the 5090 tower (image already
present locally, no pull needed), boot-to-health in 30s, queried once, then `docker stop && docker
rm` immediately — GPU back to the 10 MiB idle baseline afterward. The project's four
always-on containers were not touched: `docker ps` before and after this session shows identical
names and monotonically-increasing uptimes (23h/35h/35h/44h/45h before → same, +minutes after),
confirming continuity.

## Files

- `qwen35_gate/vendor/{run_rwkv7_qwen35.py,run_qwen35_make_pth.py}` — unmodified upstream scripts.
- `qwen35_gate/numpy_reference.py` — adapted `Qwen35` class (see module docstring for the exact
  diff from upstream).
- `qwen35_gate/mlx_probe.py` — mlx-lm live probe.
- `qwen35_gate/gate_qwen35.py` — reusable driver (numpy vs mlx-lm, optionally vs a live sglang
  server); prints a machine-readable `GATE_QWEN35_{PASS,FAIL}` marker.
- `qwen35_gate/results/qwen35_2b_eiffel_gate_20260707.json` — full three-way raw result from this
  run.
- Converted checkpoint (`qwen35_2b_text.pth`, ~4.3GB) and the source HF checkpoint directory are
  both outside the repo (scratch/cache paths), not committed — regenerate via
  `qwen35_gate/vendor/run_qwen35_make_pth.py` per the README.

## Cross-references

F0044 (MLX feasibility — first flagged the missing-oracle gap) · F0045 (MLX matched benchmark —
repeated the same flag) · F0048 (int8 tier gap) · F0049 (desktop-tier 3090 comparison) ·
`memory/project-qwen35-benchmark.md` (full round-by-round log, including the sglang
`--dtype bfloat16` boot-fix this finding's sglang leg reuses) · `bench/oracle_numpy.py` /
`mlx_port/gate_oracle.py` (the RWKV-7 oracle-gate discipline this finding extends to Qwen3.5 for
the first time).
