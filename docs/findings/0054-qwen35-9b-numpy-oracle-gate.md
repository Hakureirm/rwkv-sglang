---
doc_kind: finding
finding_id: F0054
title: "Qwen3.5-9B correctness gate against Bo Peng's independent numpy fp32 reference: PASSES against this project's live sglang bf16 serving path — top-1 exact match, identical top-10 token sets, max abs prob diff 0.0049 (well inside the 0.02 tolerance) — closing the gap F0050 explicitly left open ('the 9B tier has not been run through this gate'); required a real architecture fix, not just new constants, for 9B's asymmetric linear-attention key/value head counts (16 vs 32) that upstream's own reference script never handles"
last_verified_commit: "2c38fc5"
discovered_by: Sonnet 5 (agent-assisted, 3090 box + the 5090 tower), 2026-07-08/09
severity: info
status: open
related: [F0048, F0049, F0050]
---

# Finding F0054: Qwen3.5-9B numpy-oracle correctness gate — PASS

## Context / question being asked

F0050 built and passed a bit-exact-adjacent correctness gate for this project's Qwen3.5-2B
comparison tier: an independent, from-scratch numpy fp32 reference implementation (Bo Peng's
`run_rwkv7_qwen35.py`, vendored at `qwen35_gate/vendor/`) checked against the two live serving
paths this project actually publishes Qwen3.5-2B numbers from (mlx-lm on Apple Silicon, sglang on
CUDA). F0050 explicitly scoped itself to 2B only and flagged the gap in its own "Scope" section:
"This project also publishes Qwen3.5-9B numbers (F0048 and others) at similar weight/effort. The
9B tier has not been run through this gate." This finding closes that gap.

**Result up front: PASS.** The independent numpy fp32 reference and this project's live sglang
bf16 serving path agree on the probe `" Eiffel"`'s next-token distribution — exact top-1 match
(both pick `" Tower"`, i.e. both models confidently complete "Eiffel Tower"), an identical 10/10
top-10 token set, and a max absolute probability difference of `0.0049` across every shared token
-- comfortably inside F0050's `0.02` tolerance. Getting to that result required more than
plugging in new numbers, though: 9B's linear-attention layer has a real architectural difference
from 2B/0.8B (asymmetric key/query vs. value head counts) that broke an assumption baked into
upstream's own reference script and required an actual fix, detailed below.

## Method

Same method as F0050, minus the mlx-lm leg (Apple-Silicon-only, not applicable on this Linux/CUDA
box; this run's purpose is specifically to gate the sglang serving path this project's 9B numbers
actually come from, e.g. F0048/F0049):

1. Located the already-downloaded, previously-used `Qwen/Qwen3.5-9B` HF checkpoint at
   `~/rwkv_models/qwen3.5-9b` on the 3090 box (bf16, 4 safetensors shards,
   19GB on disk) — the same checkpoint task#40's 9B concurrency-boundary search
   (F0049) already used, no re-download needed.
2. Found the canonical project repo is `Hakureirm/rwkv-sglang` on GitHub, not a literally-named
   "rwkv-vllm" repo — the 3090 box's local `~/rwkv-vllm` working directory
   is an un-versioned (no `.git`), stale partial mirror (findings only up to F0018, no
   `qwen35_gate/` at all) and is NOT where this task's prerequisite tooling (F0050's
   `qwen35_gate/`) lives. the 3090 box does have direct GitHub HTTPS access (confirmed:
   `curl https://github.com` -> 200, and a real `git clone` of the canonical repo succeeded) even
   though prior memory notes recorded this box as GitHub-less — worth re-confirming, not
   re-asserting, per the project's own "claims need numbers" / "don't assume" discipline. Cloned
   the canonical repo fresh to `~/qwen35_9b_gate_work/rwkv-sglang` at commit
   `2c38fc5` (HEAD at the time of this run).
3. **Manually cross-checked every hardcoded architecture constant in F0050's `numpy_reference.py`
   against Qwen3.5-9B's own `config.json`** (`text_config` block), field by field, rather than
   assuming 2B's verified values carry over — see "Architecture constant audit" below. This
   found a real, previously-impossible-to-anticipate divergence (asymmetric linear-attention
   key/value head counts) that required an actual code fix, not just new numbers — see
   "Generalization findings" below.
4. Ran `run_qwen35_make_pth.py` (unmodified, per F0050's finding that it's shape-agnostic /
   key-prefix-driven) against the 9B checkpoint to produce a flat text-only `.pth`.
5. Ran the (now 9B-fixed) `numpy_reference.Qwen35` on the same probe text F0050 used
   (`" Eiffel"`, the upstream script's own default) to get a top-10 next-token distribution.
6. Booted a live sglang server for the 9B checkpoint directly on the 3090 box's existing bare-metal
   `~/envs/rwkv-sgl` venv (this box's established pattern for its own prior Qwen3.5-9B work --
   see `logs/qwen35_9b_bf16_server_3090_cg64_confirm.log` — rather than F0050's tower-side
   ephemeral-container approach; both are "the project's own established live-serving path for
   this box," just different boxes with different existing conventions), `--dtype bfloat16
   --trust-remote-code`, matching F0050's flags.
7. Queried the sglang server's native `/generate` endpoint with the SAME token IDs the numpy
   reference used (via `transformers.AutoTokenizer.encode(probe, add_special_tokens=False)`,
   bypassing each runtime's own tokenizer wrapper, exactly as F0050 did) to get the same probe's
   top-10 distribution from the live serving path.
8. Compared numpy-fp32 vs sglang-bf16: top-1 token match, top-5 token-SET match, max absolute
   probability delta on tokens shared across both top-10 lists — same PASS bar as F0050 (top-1
   exact + top-5 set exact + max abs prob diff <= 0.02).
9. Did NOT run the mlx-lm leg (out of scope for this run per task instructions — Apple-Silicon
   only, and this run's purpose is gating the sglang path specifically).

### Mid-task discovery: the numpy-reference leg does not fit on the 3090 box itself

The plan above (run everything on the 3090 box) hit two independent, genuine resource walls on that
specific box, both discovered empirically (not assumed) partway through, which forced splitting
the gate across two machines:

- **Disk**: the 3090 box's root filesystem is btrfs, 99% full (`Device allocated: 1.94TiB` of
  `1.95TiB`, only `12.92GiB` truly unallocated). The first `run_qwen35_make_pth.py` attempt
  stalled completely — confirmed via `/proc/<pid>/stack` showing the process blocked in
  `btrfs_buffered_write -> wait_on_page_writeback`, and system-wide `vmstat` showing near-zero
  disk throughput (`bi`/`bo` ~ 0) with 9 processes blocked — a known btrfs pathology near
  capacity (chunk allocation and extent-fragmentation costs balloon as unallocated space runs
  out), not a bug in the conversion script. The stuck process was killed and its partial output
  file removed.
- **RAM**: independent of the disk issue, `numpy_reference.Qwen35.__init__` upcasts every weight
  tensor to float32 (`v.detach().cpu().float().numpy()`) for genuine fp32-reference fidelity. 9B's
  text-only parameter count is 8,953,803,264 (measured directly off the checkpoint's own tensor
  shapes, categorizing every key as vision/mtp/text by prefix) — float32 alone is ~35.8GB, and
  the loading path's transient peak (original tensors + the growing float32 dict) can run higher
  still. the 3090 box has 31GB total RAM (~28GB "available"). This would not fit even if the disk
  write above had succeeded.

Both constraints are specific to the 3090 box's current state (a heavily-used, nearly-full shared
box), not to the gate methodology itself, and neither affects the sglang *serving* leg (sglang
loads bf16 weights straight onto the GPU — 17.62GB VRAM, confirmed fine on this box's 24GB 3090,
no CPU-side float32 duplication, no large intermediate file). Rather than force either constraint
(e.g. by deleting other users'/projects' files on a heavily shared research box to claw back disk
headroom, or by weakening the reference to fp16 and muddying the fp32-vs-bf16 comparison this gate
exists to make clean), the numpy-reference computation only was moved to the 5090 tower
(already used for this project's other cloud-tier work, and independently confirmed to
already hold both the 2B and 9B checkpoints locally at
`~/rwkv-sglang/models/`): 62GB RAM / 1.4TB free NVMe disk / direct GitHub+HF+PyPI
access, comfortably clear of both walls. The tower's GPU (RTX 5090) was 100% utilized by other
work at the time (unrelated active job, ~31.8/32.6GB VRAM) and was never touched — only CPU/RAM
was borrowed there, in a fresh, isolated `uv venv` kept outside that box's existing (and, for an
unrelated in-progress spec-decode branch, uncommitted) `repo/` git working tree, so as not to risk
disturbing it.

Concretely: `qwen35_gate/numpy_reference.py` and the unmodified `vendor/run_qwen35_make_pth.py`
were copied to the 5090 tower (`~/rwkv-sglang/staging/qwen35_9b_gate/`) and run there
against its own local copy of the same 9B checkpoint — verified, not assumed, to be the identical
checkpoint before trusting it: `diff` of the full `config.json` between the 3090 box's and
the tower's copies is byte-identical (not just the specific fields tabulated above), and
`md5sum` of `tokenizer.json`/`vocab.json`/`merges.txt` also matches exactly across both copies.
The probe token IDs were independently obtained straight from the 3090 box's own already-working
tokenizer install (confirming both boxes' copies of the checkpoint tokenize the probe identically:
`[242476, 300]`, same as 2B's tokenizer) so the sglang leg on the 3090 box could be queried
immediately without waiting on the tower; the two legs' results were then combined by hand
(`compare_results.py`, a standalone reimplementation of `gate_qwen35.py`'s `compare()` — same
tolerance policy, since the two legs could not run inside one shared process across two machines
that cannot directly reach each other on the network).

## Architecture constant audit: 9B vs 2B config.json, field by field

F0050 explicitly warned: "If this script is ever pointed at a different Qwen3.5 dense size
(4B/9B), these must be re-verified against that size's config.json first — do not assume they
hold." This section is that re-verification, done before writing or trusting any code change.

| constant | meaning | 2B `config.json` | 9B `config.json` | same? |
|---|---|---:|---:|---|
| `hidden_size` | C | 2048 | 4096 | NO (already handled: derived from checkpoint, not hardcoded) |
| `num_hidden_layers` | n_layer | 24 | 32 | NO (already handled: derived from checkpoint) |
| `linear_num_key_heads` | Hk | 16 | 16 | yes |
| `linear_key_head_dim` | N | 128 | 128 | yes |
| `linear_num_value_heads` | Hv | 16 | **32** | **NO — new asymmetry, see below** |
| `linear_value_head_dim` | (value head dim) | 128 | 128 | yes |
| `linear_conv_kernel_dim` | conv_len | 4 | 4 | yes |
| `num_attention_heads` | aH | 8 | **16** | **NO** |
| `num_key_value_heads` | aKV | 2 | **4** | **NO** |
| `head_dim` | aN | 256 | 256 | yes |
| `full_attention_interval` | hybrid pattern period | 4 | 4 | yes (and the literal `layer_types` array was also read for both tiers, not just the scalar — both show `linear_attention` at every index except `i % 4 == 3`, for all 24/32 layers respectively) |
| `rope_parameters.rope_theta` | rope base | 10000000 | 10000000 | yes |
| `rope_parameters.partial_rotary_factor` | -> rope_dim = aN * factor | 0.25 -> 64 | 0.25 -> 64 | yes |
| `tie_word_embeddings` | whether `lm_head.weight` is a separate tensor | true (no separate `lm_head.weight`) | false (separate `lm_head.weight` tensor present, confirmed via direct safetensors inspection: `lm_head.weight: [248320, 4096]`) | NO, but already handled (code checks for key presence, doesn't hardcode either way) |

Three real differences found: `num_attention_heads` (8->16), `num_key_value_heads` (2->4), and
`linear_num_value_heads` (16->32, while `linear_num_key_heads` stays 16 — a NEW asymmetry, not
just a bigger version of the same symmetric shape 2B had).

## Generalization findings: 2B -> 9B

`run_qwen35_make_pth.py` again needed zero changes (same conclusion F0050 reached for 0.8B->2B):
it is driven entirely by key-name prefixes, not shapes. Run against the 9B checkpoint it reported
`kept_tensors: 427` (`379` bfloat16 + `48` float32), `skipped_vision_tensors: 333`,
`skipped_mtp_tensors: 15` (427+333+15 = 775, the full checkpoint, no overlap), `numel:
8,953,803,264` — which exactly matches an independent byte-level count taken directly off the raw
checkpoint's own safetensors tensor shapes (categorizing every key as vision/mtp/text purely by
prefix, done as a sanity check before trusting the conversion script's own self-report), giving two
independently-computed numbers agreeing exactly rather than one script's output taken on faith.

`numpy_reference.py`'s `Qwen35` class needed a real, non-cosmetic fix — not just plugging in new
numbers for `aH`/`aKV` (which alone would have been a trivial change), but a genuine shape-handling
bug in the linear-attention (GDN) layer. Full detail:

**The bug upstream's script (and F0050's 2B-scoped adaptation) has**: `make_GDN`'s `layer()`
closure does `q, k, v = np.split(qkv, 3)` then reshapes all three to the SAME head count `H`. This
is only valid when the linear-attention key/query head count equals the value head count. It does
for 0.8B and for 2B (both `linear_num_key_heads == linear_num_value_heads == 16`) — which is
exactly why F0050 never surfaced this: two matching data points don't reveal a scenario neither of
them exercises. 9B breaks it (16 key/query heads, 32 value heads). Concretely, 9B's
`in_proj_qkv.weight` is `[8192, 4096]` — confirmed by direct safetensors inspection — and
`8192 = 2*(16*128) + 32*128 = 2048 + 2048 + 4096`, NOT three equal 2730.67-wide thirds, so
`np.split(qkv, 3)` would not even run (numpy raises on non-integer equal-split), let alone produce
a silently-wrong result — this fails loud, which is how the gap was caught before ever reaching
the probe-comparison step.

**Root cause, confirmed against ground truth**: read HF transformers' own
`Qwen3_5GatedDeltaNet.forward()` (`transformers/models/qwen3_5/modeling_qwen3_5.py`, the actual
model class both this project's sglang serving path and the reference HF library instantiate for
this checkpoint). It splits `mixed_qkv` into `[key_dim, key_dim, value_dim]` (unequal, ordered)
rather than three equal parts, reshapes query/key at `num_k_heads` and value at `num_v_heads`, and
-- critically — when `num_v_heads // num_k_heads > 1`, applies
`query.repeat_interleave(ratio, dim=2)` and the same for `key`, BEFORE the recurrence. This is a
GQA-style expansion applied to linear attention: fewer key/query heads than value heads, with each
key/query head's vector duplicated across the value heads it serves.

**Fix applied to `qwen35_gate/numpy_reference.py`** (F0050's 2B-only file, edited for this
finding — allowed per this task's own instructions, since 9B's constants genuinely differ):

- `self.Hk` (linear_num_key_heads) and `self.Hv` (linear_num_value_heads) are now two separate
  attributes instead of one shared `self.H`.
- `make_GDN`'s `layer()` closure now slices `qkv` into `[0:key_dim]`, `[key_dim:2*key_dim]`,
  `[2*key_dim:]` (matching HF's ordered, unequal split) instead of `np.split(qkv, 3)`; `q`/`k`
  reshape to `(Hk, N)`, `v` reshapes to `(Hv, N)`.
- When `Hv // Hk > 1`, `q` and `k` are expanded via `np.repeat(x, Hv // Hk, axis=0)` --
  numpy's `repeat` (not `tile`) duplicates each element contiguously
  (`head0,head0,head1,head1,...`), which is the exact numpy equivalent of PyTorch's
  `repeat_interleave` (confirmed by reading numpy's and PyTorch's own docs for both functions,
  not assumed from the similar-sounding name). This happens AFTER this file's existing L2-norm
  step on q/k rather than before (HF's literal order is repeat-then-normalize-inside-the-kernel);
  algebraically equivalent because L2-norm is a per-vector operation and repeat only ever
  duplicates a vector, so normalize-then-duplicate produces the same result per output head as
  duplicate-then-normalize-each-copy-independently.
- The gating terms (`w`/decay and `a`/beta) were already effectively indexed by value-head-count
  in the original code (confirmed: 9B's `dt_bias`/`A_log`/`in_proj_b`/`in_proj_a` tensors are all
  width 32 = `Hv`, not 16 = `Hk` — this was invisible in 2B/0.8B only because `Hv == Hk` there
  too), so `w`/`a` now reshape to `(Hv, 1)` instead of `(H, 1)`.
- The per-head recurrent state (`S0()`'s `"rnn"` array) is now shaped `(Hv, N, N)` instead of
  `(H, N, N)`, and the conv-state buffer width is `2*Hk*N + Hv*N` instead of `3*H*N`.
- **Correctness of this generalization was verified two ways before trusting it against 9B**: (a)
  algebraically, by hand-deriving that Bo Peng's `DPLR()` einsum formula (unchanged) is exactly
  equivalent to HF's own official `torch_recurrent_gated_delta_rule` step-by-step reference
  recurrence (also read directly from `modeling_qwen3_5.py`) under the substitution `w[h] =
  exp(g_t)`, `a[h] = beta_t`, for a state indexed at whatever head count q/k/v/w/a all agree on
  after expansion; (b) empirically, by re-deriving all the geometry constants (`Hk`, `Hv`, `N`,
  `conv_len`, `aH`, `aKV`, `aN`) straight from checkpoint tensor shapes rather than hand-copying
  numbers from `config.json`, running the updated code against the (unaffected, `Hk==Hv`) 2B
  checkpoint, and confirming the derived values exactly reproduce F0050's original hardcoded
  ones (`Hk=Hv=16, N=128, conv_len=4, aH=8, aKV=2, aN=256`) and the resulting top-10 token
  ranking is identical to F0050's committed 2B result (`qwen35_gate/results/
  qwen35_2b_eiffel_gate_20260707.json`), with logits/probabilities agreeing to 5-6 significant
  figures (residual ~1e-6 differences are fp32 accumulation-order noise from the refactored
  slicing, ~4 orders of magnitude below F0050's own 0.02 absolute-probability PASS tolerance) --
  i.e. this is a strict generalization, not a rewrite, and does not put F0050's already-published
  2B PASS verdict at any risk.

`self.aH`/`self.aKV`/`self.aN` (full-attention head geometry) and `self.Hk`/`self.Hv`/`self.N`/
`self.conv_len` (linear-attention head geometry) are now all derived directly from checkpoint
tensor shapes (mirroring how F0050 already derived `self.C`/`self.n_layer`) rather than hardcoded
per-tier numbers, specifically so a future Qwen3.5 dense tier doesn't need a third manual patch of
this same kind — while still being independently hand-cross-checked against both 2B's and 9B's
`config.json` in the table above, not trusted blind.

`gate_qwen35.py` (the reusable driver) was also given a `--skip-mlx` flag so it can run the
numpy-vs-sglang comparison alone on a box without `mlx`/`mlx_lm` installed (this box is
Linux/CUDA, not Apple Silicon); default behavior (no flag) is unchanged, preserving F0050's
original 2B usage.

## Result

Probe `" Eiffel"` tokenizes to `[242476, 300]` (`" Eiff"` + `"el"`) — independently confirmed
identical on both the 3090 box's and the tower's copies of the 9B tokenizer, and the same token IDs
F0050 got from the 2B tokenizer (the dense-family tokenizer is shared across sizes).

Resolved architecture constants the numpy reference actually ran with (printed at load time, not
just claimed): `n_layer=32 C=4096 Hk=16 Hv=32 N=128 conv_len=4 aH=16 aKV=4 aN=256` — matching the
"Architecture constant audit" table above exactly.

| rank | numpy fp32 (reference, the tower) | sglang bf16 (live, the 3090 box) |
|---|---|---|
| 1 | tok=21262 `' Tower'` p=0.13026 | tok=21262 `' Tower'` p=0.12538 |
| 2 | tok=271 `'\n\n'` p=0.04350 | tok=579 `"'s"` p=0.04333 |
| 3 | tok=11 `','` p=0.04295 | tok=11 `','` p=0.04333 |
| 4 | tok=579 `"'s"` p=0.04282 | tok=271 `'\n\n'` p=0.04333 |
| 5 | tok=369 `' is'` p=0.04062 | tok=369 `' is'` p=0.04070 |
| 6 | tok=220 `' '` p=0.03696 | tok=220 `' '` p=0.03824 |
| 7 | tok=25 `':'` p=0.02404 | tok=198 `'\n'` p=0.02319 |
| 8 | tok=198 `'\n'` p=0.02244 | tok=25 `':'` p=0.02319 |
| 9 | tok=321 `' and'` p=0.01794 | tok=321 `' and'` p=0.01806 |
| 10 | tok=11106 `' Language'` p=0.01675 | tok=11106 `' Language'` p=0.01594 |

Top-1 is exact-match on both implementations (`" Tower"` — both models confidently complete
"Eiffel Tower," a materially more confident, more semantically pointed completion than 2B's own
top-1 of a generic punctuation mark in F0050, consistent with a larger model having a stronger
grip on this named entity). Ranks 2-4 and 7-8 show a harmless reshuffle among tokens whose
probabilities are within ~0.001 of each other even within the fp32 reference alone — exactly the
bf16-rounding noise pattern F0050 already documented for 2B, not a disagreement. The **top-10
token set is identical, 10/10**, between the two implementations.

| comparison | top-1 | top-5 set | shared tokens | max abs prob diff | verdict |
|---|---|---|---|---|---|
| numpy fp32 (the tower) vs sglang bf16 (the 3090 box) | match | match | 10/10 | 0.004880 | **PASS** |

**Verdict: GATE_QWEN35_9B_PASS.** The sglang bf16 serving path this project's Qwen3.5-9B numbers
(F0048, F0049, and any future 9B benchmark this project publishes) actually run on agrees with an
independent, from-scratch numpy fp32 reference well inside the noise expected from a genuine
fp32-vs-bf16 precision gap — sub-half-a-percentage-point on the shared top-1 token, no token
appearing in one top-10 and absent from the other. F0050 retired this risk for the 2B tier; this
finding retires the same risk for 9B. Getting here required catching and fixing a genuine
architecture-generalization bug (the asymmetric linear-attention head counts, detailed above)
rather than 9B simply working out of the box — which is itself evidence this gate is doing real
work, not rubber-stamping: a shape bug that would have crashed loudly on first run (via a
non-integer `np.split`) was caught and fixed *before* it could have been mistaken for "not
applicable to 9B" or silently worked around.

## Scope / what this does NOT establish

- **9B only, and specifically the dense text tier.** This does not re-run or re-validate F0050's
  2B result (untouched, see Files below) beyond the incidental regression check described above.
  It also says nothing about any other Qwen3.5 size this project might benchmark in the future
  (e.g. a hypothetical 4B) — per both F0050's and this finding's own recurring point, every
  architecture constant must be re-checked against that tier's own `config.json`, not assumed.
- **sglang leg only, no mlx-lm leg.** Out of scope for this run by the task's own instructions
  (mlx-lm is Apple-Silicon-only; this run's purpose is specifically to gate the sglang serving
  path this project's published 9B numbers, e.g. F0048/F0049, actually come from). F0050's 2B gate
  checked three independent implementations (numpy, mlx-lm, sglang) mutually agreeing; this 9B gate
  checks two (numpy, sglang). A future 9B-on-Apple-Silicon benchmarking effort would still need its
  own mlx-lm leg before citing 9B numbers from that path with the same confidence F0050 established
  for 2B.
- **Single probe position, not a decode oracle.** Same limitation F0050 already flagged for 2B:
  this checks one forward pass's next-token distribution (matching the upstream script's own
  scope), not a multi-step greedy-decode oracle the way `bench/oracle_numpy.py` gates RWKV-7. It
  would not catch a bug that only manifests after several autoregressive steps (e.g. a state
  update that only drifts visibly after N steps) — this is an even more pointed caveat here than
  it was for 2B, given this finding's central discovery was a shape bug in exactly the recurrent
  state (`S0()`'s `"rnn"` array) that a single-step-only check could plausibly have missed if the
  bug had been more subtle (this one happened to fail loud via a non-integer `np.split`, but nothing
  guarantees a future one would).
- **Tolerance, not bit-exactness.** Same policy as F0050 and for the same reason: fp32-CPU (on
  the 5090 tower) vs bf16-CUDA (on the 3090 box) are legitimately different numeric paths, and the
  `PASS` bar (top-1 match + top-5 set match + max abs prob delta <= 0.02) is deliberately looser
  than RWKV-7's own bit-exact gate to avoid rejecting legitimate bf16 precision noise as if it
  were a real bug.
- **What this run's own two-machine split does NOT undermine**: the numpy reference and the
  sglang server ran against independently-sourced copies of the same checkpoint (the 3090 box's
  `~/rwkv_models/qwen3.5-9b` and the tower's
  `~/rwkv-sglang/models/qwen3.5-9b`) — their `config.json`s were diffed field-by-
  field (hidden_size, num_hidden_layers, linear_num_key_heads, linear_num_value_heads,
  num_attention_heads, num_key_value_heads all matched) and both independently tokenized the probe
  identically, so this is not weaker evidence than a same-machine run would have been, just
  logistically split.

## Files

- `qwen35_gate/numpy_reference.py` — edited in this finding (see Generalization findings above
  for the precise diff); F0050's 2B behavior is preserved (regression-tested, see above).
- `qwen35_gate/gate_qwen35.py` — edited in this finding to add `--skip-mlx` (backward compatible,
  default off).
- F0050's own files (`vendor/`, `mlx_probe.py`, `results/qwen35_2b_eiffel_gate_20260707.json`,
  the finding doc itself) were NOT modified.
- `compare_results.py` (new, this finding) — standalone reimplementation of `gate_qwen35.py`'s
  `compare()`, used to combine the two legs' results since they ran on separate machines that
  cannot reach each other directly; same tolerance policy, not a different/looser check.
- `query_sglang_probe.py` (new, this finding) — standalone sglang `/generate` prober, the
  sglang-only half of `gate_qwen35.py`'s leg, for running directly on a box hosting the server
  without needing the numpy/torch/transformers stack in the same process.
- Result JSONs from this run (all currently sitting in the *working directories* described below,
  not yet committed anywhere, per this task's "don't commit" constraint):
  - `numpy_9b_result.json` — produced on the tower at
    `~/rwkv-sglang/staging/qwen35_9b_gate/numpy_9b_result.json`.
  - `sglang_9b_result.json` — produced on the 3090 box at
    `~/qwen35_9b_gate_work/sglang_9b_result.json`.
  - `combined_result.json` — the two above merged by `compare_results.py` with the final
    PASS verdict; currently only on the machine that ran the comparison (the orchestrating agent's
    local working directory, not the 3090 box or the 5090 tower).
- Converted checkpoint (`qwen35_9b_text.pth`, bf16+fp32 mixed, `numel=8,953,803,264`, ~18GB) and
  the source HF checkpoint directories are outside the repo, not committed, and were deleted
  (the `.pth`) or left as pre-existing shared assets (the HF checkpoint dirs, which predate this
  task on both boxes) — regenerate the `.pth` via `qwen35_gate/vendor/run_qwen35_make_pth.py`
  per the README, on a box with enough free RAM (>~40GB recommended) for the fp32 upcast and
  enough free disk for the intermediate file — **not** the 3090 box as of this run (see "Mid-task
  discovery" above; the 5090 tower worked cleanly, ~2 minutes for the conversion, well under a minute
  for the numpy forward pass itself).
- This task's staging/scratch locations (not part of the canonical repo, left in place per this
  task's instructions for the orchestrating agent to integrate): the 3090 box's
  `~/qwen35_9b_gate_work/` (fresh clone of the canonical repo at `2c38fc5` plus the
  serve script, sglang query script, and sglang result) and the tower's
  `~/rwkv-sglang/staging/qwen35_9b_gate/` (the edited `qwen35_gate/` files, the
  numpy-leg runner script, and the numpy result) — neither is a git working tree, both are
  ordinary scratch directories.

## Cross-references

F0048 (int8/fp8 tier gap, established 9B's bf16 weight footprint on this box: 17.62GB) · F0049
(desktop-tier 3090 concurrency comparison, the 9B benchmark numbers this gate is meant to back) ·
F0050 (the 2B gate this finding extends to 9B, same method, same tolerance policy) ·
`memory/project-qwen35-benchmark.md`.
