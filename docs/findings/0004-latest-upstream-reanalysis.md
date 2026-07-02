---
doc_kind: finding
finding_id: F0004
title: "Verified latest-upstream re-analysis (vLLM / sglang / HF)"
last_verified_commit: (initial)
discovered_by: workflow rwkv7-latest-upstream-recon (5 finders + 5 adversarial verifiers), 2026-06-30
severity: info
status: open
related: [F0002, F0003]
---

# Finding F0004: Verified latest-upstream re-analysis

## Hypothesis
The dev-box's pinned vLLM build is not a sound basis for strategy; the *latest
community upstream* state of RWKV-7 across vLLM / sglang / HF must drive which
platform we integrate with. (Directive: "从社区最新版来考虑,重新分析" — reconsider
from the latest community state, re-analyze.)

## Method
Dynamic workflow: 5 parallel research finders (vLLM / sglang / HF upstream state +
rwkv-lm/albatross baselines + prior art), each producing a decision-critical
claim, then 5 independent adversarial verifiers re-checking those claims against
live GitHub (gh CLI + fresh shallow clones). 10 agents, ~334k tokens, ~8 min.
**All 5 verdicts: `confirmed`.**

## Result (verified facts, all dated late June 2026)

### vLLM — in-tree path closed upstream
- Latest release v0.24.0 (2026-06-29). No RWKV in main.
- **Both** complete RWKV-7 PRs **CLOSED 2026-06-29** by maintainer @hmellor
  (MEMBER): **#41060** "RWKV-7 Goose" (sirus20x6, ~3.9k LOC, full V1 integration:
  `models/rwkv7.py` + `rwkv7_mixer.py` + `rwkv7_attn.py` + vendored fla rwkv7 ops;
  uses HasInnerState+IsAttentionFree+SupportsMambaPrefixCaching) and **#46269**
  "RWKV7 Albatross" (Cai-z-us / LateranLab / rwkv-rs, ~90k LOC, custom CUDA +
  rapid-sampling + tool-call parser). Identical close text: checkpoints have "only
  a few hundred downloads each … not enough to warrant dedicated vLLM
  implementation/maintenance"; directed to out-of-tree plugin or transformers
  backend. **Verified via gh CLI** (state CLOSED, mergedAt null).
- #46269 = a large, well-resourced parallel RWKV-7 integration effort (also closed).

### sglang — greenfield + mature substrate (SELECTED)
- Latest release v0.5.14 (2026-06-26); main HEAD `f920a37` (2026-06-29).
- **Verified greenfield**: fresh clone + `grep -rin rwkv` → only
  `parser/conversation.py` legacy `SeparatorStyle.RWKV` enum. No RWKV model, no PR,
  **a clean starting point**.
- Mature linear-attn substrate: vendored fla kernels, `RadixLinearAttention`,
  `MambaRadixCache`/`mamba_checkpoint_pool` state cache, chunked prefill, dynamic
  batching, spec decode, PD-disagg; Qwen3-Next/Kimi-Linear/Nemotron-H live. fla
  was "adapted from vLLM's fla port" → kernel work transfers between frameworks.
- Rated the only **"attractive / medium-effort"** integration target.

### HF — modeling already solved by fla
- transformers main (HEAD 957e6032, 2026-06-29) has only native RWKV-4; rwkv7 dir
  404; RWKV-6 PR #34918 a stale draft. fla `modeling_rwkv7.py` mature + auto-
  registers (trust_remote_code). Remaining: native upstream + tested PEFT/TRL +
  quant → mostly already solved, low net-new value.

### Baselines (reframed)
- **albatross** = static-batch raw-speed kernel engine (fp16-only; **no** dynamic
  batching / chunked prefill / state cache). 5090 / RWKV-7 7.2B: ~17k tps prefill,
  ~15k tps decode, ~21k batch-prefill. ⇒ "match albatross" bounds raw per-batch
  throughput; serving features are net-new.
- **Accuracy oracle = BlinkDL `rwkv` pip (cuda fp16, `RWKV_V7_ON=1`) +
  `RWKV-v7/rwkv_v7_numpy.py`**. **fla is NOT aligned with the reference
  ("performance is quite worse") → must NOT be the accuracy oracle.** Anchor:
  World-1.5B-v3 ≈ 44.87% MMLU greedy. World tokenizer = 65,536-vocab byte trie.
- Models on ModelScope `RWKV/rwkv7-g1`; HF orgs BlinkDL/RWKV/fla-hub.

## Conclusion
sglang is the greenfield, technically-tractable target →
ADR-0001 commits to integrating RWKV-7 with sglang. Reusable references captured for the port:
`refs/pr41060-rwkv7-goose.diff` (clean structural blueprint), `refs/sglang`,
`refs/fla` (rwkv7 kernels + correct math), `refs/Albatross` (speed baseline).

## Cross-references
ADR-0001, ADR-0002, [[F0002]], [[F0003]]. Workflow result archived in session.
