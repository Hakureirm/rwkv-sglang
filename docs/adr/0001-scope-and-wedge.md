---
doc_kind: adr
adr_id: 0001
title: "Scope & rationale — RWKV-7 on sglang"
status: accepted
date: 2026-06-30
last_verified_commit: (initial)
supersedes: []
superseded_by: []
---

# ADR-0001: Scope & rationale — RWKV-7 on sglang

## Context
Goal: a production-grade RWKV-7 serving adaptation. The production bar is to match
`rwkv-lm` (accuracy) + `albatross` (speed/VRAM) across batch sizes; a serving
engine needs dynamic batching + chunked prefill + a recurrent state cache; 8/4-bit
quant no slower than 16-bit; consumer + datacenter GPU coverage. See [[F0001]] [[F0002]] [[F0003]].

Among the candidate host frameworks (HF transformers / vLLM / sglang), a
latest-upstream re-analysis (workflow `rwkv7-latest-upstream-recon`, 5 finders +
5 adversarial verifiers, **all verdicts `confirmed`**, 2026-06-30) materially
changed the picture vs the naive "vLLM is the obvious target":

- **vLLM**: NO RWKV in main. Two complete RWKV-7 PRs (#41060 Goose ~3.9k LOC;
  #46269 "Albatross" ~90k LOC, custom CUDA + tool-calling) were **both CLOSED
  2026-06-29** by maintainer @hmellor (low download counts — a maintenance/demand
  decision, not technical), who directs contributors to an **out-of-tree plugin**
  or the transformers backend. *In-tree merge is shut*, and #46269 (LateranLab /
  rwkv-rs) is a **well-resourced parallel RWKV-7 effort on the same ground**.
- **HF**: `fla`'s `modeling_rwkv7.py` is mature and auto-registers into
  transformers → the **modeling is largely already solved**; remaining work
  (native upstream + tested PEFT/TRL + quant) carries high "already-covered" risk.
- **sglang**: **verified greenfield** — a fresh clone of main (HEAD `f920a37`,
  2026-06-29) shows ZERO RWKV (only a legacy `SeparatorStyle.RWKV` chat-template
  enum). **No PR, no prior RWKV work.** Yet the linear-attention/hybrid substrate is
  **production-mature**: vendored `fla` kernels, `RadixLinearAttention`,
  **`MambaRadixCache` state cache**, chunked prefill, dynamic batching, spec
  decode, PD-disaggregation; Qwen3-Next (GDN) / Kimi-Linear (KDA) / Nemotron-H
  live. Rated the **only "attractive / medium-effort"** host.

Baseline reframings that change acceptance: **albatross is a static-batch
raw-speed kernel engine** (fp16-only; *no* dynamic batching / chunked prefill /
state cache) — so "match albatross" bounds raw per-batch throughput, and the
serving features are net-new. **Accuracy oracle = BlinkDL `rwkv` pip (cuda fp16,
`RWKV_V7_ON=1`) + `rwkv_v7_numpy.py`; `fla` is documented as NOT aligned with the
reference ("performance is quite worse") → fla must NOT be the accuracy oracle.**

## Options considered
1. **vLLM out-of-tree plugin** (fork of closed #41060). Pro: clean forkable base,
   biggest ecosystem. Con: in-tree shut; a deep parallel effort already on the same ground.
2. **HF native + PEFT/RL/quant**. Pro: foundational. Con: modeling already done
   by fla; highest "already-solved" perception risk.
3. **sglang** (CHOSEN). Pro: greenfield (no existing RWKV port); mature serving substrate
   (state cache / chunked prefill / dynamic batching already exist); reduces the
   effort to a tractable model-port; fla lineage shared with vLLM so a future
   cross-port is cheap. Con: no RWKV-specific forkable base inside sglang
   (mitigated: study closed vLLM #41060 + qwen3_next.py template); sglang merge
   policy for RWKV unverified; smaller ecosystem than vLLM.
4. **Shared core → vLLM plugin + sglang** (deferred). Highest reach, but the scope
   was deliberately narrowed to **sglang-only** for focus.

## Decision
**Scope this project to sglang only.** (Decided 2026-06-30, on the verified re-analysis.)

**Wedge (one sentence):** *the first production-grade RWKV-7 serving in sglang —
dynamic batching + chunked prefill + recurrent state cache + 8/4-bit quant —
matching rwkv-lm accuracy and albatross raw-kernel speed across batch sizes, on
consumer + datacenter GPUs.*

## Done means (falsifiable milestones)
| M | Goal | Done means (gate) |
|---|---|---|
| **M0** | Env + oracle + models | sglang(editable)+fla+rwkv-pip on the dev box; RWKV-7 0.1B/0.4B; `rwkv_v7_numpy.py` oracle runs. |
| **M1** | Minimal correctness | `models/rwkv7.py` in sglang serves RWKV-7 0.1B; **greedy tokens match the numpy/rwkv-pip oracle** over ≥1000 prompts (bit-tolerant fp32). |
| **M2** | Serving features | dynamic batching + chunked prefill + recurrent state cache correct under continuous batching; multi-request interleave matches single-request. |
| **M3** | Parity | accuracy: MMLU/lambada Δ within noise vs rwkv-lm. speed/VRAM: ≥ **albatross-on-the-same-3090** across bsz {1,16,64,320}; constant VRAM w.r.t. ctx. |
| **M4** | Quant | 8-bit & 4-bit: VRAM ↓, **not slower than fp16**. |
| **M5** | GPU coverage + deliverable | runs on consumer (3090/4090/5090) + datacenter; packaged install; README + bench cites; (stretch) sglang PR. |

## Consequences
### Positive
- Greenfield, tractable target (port on a mature substrate, no new C++/CUDA
  required for the triton path).
- Kernel/model work transfers to a future vLLM plugin (shared fla lineage).
### Negative
- No RWKV-specific forkable base inside sglang (rely on #41060 + qwen3_next.py).
- Smaller ecosystem reach than vLLM.
### Risk
- sglang's `MambaRadixCache` is shaped for Mamba2/SSM state; RWKV-7's per-head
  `[K,V]` matrix state may not fit its prefix-reuse assumptions → may need a
  state-pool adapter (addressed in ADR-0002 / a finding).
- albatross fp16 7.2B may not fit/perform on a 24GB 3090 → parity may be measured
  at 1.5B/2.9B; must re-run albatross on the 3090 for a same-HW target.

## Cross-references
[[F0001]] env · [[F0002]] architecture↔sglang mapping · [[F0003]] baselines/oracle ·
ADR-0002 integration approach · workflow `rwkv7-latest-upstream-recon`.
