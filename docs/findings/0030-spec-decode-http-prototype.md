---
doc_kind: finding
finding_id: F0030
title: "Speculative-decoding HTTP two-server prototype is the WRONG vehicle: statecache mode forces cuda-graph OFF so the plain baseline collapses to ~12 tok/s (19x slower than the 226 cuda-graph path), and per-round probe+verify HTTP overhead + a finicky radix input_top_logprobs alignment make it 0.4x plain with a residual exactness bug — viability (F0029 alpha=0.738) is unaffected; the real build must be IN-ENGINE (ADR-0006), which stays queued behind the large-batch front"
last_verified_commit: "HEAD"
discovered_by: lead (M13), 2026-07-04
severity: info
status: open
related: [F0029]
---

# Finding F0030: the HTTP two-server spec-decode prototype is structurally the wrong vehicle

## What was tried
A zero-kernel-change orchestrator (`bench/spec_decode.py`, kept as a scratch prototype — NOT
committed to the deliverable) over two state-cached sglang servers: 0.1B draft proposes K tokens
from the committed prefix, 1.5B target verifies with one extend + a probe call, rejected suffix
state expires in the radix tree (rollback == radix fork). The idea was to get a preliminary
speedup demo without building an in-engine draft/verify/rollback loop.

## Why it cannot win (the structural problem)
The state prefix cache (req#3 / F0022) only works with **radix ON, and F0022 verified that pairing
only with cuda-graph OFF** (statecache mode). So the target server runs **eager**: measured plain
bsz1 decode is **~11.7 tok/s**, vs **225.9 tok/s** with cuda-graph on (F0028) — a **~19x** penalty.
A spec-decode scheme that even tripled effective tokens-per-forward would reach ~35 tok/s, still
**6x slower than simply running plain decode with cuda-graph on**. On top of that the prototype pays
two HTTP calls per round (probe + verify) and Python-loop overhead. Net measured: **0.4x** the
(already crippled) eager plain. There is no configuration of the HTTP two-server design that beats
cuda-graph plain, because it cannot use cuda-graph and pay the cross-process cost at once.

## Plus a residual exactness bug (so its alpha is not a real measurement)
The greedy gate (spec output == plain greedy, token-for-token, same server) **FAILED** on several
prompts, and the end-to-end acceptance came out alpha≈0.06 — an order of magnitude below F0029's
cleanly-measured 0.738. That gap is the signature of an **alignment bug** in reading
`input_top_logprobs` under radix (its length/None-padding shifts with the prefix-cache boundary in
ways probe 2026-07-04 only partially pinned), NOT a real low acceptance. So the prototype's alpha is
**invalid**, not evidence against viability.

## What stands
- **Viability is unaffected.** F0029's alpha = 0.738 was measured with a clean, cache-independent
  method (draft argmax vs the target's own greedy tokens) and remains the viability number.
- **The real design was always in-engine.** ADR-0006 specifies draft + chain-verify + O(1)
  conv/temporal rollback **inside one scheduler with cuda-graph on** — which is exactly what avoids
  both failure modes here (no eager penalty, no per-round HTTP, direct state access instead of
  logprob-reverse-engineering). That build is HIGH-effort and, per
  `feedback-full-spectrum-not-single-stream`, sequenced AFTER the large-batch front. It is not
  started; this finding just rules out the HTTP shortcut.

## Lesson
The state cache makes rollback cheap, but statecache-mode's eager requirement makes any HTTP
orchestration a net loss at bsz1. Don't confuse "rollback is free" with "the prototype is fast" —
measure the baseline the prototype actually runs against (eager), not the cuda-graph number.

## Cross-references
[[F0029]] (viability, alpha=0.738 — intact) · ADR-0006 (in-engine design, queued) ·
`feedback-full-spectrum-not-single-stream` (why the build is sequenced late).
