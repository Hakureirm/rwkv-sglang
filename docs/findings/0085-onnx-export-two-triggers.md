---
finding_id: F0085
title: "Two ops, not one, made every ONNX export subtest fail: cumprod and linalg.solve_triangular"
date: 2026-08-06
status: closed
severity: medium
---

# F0085 — Two ops, not one, made every ONNX export subtest fail

**Date:** 2026-08-06 · **Repo:** transformers-rwkv7-prep / upstream PR #47780

## TL;DR

Six `RUN_SLOW` ONNX export subtests had been failing since the port existed, with
`aot_autograd expected to have an entirely functional graph, but found aten.detach_`.
The cause is **two** operations, and every earlier attempt failed because removing
either one alone changes nothing observable. Both are now replaced with exact
equivalents; the subtests go **6 failed → 8 passed, 16 subtests passed**.

## Mechanism

1. `aten.cumprod` and `aten.linalg_solve_triangular` each decompose into a graph
   that carries a **scalar tensor constant**.
2. `torch.export` wraps a lifted constant in `aten.lift_fresh`.
3. The ONNX decomposition table maps `aten.lift_fresh` (and `aten.lift`,
   `aten.detach`) to `nop_decomposition`, whose body is `return aten.alias(x)`.
4. `aten.alias` has **no entry of its own** in that table, so it survives.
5. Functionalization rewrites the surviving alias to the in-place `aten.detach_`.
6. `assert_functional_graph` rejects the graph.

Confirmed by reading the table: 1204 entries, `aten.lift_fresh.default` present,
`aten.alias.default` and `aten.detach.default` absent.

## How it was found, after several wrong turns

Bisection from a stub that exports cleanly, adding one operation at a time to the
`RWKV7_WKV_FUNCTIONS` entry:

| probe | result |
|---|---|
| bare stub, `.to()` casts, `exp`, `cumsum`, `eye`, `ones/tril`, `pad`, `einsum`, `sigmoid` | CLEAN |
| **`cumprod`** | ConversionError |
| **`linalg.solve_triangular`** | ConversionError |
| `linalg.inv`, `linalg.solve` (as replacements) | ConversionError |
| Neumann series, Newton doubling (as replacements) | CLEAN |

## The replacements are exact, not approximate

- `cumprod(exp(x))` **is** `exp(cumsum(x))`. Better conditioned too: a running
  product of factors below one underflows where the equivalent sum of logs does
  not, and `c_prev = c / w_c` becomes a subtraction rather than a division by a
  possibly-tiny value.
- `lhs` is unit lower triangular, so `qb` is nilpotent and the Neumann series for
  `(I + qb)^-1` terminates. Newton doubling reaches it in `ceil(log2(span))` steps.
  Against `solve_triangular` in float64: max abs difference **1.8e-15**, and
  `|inv @ lhs - I|` = **5.0e-16**. Also moved out of the per-chunk serial loop into
  one batched pass.

## Why every earlier attempt "failed"

The PR text listed as ruled out: the no-op `.to()` on inputs/state/output, the
no-op output slice, `solve_triangular` replaced with an explicit inverse, and the
cache. Three of those were tested **while the other trigger was still present**, so
a real improvement produced no visible change and was recorded as a dead end. The
`solve_triangular` line in particular was correct and was discarded.

**The lesson is not "test more". It is that with two independent causes, single-factor
elimination reports every true cause as false.** Nothing in the earlier method could
have found this; what found it was starting from a known-clean state and adding one
thing at a time, which makes each cause visible on its own.

## Two false alarms recorded on the way

- A "4× prefill collapse" attributed to a flag was run-order noise; an alternating
  A/B showed the same spread with the flag off.
- A "test I broke" was a stale `__pycache__` after `git checkout`: the test passes
  in isolation. See the open item below for what is actually left.

## Open

`test_the_last_real_token_is_found_by_index_not_by_float_arithmetic` passes alone
and with its neighbour, and fails in the full suite — reproducibly, with no random
ordering plugin installed. The full suite passed before this change, so the order
dependence is ours. Not yet diagnosed.
