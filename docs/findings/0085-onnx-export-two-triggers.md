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

Seven `RUN_SLOW` ONNX export subtests had been failing since the port existed, with
`aot_autograd expected to have an entirely functional graph, but found aten.detach_`.
The cause is **two** operations, and every earlier attempt failed because removing
either one alone changes nothing observable. Both are now replaced; on the upstream
PR branch the export tests go **7 failed / 12 passed → 12 passed, 16 subtests
passed**, and the model suite from 125 to 126 passing.

The first replacement for the second trigger was **exact in exact arithmetic and
broken in float32**, and shipped a NaN into a fifth of long prefills before a test
caught it. That half of the story is below, and it is the more useful half.

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
| Neumann series, Newton doubling (as replacements) | CLEAN — and numerically wrong, see below |
| block forward substitution (as replacement) | CLEAN |

## The replacements are exact, not approximate

- `cumprod(exp(x))` **is** `exp(cumsum(x))`. Better conditioned too: a running
  product of factors below one underflows where the equivalent sum of logs does
  not, and `c_prev = c / w_c` becomes a subtraction rather than a division by a
  possibly-tiny value.
- `lhs` is unit lower triangular, so `qb` is nilpotent and the series for
  `(I + qb)^-1` terminates. **This is where I went wrong** — see the next section.
  The shipped replacement is block forward substitution, block 8, batched over every
  chunk at once instead of solving inside the serial loop.

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

## The first fix was exact in exact arithmetic and useless in float32

Newton doubling `X <- X(2I - AX)` from `X0 = I - qb` terminates after
`ceil(log2(span))` steps because `qb` is nilpotent. It agreed with float64
`solve_triangular` to 1.8e-15 **on random triangular matrices**, which is what I
validated it on. On the matrices the model actually produces it is garbage:

| quantity | value on a real chunk |
|---|---|
| max abs entry of `qb` | 0.977 |
| max abs entry of the true inverse | **1.0** |
| max abs entry of `qb^32` | **1.3e11** |
| Newton result, max abs entry | 1.2e4 (answer: 1.0) |
| plain Neumann sum, max abs entry | 2.7e4 |
| float32 `solve_triangular` | 2.4e-8 from the float64 reference |
| block forward substitution, block 4 / 8 / 16 | 1.3e-7 / **6.6e-7** / 1.9e-5 |

Eleven orders of magnitude of cancellation against seven digits of mantissa. The
result then went to NaN one layer on. Measured incidence, 40 random inits per cell:

| T | 32 | 64 | 128 | 256 | 1024 |
|---|---|---|---|---|---|
| Newton | 0/40 | 0/40 | 0/40 | 1/40 | **8/40** |
| block substitution | — | — | — | — | **0/140** |

**Random triangular matrices cannot detect this, and that is the whole trap.** Their
own inverses are as large as the intermediate powers, so the cancellation is
invisible and the series looks accurate to 1e-6 — on exactly the input a test
reaches for first. The pathology needs an inverse that stays near 1 while the powers
do not, which is what the delta rule produces. I checked three synthetic
constructions before accepting that the regression test had to be end-to-end.

Cost, measured on both. CPU end to end: a T=1024 forward runs about a third slower
than on the solve. RTX 5090, the inverse step against the per-chunk solve with the
loop included, at 1.5B shapes: **2.5x at T=1024**, where 16 chunks do not fill the
card, falling to **1.2x at T=4096**, flat in batch and head count.

## Cross-checked on the GPU, because everything above was CPU

Everything was found and fixed on a CPU, and float32 cancellation is exactly the
kind of thing that reshuffles when the matmul changes. On the 5090
(`torch_device = cuda`, torch 2.11+cu130):

- Model suite **130 passed, 0 failed, 1583 subtests** (130 rather than the CPU's 126
  because the CUDA-gated tests run). Two failures on the first attempt were
  `accelerate` missing from the container, not the model.
- The same 40-seed sweep: **0/40** with block substitution, **7/40** with Newton.
  CPU said 8/40 — seed 34 is finite on one and not on the other, which is what a
  borderline cancellation should do and is a reason not to trust a single backend.
- Both seeds the regression test pins (7 and 33) NaN on CUDA as well, so the test
  discriminates on the GPU and not only where it was written.

## Two false alarms recorded on the way

- A "4× prefill collapse" attributed to a flag was run-order noise; an alternating
  A/B showed the same spread with the flag off.
- A "test I broke" was blamed on a stale `__pycache__` after `git checkout`, and
  then on suite ordering. **Both readings were wrong.** The test
  (`test_the_last_real_token_is_found_by_index_not_by_float_arithmetic`) draws its
  weights unseeded, so every process gets a different model; it failed whenever the
  draw landed in the fifth that NaNs. There was never any order dependence — the
  first "passes alone" observation was a lucky draw, and I built a whole bisection
  harness on top of it before checking that premise. Diagnosis started for real when
  the assertion was instrumented and both sides printed `nan`.

## What the regression test pins

`test_a_long_prefill_stays_finite_where_a_series_inverse_would_not`, seeded at two
of the failing draws, at T=1024. Run against all three implementations before being
committed: **fails** on Newton, **passes** on `solve_triangular` and on the block
substitution. A test that has not been run against the code it is about is not known
to have any discriminating power — the unseeded neighbour above is what happens
without that step.
