# RWKV-7 CoreML/ANE feasibility probe (Apple Silicon)

**Verdict: feasibility gate FAILED. No model was built, no tok/s is reported here.** Full evidence
and reasoning: [`docs/findings/0042-coreml-ane-feasibility.md`](../docs/findings/0042-coreml-ane-feasibility.md),
summarized in [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) §12.6.

## What this is

Before building a full RWKV-7 → CoreML converter (fixed-shape decode/prefill programs, explicit
state tensors threaded across calls — the pattern `rwkv-mobile`'s CoreML backend uses, studied for
structure only, no code copied), this probes the one question that decides whether the whole
exercise is worth it: **does RWKV-7's WKV recurrence — the actual novel numerics — genuinely
dispatch to the Apple Neural Engine, or does it silently fall back to CPU?**

`probe_ane.py` builds the WKV delta-rule state update directly in MIL (coremltools' IR builder — no
torch, no fla; the math mirrors `mlx_port/rwkv7_mlx.py`'s `_wkv_scan_pure` / `bench/oracle_numpy.py`'s
`time_mixing` exactly) at the real checkpoint geometry (0.1B: H=12,D=64; 1.5B: H=32,D=64), converts
with `compute_units=ct.ComputeUnit.CPU_AND_NE`, and asks CoreML's own
`coremltools.models.compute_plan.MLComputePlan` which device each op actually prefers — not a timing
guess.

## Result

Zero of 168 tested WKV-recurrence ops (two model sizes, single-step and a 4-step chain) ever get
`preferred_compute_device = ANE`, even under unrestricted `compute_units=ALL` (GPU and ANE both
available). Positive controls (a big batched GEMM, a decode-shaped GEMV, a conv2d) confirm the
*query itself* works — a big GEMM genuinely gets ANE + a real 1.2x speedup on this machine — so the
WKV result is a real scheduling decision (tensors too small to amortize ANE dispatch overhead), not
a broken probe or an absent ANE. Run it yourself:

```bash
pip install coremltools
python coreml_port/probe_ane.py
```

## Why the full converter wasn't built

Per the task's pre-registered stop condition: if the core recurrence falls back to CPU, building the
full model would only produce a CoreML/ANE-labeled artifact that's still CPU-bound underneath, at
the cost of real conversion/runtime complexity (fixed-shape program splitting, MultiFunctionDescriptor
combining, Objective-C state-threading on-device) — and a real risk of reporting a dishonest-looking
"ANE tok/s" number. The MLX-GPU port (`../mlx_port/`) remains the complete Apple-Silicon story;
see `docs/BENCHMARKS.md` §12.1–§12.5 for its (real, oracle-gated) numbers.

## Files

- `probe_ane.py` — the feasibility probe (self-contained; `pip install coremltools` is the only new
  dependency). Kept as a reusable artifact in case a future workload shape (batched/large-bsz decode,
  a genuinely large prefill tensor) reopens the question — see the finding's "what would change this"
  section.
