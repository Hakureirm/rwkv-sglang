# qwen35_gate

Correctness gate for Qwen3.5, adapted for this project's Qwen3.5-2B comparison tier from
[Bo Peng's independent numpy reference](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py).
This is the RWKV-7-oracle-gate discipline (`bench/oracle_numpy.py`, `mlx_port/gate_oracle.py`)
extended to the competitor model in this project's benchmark comparisons — previously Qwen3.5 had
only ever been checked for "output looks coherent," never against a numerics reference. See
[`docs/findings/0050-qwen35-numpy-oracle-gate.md`](../docs/findings/0050-qwen35-numpy-oracle-gate.md)
for the full writeup.

## Layout

- `vendor/` — Bo Peng's two scripts, unmodified, fetched 2026-07-07, kept for diffing.
- `numpy_reference.py` — the `Qwen35` class from `vendor/run_rwkv7_qwen35.py`, adapted to
  generalize from the upstream script's original 0.8B target to 2B (see module docstring for the
  exact diff — one hardcoded shape actually needed to change, the rest were verified, not assumed,
  to already generalize).
- `mlx_probe.py` — same probe, run against a live `mlx-lm` model (Apple Silicon serving path).
- `gate_qwen35.py` — driver: runs the numpy reference + the mlx-lm probe, optionally also queries
  a live sglang server (native `/generate` API, given `--sglang-url`), and prints a PASS/WARN
  verdict per pair (top-1 token match, top-5 token-set match, max abs probability delta on shared
  tokens).
- `results/` — saved JSON outputs, one per gate run.

## Usage

```bash
# 1. one-time: convert an HF Qwen3.5 checkpoint to a flat .pth
python vendor/run_qwen35_make_pth.py --model /path/to/Qwen3.5-2B --output /tmp/qwen35_2b_text.pth

# 2. gate it (numpy vs mlx-lm; add --sglang-url to also check a live sglang server)
python gate_qwen35.py --pth /tmp/qwen35_2b_text.pth --hf-dir /path/to/Qwen3.5-2B \
    [--sglang-url http://<tower>:30070] --out results/latest.json
```

Exit code 0 + `GATE_QWEN35_PASS` iff every leg that ran agrees; 1 + `GATE_QWEN35_FAIL` otherwise.

## Scope note

This checks a single probe position (next-token distribution after `" Eiffel"`, top-10), not a
24-step greedy-decode oracle like the RWKV-7 gate. That's deliberate — see the finding doc for
why exact bit-identity isn't the right bar across fp32-CPU / bf16-Metal / bf16-CUDA backends, and
what a top-1/top-5/probability-closeness gate does and doesn't guarantee. Only verified against
the 2B checkpoint so far; the 9B tier this project also benchmarks has NOT been run through this
gate yet (its own hardcoded-vs-actual shape constants would need the same by-hand verification
against its `config.json` first — do not assume F0050's "generalizes cleanly" finding extends to
9B without checking).
