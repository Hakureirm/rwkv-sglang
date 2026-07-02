# Contributing

Thanks for your interest! This project integrates RWKV-7 into sglang for production serving.
Contributions are welcome — bug reports, benchmark reproductions, kernels, docs.

## Ground rules (what keeps this repo trustworthy)

1. **Correctness is gated, not assumed.** Any change touching the model/kernel path must pass:
   - `bench/verify_m1d.py` — greedy token-exact vs the numpy oracle (fp16 + bf16, cuda-graph);
   - `bench/verify_batch.py` — dynamic-batch (identical / shared-prefix / mixed) == B=1;
   - `bench/verify_chunked_prefill.py` — multi-chunk prefill == single-shot.
   If your change is *intentionally* value-perturbing (e.g. quantization), it must instead pass
   the lm-eval parity gate (see `bench/results/lm_eval.md` for the methodology) and say so.
2. **Every performance claim carries its numbers.** State the GPU, dtype, batch size, and the
   exact script/flags; commit raw logs under `bench/results/`. No "faster" without a table.
3. **No FLA on the RWKV-7 path** (see `docs/adr/0004-no-fla-dependency.md`). New kernels are
   hand-written CUDA/Triton with a torch reference fallback and a standalone numerics test
   (pattern: `bench/verify_w4.py`).
4. **Honesty over marketing.** If a result is mixed, publish both sides (see the README's
   "one axis albatross leads" section for the house style).

## Dev workflow

```bash
# deploy the overlay into an installed sglang (v0.5.10.post1)
BOX=<host> SP=<site-packages> bash scripts/deploy.sh
# run the correctness gates (needs a CUDA GPU + a converted model)
python bench/verify_m1d.py --model <fla_dir> --fixture bench/fixtures/oracle_rwkv7_15b_eiffel.json --dtype bfloat16 --cuda-graph
```

- Docs: `docs/human/` is the readable track (中文, diagrams); `docs/` (ADR / findings /
  snapshot) is the dense engineering record. A change that alters behavior updates
  `docs/snapshot.md` in the same PR.
- Style: match the surrounding code; kernels document their numerics contract
  (exactness, accumulation order, batch invariance) in the header comment.

## Reporting issues

Include: GPU model + driver, sglang version, the exact command, and (for correctness issues)
the `verify_*` output. Benchmark disputes are welcome — attach your raw log and hardware info.
