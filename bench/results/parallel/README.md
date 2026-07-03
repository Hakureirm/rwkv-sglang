# TP / PP multi-GPU gates (1.5B, bf16)

Full data for the tensor-/pipeline-parallel verification matrix. All runs:
real multi-GPU hardware (L4 ×N), greedy fixture vs the numpy oracle
(24 tokens), **gate config = cuda-graph OFF** — the tok/s columns are
functional-verification numbers, NOT tuned throughput (single L4 *with*
cuda-graph does 76 tok/s bsz1; tuned multi-GPU throughput is a follow-up).
Design + analysis: [`../../../docs/findings/0019-tp-pp-parallel.md`](../../../docs/findings/0019-tp-pp-parallel.md).
**Raw transcripts** (engine logs + gate outputs + per-GPU memory, one file per run,
scrubbed of infra-internal paths): [`raw/`](raw/) — tp2/pp2 (`tp2_…`, `pp2_…`),
tp4+pp4+mixed (`tp4_pp4_tp2pp2_…`), tp8+pp8 (`tp8_pp8_…`), and the mixed-mode
fix verification at 0.1B and 1.5B (`tp2pp2_*_fix_verify.log`).

## Baseline regressions (tp=1, pp=1 — RTX 3090)

| model / mode | dtype | greedy |
|---|---|---|
| 0.1B default | bf16 | **EXACT 24/24** |
| 1.5B default | bf16 | **EXACT 24/24** |
| 1.5B RWKV_W8=1 | fp16 | **EXACT 24/24** |
| 0.1B post-PP (re-run) | bf16 | **EXACT 24/24** |
| 1.5B post-PP (re-run) | bf16 | **EXACT 24/24** |

## Multi-GPU matrix (1.5B bf16, L4 ×N, gate config)

| config | GPUs | greedy | bsz1 tok/s | bsz8 | bsz32 | per-GPU mem MiB (nvidia-smi) |
|---|---|---|---|---|---|---|
| tp=2       | 2× L4 | **EXACT 24/24** | 20.6 | 161.9 | 644.4 | (see note) |
| pp=2       | 2× L4 | **EXACT 24/24** | 17.3 | 133.1 | 525.1 | (see note) |
| tp=4       | 4× L4 | **EXACT 24/24** | 14.5 | 111.3 | 452.4 | 7241 ×4 |
| pp=4       | 4× L4 | **EXACT 24/24** | 15.1 | 117.7 | 446.5 | 3149 / 2879 / 2879 / 2897 |
| tp=2×pp=2  | 4× L4 | **EXACT 24/24** (fixed) | 14.4 | 110.3 | 442.0 | 4379 / 4379 / 4279 / 4279 |
| tp=8       | 8× L4 | **EXACT 24/24** | 15.1 | 115.9 | 454.4 | 7103 ×8 |
| pp=8       | 8× L4 | **EXACT 24/24** | 11.9 | 92.2 | 358.8 | 2143 / 1873 ×6 / 1891 |

- **Pure TP is greedy-EXACT at 2, 4 and 8 ranks; pure PP is greedy-EXACT at 2, 4
  and 8 stages.** 32 heads ÷ tp8 = 4 heads/rank; 24 layers ÷ pp8 = 3 layers/stage.
- **Mixed tp×pp: FIXED** (was 12/24 @token 12, deterministic, dtype-independent —
  0.1B failed at token 5, fp32 at the same token 12). Root cause: sglang's PP
  tensor-dict transfer chunk-sends each tensor as `reshape(tp,-1)[tp_rank]` and
  reassembles rank-by-rank on receive — lossless ONLY for tp-REPLICATED tensors
  (llama's hidden/residual). Our `v_first` is the LOCAL head slice under tp>1,
  so the reassembly produced a franken-tensor (half from each rank; measured:
  both receivers got identical wrong checksums ≠ either sender). Fix: all-gather
  v_first to full width before the boundary, slice per-rank after (rwkv7.py).
  Re-verified: 0.1B AND 1.5B tp2×pp2 greedy **24/24 EXACT**. Upstream-relevant:
  any model shipping non-replicated proxy tensors across PP hits this.
- Memory shows the expected split: PP divides weights across stages
  (pp8: ~1.9 GB/GPU), TP replicates less than it divides at this size because
  the mem_fraction-static pool dominates (tp: ~7.1 GB/GPU incl. state pool).
- gate-config tok/s decreases with more ranks (comm overhead, no cuda-graph) —
  expected for a 1.5B model that fits on one card; multi-GPU pays off for
  models/batches that DON'T fit (7.2B multi-GPU = follow-up).

Note: the 2-GPU runs predate the nvidia-smi per-GPU memory capture (added for
the 4/8-GPU suite); their engine-subprocess allocations were not visible to the
parent's torch accounting.

## Reproduce (any multi-GPU box)

```bash
python bench/verify_tp.py --model <fla-dir> \
  --fixture bench/fixtures/oracle_rwkv7_15b_eiffel.json --tp 2   # or 4 / 8
# pp / mixed: sgl.Engine(..., tp_size=T, pp_size=P) with the same fixture compare
```
