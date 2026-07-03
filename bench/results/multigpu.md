# RWKV-7 × sglang — multi-GPU correctness + speed

Goal: broad GPU coverage (consumer + datacenter). We prove the deliverable (`sglang_overlay/`
over sglang **0.5.10.post1**) is **greedy-EXACT** vs the numpy-oracle fixture and measure decode
speed across consumer + datacenter GPU architectures (Turing / Ampere / Ada / Hopper / Blackwell), with **no
per-arch code change**. Each GPU was benchmarked on a real instance of that card.

- **Correctness gate** per GPU = `bench/verify_m1d.py` greedy **EXACT** vs fixture, at **bf16 +
  cuda-graph** (production config). **Speed** = `bench/throughput.py --cuda-graph --disable-radix-cache`.
- **int4** = our hand-written weight-only GEMV (`RWKV_W4=1`, `bench/quant_w4.py` RTN g64); see
  [`../../docs/findings/0017-w4-int4-quant.md`](../../docs/findings/0017-w4-int4-quant.md).
- Model = the same checkpoints the fixtures were generated from (`BlinkDL/rwkv7-g1`, converted with
  `tools/convert_rwkv7_blinkdl_to_fla.py`). Image = a CUDA-devel base + sglang 0.5.10.post1 + the
  overlay (== `scripts/deploy.sh`). Raw per-GPU JSON: [`allcards.json`](allcards.json).

## 1. Comprehensive 1.5B sweep — bf16 + **int4**, 10 GPU types (Turing → Blackwell)

bf16 greedy-EXACT gate + decode tok/s (bsz 1/8/32), plus our int4 decode (bsz1) and its speedup
over bf16:

| GPU | sm | bf16 greedy | bf16 decode 1/8/32 | int4 decode bsz1 | int4/bf16 (bsz1) | int4 peak VRAM (MiB) |
|---|---|---|---|---|---|---|
| T4 | 7.5 (Turing) | **24/24** | 65.1 / 446.5 / 592.3 | 114.9 | **1.77×** | 4661 |
| L4 | 8.9 (Ada) | **24/24** | 75.5 / 520.8 / 737.0 | 154.8 | **2.04×** | 6597 |
| A10G | 8.6 (Ampere) | **24/24** | 105.4 / 767.2 / 985.5 | 198.2 | **1.88×** | 6649 |
| A100-40GB | 8.0 (Ampere) | **24/24** | 161.6 / 1223.1 / 4369.6 | 204.9 | 1.27× | 11084 |
| A100-80GB | 8.0 (Ampere) | **24/24** | 166.3 / 1340.8 / 4416.9 | 204.9 | 1.23× | 21108 |
| L40S | 8.9 (Ada) | **24/24** | 171.3 / 1090.1 / 4150.0 | 287.9 | **1.68×** | 12265 |
| H100 | 9.0 (Hopper) | **24/24** | 229.7 / 1788.1 / 6569.2 | 261.2 | 1.14× | 21042 |
| H200 | 9.0 (Hopper) | **24/24** | 241.6 / 1875.4 / 6937.6 | 262.7 | 1.09× | 35782 |
| B200 | 10.0 (Blackwell) | **24/24** | 217.4 / 1801.1 / 7213.1 | 248.7 | 1.14× | 45437 |
| RTX PRO 6000 | 12.0 (Blackwell) | **24/24** | 201.3 / 1167.1 / 5469.4 | 284.2 | **1.41×** | 24847 |

- **bf16 is greedy-EXACT on all 10 GPU types across 7 SM generations** (7.5 / 8.0 / 8.6 / 8.9 /
  9.0 / 10.0 / 12.0 — Turing → Blackwell) — broad-GPU-coverage correctness holds universally; the
  RWKV-7 WKV + fused-glue Triton kernels JIT-compiled and ran on every SM generation with no
  per-arch change (only sgl_kernel needs the `libnuma1` system lib in the image).
- **Blackwell rows** (B200, RTX PRO 6000) were measured on a CUDA **12.8** devel image with the
  torch **cu128** wheel (Blackwell needs sm100/sm120 kernels + a 12.8 nvcc for the int4 JIT);
  the other rows used the CUDA 12.4 image with the same sglang/torch versions — the base-image
  difference affects only the JIT toolchain, not the model stack. B200 peak prefill:
  **103,022 tok/s** @bsz32 (new peak, vs H200's 78,268).
- **Turing caveat (honest):** sm75 has **no native bf16 compute** — on T4, bf16 runs via
  fp32-conversion emulation. It is numerically exact (24/24 above), and we measured the fp16
  (Turing's natural dtype) baseline for a fair comparison: **T4 fp16 is also greedy-EXACT 24/24**,
  and fp16 decode ≈ bf16 decode (65.4/446.0/604.6 vs 65.1/446.5/592.3 at bsz 1/8/32) — the
  emulation cost is negligible here because decode is bandwidth-bound. **T4 int4 vs fp16**
  (`gemm_w4_small` kernel): 116.0/354.5/642.4 → **1.77× at bsz1**, 0.79× at bsz8 (T4's scalar-FMA
  throughput moves the small-M crossover earlier than on the 3090 — per-arch cutover tuning is a
  noted follow-up), 1.06× at bsz32; int4 peak VRAM 4609 vs fp16 5597 MiB (bsz1). Raw:
  `allcards.json` entry `T4-fp16`.
- **int4 runs on all 10, including Turing (T4 sm7.5)** — the kernel uses plain vectorized loads, no
  `cp.async` (so it is not limited to sm80+). int4 bsz1 is faster than bf16 on every card; the
  speedup is **largest on bandwidth-starved cards** (L4 2.04×, A10G 1.88×, T4 1.77×) and smallest on
  compute-rich Hopper (H100 1.14×, H200 1.09×) — the bandwidth-bound signature. int4 greedy matches
  the 3090 bit-for-bit (arch-consistent kernel). int4 at bsz32 ties/beats bf16 on bandwidth-starved
  cards (T4 614 vs 592) but is slower on compute-rich cards (batch is compute-bound + the M>1 dequant
  fallback; fused int4 GEMM is the endgame — see F0017).

### 1b. Full precision × batch grid (1.5B, decode + prefill tok/s, bsz 1/8/32)

Every precision on every card. bf16 decode + prefill:

| GPU | arch | bf16 decode 1/8/32 | bf16 prefill 1/8/32 |
|---|---|---|---|
| T4 | Turing 7.5 | 65 / 446 / 592 | 4,909 / 5,590 / 5,562 |
| L4 | Ada 8.9 | 76 / 521 / 737 | 8,996 / 11,197 / 11,326 |
| A10G | Ampere 8.6 | 105 / 767 / 986 | 10,886 / 13,091 / 13,258 |
| A100-40GB | Ampere 8.0 | 162 / 1,223 / 4,370 | 13,591 / 32,919 / 34,328 |
| A100-80GB | Ampere 8.0 | 166 / 1,341 / 4,417 | 9,086 / 34,436 / 37,056 |
| L40S | Ada 8.9 | 171 / 1,090 / 4,150 | 23,364 / 39,156 / 40,806 |
| H100 | Hopper 9.0 | 230 / 1,788 / 6,569 | 23,874 / 71,690 / 74,109 |
| H200 | Hopper 9.0 | 242 / 1,875 / 6,938 | 24,830 / 74,319 / 78,268 |
| B200 | Blackwell 10.0 | 217 / 1,801 / 7,213 | 24,547 / 93,262 / 103,022 |
| RTX PRO 6000 | Blackwell 12.0 | 201 / 1,167 / 5,469 | 23,952 / 54,516 / 57,408 |

int8 (w8a8) + int4 decode:

| GPU | int8 decode 1/8/32 | int4 decode 1/8/32 |
|---|---|---|
| T4 | — (needs sm80+) | 115 / 196 / 614 |
| L4 | 106 / 792 / 505 | 155 / 371 / 752 |
| A10G | 142 / 1,085 / 620 | 198 / 384 / 922 |
| A100-40GB | 174 / 1,342 / 4,535 | 205 / 628 / 2,383 |
| A100-80GB | 178 / 1,394 / 4,733 | 205 / 638 / 2,421 |
| L40S | 201 / 1,511 / 5,069 | 288 / 763 / 2,944 |
| H100 | 218 / 1,688 / 6,266 | 261 / 848 / 3,283 |
| H200 | 225 / 1,739 / 6,475 | 263 / 912 / 3,537 |
| B200 | — (sm100 unsupported) | 249 / 1,542 / 4,385 |
| RTX PRO 6000 | — (sm120 unsupported) | 284 / 1,501 / 3,191 |

**int8 (w8a8) is bounded to sm80–90** by sgl-kernel's cutlass int8 GEMM: below (T4 sm75) it
crashes with cutlass `Error Internal`; above (B200 sm100 / RTX PRO 6000 sm120) it raises an
explicit `NotImplementedError: No implemented int8_scaled_mm for current compute capability` —
an upstream kernel-coverage limit, not an RWKV-path issue. **Our int4 runs on all 10** (it JIT
builds per-arch). int8 greedy is exact at 7.2B; at 1.5B it has the usual small-model quant drift
(`comparison_clean.md`). Peaks: **B200 prefill 103,022 tok/s @bsz32; B200 bf16 decode 7,213 tok/s**.

## 2. 0.1B correctness (all architectures)

0.1B, bf16 + cuda-graph, radix off. `verify_m1d.py` greedy EXACT vs `oracle_rwkv7_01b_eiffel.json`:

| GPU | sm | greedy EXACT | decode tok/s (bsz1 / bsz8) |
|---|---|---|---|
| T4 | 7.5 | **24/24** | 276.9 / 2130.4 |
| L4 | 8.9 | **24/24** | 401.2 / 2808.9 |
| A10G | 8.6 | **24/24** | 463.7 / 3175.5 |
| A100-40GB | 8.0 | **24/24** | 427.0 / 3177.3 |
| H100 | 9.0 | **24/24** (via the 1.5B gate) | — |

Greedy-EXACT held on every architecture — Turing, Ampere (consumer sm86 + datacenter sm80), Ada, Hopper.

## 3. Quantization notes across GPUs
- **int8 (w8a8) requires sm80–90** (both ends bounded by sgl-kernel's cutlass int8 GEMM): on T4
  (sm75) the load fails inside cuda-graph capture with `gemm execution failed: Error Internal`
  (no Turing config); on B200 (sm100) / RTX PRO 6000 (sm120) it raises
  `NotImplementedError: No implemented int8_scaled_mm for current compute capability` — both are
  upstream kernel-coverage limits (the `rc=-9` rows in `allcards.json`). Outside sm80–90, use our
  **int4** (works on all 10 GPUs, faster than bf16 at bsz1; fp16≈bf16 verified on T4) or fp16.
- **int8 (w8a8)** runs on Ampere / Ada / Hopper; on Hopper it is ~neutral vs bf16 at 1.5B (bf16
  tensor cores already saturate), so int8's value there is VRAM (−41–46% weights), not decode speed —
  its cross-precision decode win vs albatross-fp16 is at 7.2B (see [`comparison_clean.md`](comparison_clean.md)).
- **int4** — see §1 and [`w4/`](w4/): bsz1 faster than bf16 on every arch (fp16≈bf16 verified on T4) + ~4× weight-VRAM cut. **7.2B int4 verified live on a 16 GB T4**: greedy 8/8 EXACT, 32.9 tok/s bsz1, peak 6,735 MiB (`allcards.json`: `T4-72b-w4`).
- **fp8 (Hopper)** — **not feasible with the deliverable as-is**: sglang's dynamic-fp8 registers
  runtime `weight_scale` params the strict `load_weights` counts as "not loaded" (int8 works because
  its offline converter bakes the scales in). Would need an offline fp8 converter or a relaxed loader.

## 4. Reproduce
```bash
# per GPU, after deploy.sh has applied the overlay onto sglang 0.5.10.post1:
python bench/verify_m1d.py --model <fla_dir> --fixture bench/fixtures/oracle_rwkv7_15b_eiffel.json --dtype bfloat16 --cuda-graph
python bench/throughput.py --model <fla_dir> --dtype bfloat16 --batch-sizes 1,8,32 --cuda-graph --disable-radix-cache
# int4:
python bench/quant_w4.py --model <fla_dir> --out <w4_dir> --group 64
RWKV_W4=1 python bench/throughput.py --model <w4_dir> --dtype float16 --batch-sizes 1,8,32 --cuda-graph --disable-radix-cache
```
