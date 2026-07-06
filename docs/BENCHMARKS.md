# Benchmarks — the full picture

Every measured axis of this project, in readable form. Each table states its setup and links
the committed raw output. Methodology details and negative results live in the dated reports
under [`findings/`](findings/); this page is the summary you can actually read.

**Engine versions.** Since 2026-07-05 all new measurements run on **sglang main**; earlier
numbers were measured on v0.5.10 and are kept, marked "(v0.5.10)". Where both exist we show
both — the migration itself changed nothing for correctness (verified) and made the 3090
slightly faster across the board.

**Two timing windows** (do not compare across them):

| | steady-state | wall-clock |
|---|---|---|
| plain meaning | "tokens per second once it's running" | "tokens per second from request sent to answer complete" |
| includes prompt reading (TTFT)? | no | yes |
| tool | `bench/serving_scale.py` | `bench/bsz_throughput.py` (64-in/256-out) |
| used in | §3 single-request ladder, §4 quant | §5 sweeps, §6 fleet, §7 Albatross |

Same config, same GPU: steady-state reads ~3% higher. Every table below says which window it uses.

All comparison tables use **cuda-graph ON** (the production decode path — serving_scale.py,
run_clean_comparison.py, and bsz_throughput.py via serve.sh). A separate script, throughput.py,
reports an **eager (cuda-graph OFF) baseline** used only for internal kernel-development tracking;
its numbers are ~2× lower for batched decode and are never quoted against the cuda-graph tables.

---

## 1. Correctness (the gate everything else stands on)

Greedy decoding is compared token-by-token against a pure-numpy fp32 reference implementation
(`bench/oracle_numpy.py`). A config ships only if it matches 24/24 tokens.

| what | result | raw |
|---|---|---|
| 0.1B / 1.5B / 7.2B, CUDA, fp16+bf16 | 24/24 exact, both sglang main and v0.5.10, RTX 3090 and RTX 5090 | `bench/results/greedy_gates_5090.log`, gate logs per finding |
| dynamic batching / chunked prefill / CUDA graphs | exact (mixed-batch and shared-prefix cases included) | F0022 |
| int8 (w8g64) | 24/24 exact — quantization is greedy-lossless; re-verified on sglang main (RTX 5090, fp16) | §4, F0015, `bench/results/quant_oracle_gates_5090main.log` |
| TP 2/4/8, PP 2/4/8, tp2×pp2 | 24/24 exact on real multi-GPU; **on main under cuda-graph ON, TP=2 and PP=2 are greedy 24/24 == 1-GPU + deterministic** (2×L4; fixed a PP+cuda-graph capture crash — F0036) | F0019, F0036 |
| Apple Silicon (MLX), 0.1B + 1.5B | 24/24 exact, both the pure-ops path and the custom Metal kernel | [`../mlx_port/`](../mlx_port/) |
| batch-position independence | outputs identical whether a request runs alone or inside a batch (prefix ≥ 4 tokens guaranteed; beyond that is bf16 accumulation, same as any engine) | test/registered/models/test_rwkv7.py |

Drift protection: after every major engine change the compression ruler is re-run — the last
full re-run was **bit-identical** (pooled 0.6085, drift −0.0000 over ~7.5M tokens).

## 2. Accuracy rulers (official RWKV evaluation definitions)

**Compression rate** (bits per byte on fresh corpora, lower is better; tokenizer-independent,
15 corpora × 500 documents). **Re-measured in full on sglang main (2026-07-05): every value below reproduced to the 4th decimal on BOTH the RTX 5090 and RTX 3090** — different silicon, different engine version, same pooled cross-entropy over ~7.5M tokens (`bench/results/uncheatable_full_*_5090main.json`, `*_3090main.json`):

| precision | pooled bpb | vs fp16 |
|---|---|---|
| fp16 | **0.6085** | — |
| int8 w8g64 | 0.6086 | +0.0001 (lossless in practice) |
| int8 w8a8 (throughput path) | 0.6161 | +0.0076 |
| int4 GPTQ | 0.6514 | +0.0429 |
| 7.2B fp16 | **0.5413** | −0.0672 vs 1.5B (bigger model compresses better) |
| 7.2B w8a8 | 0.5454 | +0.0041 vs 7.2B fp16 |
| 7.2B w4 (rwkv_w4, g64) | 0.5615 | +0.0202 vs 7.2B fp16 |

**Quantization costs less at 7.2B than at 1.5B.** w8a8: +0.0041 (7.2B) vs +0.0076 (1.5B); int4:
+0.0202 (7.2B, plain RTN `rwkv_w4`) vs +0.0429 (1.5B, the stronger GPTQ) — the 7.2B RTN checkpoint
degrades *less than half* as much as the 1.5B GPTQ one despite the weaker quantizer, i.e. the larger
model absorbs low-bit weights markedly better (`bench/results/uncheatable_full_{w4,w8a8}_7.2b_5090main.json`).

Per-corpus table (all 15, no cherry-picking) and the position-curve (proof the recurrent state
keeps absorbing context: 3.65 bits at position 0-64 → 2.24 bits past 1024) are in
`bench/results/uncheatable_*` and F-series reports.

**MATH500** (faithful port of Albatross's `eval_math500.py`: same prompt, sampling, grader,
1500-token budget):

| metric | value | note |
|---|---|---|
| avg@64 (v0.5.10) | **0.4060** (12,991 / 32,000 generations) | 500 problems × 64 samples |
| pass@64 (v0.5.10) | 0.6980 | ≥1 correct in 64 |
| greedy avg@1, v0.5.10 | 0.3920 (196/500) | deterministic |
| greedy avg@1, **main** | **0.3940 (197/500)** | Δ +0.0020, far inside the ±0.0220 noise band → no regression (`bench/results/math500_greedy_5090main.json`) |
| avg@64, **main** | **0.4042** (RTX 5090) / **0.4063** (RTX 3090) | both inside the ±0.0027 per-run band around v0.5.10's 0.4060 → no regression on either card (`bench/results/math500_avg64_{5090main,3090main}.json`) |
| **7.2B** greedy avg@1, main | **0.6320 (316/500)** | the flagship: +23.8pt over 1.5B's 0.3940 — a much stronger reasoner, at 3,248 tok/s on the 5090 (`bench/results/math500_greedy_7.2b_5090main.json`) |
| **w8a8** avg@64, main | **0.3812** (12,197/32,000) | vs fp16 0.4042 = **−2.3pt** — a real int8 reasoning cost the low-variance ruler resolves; compression (0.6161) and greedy hid it (`bench/results/math500_avg64_w8a8_5090main.json`) |
| **w8a8** greedy avg@1, main | 0.3800 (190/500) | vs fp16 0.3940 = −1.4pt (within 1 binomial SE at n=500) |
| **int4 GPTQ** greedy avg@1, main | 0.1560 (78/500) | vs fp16 0.3940 = **−24pt collapse** — perplexity-style rulers badly understate int4's reasoning damage (see §4 warning) |

The three quantization tiers on the *reasoning* ruler, ordered by damage: w8g64 (weight-only,
greedy-lossless) → w8a8 (−2.3pt) → int4 (−24pt). Compression rate alone would rank them
+0.0001 / +0.0076 / +0.0429 — the same order but wildly understating int4, which is why MATH500
is the ruler that decides quantization quality here.

## 3. Single-request speed ladder (steady-state, 1.5B fp16)

Each row adds one hand-written kernel set on top of the previous. RTX 5090, sglang main;
the 3090 column is the v0.5.10 historical ladder for lineage.

| config | RTX 3090 (v0.5.10) | RTX 5090 (main) | 5090 vs its baseline |
|---|---|---|---|
| no fast kernels | 166.5 | 261.9 | — |
| + fused GEMV + sparse FFN | 202.9 | 311.1 | +18.8% |
| + fused LoRA chain | 226.5 | 341.6 | +30.4% |
| **+ fused token-shift glue + launch autotune (full stack)** | — | **409.8** | **+56.5%** |
| int8 w8g64 (prequantized) | 227.4 | **461.9** | +76.4% |
| int4 (prequantized) | 259.1 | **548.8** | +109.5% |

Raw: `bench/results/ladder_*_5090.log`. The 3090-on-main ladder is being re-measured.

## 4. Quantization (what you trade and what you get)

Three modes, all with hand-written kernels, all arch-portable (JIT per GPU):

| mode | accuracy cost | when it wins | checkpoints |
|---|---|---|---|
| int8 w8g64 (weight-only) | none measurable (greedy-exact; compression +0.0001) | small-batch speed (+13% over fp16 full stack at bsz1 on 5090) + half the weight bytes | ModelScope `Hakureirm/rwkv7-g1-1.5b-w8g64` |
| int8 w8a8 (tensor-core) | compression 0.6161 (+0.0076, == cutlass); **MATH500 avg@64 0.3812 vs fp16 0.4042 = −2.3pt** (the low-variance ruler resolves a real reasoning cost the compression rate and greedy hid — same pattern as int4, far milder) | large-batch throughput king on sm80–90 (3090 peak 9,851 tok/s, 64-in/256-out). On sm120/Blackwell the upstream cutlass op does not exist; rwkv-sglang's own s8-wmma kernel (register-blocked V2, bit-exact gate, batch-invariant) now serves the tier there — the int8 GEMM beats fp16 cuBLAS at M≥512 (1.03–1.55×), while e2e peak is 20,991 tok/s (@c512, 64-in/256-out) = 0.9466× fp16 (the residual gap is the per-token activation-quant launch tax against an already-tuned fp16 baseline) | box-relayable |
| int4 GPTQ | lambada −1.28pt at 7.2B (RTN would be −2.64) | lowest VRAM: 7.2B in 4.6 GB weights, serves on a 16 GB T4 at 32.9 tok/s | ModelScope `Hakureirm/rwkv7-g1-{1.5b,7.2b}-w4gptq` |

Prequantized checkpoints are required (the loader reads qweight/scale keys; pointing the
quant flags at an fp16 dir errors out by design).

**Where int8 is decisive — 7.2B on a single 32 GB 5090 (measured 2026-07-06).** RWKV-7
state is constant-size, so the state-pool slot count is the max concurrency (per-request
state ≈ 33 MB, identical for both — it is fp32 model state, independent of weight
quantization). fp16 weights (14.4 GB) leave the pool room for only 221 concurrent and it
OOMs above; w8a8 weights (7.75 GB) free enough headroom for 640. Same launch, cuda-graph
ON, 64-in/256-out:

| 7.2B on one 5090 | max concurrency | peak output throughput |
|---|---|---|
| fp16 | 221 | 5,983 tok/s @c192 |
| **w8a8** | **640 (2.90×)** | **7,587 tok/s @c640 (1.268×, still climbing at 640)** |

Full concurrency sweep (output tok/s) — fp16 tops out and OOMs where w8a8 keeps scaling:

| concurrency | 1 | 128 | 192 | 221 | 320 | 448 | 512 | 576 | 640 |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | 124 | 5,668 | 5,983 | 5,747 | — OOM above 221 → | | | | |
| **w8a8** | 60 | 4,657 | — | 5,342 | 6,304 | 6,679 | 6,997 | 7,346 | **7,587** |

w8a8's curve is still rising at 640 (its own memory ceiling: 20.03 GB state pool, 1.92 GB
free); the 7,587 is a memory-bound floor, not a compute plateau. So int8 serves 7.2B at
**2.90× the concurrency and a 26.8% higher peak than fp16 can reach on this card** — fp16 is
pinned at the memory limit. Honest mechanism: at matched concurrency ≤221 fp16 is faster
per step (no activation-quant tax); w8a8 wins purely by reaching concurrency fp16 physically
cannot. Raw: `bench/results/72b/`.

**An honest int4 warning (measured 2026-07-05):** perplexity-style metrics understate int4's
damage to multi-step reasoning. On the 1.5B GPTQ checkpoint, compression looks mild (0.6514)
but **MATH500 greedy collapses to 0.1560** (78/500, vs fp16's 0.3940) — the quantized model
loses the thread mid-derivation and rambles to the token cap (60% truncation vs 14%). Treat
1.5B int4 as a memory tool for non-reasoning workloads; the 7.2B GPTQ (much smaller lambada
loss) is being re-checked on the same ruler. Raw: `bench/results/math500_greedy_w4gptq_5090main.json`.

**The sm120 w8a8 kernel (GEMM microbench).** Upstream cutlass `int8_scaled_mm` does not
compile for sm120, so on Blackwell consumer cards our hand-written s8-wmma GEMM (register-
blocked "V2", bit-exact vs a per-row reference, batch-invariant) is the only int8 path. It
beats fp16 cuBLAS on the projection shapes at decode/prefill batch (RTX 5090, standalone
GEMM, × = our speedup over fp16):

| projection shape | M=512 | M=1024 | M=4096 |
|---|---|---|---|
| attn 2048×2048 | 1.08× | 1.33× | 1.52× |
| ffn.k 8192×2048 | 1.45× | 1.52× | 1.55× |
| ffn.v 2048×8192 | 1.03× | 1.28× | 1.53× |

The GEMM wins, but 1.5B e2e is 0.9466× fp16 (§5): the per-token activation-quant launch,
not amortized across ~144 heterogeneous decode kernels, plus an already-excellent fp16
baseline, eat the kernel's margin. That tax is latent on the VRAM-bound 7.2B case above,
where int8's real win (2.90× concurrency) lives. Raw: `bench/verify_w8a8.py --bench`.

## 5. Serving throughput (wall-clock, 64-in/256-out, concurrency sweep)

| config | RTX 3090 main | RTX 5090 main |
|---|---|---|
| plain fp16, single request | 153.7 | 256.8 |
| plain fp16, peak | 7,205.5 @ 384 conc | 22,090.8 @ 512 |
| full kernel stack, single request | 230.7 | 397.3 |
| full kernel stack, peak | 7,257.7 @ 384 | **22,175.3 @ 512** |
| int8 w8a8 + fused glue, peak | **9,850.9 @ 256** | 20,991 @ 512 (own s8-wmma kernel V2; 0.9466× fp16 — GEMM >fp16, e2e just under) |

v0.5.10 reference points: 3090 plain peak was 6,885, w8a8+glue 9,686 — the main migration
alone made the 3090 faster. Raw: `bench/results/bsz_sweep_*_{3090main,5090}.json`.

Known pitfall reproduced on main: sglang defaults `cuda_graph_max_bs` to 24 for this model
family, silently falling back to eager above it — always set `--cuda-graph-max-bs` explicitly
(serve.sh does).

## 6. The 10-GPU fleet (same code, same recipe, every card)

1.5B fp16 full stack on sglang main, wall-clock. **Single-request = bsz1 sustained decode
(steady state); peak = best total throughput over a 64-in/256-out concurrency sweep** (capped
at 384 concurrency on the fleet, 512 on the workstation 5090):

| GPU | arch | single request | peak |
|---|---|---|---|
| T4 | sm75 | 97.1 | 3,176 |
| L4 | sm89 | 102.2 | 4,674 |
| A10G | sm86 | 168.3 | 6,627 |
| A100-40GB | sm80 | 257.0 | 17,042 |
| A100-80GB | sm80 | 278.9 | 18,420 |
| L40S | sm89 | 238.0 | 13,352 |
| H100 | sm90 | 361.1 | 28,578 |
| H200 | sm90 | 399.3 | 32,289 |
| B200 | sm100 | 381.6 | **40,544** |
| RTX PRO 6000 | sm120 | 315.0 | 21,566 |
| RTX 5090 (workstation) | sm120 | **397.3** | 22,175 |

Notable: at single-request the consumer RTX 5090 matches H200 (397.3 vs 399.3) and beats
H100 and B200 — single-stream decode is a memory-bandwidth story and GDDR7 delivers. Raw:
`bench/results/fleet_main_10cards.json`.

## 6b. Multi-GPU: TP / PP (verified on main, cuda-graph ON)

Tensor- and pipeline-parallel, on sglang main under the production cuda-graph path (F0019's
matrix was cuda-graph OFF). 1.5B bf16, 2×L4, wall-clock tok/s, **64-in/256-out** (c1/c8/c32/c64
= concurrency). **TP=2 and PP=2 are both greedy 24/24 identical to single-GPU and
deterministic** — multi-GPU changes nothing about the output. (Getting PP here first required
fixing a cuda-graph capture crash — F0036.)

| config | greedy vs 1-GPU | c1 | c8 | c32 | c64 (peak) | vs tp=1 |
|---|---|---|---|---|---|---|
| tp=1 (1 GPU) | reference | 72.6 | 482.3 | 1,612.9 | 2,582.6 | — |
| **tp=2** | **24/24 exact** | 105.3 | 655.9 | 2,008.6 | **3,026.2** | **1.17×** |
| **pp=2** | **24/24 exact** | 65.4 | 367.7 | 1,365.5 | 2,288.8 | 0.89× |

Honest read: at 1.5B on PCIe-connected L4s, TP=2 buys ~1.17× at c64; PP=2 is 0.89×
(pipeline bubbles dominate at this model size — PP's job is fitting a model larger than one
card, not per-token speedup for a small one). The value is that both are **correct and
production-viable** on main; scaling for models that actually need multiple cards (7.2B+,
NVLink) is a follow-up. Raw: `bench/results/tppp_l4_main.json`.

## 7. Comparison with Albatross (BlinkDL's official speed reference)

Albatross is a forward-loop benchmark (no scheduler, no dynamic batching, no API); this
comparison answers exactly one question — raw single-stream speed — with the same 1.5B
weights file on every card. Its shipped constants were tuned by the author on his own
RTX 5090, so "stock" is its best case there and its out-of-box state everywhere else.
Timing note: the Albatross column excludes prompt reading, ours includes it (~3% against us),
so these ratios are conservative lower bounds.

| GPU | Albatross (tok/s) | ours (tok/s) | ours / Albatross |
|---|---|---|---|
| T4 | **stock kernel won't compile** (sm80+ `cp.async`; removable — see note†) | 97.1 | out-of-box, only we run |
| L4 | 113.5 | 102.2 | **0.9004** |
| A10G | 203.4 | 168.3 | 0.8274 |
| L40S | 291.8 | 238.0 | 0.8156 |
| A100-40GB | 341.3 | 257.0 | 0.7530 |
| A100-80GB | 385.5 | 278.9 | 0.7235 |
| RTX 3090 | 309.2 (we re-tuned it for this card) | 230.7 | 0.7461 |
| RTX PRO 6000 | 457.4 | 315.0 | 0.6887 |
| RTX 5090 (author's own card) | 553.9 | 397.3 | 0.7173 |
| H100 | 607.3 | 361.1 | 0.5946 |
| H200 | 684.3 | 399.3 | 0.5835 |
| B200 | 744.0 | 381.6 | 0.5129 |

† The T4 gap is the *shipped* Albatross WKV kernel's `cp.async` (an sm80+ instruction); BlinkDL
notes this is removable — a patched kernel runs on T4 — so it's a packaging limit, not a fundamental
one. The claim here is strictly out-of-the-box: our stack serves T4 unmodified, and we have not
benchmarked a hand-patched Albatross on T4.

How to read it: the gap tracks memory bandwidth. On inference cards we are close (0.90 on
L4); on HBM monsters its whole-layer fused kernel stretches ahead (0.51 on B200) because our
per-operator launch overhead grows in relative terms as compute gets faster — which is
precisely what our next speed increment (CUDA graphs + deeper fusion) targets. Meanwhile our
**int4 path reaches 0.9908× of Albatross's fp16 on the author's own 5090** (548.8 vs 553.9,
cross-precision), and the T4 row shows the coverage difference. Raw:
`bench/results/albatross_fleet_10cards.json` + per-run logs.

One more finding: on CUDA 12.9 the constants Albatross ships are no longer optimal even on
the 5090 they were tuned for. We went further and **re-tuned Albatross for this card
ourselves** (14 dispatch-table edits, every one verified numerically and end-to-end, one
false win from an L2-resident microbench caught and reverted — full diff and evidence in
`bench/results/albatross_5090/`). Result, stock → re-tuned on the RTX 5090 (median of 3):

| model | batch | decode | prefill |
|---|---|---|---|
| 0.1b | 1 / 8 / 32 | +0.0% / **+11.0%** / +0.0% | +0.9% / +2.2% / **+13.4%** |
| 1.5b | 1 / 8 / 32 | +0.0% / **+6.6%** / **+7.9%** | +5.2% / +1.9% / +2.9% |
| 7.2b | 1 / 8 / 32 | +0.0% / +3.5% / +0.0% | +1.2% / +2.9% / **+5.0%** |

Single-stream decode does not move (memory-bandwidth wall — stock 7.2b at 147.0 tok/s already
exceeds the author's own published 144.04); the batch shapes gain up to 13%. Our single-request
ratios above are against the stock numbers; against the re-tuned track they are unchanged at
bsz1 (554.0 vs 553.9). Our launch parameters re-select at warmup on any card+CUDA — the design
difference the next table quantifies.

## 7b. Comparison with vllm-rwkv (the community vLLM fork)

Measured 2026-07-06 under strictly equal conditions: same GPUs (RTX 3090 + RTX 5090), same
weights file (tensor-verified), same client logic (the sweep client ported to the vllm-rwkv OpenAI
endpoint, identical 64-in/256-out protocol), vllm-rwkv at its documented best config.
Disclosure: the vllm-rwkv tip (`4bf0239a1`) crashes on the first decode as shipped (an
interface mismatch introduced by its automated upstream rebase); all vllm-rwkv numbers below
required a documented 2-line compatibility fix to run at all. That branch force-push rebases
daily — pin commits when reproducing.

**Correctness:** vllm-rwkv's fp16 engine also reproduces the fp32 numpy-oracle fixture 24/24
token-exactly on both GPUs — two independent engines converging on the same reference is
mutual validation, recorded plainly to vllm-rwkv's credit.

**Throughput, vllm-rwkv / rwkv-sglang full-stack (wall-clock, in64/out256):**

| concurrency | RTX 5090 | RTX 3090 |
|---|---|---|
| 1 | 1.1352 (vllm-rwkv leads) | **0.8114 (rwkv-sglang leads: 230.7 vs 187.2)** |
| 8 | **0.9204 (rwkv-sglang leads)** | **0.8947 (rwkv-sglang leads)** |
| 32 | **0.9866** | **0.9980** |
| 64 | **0.9858** | **0.9557** |
| 128 | 1.0507 | **0.9745** |
| 256 | 1.2194 | 1.0985 |
| peak (512/384) | 1.2621 (27,988 vs 22,175) | 1.1702 (8,493 vs 7,258) |

**Reading it honestly:** vllm-rwkv's kernels are Albatross's (ported file-by-file), so single-stream
tracks the Albatross baseline — vllm-rwkv leads bsz1 on the 5090; on the 3090 rwkv-sglang's hand-written
GEMV stack beats the port outright. rwkv-sglang leads the c8–64 middle on the 5090. **vllm-rwkv leads
high concurrency on both cards (up to 1.26×)** — that is the real result of this comparison
and rwkv-sglang's next kernel target. Two counters already exist: on the 3090 rwkv-sglang's int8 w8a8 peak
(9,851) beats vllm-rwkv's fp16 peak (8,583) by **1.1477×**; on the 5090 the upstream cutlass
int8 op does not exist; rwkv-sglang's own s8-wmma kernel (V1) now runs the tier there end-to-end
(20,991 @c512 = 0.9466× fp16; the int8 GEMM itself is 1.03–1.55× fp16 at M≥512) — the availability gap is closed, and the 3090
ratio is 1.38×) is the single highest-leverage speed item. Raw:
`bench/results/vllmrwkv/` (correctness JSONs with full token ids + both sweeps per card).


> The 3090 column is the clean re-measurement (`_v2`, max_num_seqs sized to 24GB): it
> reproduces the first box run within ~2% at every point (if anything slightly lower), so the
> earlier numbers were sound — the 3090/5090 asymmetry is a real hardware effect (higher
> bandwidth favors the fused-layer kernels at high concurrency), not a config artifact.

## 7c. Real-workload comparison (ShareGPT, variable-length conversations)

The synthetic sweep above uses one fixed shape (64-in/256-out). Real serving is
variable-length, which stresses the scheduler differently. Same neutral client
(`sglang.bench_serving`), same ShareGPT file, same 500 prompts, same weights, each engine
at its best config. Two load levels: peak (all requests at once) and steady (16 req/s). Equal-conditions proof:
all 8 runs processed exactly 168,913 input tokens and generated exactly 109,861 output tokens
— same prompts in, same tokens out (identical weights + greedy + ignore_eos).

**Output throughput (tok/s) and latency, RTX 5090:**

| load | engine | output tok/s | median TTFT | p99 inter-token |
|---|---|---|---|---|
| peak | rwkv-sglang | **9,602** | **2,503 ms** | **20.5 ms** |
| peak | vllm-rwkv | 8,865 | 3,458 ms | 370.8 ms |
| 16 req/s | rwkv-sglang | 3,300 | 31.6 ms | 37.8 ms |
| 16 req/s | vllm-rwkv | 3,351 | 24.1 ms | 22.7 ms |

**RTX 3090:**

| load | engine | output tok/s | median TTFT | p99 inter-token |
|---|---|---|---|---|
| peak | rwkv-sglang | **3,974** | **7,297 ms** | **717 ms** |
| peak | vllm-rwkv | 2,805 | 12,750 ms | 1,595 ms |
| 16 req/s* | rwkv-sglang | 2,477 | **316 ms** | 1,239 ms |
| 16 req/s* | vllm-rwkv | 2,600 | 375 ms | 375 ms |

*The 3090 can't actually sustain 16 req/s on this model (both engines top out ~11–12 req/s),
so this row is mild overload, not true steady state. The 5090 handles 16 req/s comfortably.

**The reversal — and it's the point.** On the *synthetic fixed-shape* sweep, vllm-rwkv led
high concurrency (its Albatross kernels + decode-wave batching like uniform shapes). On
*real variable-length* load at peak, **rwkv-sglang leads throughput on both cards** (1.08× on
the 5090, 1.42× on the 3090) with lower median time-to-first-token — sglang's continuous
dynamic batching packs uneven requests without the bubbles a wave scheduler leaves on
variable shapes. At steady 16 req/s the two are within a few percent on throughput, and
tail latency is mixed (vllm-rwkv's steady-state inter-token tail is tighter; rwkv-sglang's
peak-load tail is far tighter on the 5090). Net: for realistic mixed-length serving at high
load, rwkv-sglang is ahead; at light steady load they trade. Raw: `bench/results/realload/`.

## 8. Launch autotune across cards (why hardcoded constants don't travel)

Kernel-level A/B of our GEMV launch autotune vs the built-in heuristic (interleaved 4-pass
median; only the numerically-safe axis is tuned by default). Gain = time saved on that shape:

| GPU | att_rkvo / ffn_key / ffn_value | takeaway |
|---|---|---|
| T4 | +7.6% / +5.6% / +2.5% | wins where the heuristic misses |
| L4 | +0.1% / +11.3% / **+24.1%** | biggest win |
| A10G | +0.1% / +0.3% / +2.1% | near-parity |
| A100-40/80 | ≤ +4.9% | mixed |
| L40S | +0.0% / +9.2% / +2.6% | wins |
| H100 / H200 / B200 | ≈ 0 | heuristic already optimal |
| RTX 3090 | 0% ± noise | honest zero (serving-level, 7 runs) |
| RTX 5090 | +0.0% / +3.2% / +5.0% | tile choice differs from heuristic at 170 SMs |

Raw: `bench/results/autotune_ab_9cards.json`, `autotune_ab_5090.json`. F0025 has the
methodology (including the clock-ramp artifact that forced the interleaved design).

## 9. Latency under real load

**Poisson arrivals** (requests arrive at a fixed average rate; 512-in/256-out; RTX 5090 main):

| arrival rate | output tok/s | TTFT p50 / p99 | per-token p50 / p99 |
|---|---|---|---|
| 2 req/s | 524 | 23.6 / 43.4 ms | 3.8 / 5.1 ms |
| 8 req/s | 2,047 | 26.6 / 52.2 ms | 5.1 / 5.5 ms |
| 16 req/s | 3,977 | ~27 / ~52 ms | ~5 / ~5.5 ms |
| 300 at once | 11,865 | 1.7 / 3.3 s | 18.6 / 24.7 ms |

No queueing below 16 req/s — first-token latency stays ~26 ms. The 3090 (v0.5.10) reference
had 302 ms TTFT at 16 req/s. Raw: `bench/results/pd_mixed_5090.json`, `pd_mixed_3090main.json`.

**ShareGPT** (real conversation lengths, standard `bench_serving`, 500 requests, RTX 5090):
peak 9,845.6 output / 27,527.7 total tok/s; at 16 req/s median TTFT 32.3 ms. Raw:
`bench/results/sharegpt_{peak,r16}_5090.log`.

## 10. The structural advantage: constant-size state

| scale axis | baseline | scaled | extra peak VRAM |
|---|---|---|---|
| concurrency 1 → 256 (1.5B, 3090) | 12,420 MiB | 12,622 MiB | **+202 MiB** |
| context 1K → 64K (1.5B) | 12,364 MiB | 12,368 MiB | **+4 MiB** |
| context 1K → 32K (7.2B) | 17,866 MiB | 17,866 MiB | **+0 MiB** |
| concurrency 1 → 64 (7.2B, 24 GB card) | 46.6 tok/s | 1,802.7 tok/s | +308 MiB |

A Transformer's KV cache grows on both axes; RWKV-7's state does not. This is why a single
32 GB 5090 serves **640 concurrent 7.2B streams** with w8a8 (§4) — the state pool is the only
thing that scales with concurrency, and it is tiny and fixed-per-request. (The VRAM-growth
rows above are v0.5.10 measurements; unchanged by design on main.)

## 11. Speculative decoding (phase 1)

Draft model proposes K tokens, target verifies them in one pass, rejected tokens roll back by
restoring an O(1) state snapshot. Status: functional; 9/10 gate prompts token-identical to
normal decoding, mean 3.17 tokens accepted per round (measured acceptance rate α = 0.738).
The single differing token was traced to float rounding-order (the probe: it occurred exactly
at the sequence's smallest top-2 logit gap, 0.005 nats) — the verify's M=K GEMM reduces in a
different order than the M=1 baseline decode. The exactness fix is built and gated: `gemv_mb`,
a batch-invariant M-row GEMV whose every row is bit-identical to the decode kernel (`gemv_m1`)
— routing the verify's projections through it makes spec-on ≡ spec-off. Remaining: wire it in,
port the worker to sglang main, and add the draft/verify CUDA graphs (the speedup). Full
analysis: [F0031](findings/0031-spec-decode-increment-i.md), F0029 (viability), ADR-0006.

## 12. Apple Silicon (MLX)

Native implementation, custom Metal WKV kernel, gated by the same numpy reference:

Default (Metal WKV) path, Apple M5, 32 GB unified memory, MLX 0.31.2 (decode median of 5, prefill median of 3, lightly-loaded host):

| | 0.1B | 1.5B | 7.2B |
|---|---|---|---|
| greedy vs oracle | 24/24 | 24/24 | 8/8¹ |
| decode, single stream | 294 tok/s | 32 tok/s | 7.6 tok/s |
| prompt reading (1024 tok) | 10,457 tok/s | 1,787 tok/s | 453 tok/s |
| peak memory | 0.54 GiB | 3.38 GiB | 14.6 GiB |

¹ the 7.2B oracle fixture is 8 tokens; 0.1B/1.5B are 24 — all three token-exact on BOTH the pure-ops
and Metal paths. **7.2B fp16 fits in 32 GB unified memory with headroom.**

The fused **Metal WKV kernel is the default**: it reads the prompt **4.4–8.1× faster** than the
pure-ops scan (0.1B 8.1× / 1.5B 5.6× / 7.2B 4.4×) at equal-within-noise decode (both bandwidth-bound
on the per-token weight read). `RWKV_MLX_WKV=pure` is a JIT-free fallback. Peak memory is measured
per-model (a prior number double-counted a retained compiled-decode closure — corrected). See
[F0037](findings/0037-mlx-fused-metal-default.md) and [`../mlx_port/`](../mlx_port/).

**MLX now matches the CUDA platform's coverage** (F0038–F0041, all oracle-exact 24/24 on the fp16
default): M5 kernel profiling (bsz1 decode is weight-bandwidth-bound, ~79% of the hard ceiling) plus a
bit-exact decay-precompute WKV win; **opt-in w8/w4 weight quant** (`RWKV_MLX_QUANT`, `mx.quantize` g64,
mirrors CUDA w8g64/w4g64) — **w8 greedy-lossless, decode +49% (1.5B) / +68% (7.2B), peak memory −33%
(w8) / −55% (w4)**; a direct-call compression-rate ruler (w8 +0.0003 bpb = lossless, w4 +0.0504 —
matching the CUDA column); and a real ShareGPT single-stream bench (w8 halves inter-token latency).
fp16/bf16 remains the exact default.

---

*In-progress (this page is updated as they land): MATH500 avg@64 and full compression on
main for both GPUs; 3090-on-main ladder; per-size decode/prefill grid vs Albatross retuned.*
