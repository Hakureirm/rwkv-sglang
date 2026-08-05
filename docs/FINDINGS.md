# Findings index

Dated measurement reports, including the negative results. Each is self-contained:
methodology, raw data pointers, and what was retracted when a prediction failed.
Generated from front-matter by `python3 tools/gen_findings_index.py` — edit the
findings, not this table.

| id | finding | status |
|---|---|---|
| F0001 | [Dev box & environment recon](findings/0001-dev-box-and-env-recon.md) | open |
| F0002 | [RWKV-7 architecture & vLLM component mapping](findings/0002-rwkv7-architecture-and-vllm-mapping.md) | open |
| F0003 | [Parity baselines (rwkv-lm, albatross) & acceptance test definition](findings/0003-parity-baselines-and-acceptance.md) | open |
| F0004 | [Verified latest-upstream re-analysis (vLLM / sglang / HF)](findings/0004-latest-upstream-reanalysis.md) | open |
| F0005 | [M1 complete — RWKV-7 0.1B runs in sglang, exact greedy-match vs oracle](findings/0005-m1-complete.md) | closed_by_m1 |
| F0006 | [M2-baseline — bf16 + 1.5B exact greedy-match; throughput baseline; decode is eager-bound](findings/0006-m2-baseline-throughput.md) | open |
| F0007 | [Albatross speed baseline on our 3090; quantified gap; kernel-vendoring path](findings/0007-albatross-3090-baseline.md) | open |
| F0008 | [M2b — CUDA graph for RWKV-7 decode: 7.5-21× speedup, exact, gap vs albatross → ~2-3×](findings/0008-m2b-cudagraph.md) | open |
| F0009 | [7.2B exact + dynamic-batch correctness (radix auto-off) + full ours-vs-albatross table (gap shrinks with sc...](findings/0009-7.2b-comparison-radix.md) | open |
| F0010 | [M3b — deliverable is 100% FLA-free (own WKV kernel for decode+prefill), zero speed cost](findings/0010-m3b-de-fla-complete.md) | closed_by_m3b |
| F0011 | [M4 — w8a8-int8 quant: decode FASTER than bf16 + weight bytes −41-46%, accuracy preserved at scale](findings/0011-m4-quant.md) | open |
| F0012 | [Multi-GPU coverage — greedy-EXACT on 10 GPU types / 7 SM generations (Turing→Blackwell: T4/L4/A10G/A100-40/...](findings/0012-multigpu-coverage.md) | open |
| F0013 | [Elementwise fusion (+5-11% decode, EXACT) + the bit-exact↔speed ceiling; speed standing vs albatross](findings/0013-fusion-and-speed-standing.md) | open |
| F0014 | [Clean same-precision standing vs albatross (honest): raw speed loses, accuracy ties, VRAM/int8/serving win](findings/0014-clean-same-precision-standing.md) | open |
| F0015 | [CUDA endgame result: fused fp16 GEMV is greedy-EXACT and +5-9% bsz1 decode at 1.5B/7.2B, but cuda-graph amo...](findings/0015-cuda-endgame-result.md) | open |
| F0016 | [Serving-scale measured: decode throughput scales ~50× with concurrency (166→8298 tok/s, bsz 1→128) at flat ...](findings/0016-serving-scale-wedge.md) | open |
| F0017 | [Hand-written weight-only int4: faster than (or ties) fp16 at EVERY bsz≤32 (1.03–1.56×; gemv_m1 + gemm_w4_sm...](findings/0017-w4-int4-quant.md) | open |
| F0018 | [Hand-written weight-only int8 (w8a16): greedy-EXACT 24/24 (lossless in practice), faster than (or tied with...](findings/0018-w8-weight-only.md) | open |
| F0019 | [TP + PP multi-GPU: head-parallel tensor parallelism and layer-partition pipeline parallelism, both greedy t...](findings/0019-tp-pp-parallel.md) | open |
| F0020 | [Fused LoRA kernel (lora4_m1): all four LoRA chains in 2 launches — fp16 bsz1 decode 203.0 → 226.5 tok/s (+1...](findings/0020-fused-lora.md) | open |
| F0022 | [State prefix cache (req#3): RWKV-7 routed through sglang's state-aware MambaRadixCache — greedy-EXACT on sh...](findings/0022-state-prefix-cache.md) | open |
| F0023 | [Albatross-vs-ours kernel audit (GEMV / GEMM / LoRA / layer-glue), line-by-line: tests Bo's 'GEMV/GEMM/LoRA ...](findings/0023-albatross-kernel-audit.md) | open |
| F0024 | [MATH500 avg@64 (Bo's decreed accuracy metric, req#7) on RWKV-7 1.5B via our sglang stack: avg@64 = 40.60% (...](findings/0024-math500-avg64.md) | open |
| F0025 | [Serving eval: PD-mixed (open-loop Poisson) tail latencies + arch-aware GEMV launch autotune (A-segment): ge...](findings/0025-pd-mixed-and-gemv-autotune.md) | open |
| F0026 | [R2 fused paged layer-boundary glue (shift_lerp6): fuses the paged token-shift (gather+scatter, dropping the...](findings/0026-r2-fused-glue.md) | open |
| F0027 | [R4-B cross-arch occupancy (5 GPUs Turing→Blackwell, real cards): MEASURED validation of the launch-tuning t...](findings/0027-crossarch-occupancy.md) | open |
| F0028 | [Full-stack composition + per-bsz gating: all hand kernels compose greedy-EXACT; the fused LoRA is M-gated (...](findings/0028-fullstack-mgate.md) | open |
| F0029 | [Speculative-decoding viability (req#6): 0.1B-class RWKV-7 draft vs 1.5B target per-token greedy acceptance ...](findings/0029-spec-decode-viability.md) | open |
| F0030 | [Speculative-decoding HTTP two-server prototype is the WRONG vehicle: statecache mode forces cuda-graph OFF ...](findings/0030-spec-decode-http-prototype.md) | open |
| F0031 | [F0031 — Chain speculative decoding increment (i): functional in-engine worker; gate 9/10 token-identical, t...](findings/0031-spec-decode-increment-i.md) | open |
| F0032 | [F0032 — Equal-conditions comparison with vllm-rwkv, and the synthetic-vs-real-load reversal](findings/0032-vllmrwkv-showdown-realload-reversal.md) | open |
| F0033 | [F0033 — sm120 int8 tensor-core MMA: feasible with standard wmma, 1.9933× fp16 throughput](findings/0033-sm120-int8-mma-feasibility.md) | open |
| F0034 | [F0034 — w8a8 V2 register-blocked GEMM, the activation-quant tax, and where int8 is decisive](findings/0034-w8a8-v2-register-blocked-and-quant-tax.md) | open |
| F0035 | [F0035 — 7.2B on a single 32 GB 5090: int8 unlocks 2.90× concurrency and a 26.8% higher peak fp16 cannot reach](findings/0035-7b-int8-concurrency-headroom.md) | open |
| F0036 | [F0036 — PP + cuda-graph was broken on main (v_first proxy); fix VERIFIED + first TP/PP production throughput](findings/0036-pp-cudagraph-vfirst-fix.md) | open |
| F0037 | [MLX Apple-Silicon: fused-Metal WKV becomes the default — 5.5–8.5× faster prefill at equal-within-noise bsz1...](findings/0037-mlx-fused-metal-default.md) | open |
| F0038 | [MLX Apple-Silicon (M5): single-stream hotspot profiling + a bit-exact WKV kernel win — decode is already at...](findings/0038-mlx-m5-kernel-profiling.md) | open |
| F0039 | [MLX Apple-Silicon weight quantization (w8g64 / w4g64): the decode-bandwidth lever F0038 pointed at — w8 is ...](findings/0039-mlx-weight-quantization.md) | open |
| F0040 | [MLX Apple-Silicon accuracy ruler: uncheatable-eval compression rate (bits/byte) via a direct-call harness —...](findings/0040-mlx-compression-rate.md) | open |
| F0041 | [MLX Apple-Silicon real-workload (ShareGPT, bsz1 single-stream): realistic prefill+decode throughput and the...](findings/0041-mlx-sharegpt-realload.md) | open |
| F0042 | [CoreML/ANE feasibility probe for RWKV-7's WKV recurrence: FAIL — 0/47 ops ever prefer the Neural Engine at ...](findings/0042-coreml-ane-feasibility.md) | open |
| F0043 | [Asymmetric (scale+zero) GPTQ for w4, zero kernel changes: closes 27-35% of the fp16 gap across lambada/comp...](findings/0043-w4-asym-gptq.md) | open |
| F0044 | [Qwen3.5 runs on MLX today via mlx-lm 0.31.3 out of the box: native hybrid Gated-DeltaNet + interleaved full...](findings/0044-qwen35-mlx-feasibility.md) | open |
| F0045 | [Qwen3.5-2B vs RWKV-7 1.5B, matched MLX benchmark (same M5, same bench_mlx.py protocol, multi-run): RWKV-7 w...](findings/0045-qwen35-mlx-matched-benchmark.md) | open |
| F0046 | [RWKV_SPEC on sglang main, Strategy B built for real: 10/10 correctness gate (spec-on == spec-off token-iden...](findings/0046-spec-decode-strategy-b-build.md) | open |
| F0047 | [F0047 — RWKV-7 7.2B fp16 full-stack peak was undertested: true peak is 6,709 tok/s @ c320, not 5,983 @ c192](findings/0047-fp16-72b-concurrency-correction.md) | open |
| F0048 | [F0048 — Qwen3.5 has no viable same-tier "int8" comparison point; FP8 is the closest sglang-native substitut...](findings/0048-qwen35-int8-tier-gap.md) | open |
| F0049 | [Desktop-GPU tier (RTX 3090, 24GB) of the RWKV-7 vs Qwen3.5 comparison: same-precision bf16 peak concurrency...](findings/0049-qwen35-desktop-tier-3090.md) | open |
| F0050 | [Qwen3.5-2B correctness gate against Bo Peng's independent numpy fp32 reference: PASSES on both live serving...](findings/0050-qwen35-numpy-oracle-gate.md) | open |
| F0051 | [High-bandwidth-card decode gap (reverse-overtake W1): real H100 kernel-launch profile of the deployed fused...](findings/0051-lora-gate-fusion-highbw.md) | open |
| F0052 | [High-bandwidth-card decode gap (reverse-overtake W1 cont.): epilogue-fusing the FFN relu()**2 activation in...](findings/0052-sqrelu-epilogue-fusion-highbw.md) | open |
| F0053 | [Qwen3.5-2B MATH500 avg@64 (chatml_thinking) was run with RWKV-tuned sampling params left in place and zero ...](findings/0053-qwen35-math500-sampling-fix.md) | open |
| F0054 | [Qwen3.5-9B correctness gate against Bo Peng's independent numpy fp32 reference: PASSES against this project...](findings/0054-qwen35-9b-numpy-oracle-gate.md) | open |
| F0055 | [w4a8 large-M tensor-core path (task#52): kills the w4 M=64 concurrency cliff (c66 622.8->931.4 tok/s, peak ...](findings/0055-w4a8-large-m-tc.md) | closed |
| F0056 | [W1' serving fixes: internal step-time profiling on the 7.2B fp16 decode step (bs=320, shape A 128in/1280out...](findings/0056-w1prime-serving-fixes.md) | closed |
| F0057 | [RWKV_STATE_FP16 long-context positional compression gate (7.2B fp16, 3090, full N=7500 UncheatableEval corp...](findings/0057-state-fp16-positional-compression-gate.md) | closed |
| F0058 | [task #54 hand-CUDA WKV decode kernel: bit-exact (zero differing bytes) vs the Triton kernel on BOTH state d...](findings/0058-wkv-hand-cuda.md) | closed |
| F0059 | [F0059 — sglang_overlay drift debt: ground truth, categorization, and resync](findings/0059-overlay-resync.md) | closed |
| F0060 | [Megakernel Stage-A (#50): the 3090 bsz1 decode profile re-frames the endgame — the M==1 GEMVs already run a...](findings/0060-megakernel-stage-a-rkv-fusion.md) | open |
| F0061 | [Megakernel Stage-A2 (#50): the next fused-block components on the 3090 — (1) o_proj folded into a role-gene...](findings/0061-megakernel-stage-a2-oproj-shift-lora.md) | open |
| F0062 | [cp.async race-class audit vs Albatross ff144b6b ('bvec' zero-fill race, 2026-07-14): ALL our cp.async sites...](findings/0062-cp-async-zfill-race-audit.md) | closed |
| F0063 | [Megakernel sm120 assembly (#50): the PDL chain is LIVE — griddepcontrol wait/launch_dependents wired across...](findings/0063-sm120-pdl-chain-flagship.md) | closed |
| F0064 | [GEMV weight-stream bandwidth (#50 follow-on): the flagship bsz1 gap to Bo (D=87.9% of 155.2) lives in BUSY/...](findings/0064-gemv-weight-stream-bandwidth.md) | closed |
| F0065 | [Stage-B opener (#57): the bsz1 small-kernel BUSY is a LATENCY problem before it is a fusion problem — add_l...](findings/0065-smallkernel-latency-round.md) | closed |
| F0066 | [Stage-B fusion round (#57, after F0065): (a) fused add_ln+token-shift+lerp boundary kernel — ONE launch rep...](findings/0066-boundary-fusion-round.md) | closed |
| F0069 | [#59 public-number re-measure. Three results. (1) The published flagship numbers are NOT stale — 7.2B 142.8 ...](findings/0069-public-number-conventions.md) | closed |
| F0070 | [Scoring the HuggingFace RWKV-7 port on Bo's own ruler instead of against itself. Every correctness check th...](findings/0070-porting-accuracy-and-a-dead-harness-path.md) | open |
| F0071 | [OUR OWN HuggingFace RWKV-7 port (transformers-rwkv PR#2, written by this project) scaled the ln_x GroupNorm...](findings/0071-hf-port-groupnorm-epsilon.md) | closed |
| F0074 | [F0074 — rwkv_lightning: the serving layer albatross points at, and what it does not publish](findings/0074-rwkv-lightning-survey.md) | open |
| F0077 | [RWKV_SPEC completed to a real net win: the draft rollback was off by one token (accept 1.2 vs alpha 0.7, pe...](findings/0077-spec-decode-draft-desync-fix.md) | open |
| F0078 | [F0078 — the main-line graft was stale, and the profile said so](findings/0078-main-line-graft-was-stale.md) | open |
| F0079 | [F0079 — the large-batch step is GEMM-bound, and the megakernel line does not reach it](findings/0079-large-batch-is-gemm-bound.md) | open |
| F0080 | [F0080 — the channel-mix sparsity is mostly input-dependent, so it does not batch](findings/0080-ffn-sparsity-does-not-batch.md) | open |
| F0081 | [Mixed-precision by layer for int4: testing the claim that protecting {0, N/4, 3N/4, N-1} rescues the reason...](findings/0081-int4-layer-protection.md) | open |
| F0082 | [At 1.5B, our calibrated int4 (GPTQ) is 6.9 points WORSE on MATH500 than plain round-to-nearest — separated,...](findings/0082-gptq-loses-to-rtn-on-math500.md) | open |
| F0083 | [At 1.5B a non-uniform 4-bit lattice beats int4 by 4.4-4.6 points of MATH500 (separated), and total weight e...](findings/0083-grid-and-group-size.md) | open |
| F0085 | [Two ops, not one, made every ONNX export subtest fail: cumprod and linalg.solve_triangular](findings/0085-onnx-export-two-triggers.md) | closed |
| F0086 | [A measured +10.7% never shipped: the default was raised in the tree that was about to be deleted](findings/0086-the-improvement-was-committed-to-the-retired-tree.md) | closed |
| F0087 | [W1'': the 2<=T<=gate band was doing in torch what both of its neighbours had a kernel for — +5.2% at bs8, +...](findings/0087-the-band-between-the-two-fused-paths.md) | closed |
| F0088 | [lora4_mn stage2 re-read its weights once per token: −33% on the kernel, +1.6% on the step, and the gate sti...](findings/0088-lora-stage2-m-inner.md) | closed |
| F0089 | [Single-stream fp16 has 17% of headroom, not a factor — and int4, already faster, has 38%: measured against ...](findings/0089-the-single-stream-roof.md) | closed |
| F0090 | [int4 wins below c=32 by up to 1.75x and loses above c=64 by up to 1.87x — its headroom is bandwidth headroo...](findings/0090-int4-is-a-low-concurrency-lever.md) | closed |
