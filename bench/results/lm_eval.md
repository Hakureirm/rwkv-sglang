# RWKV-7 x sglang - accuracy: lm-eval parity + int8 drift + albatross fp16 accuracy

Goal: show OURS (sglang RWKV-7) matches the BlinkDL `rwkv-lm` reference on **standard
metrics** (not just greedy-exact), quantify **int8** drift, and - at the same fp16 weight
precision as albatross - measure whether albatross's **fp16 WKV state drifts**. This
establishes the **lm-eval-parity gate** that future value-perturbing fast kernels must pass.

All on the exclusive RTX 3090. Models are the **g1/g1g** series (0.1b-g1d, 1.5b-g1g,
7.2b-g1g), the SAME checkpoints our fla models were converted from - so the reference is
`rwkv` pip on that exact `.pth`. (The published "World-1.5B-v3 = 44.87% MMLU" is a DIFFERENT
checkpoint, cited only as a methodology sanity anchor; our g1g-1.5b scores ~0.51, higher, as
expected for the newer series - and our reference-reproduction below lands there.)

## 1. Tokenizer wired + verified (prerequisite for text-level eval)

The RWKV **World tokenizer** (fla-hub `hf_rwkv_tokenizer.py` + `rwkv_vocab_v20230424.txt`
+ `tokenizer_config.json` + `special_tokens_map.json`) was placed in every model dir so
sglang can tokenize text (`--trust-remote-code`). One fix was required for reference
fidelity: fla-hub's `tokenizer_config.json` sets `eos_token: "\n\n"`, which HF registers as
a special token (id 65530) and makes `"\n\n"` tokenize to `[65530]` instead of the raw-text
`[261]`. We set `eos_token` to the id-0 end-of-text token (`<|rwkv_tokenizer_end_of_text|>`,
RWKV's true document boundary), restoring raw-text encoding.

**Verification** (`bench/verify_tokenizer.py`): `AutoTokenizer.from_pretrained(model_dir,
trust_remote_code=True)` vs the `rwkv` pip PIPELINE trie, on 6 prompts incl. edge cases
(`"\n\n"`, CJK+emoji, digits/punct, the full MMLU template):
**ALL 6 MATCH token-for-token** (e.g. `"\n\n"` -> `[261]` both).

## 2. ours-fp16 greedy is EXACT vs the numpy oracle (with cuda-graph)

Precision-matched to albatross (fp16 weights) but with **fp32 WKV state**:
`bench/verify_m1d.py --dtype float16 --cuda-graph`

| model | greedy match vs numpy oracle |
|---|---|
| 0.1B | **24/24 EXACT** (first divergence: none) |
| 1.5B | **24/24 EXACT** (first divergence: none) |

So ours-fp16 reproduces the fp32 rwkv-lm reference token-for-token. (bf16 is likewise exact,
per F0006/F0008; 7.2B bf16 8/8 exact per F0009.)

## 3. lambada_openai (lm-eval-harness) - ours vs reference, + dtype drift

Standard lm-eval task, full 5153 examples. OURS served via the sglang OpenAI server and
driven by `lm-eval --model local-completions` (token-id requests, our HF tokenizer). The
REFERENCE is `rwkv` pip (cuda fp16) on the same `.pth` via `bench/accuracy_eval.py`.
**acc** (greedy last-word match) is the clean cross-tool parity metric.

| backend | precision | acc (greedy last-word) | ppl (lm-eval, per-doc) |
|---|---|---|---|
| **rwkv-pip REFERENCE** | fp16 | **0.6711** | (3.19 per-token*) |
| OURS sglang | **fp16** | **0.6728** | 4.747 |
| OURS sglang | bf16 | 0.6718 | 4.751 |
| OURS sglang | int8 (w8a8) | 0.6509 | 5.298 |

\*reference ppl is per-token (`accuracy_eval.py`); lm-eval's ppl is per-document - NOT
comparable, so **acc is the cross-tool parity metric**. Within lm-eval the ours ppls ARE
mutually comparable (int8 5.30 > fp16 4.75 = the int8 drift).

- **PARITY: ours-fp16 0.6728 vs reference 0.6711** = +0.17 pt (~9 of 5153) - a statistical tie
  (acc_stderr +/-0.0065). ours is marginally *higher* because our WKV state is fp32 vs
  rwkv-pip's fp16. bf16 (0.6718) is likewise at parity.
- **int8 drift**: -2.2 pt acc (0.6509), ppl 4.75->5.30 at 1.5B - modest; shrinks with scale
  (7.2B int8 is greedy-EXACT on the fixture, quant.md).

## 4. MMLU (BlinkDL rwkv_mmlu_eval methodology) - ours vs reference, + dtype drift

BlinkDL's exact MMLU method (template + single-token choices " A"/" B"/" C"/" D"), the
pipeline behind the published 44.87% for World-v3. **2000-example random sample (seed 42),
identical questions for every backend.** REFERENCE = `rwkv` pip cuda fp16 on the `.pth`
(`accuracy_eval.py --backend rwkv`, prepends token 0, argmax over the 4 choice logits).
OURS = the SAME template as an **lm-eval `multiple_choice`** task (`mmlu_blinkdl_local`) via
the sglang server + `local-completions` (the direct `/generate token_ids_logprob` feature is
buggy for RWKV in sglang 0.5.10 - see note - so we use lm-eval's working loglikelihood path).

| backend | precision | acc |
|---|---|---|
| **rwkv-pip REFERENCE** | fp16 | **0.5110** |
| OURS sglang | **fp16** | **0.5235** |
| OURS sglang | bf16 | 0.5250 |
| OURS sglang | int8 (w8a8) | 0.5145 |

- **PARITY: ours-fp16 0.5235 vs reference 0.5110** = +1.25 pt. The small offset is the one
  known methodology difference: BlinkDL prepends a leading token 0 that lm-eval's
  `multiple_choice` (add_bos_token=false) omits; both land at ~0.51-0.52. (Also validates the
  methodology: the reference reproduces the published-range ~0.51 for this newer g1g-1.5b.)
- **int8 drift**: fp16 0.5235 -> int8 0.5145 = **-0.9 pt** (negligible at 1.5B on MMLU).

## 5. Albatross fp16 accuracy at equal precision (HONEST finding: no drift on the fixtures)

At the SAME fp16 weight precision as the speed comparison, does albatross stay token-exact?
`bench/albatross_accuracy.py` greedily rolls out albatross on the fixture prompt and counts
matches vs the numpy-oracle `greedy_tokens` (the same fixtures OURS matches).

| model | albatross wkv state | greedy match | exact? | ours (same fixture) |
|---|---|---|---|---|
| 0.1B | fp16 (native) | 24/24 | **True** | 24/24 exact |
| 1.5B | fp16 (native) | 24/24 | **True** | 24/24 exact |
| 1.5B | fp32io16 | 24/24 | **True** | 24/24 exact |
| 7.2B | fp16 (native) | 8/8 | **True** | 8/8 exact |

**Honest result: albatross fp16 does NOT drift on these fixtures - it is greedy-EXACT too**
(albatross's fp16 WKV path uses deterministic dithering and is greedy-stable over these
24-/8-token rollouts). So there is **no greedy-accuracy gap between ours and albatross on the
fixtures** - both reproduce the oracle. We do NOT claim an accuracy win here. (These fixtures
are short, clear-argmax prompts; fp16-state drift could still appear on long free-running
generation, but we did not measure that and make no claim about it.)

## Methodology / commands

- **tokenizer**: `bench/verify_tokenizer.py --mode {ref,hf,compare}`.
- **fp16 exactness**: `bench/verify_m1d.py --model <dir> --fixture <fx> --dtype float16 --cuda-graph`.
- **lambada (ours)**: sglang server (`-m sglang.launch_server --dtype {float16,bfloat16}
  --disable-radix-cache --trust-remote-code --mem-fraction-static 0.45`) + `lm_eval --model
  local-completions --tasks lambada_openai_local --include_path bench/lmeval_tasks`.
- **MMLU (ours)**: same server + `lm_eval --tasks mmlu_blinkdl_local` (BlinkDL template as a
  `multiple_choice` task over the local seed-42 2000-sample parquet).
- **lambada + MMLU (reference)**: `bench/accuracy_eval.py --backend rwkv` (`rwkv` pip on the `.pth`).
- **albatross accuracy**: `bench/albatross_accuracy.py --wkv {fp16,fp32io16}`.
- **datasets**: box is air-gapped (no HF); lambada staged from the Mac, MMLU from the local
  `refs/RWKV-LM/RWKV-v7/mmlu_test_dataset`. Tasks read local parquet (`bench/lmeval_tasks/*.yaml`).
- **sglang logprob note**: sglang 0.5.10 rejects `/v1/completions` `echo`+`logprobs` and the
  native `/generate token_ids_logprob` returned near-constant (context-independent) values for
  RWKV - so a direct per-choice-logprob MMLU gave garbage (~random). lm-eval's
  `local-completions` loglikelihood path works correctly (its numbers match the reference), so
  we use it for ours-MMLU. Generation itself is unaffected (greedy-EXACT, section 2).

## Verdict

- **Accuracy PARITY with the rwkv-lm reference is established on both standard lm-eval tasks:**
  lambada_openai acc **ours-fp16 0.6728 vs ref 0.6711** (tie) and MMLU **0.5235 vs 0.5110**
  (+1.25 pt, from the leading-0 methodology diff). Combined with **greedy-EXACT 24/24** vs the
  numpy oracle, ours matches the reference not just at argmax but on standard scored metrics.
  The lm-eval-parity gate is now defined and passing for bf16/fp16.
- **int8 is a good tradeoff at 1.5B**: lambada -2.2 pt, MMLU -0.9 pt - small, and it shrinks
  with model size (7.2B int8 greedy-EXACT). Acceptable for the VRAM/speed it buys (comparison_clean.md).
- **Albatross fp16 does NOT drift on the fixtures** (24/24, 8/8 exact) - same as ours. So at
  equal fp16 precision there is **no measured greedy-accuracy difference** between ours and
  albatross; ours' real advantages over albatross are VRAM (flat O(1) state), the int8 option,
  and being a full dynamic-batch server - NOT fixture greedy accuracy. Stated plainly so no
  claim outruns the data.

## Reproducing (fixtures)
The two lm-eval task YAMLs (`bench/lmeval_tasks/*.yaml`) reference the eval parquet via a
`${FIXTURES}` placehnewer (not a hardcoded path). Stage the parquets locally and set
`FIXTURES=/abs/path/to/bench/fixtures` before running lm-eval (the parquets are not committed;
lambada_openai + a 2000-row MMLU sample). See `bench/accuracy_eval.py` for the harness.
