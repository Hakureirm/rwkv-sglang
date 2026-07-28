---
doc_kind: finding
finding_id: F0070
title: "Scoring the HuggingFace RWKV-7 port on Bo's own ruler instead of against itself. Every correctness check the port carried compared it to ITSELF — prefill against token-by-token decoding, the sparse projection against the dense one — and a self-consistently wrong model passes all of them. Measured on UncheatableEval compression over the same 300 documents in one session, this port and the sglang reference agree on 14 of 15 corpora to four decimals and by 1e-4 on the last, mean bpb 0.5281 either way. Two by-products: most of those documents exceed one 4000-token chunk, so the chunk-parallel prefill got a long-context equivalence check on real text; and `bench/uncheatable_eval.py --launch` turns out to be a dead path on current sglang (device-side assert in `logits_processor._get_pruned_states`, server exits -9) that nobody had hit because everyone starts the server by hand."
status: OPEN (2026-07-28) — compression and greedy-generation agreement both closed (bpb identical on 14/15 corpora, 36/40 generations token-identical with the four exceptions measured as fp16-step near-ties); ShareGPT not started
discovered_by: Opus 5 (1M), 2026-07-28
severity: info
related: [F0069, F0050, F0057]
machine: 5090 tower, one-shot containers; HF port scored in-process, sglang reference served through scripts/serve.sh
---

# Finding F0070: what the port's correctness tests could not prove

## 0. The gap

The transformers RWKV-7 port carried a reasonable-looking correctness suite: prefill
must equal token-by-token decoding, the chunked recurrence must reproduce the
sequential one, batch rows must not influence each other, the sparse projection must
agree with the dense one, two conversion routes must produce the same logits.

Every one of those compares the port **to itself**. They catch an implementation that
contradicts itself, which is most implementation bugs — but a model that is
self-consistently wrong passes all of them, and so would a port that had, say, the
decay constant slightly off in a way both paths shared. The request this work answers
asked for inference that is *precision-aligned*, and none of the above measures
alignment with anything external.

The project already has the right instrument and was not pointing it at the port:
**UncheatableEval compression** — mean bits per byte on corpora that post-date the
training data — is the ruler this repo treats as decisive, and it is the RWKV
project's own.

## 1. Protocol

Reproduced from `bench/uncheatable_eval.py` rather than reinvented, because a number
measured under a different protocol is not comparable to one measured under this one:

* every chunk is prepended with token 0, the world tokenizer's BOS/EOD, which is what
  resets the recurrent state in the reference
* documents are split at `ctx_len` 4000 and each chunk starts from a fresh state
* logprobs come from a float32 `log_softmax`
* per corpus, `bpb = (mean per-document NLL in nats / mean utf-8 document bytes) / ln 2`

The same 300 documents (15 corpora × the first 20) went through both implementations
in one session, on the same weights. A subset, chosen so both sides could run it
back-to-back; it is therefore **not** comparable to the published full-corpus figure
and is not offered as one. What it establishes is column-to-column agreement.

## 2. Result

| corpus | HF port | sglang reference |
|---|---:|---:|
| ao3_english | 0.7629 | 0.7629 |
| ao3_nonenglish | 1.0286 | 1.0286 |
| arxiv_cs | 0.5293 | 0.5293 |
| arxiv_math | 0.5094 | 0.5094 |
| arxiv_other | 0.4861 | 0.4861 |
| arxiv_physics | 0.5551 | 0.5551 |
| bbc_news | 0.6308 | 0.6308 |
| biorxiv_all | 0.5547 | 0.5547 |
| github_cpp | 0.2847 | 0.2847 |
| github_javascript | 0.2659 | 0.2659 |
| github_markdown | 0.5395 | 0.5395 |
| **github_other** | **0.3009** | **0.3008** |
| github_python | 0.2823 | 0.2823 |
| wikipedia_english | 0.6129 | 0.6129 |
| wikipedia_nonenglish | 0.5780 | 0.5780 |
| **mean** | **0.5281** | **0.5281** |

Fourteen of fifteen agree to four decimals, the last by 1e-4, max |diff| across all
fifteen = 0.0001. Raw: `bench/results/f0070/`.

A weak sanity check worth stating because it was checked before the comparison was
trusted: the per-corpus ordering is the one text compressibility predicts — JavaScript
and Python lowest at 0.27, non-English fiction highest at 1.03. A port with a broken
tokenizer or a misaligned logprob index would not produce that ordering by accident,
so an implausible spread would have sent this back to debugging rather than forward
to a comparison.

## 3. The by-product that is worth as much as the headline

Most of these documents are longer than 4000 tokens, so scoring them exercises the
chunk-parallel prefill **across chunk boundaries, on real text, 300 times**. The
port's own test suite checks that at chunk sizes 1/4/16/64 on a synthetic sequence of
a length none of them divides — a good test, but one built from the same
understanding that produced the code. Bo's corpora were not.

The agreement therefore also says the chunk carry is right, which the port could
previously only assert against itself.

## 3b. Long-run generation: 36/40 token-identical, and the four exceptions explained

Compression scores what the model assigns to text that already exists; it never feeds
the model its own output. Greedy generation does, 1500 steps deep, so any difference
in the state carry, the token shift or the `v_first` hand-off compounds instead of
averaging out. The same 40 MATH500 problems were decoded greedily by both
implementations, same prompt construction (`[0] + "User: ...\n\nAssistant:
<think></think"`), same 1500-token budget:

* **36 of 40 are token-identical** over the whole overlapping length
* 60,000 tokens compared

Four diverged, and rather than reporting that as a rate, the top-2 gap at each
divergence was measured by replaying the agreed prefix:

| problem | step | top-1 p | top-2 p | logit gap |
|---|---:|---:|---:|---:|
| 21 | 2 | 0.2515 | 0.2502 | 0.0049 |
| 26 | 47 | 0.3968 | 0.3959 | 0.0022 |
| 22 | 159 | 0.3020 | 0.3006 | 0.0049 |
| 17 | 155 | 0.4449 | 0.4415 | 0.0078 |

Every one is a near-tie: the two candidates are within 0.0035 of probability, and the
gaps are 0.002197 / 0.004883 / 0.007812 — integer multiples of 2^-9, i.e. sitting on
fp16's quantisation step at that magnitude. This is not two models disagreeing; it is
one model at a point where the coin is standing on its edge, and reduction order
decides which way it falls. Two of the four re-converge to identical tokens
immediately afterwards, which is what a near-synonym substitution looks like.

Worth noting which way: the probe ran on the HF side, and at all four points *its own*
top-1 is the token the reference emitted. Both implementations agree on who should
win; they differ only in the path that accumulated to that step.

**A defect in the harness, not the port.** 33 of the 40 comparisons initially looked
like divergences at exactly the reference's stop position. They were not: the
reference honours EOS and the ad-hoc HF generation loop only checked for the
`"\nUser:"` stop string, so it ran on to the 1500 cap every time. Anyone reusing that
script would measure a false divergence rate; the fix is to stop on the EOS id as
well.

## 3c. The third shape, and the bug only it could find

Compression and greedy generation both run one sequence at a time. Batching is a third
shape, and the RWKV-7 port had just gained two routes into it — `attention_mask` for a
padded batch and `cu_seq_lens` for a packed one, both added the same day. Neither had
been exercised on anything but synthetic sequences of length 4.

ShareGPT is normally a serving benchmark here (`sglang.bench_serving` against a live
server), which does not apply to a model class with no scheduler; the number would be
incomparable and uninformative. Its length *distribution* does apply: 48 real prompts
from 5 to 1024 tokens, batched four at a time, each route checked against the same
prompt run alone.

The first run returned **NaN for every row with any padding** — every batched
`generate` call an fp16 user would make. Blanking a pad position makes `k` exactly
zero, so `kk = normalize(k * k_k)` normalises a zero vector; `F.normalize` divides by
`max(norm, 1e-12)`, and 1e-12 is below the smallest fp16 subnormal, so in fp16 the
divisor is a true zero.

The port's own padding test did not catch it because it builds a tiny model in **fp32**,
where 1e-12 is representable and the identical code path is fine. Three failures of the
same shape landed in one session: a test suite that skips the sparse path on CPU, a
scratch tensor shadowing the method that used it (invisible without a GPU), and this.
The common cause is that the cheap test environment differs from the deployed one in
exactly the dimension the bug lives in.

After the fix (`torch.where`, not a multiply — `NaN * 0` is still NaN), same prompts,
pad fractions up to 97%:

| route | prediction agreement vs run-alone | max abs logit diff |
|---|---:|---:|
| padded + `attention_mask` | 48/48 | 0.031 |
| packed + `cu_seq_lens` | 48/48 | 0.062 |

one to two fp16 ULP at this model's logit magnitude (±49).

## 4. `bench/uncheatable_eval.py --launch` is a dead path

Trying to run the reference side the documented way fails:

```
torch.AcceleratorError: CUDA error: device-side assert triggered
  logits_processor.py:461 in _get_pruned_states
    last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
RuntimeError: server exited early with code -9
```

The harness's own `launch_server` starts sglang with only `--dtype`,
`--trust-remote-code`, `--disable-radix-cache` and `--mem-fraction-static`. The RWKV-7
overlay needs more than that, and `scripts/serve.sh` supplies it — starting the server
that way and pointing the harness at it with `--host/--port` works on the first
attempt. Nobody had hit this because in practice the server is always started by hand.

Two smaller things in the same area, recorded so the next person does not spend the
runs: the harness's grader imports `math_verify`, which is not in the serving image
(and is not needed if what you want is generations rather than scores); and its
sampling flag is `--samples`, not `--num-samples`.

## 5. Method note

The instrument was checked before its output was believed, twice, and both checks
were worth the minute they cost. The first version of the wait condition for the
reference run grepped for `bpb`, and the run's own banner line contained
`===SGLANG_BPB===`, so it fired immediately and reported an empty result as if the
job had finished. The second waited on container exit instead. This is the same class
of error as the vacuous gates in [[F0069]] — the assertion was fine and the subject
was absent — and it is now three sessions running that it has cost real time. The
cheap habit that catches it: before believing a green, ask what this check would have
printed if the thing it watches had crashed.
