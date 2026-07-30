---
doc_kind: finding
finding_id: F0071
title: "OUR OWN HuggingFace RWKV-7 port (transformers-rwkv PR#2, written by this project) scaled the ln_x GroupNorm epsilon by num_heads where the reference hardcodes 64e-5 = head_dim x 1e-5. Those agree at exactly one width — hidden_size 4096 — which is the 7.2B, which is the checkpoint every accuracy measurement of that port had used. Scored here against bench/oracle_numpy.py at the two non-square widths it was never scored at: with head_dim the port is 32/32 token-identical to the oracle at both 0.1B and 1.5B (logits within 5e-05); with num_heads it is 11/32 and 30/32, and the text it produces is different and wrong. Not a precision wobble — trained per-head variances are ~500x below the epsilon, so the constant is the whole denominator and a wrong one rescales the normalisation by up to 2.31x."
status: CLOSED (2026-07-29) — fixed in transformers-rwkv PR#2 (commit 6a56130), verified against the oracle at 0.1B and 1.5B
discovered_by: Opus 5 (1M), 2026-07-29, while clearing a flagged-but-unverified audit item
severity: P0 (wrong output on every width except 7.2B)
related: [F0070]
machine: 5090 tower, one-shot lmsysorg/sglang:dev-cu12 containers, fp32 on both sides
---

# Finding F0071: the epsilon that was only right on the model we measured

## 0. Whose bug this is

Ours. The port in question is `transformers-rwkv` PR#2, written by this project two days
before this finding; the bug went in with its first commit and came out a day later in
`6a56130`, before anyone outside had read it. This document originally led with "the
HuggingFace RWKV-7 port", which reads as an audit of somebody else's code, and that
framing is corrected here rather than left to a reader to work out from the status line.

The correct reading was also never obscure. BlinkDL's reference carries
`nn.GroupNorm(H, C, eps=64e-5)` with `# !!! notice eps value !!!` beside it, fla has
shipped `eps=self.head_dim * norm_eps` for months, and two implementations in this very
repository already had it right. Getting it wrong was not subtle and finding it is not a
discovery.

## 0b. Why it is still worth writing up

The bug itself is one multiplication. What makes it worth a finding is the shape of
the miss: **the port was measured, the measurement was sound, and the measurement was
blind to this by construction.** F0070 scored the port on Bo's own ruler precisely
because self-comparison could not catch a self-consistently wrong model. It ran on the
7.2B. This bug is invisible on the 7.2B and only on the 7.2B.

## 1. Hypothesis

The RWKV-7 reference hardcodes `nn.GroupNorm(H, C, eps=64e-5)` for `ln_x`. The port
wrote that constant as `norm_eps * num_heads`. The alternative reading is
`norm_eps * head_dim`. At the reference's head_dim of 64 and `norm_eps` of 1e-5, the
second reading reproduces 64e-5 exactly; the first only does so when `num_heads` also
happens to be 64.

`num_heads = hidden_size / head_dim`, so with head_dim fixed at 64 the two readings
coincide at `hidden_size = 4096` and nowhere else:

| model | hidden | num_heads | port's eps | reference eps | ratio |
|---|---|---|---|---|---|
| 0.1B | 768 | 12 | 1.2e-4 | 6.4e-4 | 5.33x |
| 0.4B | 1024 | 16 | 1.6e-4 | 6.4e-4 | 4.00x |
| 1.5B | 2048 | 32 | 3.2e-4 | 6.4e-4 | 2.00x |
| 2.9B | 2560 | 40 | 4.0e-4 | 6.4e-4 | 1.60x |
| **7.2B** | **4096** | **64** | **6.4e-4** | **6.4e-4** | **1.00x** |

Two independent implementations in this repository already resolve the ambiguity in
favour of head_dim, and both were validated against the numpy oracle:
`sglang_overlay/.../models/rwkv7.py` builds `nn.GroupNorm(..., eps=self.head_dim *
config.norm_eps)`, and `mlx_port/rwkv7_mlx.py` documents `eps = head_dim * norm_eps
(64e-5 for hd=64)`. So the fix direction was never in doubt; what was unmeasured was
the size of the error and whether anything else in the port was width-dependent.

## 2. Method

`bench/oracle_numpy.py` is the fp32 numpy reference the sglang backend and the MLX
port were themselves validated against, and it hardcodes the reference's `64e-5`
rather than deriving it — so it is a genuine third party to this question.

For each of the two non-square widths:

1. Load the native `.pth` with the oracle's loader; load the converted HuggingFace
   copy with the port, both fp32.
2. **Verify the two files are the same model** rather than assuming it — assert
   `emb.weight` is bit-identical between them. A comparison against a checkpoint that
   merely resembles the other proves nothing.
3. Prompt `"\nThe Eiffel Tower is located in the city of"`, 32 greedy tokens.
4. Run the port twice, once with each candidate epsilon assigned explicitly to every
   `ln_x`, so the result says which reading the oracle agrees with instead of assuming
   the answer, and so it does not depend on which revision of the port is installed.
5. Report token agreement, first divergence, and max abs logit delta on the
   prompt-final logits — the last being independent of whether the greedy sequence
   happens to be degenerate.

Step 5's last clause is not decoration. The first run of this used arbitrary token ids
as the prompt; the oracle's greedy output was the repeating cycle
`[97, 12, 7, 97, 12, 7, ...]`, on which "32/32" is nearly free. Real text was
substituted before any of the numbers below were recorded.

## 3. Result

Both widths, oracle-vs-port, `emb_bit_identical_to_pth: true` on both:

| width | heads / head_dim | eps | tokens vs oracle | first divergence | max abs logit delta |
|---|---|---|---|---|---|
| 0.1B (768) | 12 / 64 | **6.4e-4 (head_dim)** | **32/32** | — | **5e-05** |
| 0.1B (768) | 12 / 64 | 1.2e-4 (num_heads) | 11/32 | token 9 | 4.318 |
| 1.5B (2048) | 32 / 64 | **6.4e-4 (head_dim)** | **32/32** | — | **3e-05** |
| 1.5B (2048) | 32 / 64 | 3.2e-4 (num_heads) | 30/32 | token 14 | 0.527 |

What the two arms actually say:

```
oracle / fixed, 0.1B:   " Paris, France. It is a tower that was built in 1889 and is ..."
old eps,        0.1B:   " Paris, France. It is a tower that stands on the site of the"

oracle / fixed, 1.5B:   " Paris, France.\nThe Eiffel Tower is a symbol of Paris and France."
old eps,        1.5B:   " Paris, France.\nThe Eiffel Tower is a symbol of France and Paris."
```

The severity tracks the ratio column of §1 — 5.33x at 0.1B gives 11/32 and a 4.3 logit
delta, 2.00x at 1.5B gives 30/32 and 0.53. That the damage scales with the ratio the
mechanism predicts is the internal consistency check on this whole finding.

The 1.5B row is the more instructive one. 30/32 with a swapped word order reads like
sampling noise. Nothing about it would make a reader suspect the normalisation
constant, and no eyeball comparison would have caught it.

### Why an epsilon mattered this much

An epsilon is normally negligible: `sqrt(var + eps)` is `sqrt(var)` whenever the
variance is large. On trained weights the variance going into `ln_x` is not large.
Measured at 0.1B layer 0: **median per-head variance 1.345e-06**, minimum 4.196e-12,
**84.4% of heads below the epsilon itself**. So `sqrt(var + eps)` is effectively
`sqrt(eps)`, the constant *is* the denominator, and a wrong one rescales the entire
normalisation by `sqrt(64/12) = 2.31x` rather than perturbing it.

This also explains the reference's choice of 64e-5, which is 64x a conventional 1e-5
and looks arbitrary until you see the variances it sits next to. It is not a numerical
guard; it is load-bearing.

Secondary measurement, same 0.1B in fp16 over a 512-token forward, the two epsilons
against each other rather than against the oracle: max abs logit delta 2.797 on a
magnitude of 48.03 (5.8% relative), mean 0.154, **42 of 512 positions change their
argmax**, greedy agreement 18.75% over 64 tokens.

## 4. Conclusion

Fixed in `transformers-rwkv` PR#2 as commit `6a56130`, written as
`config.norm_eps * config.head_dim` — the axis GroupNorm actually reduces over, so the
constant now tracks the reference at any width instead of at one. The regression test
asserts the constant directly rather than a behaviour, because the way this breaks
again is somebody tidying it back to `num_heads`; its config is deliberately
non-square and asserts that too, since a square one would make the assertion vacuous.

Three things this leaves on the record.

**A measurement is only as general as the configuration it ran on.** F0070 was built
specifically to escape self-comparison, and it did — it just did so at one width, and
the width it picked was the one that hides this. "Validated against the reference" was
true and still permitted a P0. The 0.1B conversion check did run at a non-square width
but compared two conversion routes into the same model, so both sides carried the same
epsilon and it cancelled.

**The fix's correctness never needed the experiment; its severity did.** The oracle
hardcodes 64e-5 and `head_dim * norm_eps` reproduces it identically at every RWKV-7
width, so the direction was settled by inspection. What was worth a GPU was the size —
the difference between "a rounding artefact nobody would notice" and "the model says
something else", and it turned out to be the second.

**The port is now oracle-exact at three widths, not one.** 0.1B and 1.5B token-identical
here, 7.2B by compression agreement in F0070. Any future claim about this port should
name the width it was measured at.

## 5. Cross-references

- F0070 — the accuracy work this closes a hole in; its 7.2B choice is the reason this
  survived.
- `bench/results/f0071/eps_oracle_0.1b.txt` — raw output for both widths.
- `transformers-rwkv` PR#2 commit `6a56130` — the fix and its test.
- `sglang_overlay/sglang/srt/models/rwkv7.py`, `mlx_port/rwkv7_mlx.py` — the two
  in-repo implementations that already had it right.
