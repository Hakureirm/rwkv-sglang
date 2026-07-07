"""Live-serving-path probe: get top-10 next-token logprobs for the same probe
text from an actually-running Qwen3.5-2B backend, via mlx-lm on Apple Silicon.

Deliberately uses `transformers.AutoTokenizer` directly (the same call
`numpy_reference.py` makes) rather than mlx-lm's own tokenizer wrapper, so any
difference found downstream is isolated to model math (numpy fp32 einsum vs
mlx-lm's bf16/fp32 Metal kernels), not to a tokenization mismatch between the
two harnesses.

Uses mlx_lm's public `generate_step` generator with `max_tokens=1`: the first
yielded `(token, logprobs)` pair is exactly "next-token log-distribution after
consuming the full prompt" -- the same semantic position
`numpy_reference.Qwen35.forward()` returns logits for. `logprobs` there is
`log_softmax(logits)` (see mlx_lm.generate.generate_step source), which
preserves top-k ranking and gives probabilities directly comparable to the
numpy reference's own `prob` column, without needing raw-logit scale/offset
to line up bit-for-bit across two different math backends.
"""

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.utils import load
from transformers import AutoTokenizer


def top_logprobs(logprobs, k=10):
    idx = mx.argpartition(-logprobs, k)[:k]
    idx = idx[mx.argsort(-logprobs[idx])]
    idx = [int(i) for i in idx.tolist()]
    lp = logprobs[mx.array(idx)].tolist()
    return [(i, float(v), float(mx.exp(mx.array(v)).item())) for i, v in zip(idx, lp)]


def report(model_dir, probe_text=" Eiffel"):
    print(f"loading mlx-lm model: {model_dir}")
    model, _mlx_tokenizer = load(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    probe_tokens = tokenizer.encode(probe_text, add_special_tokens=False)
    prompt = mx.array(probe_tokens)

    token, logprobs = next(generate_step(prompt, model, max_tokens=1))
    mx.eval(logprobs)
    rows = top_logprobs(logprobs)

    print(f"\n== Qwen3.5 (mlx-lm live, bf16 native checkpoint) top-10 logprobs ==")
    print(f"text: {probe_text!r}")
    print(f"tokens: {probe_tokens}")
    for rank, (tid, logprob, prob) in enumerate(rows, 1):
        print(f"{rank}: token={tid} logprob={logprob:.6f} prob={prob:.8f} text={tokenizer.decode([tid])!r}")
    return probe_tokens, rows


if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/qwen35_mlx_test/Qwen3.5-2B"
    report(model_dir)
