"""Speculative-decoding VIABILITY measurement for RWKV-7 (req#6 / ADR-0006 step 1).

Measures the per-token acceptance rate α: how often the 0.1B draft's greedy argmax
matches the target's greedy token GIVEN the same prefix — the number that sets the
spec-decode speedup ceiling. Cheap: /generate return_logprob top-1, no rollback yet.

Two phases (two sglang servers can't cleanly share one GPU, so run sequentially):
  --mode target --dump D.json : target greedy → (prompt_ids, T) per prompt, saved to D.
  --mode draft  --dump D.json : feed draft (prompt+T), logprob_start_len=len(prompt),
      input_top_logprobs[j][0][1] = draft argmax predicting T[j]; α = mean(argmax==T[j]).

Confirmed API (sglang logits_processor prunes rows [S..L-1], output processor prepends
None and pops the sampled position): with logprob_start_len=S, input_top_logprobs[i] is
the model's top-k FOR full-position S+i given prefix[0..S+i-1]; entry 0 is None, so the
first target token per prompt is unscored. token_id at index 1.

Usage:
  python bench/spec_accept.py --mode target --port 30070 --dump /tmp/spec.json --gen-len 128
  python bench/spec_accept.py --mode draft  --port 30071 --dump /tmp/spec.json
"""
import argparse, json, statistics
import requests

PROMPTS = [
    "User: What is the capital of France?\n\nAssistant:",
    "User: Write a haiku about autumn.\n\nAssistant:",
    "User: Explain why the sky is blue in one sentence.\n\nAssistant:",
    "User: Solve for x: 2x + 6 = 14.\n\nAssistant: <think></think",
    "User: List three prime numbers.\n\nAssistant:",
    "User: Translate 'good morning' to Spanish.\n\nAssistant:",
    "User: Who wrote Romeo and Juliet?\n\nAssistant:",
    "User: What is 15 percent of 200?\n\nAssistant: <think></think",
]


def _post(sess, url, body):
    r = sess.post(url, json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d[0] if isinstance(d, list) else d


def phase_target(sess, url, gen_len, dump):
    rows = []
    for p in PROMPTS:
        gen = _post(sess, url, {"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": gen_len}})
        T = gen["output_ids"]
        lp = _post(sess, url, {"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                               "return_logprob": True, "logprob_start_len": 0})
        prompt_ids = [e[1] for e in lp["meta_info"]["input_token_logprobs"]]
        rows.append({"prompt_ids": prompt_ids, "T": T})
        print(f"  target: |prompt|={len(prompt_ids)} |T|={len(T)}", flush=True)
    json.dump(rows, open(dump, "w"))
    print(f"dumped {len(rows)} rows -> {dump}")


def phase_draft(sess, url, dump):
    rows = json.load(open(dump))
    accepts = []
    for row in rows:
        prompt_ids, T = row["prompt_ids"], row["T"]
        if not T:
            continue
        full = prompt_ids + T
        # input_top_logprobs[i] = model's argmax FOR position i (given prefix[0..i-1]).
        # T[j] is at full-position len(prompt_ids)+j, so start at len(prompt_ids):
        # returned itl[j] = prediction of T[j] given prompt + T[0..j-1].
        start = len(prompt_ids)
        d = _post(sess, url, {"input_ids": full, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                              "return_logprob": True, "top_logprobs_num": 1, "logprob_start_len": start})
        itl = d["meta_info"]["input_top_logprobs"]
        pred = [e[0][1] if e else None for e in itl]  # draft argmax for T[j]
        for j in range(len(T)):
            if j < len(pred) and pred[j] is not None:
                accepts.append(1 if pred[j] == T[j] else 0)
        print(f"  draft: |T|={len(T)} running α={statistics.mean(accepts):.3f} (n={len(accepts)})", flush=True)
    alpha = statistics.mean(accepts) if accepts else 0.0
    print(f"\n=== per-token acceptance α = {alpha:.4f}  (n={len(accepts)} tokens) ===")
    for K in (2, 4, 8):
        tpf = sum(alpha ** i for i in range(K + 1))  # 1 + α + ... + α^K
        print(f"  block K={K}: ~{tpf:.2f} target-tokens / target-forward")
    print("\nNet speedup ≈ tpf / (1 + K·draft_frac); draft(0.1B) ~1/15 of 1.5B, ~1/70 of 7.2B.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("target", "draft"), required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gen-len", type=int, default=128)
    a = ap.parse_args()
    sess = requests.Session()
    url = f"http://{a.host}:{a.port}/generate"
    if a.mode == "target":
        phase_target(sess, url, a.gen_len, a.dump)
    else:
        phase_draft(sess, url, a.dump)


if __name__ == "__main__":
    main()
