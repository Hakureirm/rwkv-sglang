"""Speculative decoding for RWKV-7, PRELIMINARY build (req#6 / ADR-0006 / F0029).

Zero-kernel-change design: a thin greedy chain-verify orchestrator over TWO
state-cached sglang servers (statecache mode: MambaRadixCache ON, cuda-graph
OFF — the F0022-verified pairing). The radix state cache is what makes this
cheap at the HTTP level:
  * draft proposes K tokens from the committed prefix — prefix state is a cache
    hit, so it only decodes K steps;
  * target verifies with ONE extend over committed+draft_K — again a prefix
    cache hit, so it computes exactly the K new positions (+1 sampled);
  * rejected suffix states simply stay in the radix tree until evicted:
    rollback == radix fork. No state snapshot/restore machinery at all.

Verify protocol (semantics pinned empirically — probe 2026-07-04):
  Under radix-ON, input_top_logprobs covers ONLY the actually-computed
  (uncached) suffix, logprob_start_len is effectively clamped to the prefix-
  cache boundary, and entry 0 of the computed region is always None — so the
  FIRST fresh position (draft[0]) is structurally unscorable from the verify
  call, and interior start values can 400. We therefore issue TWO independent
  target calls per round, in parallel:
    probe : {input_ids: committed, max_new_tokens: 1}  -> t* = target's greedy
            token at position len(committed) (prefix fully cached, ~1 step);
    verify: {input_ids: committed+draft, max_new_tokens: 1, return_logprob,
            top_logprobs_num: 1, logprob_start_len: len(committed)}
            -> itl[i] (i>=1) = target argmax for draft[i].
  Accept: draft[0] must equal t*; then the longest prefix with
  itl[i]==draft[i]. Commit draft[:J] + bonus, where bonus = t* when J==0 else
  itl[J] when J<K; when everything matched, commit all K (the next round's
  probe supplies the continuation). The verify call's sampled output token is
  never used (probe found same-input greedy output can differ with
  logprob_start_len — reported upstream separately). Every committed token is
  the target's greedy argmax given its committed prefix, by construction.

EXACTNESS: by construction every committed token is the target's greedy argmax
given the committed prefix — the gate below checks spec output == plain greedy
from the same target server, token for token.

Usage (servers up in statecache mode, see scripts/serve.sh MODE=statecache):
  python bench/spec_decode.py --draft-port 31041 --target-port 31042 \
      --k 4 --max-new 256 [--gate] [--baseline-port 31043] [--out results.json]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests

PROMPTS = [
    "User: What is the capital of France?\n\nAssistant:",
    "User: Explain why the sky is blue in one sentence.\n\nAssistant:",
    "User: Write a haiku about autumn.\n\nAssistant:",
    "User: Solve for x: 2x + 6 = 14.\n\nAssistant: <think></think",
    "User: List three prime numbers and briefly say why they are prime.\n\nAssistant:",
    "User: Translate 'good morning' to Spanish and French.\n\nAssistant:",
    "User: Who wrote Romeo and Juliet, and in which century?\n\nAssistant:",
    "User: What is 15 percent of 200? Show the steps.\n\nAssistant: <think></think",
]
EOD = 0  # RWKV world-tokenizer end-of-document token


def _post(sess, port, body):
    r = sess.post(f"http://127.0.0.1:{port}/generate", json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d[0] if isinstance(d, list) else d


def tokenize_via_server(sess, port, text):
    """Token ids of `text` per the server's tokenizer (no local tokenizer dep)."""
    d = _post(sess, port, {"text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                           "return_logprob": True, "logprob_start_len": 0})
    return [e[1] for e in d["meta_info"]["input_token_logprobs"]]


def plain_greedy(sess, port, prompt_ids, max_new):
    t0 = time.time()
    d = _post(sess, port, {"input_ids": prompt_ids,
                           "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new,
                                               "stop_token_ids": [EOD]}})
    dt = time.time() - t0
    out = d["output_ids"]
    return out, dt


def spec_greedy(sess, dport, tport, prompt_ids, max_new, K):
    """Greedy chain-verify loop. Returns (tokens, wall, stats)."""
    committed = list(prompt_ids)
    gen = []
    stats = {"rounds": 0, "draft_tokens": 0, "accepted": 0, "target_calls": 0}
    t0 = time.time()
    while len(gen) < max_new:
        k_eff = min(K, max_new - len(gen))
        # 1) draft proposes k_eff tokens from the committed prefix (cache-hit decode)
        d = _post(sess, dport, {"input_ids": committed,
                                "sampling_params": {"temperature": 0.0, "max_new_tokens": k_eff,
                                                    "stop_token_ids": [EOD]}})
        draft = d["output_ids"]
        stats["draft_tokens"] += len(draft)
        if not draft:  # draft immediately at EOD: fall back to one target step
            out, _ = plain_greedy(sess, tport, committed, 1)
            tok = out[0] if out else EOD
            gen.append(tok); committed.append(tok)
            stats["rounds"] += 1; stats["target_calls"] += 1
            if tok == EOD:
                break
            continue
        # 2) TWO independent target calls, in parallel:
        #    probe -> t* (target token at len(committed); fills the itl-None hole)
        #    verify -> itl scores for draft[1..]
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_probe = ex.submit(_post, sess, tport, {
                "input_ids": list(committed),
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
            f_verify = ex.submit(_post, sess, tport, {
                "input_ids": committed + draft,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                "return_logprob": True, "top_logprobs_num": 1,
                "logprob_start_len": len(committed)})
            pr, v = f_probe.result(), f_verify.result()
        t_star = pr["output_ids"][0] if pr.get("output_ids") else None
        itl = v["meta_info"]["input_top_logprobs"]
        preds = [e[0][1] if e else None for e in itl]
        stats["rounds"] += 1
        stats["target_calls"] += 2
        if t_star is None:
            break
        # Cache-shape-independent alignment: itl covers the computed SUFFIX of
        # the verify input; itl[i] predicts input position base+i, where
        # base = len(input) - len(itl). draft[j] sits at len(committed)+j, so
        # its score lives at idx = len(committed)+j - base (usable when >=1).
        base = len(committed) + len(draft) - len(itl)

        def target_says(j):
            """Target's greedy token for draft position j, or None if unscored."""
            if j == 0:
                return t_star  # the probe always covers position 0
            idx = len(committed) + j - base
            if 1 <= idx < len(preds):
                return preds[idx]
            return None

        J = 0
        bonus = None
        while J < len(draft):
            t = target_says(J)
            if t is None:
                break  # unscorable (cache clamp): stop chain, no bonus
            if t != draft[J]:
                bonus = t  # first mismatch: target's own token, safe to commit
                break
            J += 1
        stats["accepted"] += J
        new = draft[:J] + ([bonus] if bonus is not None else [])
        gen.extend(new)
        committed.extend(new)
        if new and new[-1] == EOD:
            break
    wall = time.time() - t0
    if len(gen) > max_new:  # trim overshoot for apples-to-apples with plain
        gen = gen[:max_new]
    return gen, wall, stats


def main():
    ap = argparse.ArgumentParser(description="RWKV-7 speculative decoding (preliminary)")
    ap.add_argument("--draft-port", type=int, required=True)
    ap.add_argument("--target-port", type=int, required=True)
    ap.add_argument("--baseline-port", type=int, default=None,
                    help="optional plain-throughput-mode target for the honest best-plain reference")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--gate", action="store_true",
                    help="assert spec output == plain greedy from the SAME target server")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sess = requests.Session()

    rows = []
    tot = {"spec_tok": 0, "spec_s": 0.0, "plain_tok": 0, "plain_s": 0.0,
           "base_tok": 0, "base_s": 0.0, "accepted": 0, "rounds": 0, "draft_tokens": 0}
    gate_pass = True
    for p in PROMPTS:
        ids = tokenize_via_server(sess, a.target_port, p)
        spec, s_wall, st = spec_greedy(sess, a.draft_port, a.target_port, ids, a.max_new, a.k)
        plain, p_wall = plain_greedy(sess, a.target_port, ids, a.max_new)
        identical = spec == plain
        gate_pass &= identical
        row = {"prompt": p[:40], "n_spec": len(spec), "n_plain": len(plain),
               "identical": identical, "spec_tok_s": len(spec) / s_wall if s_wall else 0,
               "plain_tok_s": len(plain) / p_wall if p_wall else 0,
               "alpha_round": st["accepted"] / max(1, st["draft_tokens"]),
               "tokens_per_target_pair": (len(spec) / max(1, st["rounds"]))}
        if a.baseline_port:
            base, b_wall = plain_greedy(sess, a.baseline_port, ids, a.max_new)
            row["baseline_tok_s"] = len(base) / b_wall if b_wall else 0
            tot["base_tok"] += len(base); tot["base_s"] += b_wall
        rows.append(row)
        tot["spec_tok"] += len(spec); tot["spec_s"] += s_wall
        tot["plain_tok"] += len(plain); tot["plain_s"] += p_wall
        tot["accepted"] += st["accepted"]; tot["rounds"] += st["rounds"]
        tot["draft_tokens"] += st["draft_tokens"]
        print(f"  [{row['prompt']:40s}] spec {row['spec_tok_s']:6.1f} tok/s | plain {row['plain_tok_s']:6.1f}"
              f" | acc/draft {row['alpha_round']:.3f} | tok/round {row['tokens_per_target_pair']:.2f}"
              f" | identical={identical}", flush=True)

    spec_tps = tot["spec_tok"] / tot["spec_s"] if tot["spec_s"] else 0
    plain_tps = tot["plain_tok"] / tot["plain_s"] if tot["plain_s"] else 0
    summary = {
        "k": a.k, "max_new": a.max_new,
        "spec_tok_s": round(spec_tps, 1), "plain_same_server_tok_s": round(plain_tps, 1),
        "speedup_vs_same_server_plain": round(spec_tps / plain_tps, 3) if plain_tps else None,
        "alpha_accept_per_draft": round(tot["accepted"] / max(1, tot["draft_tokens"]), 4),
        "tokens_per_round": round(tot["spec_tok"] / max(1, tot["rounds"]), 3),
        "gate": ("PASS (token-identical on all prompts)" if gate_pass else "FAIL") if a.gate else "not run",
        "rows": rows,
    }
    if a.baseline_port and tot["base_s"]:
        base_tps = tot["base_tok"] / tot["base_s"]
        summary["baseline_throughputmode_tok_s"] = round(base_tps, 1)
        summary["speedup_vs_throughputmode_plain"] = round(spec_tps / base_tps, 3)
    print("\n=== SPEC-DECODE SUMMARY ===")
    for k, v in summary.items():
        if k != "rows":
            print(f"  {k}: {v}")
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    if a.gate and not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
