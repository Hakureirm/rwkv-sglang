#!/usr/bin/env python3
"""
MLX-port greedy correctness gate against the repo's numpy-oracle fixtures.

Consumes `bench/fixtures/oracle_rwkv7_*_eiffel.json` exactly like
`bench/greedy_check.py` does: the fixture's `prompt_tokens` are fed directly
(they are the pinned ground-truth encoding — note the 0.1B fixture's
literal-backslash-n quirk documented in its `_comment`, which is why encoding
the prompt text is only a cross-check, never the gate input) and the 24
`greedy_tokens` must match token-by-token.

Gate = 24/24 EXACT or fail; no benchmark number may be published from a
configuration that has not passed here (both WKV paths are gated).

  python gate_oracle.py --model /tmp/mlx_models/rwkv7-0.1b-fla \
      --fixture ../bench/fixtures/oracle_rwkv7_01b_eiffel.json \
      [--dtype bfloat16] [--wkv pure,metal] [--tag 01B]

Prints machine-readable markers: GATE_<TAG>_{PASS,FAIL} per WKV mode, plus a
STEP_PREFILL_CROSSCHECK line (vectorized chunked prefill vs oracle-style
token-by-token prompt feed must produce the same 24 tokens).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_mlx import load_model


class WorldTokenizer:
    """Standalone RWKV World tokenizer (rwkv_vocab_v20230424.txt), no
    transformers dependency. Same greedy longest-match the shipped
    hf_rwkv_tokenizer.py trie performs; used only for text display and an
    informational encode cross-check (the gate feeds fixture prompt_tokens)."""

    def __init__(self, vocab_path):
        self.idx2b, self.b2idx = {}, {}
        for line in open(vocab_path, encoding="utf-8").read().splitlines():
            i = int(line[: line.index(" ")])
            tok = eval(line[line.index(" "): line.rindex(" ")])
            tok = tok.encode("utf-8") if isinstance(tok, str) else tok
            assert isinstance(tok, bytes)
            assert len(tok) == int(line[line.rindex(" "):])
            self.idx2b[i] = tok
            self.b2idx[tok] = i
        self.max_len = max(len(b) for b in self.b2idx)

    def encode(self, s):
        src, out, i = s.encode("utf-8"), [], 0
        while i < len(src):
            for ln in range(min(self.max_len, len(src) - i), 0, -1):
                tok = self.b2idx.get(src[i: i + ln])
                if tok is not None:
                    out.append(tok)
                    i += ln
                    break
        return out

    def decode(self, ids):
        return b"".join(self.idx2b[i] for i in ids).decode(
            "utf-8", errors="replace"
        )


def run_gate(model_dir, fixture_path, dtype, wkv, tag, cross_check=True):
    fx = json.load(open(fixture_path))
    prompt_tokens = fx["prompt_tokens"]
    expected = fx["greedy_tokens"]
    n = len(expected)

    tok = None
    vocab = os.path.join(model_dir, "rwkv_vocab_v20230424.txt")
    if os.path.exists(vocab):
        tok = WorldTokenizer(vocab)
        enc = tok.encode(fx["prompt"])
        note = "MATCH" if enc == prompt_tokens else (
            "DIFFERS (known fixture quirk; gate uses fixture prompt_tokens)"
        )
        print(f"[{tag}] tokenizer encode(prompt) vs fixture prompt_tokens: {note}")

    print(f"[{tag}] loading {model_dir} dtype={dtype} wkv={wkv}")
    model = load_model(model_dir, dtype=dtype, wkv=wkv)
    print(f"[{tag}] n_layer={model.n_layer} n_embd={model.n_embd} "
          f"n_head={model.n_head} head_dim={model.head_dim}")
    assert [model.n_layer, model.n_embd, model.n_head, model.head_dim] == [
        fx["arch"]["n_layer"], fx["arch"]["n_embd"],
        fx["arch"]["n_head"], fx["arch"]["head_size"],
    ], "checkpoint arch does not match fixture arch"

    got, _ = model.generate(prompt_tokens, n)
    n_match = sum(1 for i in range(n) if got[i] == expected[i])
    div = next((i for i in range(n) if got[i] != expected[i]), None)
    ok = got == expected
    print(f"GATE_{tag}_{'PASS' if ok else 'FAIL'} {n_match}/{n} "
          f"first_div={div} (dtype={dtype} wkv={wkv})")
    print("GOT ", got)
    print("EXP ", expected)
    if tok is not None:
        print(f"GOT text: {tok.decode(got)!r}")

    if cross_check:
        # Same 24 tokens must come out when the prompt is fed token-by-token
        # through the compiled decode step (oracle-style) instead of the
        # vectorized chunked prefill — gates both prompt paths in one run.
        got2, _ = model.generate(prompt_tokens, n, prefill_mode="step")
        cc = got2 == got
        print(f"STEP_PREFILL_CROSSCHECK_{tag}_"
              f"{'PASS' if cc else 'FAIL'} (step-fed prompt, wkv={wkv})")
        if not cc:
            print("STEP GOT ", got2)
        ok = ok and cc
    return ok


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fixtures = os.path.join(here, "..", "bench", "fixtures")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model dir (default: both)")
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--wkv", default="pure,metal",
                    help="comma list of WKV paths to gate")
    ap.add_argument("--models-root", default="/tmp/mlx_models")
    args = ap.parse_args()

    if args.model:
        if not args.fixture:
            ap.error("--model requires --fixture")
        jobs = [(args.model, args.fixture,
                 args.tag or os.path.basename(args.fixture))]
    else:
        jobs = [
            (os.path.join(args.models_root, "rwkv7-0.1b-fla"),
             os.path.join(fixtures, "oracle_rwkv7_01b_eiffel.json"), "01B"),
            (os.path.join(args.models_root, "rwkv7-1.5b-fla"),
             os.path.join(fixtures, "oracle_rwkv7_15b_eiffel.json"), "15B"),
        ]
        # 7.2B is optional (14 GB): gated only when the weights are present.
        _m72 = os.path.join(args.models_root, "rwkv7-7.2b-fla")
        if os.path.isdir(_m72):
            jobs.append(
                (_m72, os.path.join(fixtures, "oracle_rwkv7_72b_eiffel.json"), "72B")
            )

    all_ok = True
    for model_dir, fixture, tag in jobs:
        for wkv in args.wkv.split(","):
            all_ok &= run_gate(model_dir, fixture, args.dtype, wkv, tag)
    print("GATE_ALL_PASS" if all_ok else "GATE_ALL_FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
