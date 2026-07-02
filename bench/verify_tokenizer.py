#!/usr/bin/env python3
"""
Verify the RWKV World tokenizer wired into the model dir (fla-hub
`hf_rwkv_tokenizer.py` + `rwkv_vocab_v20230424.txt`) produces the SAME token ids
as the BlinkDL `rwkv` pip PIPELINE tokenizer. This is the gate that lets sglang
tokenize text for lm-eval and produce reference-comparable inputs.

Two envs, one comparison:
  # reference ids from the rwkv pip PIPELINE (oracle env has `rwkv`)
  ~/envs/rwkv-ref/bin/python bench/verify_tokenizer.py --mode ref \
      --vocab /home/user/rwkv_models/rwkv7-1.5b-fla/rwkv_vocab_v20230424.txt \
      --out /tmp/tok_ref.json
  # HF ids from AutoTokenizer(model_dir, trust_remote_code=True) (sglang env has transformers)
  ~/envs/rwkv-sgl/bin/python bench/verify_tokenizer.py --mode hf \
      --model /home/user/rwkv_models/rwkv7-1.5b-fla --out /tmp/tok_hf.json
  # assert equality
  python bench/verify_tokenizer.py --mode compare --a /tmp/tok_ref.json --b /tmp/tok_hf.json
"""
import argparse
import json

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "User: You are a very talented expert in abstract algebra. Answer this question:\n"
    "What is the order of the element 2 in Z/7Z under addition?\n"
    "A. 3\nB. 7\nC. 2\nD. 14\n\nAssistant: The answer is",
    "In 2026, RWKV-7 (Goose) is a linear-attention RNN language model.",
    "Hello, 世界! 这是一个 RWKV tokenizer 测试 with mixed 中英文 and emoji 🚀.",
    "\n\n",
    "1234567890 !@#$%^&*() the end.",
]


def mode_ref(args):
    # rwkv pip PIPELINE tokenizer over rwkv_vocab_v20230424
    try:
        from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
        tok = TRIE_TOKENIZER(args.vocab)
        enc = lambda s: tok.encode(s)
    except Exception:
        # fall back to the PIPELINE path (model can be None; only tokenizer is used)
        from rwkv.utils import PIPELINE
        pipe = PIPELINE(None, "rwkv_vocab_v20230424")
        enc = lambda s: pipe.encode(s)
    out = {p: enc(p) for p in PROMPTS}
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)
    for p, ids in out.items():
        print(repr(p[:40]), "->", ids[:12], "..." if len(ids) > 12 else "", f"(len {len(ids)})")


def mode_hf(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # add_bos_token=false in tokenizer_config -> encode adds no special tokens,
    # matching the raw rwkv pip PIPELINE.encode.
    out = {p: tok.encode(p) for p in PROMPTS}
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)
    for p, ids in out.items():
        print(repr(p[:40]), "->", ids[:12], "..." if len(ids) > 12 else "", f"(len {len(ids)})")


def mode_compare(args):
    a = json.load(open(args.a))
    b = json.load(open(args.b))
    ok = True
    for p in a:
        if a[p] != b.get(p):
            ok = False
            print("MISMATCH for", repr(p[:50]))
            print("  ref:", a[p])
            print("  hf :", b.get(p))
    if ok:
        print("ALL %d PROMPTS MATCH: HF tokenizer == rwkv pip PIPELINE (token-for-token)"
              % len(a))
    else:
        raise SystemExit("tokenizer MISMATCH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("ref", "hf", "compare"))
    ap.add_argument("--vocab", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--a", default="")
    ap.add_argument("--b", default="")
    args = ap.parse_args()
    {"ref": mode_ref, "hf": mode_hf, "compare": mode_compare}[args.mode](args)


if __name__ == "__main__":
    main()
