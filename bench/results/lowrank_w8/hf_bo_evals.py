# Bo's two decreed metrics, three arms, one protocol: uncheatable-eval (fresh-corpus
# compression, methodology replicated from bench/uncheatable_eval.py which itself is
# a line-audited port of Jellyfish042/uncheatable_eval) and MATH500 greedy
# completions (prompt convention of bench/math500_avg64.py). Arms and the int8
# round-trip come from lambada_lora_w8.py -- one quantizer, three evals.
import json
import math
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, "/data")
from lambada_lora_w8 import apply_arm  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TOK_DIR = "/data/_tfrwkv_ckpt"
MODEL_DIR = os.environ.get("MODEL_DIR", "/data/hubout/rwkv7-1.5b-hf")
TAG = os.environ.get("TAG", "")
UNCHEAT_DIR = pathlib.Path("/data/rwkv-sglang/repo/bench/data/uncheatable")
MATH_PATH = "/data/rwkv-sglang/repo/bench/data/MATH500.jsonl"
OUT_DIR = pathlib.Path("/data/bo_evals")
CHUNK = 4000
MATH_BATCH = 24
MATH_MAXNEW = 1500
LN2 = math.log(2)


@torch.no_grad()
def uncheatable(model, tok, device) -> dict:
    per_ds, tot_nll, tot_bytes, tot_docs = {}, 0.0, 0, 0
    for f in sorted(UNCHEAT_DIR.glob("*.json")):
        docs = json.load(open(f))
        nll_sum, byte_sum = 0.0, 0
        for text in docs:
            ids = tok.encode(text, add_special_tokens=False)
            byte_sum += len(text.encode("utf-8"))
            for b in range(0, len(ids), CHUNK):
                chunk = [0] + ids[b : b + CHUNK]
                x = torch.tensor([chunk], device=device)
                logits = model(input_ids=x).logits[0].float()
                lp = torch.log_softmax(logits[:-1], dim=-1)
                tgt = torch.tensor(chunk[1:], device=device)
                nll_sum += float(-lp.gather(1, tgt[:, None]).sum())
        n = len(docs)
        per_ds[f.stem] = {"docs": n, "pooled_bpb": nll_sum / byte_sum / LN2}
        tot_nll += nll_sum
        tot_bytes += byte_sum
        tot_docs += n
        print(f"  uncheatable {f.stem}: bpb {per_ds[f.stem]['pooled_bpb']:.4f} ({n} docs)", flush=True)
    overall = {"pooled_bpb": tot_nll / tot_bytes / LN2,
               "pooled_compression_rate": tot_nll / tot_bytes / LN2 * 0.125 * 100,
               "total_docs": tot_docs}
    print(f"  uncheatable OVERALL pooled_bpb {overall['pooled_bpb']:.4f}", flush=True)
    return {"datasets": per_ds, "overall": overall}


@torch.no_grad()
def math500(model, tok, device, out_path) -> None:
    tasks = [json.loads(l) for l in open(MATH_PATH)]
    prompts = []
    for t in tasks:
        problem = t["problem"].strip().replace("\r\n", "\n")
        prompts.append([0] + tok.encode(f"User: {problem}\n\nAssistant: <think></think", add_special_tokens=False))
    out = open(out_path, "w")
    t0 = time.time()
    for s in range(0, len(tasks), MATH_BATCH):
        batch = prompts[s : s + MATH_BATCH]
        width = max(len(p) for p in batch)
        ids = torch.zeros(len(batch), width, dtype=torch.long)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for i, p in enumerate(batch):  # left-pad so every row ends at the same step
            ids[i, width - len(p):] = torch.tensor(p)
            mask[i, width - len(p):] = 1
        gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                             max_new_tokens=MATH_MAXNEW, do_sample=False, pad_token_id=0)
        for i in range(len(batch)):
            toks = gen[i, width:].tolist()
            text = tok.decode(toks)
            completion = text.split("\nUser:", 1)[0]
            if completion.startswith(">"):
                completion = completion[1:]
            idx = s + i
            out.write(json.dumps({"idx": idx, "answer": tasks[idx]["answer"],
                                  "completion": completion.strip()}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"  math500 {min(s + MATH_BATCH, len(tasks))}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    out.close()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)
    arms = sys.argv[1:] or ["baseline", "big-w8", "big+lora-w8"]
    for arm in arms:
        print(f"== arm {arm} ==", flush=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to("cuda").eval()
        apply_arm(model, arm, model.config.hidden_size)
        u = uncheatable(model, tok, "cuda")
        json.dump(u, open(OUT_DIR / f"{TAG}{arm.replace('+','_')}_uncheatable.json", "w"), indent=1)
        math500(model, tok, "cuda", OUT_DIR / f"{TAG}{arm.replace('+','_')}_math500.jsonl")
        print(f"RESULT {arm}: uncheatable pooled_bpb {u['overall']['pooled_bpb']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
