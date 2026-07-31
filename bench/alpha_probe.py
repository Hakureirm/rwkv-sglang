# Independent alpha measurement: target greedy trajectory, draft teacher-forced
# argmax agreement -- the F0029 quantity, on the exact serving workload.
# usage: alpha_probe.py <tokenizer_dir> <target_hf_dir> <draft_hf_dir>
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
TOK_DIR, TGT_DIR, DRF_DIR = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)
tgt = AutoModelForCausalLM.from_pretrained(TGT_DIR, dtype=torch.float16).cuda().eval()
drf = AutoModelForCausalLM.from_pretrained(DRF_DIR, dtype=torch.float16).cuda().eval()

prompts = [
    "User: Tell me a long story about a dragon who learns to paint.\n\nAssistant:",
    "User: Describe the causes and consequences of the industrial revolution.\n\nAssistant:",
    "The Industrial Revolution began in Britain in the late 18th century and",
]
for p in prompts:
    ids = [0] + tok.encode(p, add_special_tokens=False)
    with torch.no_grad():
        gen = tgt.generate(input_ids=torch.tensor([ids], device="cuda"), max_new_tokens=200, do_sample=False)
        full = gen[0].tolist()
        traj = torch.tensor([full], device="cuda")
        dlog = drf(input_ids=traj).logits[0]
    n_prompt = len(ids)
    pred = dlog[n_prompt - 1 : -1].argmax(dim=-1)
    gold = torch.tensor(full[n_prompt:], device="cuda")
    alpha = (pred == gold).float().mean().item()
    print(f"alpha={alpha:.3f}  ({p[:40]!r})")
