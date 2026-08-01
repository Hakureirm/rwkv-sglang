#!/usr/bin/env python3
"""Is the channel-mix sparsity input-dependent, or is it the same dead channels every time?

F0079's corrected conclusion says large-batch effort belongs in removing work. The sparse
channel-mix already removes ~90% of the value-projection's weight reads -- but only at M==1,
because the kernel skips rows of the weight whose activation is zero, and at M>1 a row can
only be skipped if it is zero for EVERY request in the batch.

Which of two worlds we are in decides whether that kernel can ever batch:

  * input-dependent zeros: if each row is ~90% zero independently, the chance a channel is
    zero across 320 rows is 0.9^320, i.e. never. Batched sparsity is dead on arrival and the
    honest thing is to write that down and stop.
  * channel-intrinsic zeros: if the same channels are dead regardless of input, the union
    stays sparse at any batch size, and those channels can be pruned statically -- which is
    better than a sparse kernel, because it costs nothing at runtime.

This measures the union directly: run several unrelated prompts, capture relu(k)^2 per FFN
layer, and report per-row zero fraction against the fraction of channels zero in ALL rows.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/data/rwkv-sglang/models/rwkv7-1.5b-fla"
PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "In 1789 the French Revolution began when",
    "The chemical symbol for gold is Au, and for silver it is",
    "User: Explain how a refrigerator works.\n\nAssistant: A refrigerator moves heat",
    "床前明月光，疑是地上霜。举头望明月，",
    "SELECT name, COUNT(*) FROM orders GROUP BY",
    "The mitochondrion is often called the powerhouse of the",
]

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, dtype=torch.float16
).cuda().eval()

acts = {}


def hook(name):
    def fn(mod, inp, out):
        # channel-mix key projection output; the kernel's skip test is relu(k)^2 == 0,
        # which is relu(k) == 0, i.e. k <= 0
        a = (torch.relu(out.detach().float()) ** 2).reshape(-1, out.shape[-1])
        acts.setdefault(name, []).append((a == 0))
    return fn


handles = []
for n, m in model.named_modules():
    if n.endswith("ffn.key") or n.endswith("feed_forward.key"):
        handles.append(m.register_forward_hook(hook(n)))
print(f"hooked {len(handles)} channel-mix key projections")
assert handles, "no ffn key modules found -- check module naming"

with torch.no_grad():
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        model(ids)

print(f"\n{'layer':>28}  {'rows':>6}  {'per-row zero':>12}  {'zero in ALL rows':>16}")
tot_row, tot_union, n = 0.0, 0.0, 0
for name, chunks in acts.items():
    z = torch.cat(chunks, 0)               # [rows, inter] bool, True = zero
    per_row = z.float().mean().item()
    union = z.all(dim=0).float().mean().item()   # channels zero for every row seen
    tot_row += per_row; tot_union += union; n += 1
    if n <= 6 or n % 8 == 0:
        print(f"{name:>28}  {z.shape[0]:>6}  {per_row:>11.1%}  {union:>15.1%}")
print(f"\n{'MEAN over ' + str(n) + ' layers':>28}  {'':>6}  {tot_row/n:>11.1%}  {tot_union/n:>15.1%}")
print("\nreading: if 'zero in ALL rows' collapses toward 0 while per-row stays high, the")
print("sparsity is input-dependent and a batched sparse kernel cannot skip anything.")
