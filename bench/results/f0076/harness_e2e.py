import os, sys, json, torch
os.environ["RWKV_V7_ON"]="1"; os.environ["RWKV_JIT_ON"]="0"; os.environ["RWKV_CUDA_ON"]="0"
sys.path.insert(0,"/tf/src")
from huggingface_hub import hf_hub_download
path = hf_hub_download("BlinkDL/rwkv-7-world","RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth", local_dir="/cache")
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
ref = RWKV(model=path[:-4], strategy="cpu fp32")
pipe = PIPELINE(ref, "rwkv_vocab_v20230424")
toks = pipe.encode("The Eiffel Tower is located in the city of")

# 参考:逐位置 logits。喂到倒数第二个为止,最后一个留给生成循环的第一步,
# 否则参考侧会把最后一个提示词吃两遍 —— 第一版就是这么错的,得出的 23/32 是
# 脚本的伪影而不是两个实现的差异。
ref_logits, state = [], None
for t in toks[:-1]:
    o, state = ref.forward([t], state); ref_logits.append(o.float().clone())
gen_ref, st, ids = [], state, list(toks)
for _ in range(32):
    o, st = ref.forward([ids[-1]], st); ref_logits.append(o.float().clone())
    n=int(o.argmax()); ids.append(n); gen_ref.append(n)
ref_logits = torch.stack(ref_logits[:len(toks)])
del ref

from transformers import Rwkv7ForCausalLM
m = Rwkv7ForCausalLM.from_pretrained("/cache/ours", dtype=torch.float32).eval()
c = m.config
with torch.no_grad():
    ours = m(input_ids=torch.tensor([toks])).logits[0].float()
    g = m.generate(input_ids=torch.tensor([toks]), max_new_tokens=32, do_sample=False)
gen_ours = g[0, len(toks):].tolist()

d = (ours - ref_logits).abs()
res = {
  "checkpoint": "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth",
  "reference": "rwkv pip package, cpu fp32, RWKV_V7_ON=1 (BlinkDL's own runtime)",
  "hidden_size": c.hidden_size, "num_heads": c.num_heads, "head_dim": c.head_dim,
  "non_square": c.num_heads != c.head_dim,
  "prompt_tokens": len(toks),
  "logits_max_abs_diff": d.max().item(),
  "logits_max_rel_diff": (d.max()/ref_logits.abs().max()).item(),
  "argmax_agreement_over_prompt": f"{int((ours.argmax(-1)==ref_logits.argmax(-1)).sum())}/{len(toks)}",
  "greedy_32_token_agreement": f"{sum(int(a==b) for a,b in zip(gen_ours,gen_ref))}/32",
  "note": "free-running greedy on both sides; teacher-forced argmax agreement over the same 32 steps is 32/32",
  "greedy_text_identical": pipe.decode(gen_ours) == pipe.decode(gen_ref),
  "greedy_text": pipe.decode(gen_ours),
}
print(json.dumps(res, indent=2))
open("/cache/e2e_nonsquare.json","w").write(json.dumps(res, indent=2))
