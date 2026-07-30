import os, sys, torch
os.environ["RWKV_V7_ON"]="1"; os.environ["RWKV_JIT_ON"]="0"; os.environ["RWKV_CUDA_ON"]="0"
sys.path.insert(0,"/tf/src")
from huggingface_hub import hf_hub_download
path = hf_hub_download("BlinkDL/rwkv-7-world","RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth", local_dir="/cache")

from rwkv.utils import PIPELINE
from rwkv.model import RWKV
ref = RWKV(model=path[:-4], strategy="cpu fp32")
pipe = PIPELINE(ref, "rwkv_vocab_v20230424")
PROMPT = "The Eiffel Tower is located in the city of"
toks = pipe.encode(PROMPT)
print("prompt ids:", toks)

# 官方:贪心 20 步
state=None; out=None; ids=list(toks); gen=[]
for i in range(20):
    out, state = ref.forward(ids if i==0 else [ids[-1]], state)
    nxt = int(out.argmax()); ids.append(nxt); gen.append(nxt)
print("\n[官方运行时] ", repr(pipe.decode(gen)))

# 我们的
from transformers import Rwkv7ForCausalLM
m = Rwkv7ForCausalLM.from_pretrained("/cache/ours", dtype=torch.float32).eval()
with torch.no_grad():
    g = m.generate(input_ids=torch.tensor([toks]), max_new_tokens=20, do_sample=False)
ours = g[0, len(toks):].tolist()
print("[我们的实现] ", repr(pipe.decode(ours)))
