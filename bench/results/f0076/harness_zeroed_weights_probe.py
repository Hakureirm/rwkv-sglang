import sys, torch
sys.path.insert(0,"/tf/src")
from transformers import Rwkv7ForCausalLM
from safetensors.torch import load_file
m = Rwkv7ForCausalLM.from_pretrained("/cache/ours", dtype=torch.float32).eval()
disk = load_file("/cache/ours/model.safetensors")
zeroed, mismatched = [], []
sd = m.state_dict()
for k, v in sd.items():
    if k not in disk: continue
    if v.abs().max() == 0 and disk[k].abs().max() != 0:
        zeroed.append(k)
    elif not torch.equal(v.float(), disk[k].float()):
        mismatched.append(k)
print(f"加载后被清零的参数: {len(zeroed)}")
for k in zeroed[:12]: print("   ", k)
print(f"加载后与磁盘不一致的: {len(mismatched)}")
for k in mismatched[:8]: print("   ", k)
print(f"共 {len(sd)} 个参数, 磁盘 {len(disk)} 个")
