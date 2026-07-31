"""Long-form speed probe: realistic generation workloads (story / explanation /
reasoning / code), 256 new tokens each -- the regime speculative decoding is
for. Same server-side e2e metric as bench/spec_speed.py.
"""
import argparse, json, statistics
import requests

PROMPTS = [
    ("story", "User: Tell me a long story about a dragon who learns to paint.\n\nAssistant:"),
    ("explain", "User: Explain in detail how a refrigerator works, covering the compressor, condenser and evaporator.\n\nAssistant:"),
    ("math think", "User: A train travels 120 km in 1.5 hours, then 80 km in 0.5 hours. What is its average speed for the whole trip?\n\nAssistant: <think></think"),
    ("code", "User: Write a Python function that merges two sorted lists into one sorted list, with comments.\n\nAssistant:"),
    ("history", "User: Describe the causes and consequences of the industrial revolution.\n\nAssistant:"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--gen-len", type=int, default=256)
a = ap.parse_args()

s = requests.Session()
url = f"http://127.0.0.1:{a.port}/generate"
s.post(url, json={"text": "warmup", "sampling_params": {"temperature": 0.0, "max_new_tokens": 8}})
vals = []
for name, p in PROMPTS:
    r = s.post(url, json={"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": a.gen_len}}).json()
    mi = r["meta_info"]
    tps = mi["completion_tokens"] / mi["e2e_latency"]
    acc = mi.get("spec_accept_length", None)
    acc_s = f"  accept={acc:.2f}" if acc else ""
    vals.append(tps)
    print(f"  [{a.tag}] {name:12s} n={mi['completion_tokens']:4d}  {tps:7.1f} tok/s{acc_s}", flush=True)
print(f"=== {a.tag}: median {statistics.median(vals):.1f} tok/s  mean {statistics.fmean(vals):.1f} tok/s ===", flush=True)
