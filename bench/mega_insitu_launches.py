#!/usr/bin/env python3
"""In-situ launch-count probe for the megakernel path (task #50 / F0060 §6).

Builds the REAL deployed Rwkv7DecoderLayer with the shipping fast-path flag set
and torch-profiles one decode step, printing the per-layer CUDA kernel histogram.
Run twice (RWKV_MEGA 0 then 1) to read the launch-count delta and confirm the
grouped kernel routing (r/k/v G=3 + o_proj G=1 -> gemv_grouped_m1).

  python /tmp/mega_insitu_launches.py <model_dir> <mega:0|1>
"""
import os
import sys

import torch

# F0060 §1 shipping flag set (the deployed decode configuration).
for k in ["RWKV_FAST_LINEAR", "RWKV_SPARSE_FFN", "RWKV_FUSED_LORA", "RWKV_FUSED_GLUE",
          "RWKV_GEMV_AUTOTUNE", "RWKV_FUSED_GATES", "RWKV_FUSED_SQRELU",
          "RWKV_FUSED_ADDLN", "RWKV_FUSED_GNGC", "RWKV_WKV_CUDA"]:
    os.environ[k] = "1"
MEGA = sys.argv[2] if len(sys.argv) > 2 else "0"
os.environ["RWKV_MEGA"] = MEGA
sys.path.insert(0, "/tmp")
import pc  # noqa: E402

from sglang.srt.models import rwkv7  # noqa: E402

dtype = torch.float16
cfg, _ = pc.load_cfg(sys.argv[1] if len(sys.argv) > 1 else "/models/rwkv7-1.5b-fla")
torch.manual_seed(0)
layer = rwkv7.Rwkv7DecoderLayer(cfg, 1).cuda().to(dtype).eval()
be, fb = pc.make_state(cfg, 1, True, dtype)
x = torch.randn(1, cfg.hidden_size, dtype=dtype, device="cuda")
vf = torch.randn(1, cfg.hidden_size, dtype=dtype, device="cuda")


def step():
    with torch.no_grad():
        return layer(fb, x.clone(), vf.clone())


for _ in range(20):  # warm (lazy caches: mix6, lora pack, autotune)
    step()
torch.cuda.synchronize()

from torch.profiler import ProfilerActivity, profile  # noqa: E402

N = 50
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(N):
        step()
    torch.cuda.synchronize()

rows = []
total = 0
for e in prof.key_averages():
    c = getattr(e, "device_type", None)
    # CUDA kernels only (exclude host ops / memcpy where possible)
    if e.self_device_time_total <= 0:
        continue
    per = e.count / N
    total += per
    rows.append((per, e.key, e.self_device_time_total / N))
rows.sort(reverse=True)
print(f"# model={sys.argv[1] if len(sys.argv) > 1 else '1.5b'} RWKV_MEGA={MEGA} "
      f"H={cfg.hidden_size}")
print(f"# per-layer CUDA launches: {total:.1f}")
for per, name, us in rows[:22]:
    print(f"  {per:5.2f}  {us:8.2f}us  {name[:70]}")
