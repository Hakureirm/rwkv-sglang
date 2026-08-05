"""How much headroom does single-stream decode actually have?

A bsz1 decode step reads every weight exactly once and almost nothing else, so
its floor is set by memory bandwidth, not by the kernels. Before spending more
effort on fusion it is worth knowing whether the remaining gap to Albatross is
7% of a roof we are already near, or a factor we are leaving on the table.

Measures the card's achievable read bandwidth rather than quoting the spec
number, because the spec number is not what a kernel gets.
"""
import sys

import torch

dev = "cuda"
print(torch.cuda.get_device_name(0), "| torch", torch.__version__)


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3  # seconds


# ---- achievable read bandwidth: a large reduction touches each byte once ----
print("\n-- achievable read bandwidth --")
best = 0.0
for gb in (1, 2, 4):
    n = gb * (1 << 30) // 2
    x = torch.empty(n, dtype=torch.float16, device=dev).fill_(1.0)
    t = timed(lambda: x.sum(), iters=20)
    bw = (n * 2) / t / 1e9
    best = max(best, bw)
    print(f"  sum over {gb} GiB fp16: {t*1e3:7.3f} ms -> {bw:7.1f} GB/s")
    del x
    torch.cuda.empty_cache()

# a GEMV chain is closer to what decode does: many separate weight reads
print("\n-- GEMV chain (closer to a decode step's access pattern) --")
H = 2048
n_mat = 64
mats = [torch.randn(H, H, device=dev, dtype=torch.float16) for _ in range(n_mat)]
v = torch.randn(1, H, device=dev, dtype=torch.float16)


def chain():
    out = v
    for m in mats:
        out = out @ m
    return out


t = timed(chain, iters=20)
byts = n_mat * H * H * 2
print(f"  {n_mat} x [1,{H}]@[{H},{H}]: {t*1e3:7.3f} ms -> {byts/t/1e9:7.1f} GB/s")
gemv_bw = byts / t / 1e9

# ---- the roof ----
print("\n-- the roof for bsz1 decode --")
weights_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 3.06
for name, bw in (("peak-ish (big reduction)", best), ("GEMV-chain achievable", gemv_bw)):
    step_s = weights_gb * 1e9 / (bw * 1e9)
    print(f"  at {bw:7.1f} GB/s ({name}): {step_s*1e6:7.1f} us/token -> {1/step_s:7.1f} tok/s")

print("\n-- where we and Albatross sit --")
for who, toks in (("ours (fp16)", 514.6), ("Albatross (fp16)", 554.11), ("ours (int4)", 742.6)):
    step_s = 1.0 / toks
    print(f"  {who:>18}: {step_s*1e6:7.1f} us/token -> {weights_gb / step_s:7.1f} GB/s effective")
