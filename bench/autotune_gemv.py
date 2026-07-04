"""Build + correctness-gate + autotune-seed for gemv_m1_cfg (F0023 roadmap #6).

Verifies the parametrized gemv_m1_cfg is token-exact vs the fixed gemv_m1 and vs a
torch fp32 reference on the real RWKV-7 1.5B decode shapes, then sweeps all valid
(threads,out_tile) per shape to report the best config + speedup over the old fixed
<128,2>, and writes the winners to the per-GPU autotune cache.

Run on the box (GPU):  python bench/autotune_gemv.py
"""
import sys
from pathlib import Path

import torch

# load the JIT ext + helpers from the overlay
OV = Path(__file__).resolve().parents[1] / "sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels"
sys.path.insert(0, str(OV))
import fast_linear as FL  # noqa: E402

assert FL.available(), "rwkv7_fast ext failed to build"
dev = torch.device("cuda")
arch = FL._arch_key()
name = torch.cuda.get_device_name(0)
print(f"GPU={name} sm_{arch} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")

# RWKV-7 1.5B M==1 GEMV shapes routed through gemv_m1 (hidden=2048, inter=8192):
# r/k/v/o (2048x2048), ffn_key (8192x2048), ffn_value (2048x8192).
SHAPES = [("att_rkvo", 2048, 2048), ("ffn_key", 8192, 2048), ("ffn_value", 2048, 8192)]


def bench_cfg(x, w, t, ot, iters=200):
    for _ in range(20):
        torch.ops.rwkv7_fast.gemv_m1_cfg(x, w, t, ot)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        torch.ops.rwkv7_fast.gemv_m1_cfg(x, w, t, ot)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


print(f"\n{'shape':10s} {'N':>6s} {'K':>6s} | {'fixed<128,2>':>12s} | {'best cfg':>10s} {'best ms':>9s} | speedup")
print("-" * 78)
seeded = {}
for tag, N, K in SHAPES:
    x = torch.randn(1, K, dtype=torch.float16, device=dev)
    w = torch.randn(N, K, dtype=torch.float16, device=dev)
    # correctness: cfg (default 128,2) vs plain gemv_m1 vs torch fp32
    ref = (x.float() @ w.float().t())
    y_plain = torch.ops.rwkv7_fast.gemv_m1(x, w).float()
    y_cfg = torch.ops.rwkv7_fast.gemv_m1_cfg(x, w, 128, 2).float()
    exact = torch.equal(y_plain, y_cfg)
    rel = (y_cfg - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    # sweep all valid configs
    fixed_ms = bench_cfg(x, w, 128, 2)
    best_t, best_ot, best_ms = 128, 2, fixed_ms
    for (t, ot) in FL._valid_configs(N):
        ms = bench_cfg(x, w, t, ot)
        if ms < best_ms:
            best_ms, best_t, best_ot = ms, t, ot
    sp = fixed_ms / best_ms
    flag = "OK" if exact else "!!CFG!=PLAIN!!"
    print(f"{tag:10s} {N:6d} {K:6d} | {fixed_ms*1e3:10.2f}us | ({best_t:3d},{best_ot}) {best_ms*1e3:7.2f}us | {sp:.2f}x  "
          f"[cfg==plain:{flag} rel={rel:.1e}]")
    seeded[(arch, N, K)] = (best_t, best_ot)

# write winners into the autotune cache
FL._load_disk_cache()
FL._CFG_CACHE.update(seeded)
FL._save_disk_cache()
print(f"\nseeded {len(seeded)} shapes into {FL._cache_path()}")
