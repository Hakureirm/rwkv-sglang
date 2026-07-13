import time

import torch

from verify_w4a8 import build, quantize_w4

build()
dev = "cuda"
torch.manual_seed(0)


def t(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


print(f"{'shape':>16} {'M':>4} {'a0 ms':>8} {'a1 ms':>8} {'a0/a1':>6}")
for n, k in [(2048, 2048), (2048, 8192), (8192, 2048), (4096, 4096),
             (4096, 16384), (16384, 4096)]:
    W = (torch.randn(n, k, device=dev) * 0.02).half()
    qw, sc, _ = quantize_w4(W)
    st = sc.t().contiguous()
    for m in (66, 96, 128, 192, 256, 320, 384, 512):
        xq = torch.randint(-128, 128, (m, k), dtype=torch.int8, device=dev)
        xs = torch.rand(m, dtype=torch.float32, device=dev) * 0.02
        t0 = t(lambda: torch.ops.rwkv7_w4.gemm_w4a8_tc(xq, qw, st, xs, 0))
        t1 = t(lambda: torch.ops.rwkv7_w4.gemm_w4a8_tc(xq, qw, st, xs, 1))
        print(f"{n}x{k:>6} {m:>4} {t0:8.4f} {t1:8.4f} {t0 / t1:6.2f}")
