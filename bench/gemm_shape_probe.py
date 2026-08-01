#!/usr/bin/env python3
"""Is the 155 TFLOP/s in F0079 a shape limit or a dispatch problem?

F0079 measured the c=320 decode step spending 58% of its time in one cutlass GEMM at
roughly a third of this card's dense fp16 peak, with Ampere-era tiles and split-K
reductions visible in the trace. Before anyone writes a kernel, the question to settle
is which of two very different things that means:

  * the shape is simply unfriendly -- M=320 against K=N=4096 is skinny, tile
    quantisation and wave quantisation eat the difference, and no library would do
    better; or
  * the library is dispatching badly for THIS shape on THIS architecture, in which
    case there is something to recover without writing anything at all.

The referee is the same card running a shape it likes. If a large square GEMM reaches
a much higher rate, the deficit is the shape's; if it lands in the same band, the
ceiling is the card's and F0079's headroom is illusory.

Also sweeps M to find where the projection shapes stop being skinny, and toggles
fp16 reduced-precision reduction, which is the one split-K-related knob torch exposes.
"""
import torch, time

assert torch.cuda.is_available()
dev = torch.cuda.get_device_properties(0)
print(f"{dev.name}  SMs={dev.multi_processor_count}")


def bench(a, b, n=30):
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        a @ b
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def report(tag, M, K, N, flops_scale=1):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    dt = bench(a, b)
    tflops = 2 * M * K * N * flops_scale / dt / 1e12
    print(f"  {tag:34s} M={M:6d} K={K:5d} N={N:6d}   {dt*1e3:8.3f} ms   {tflops:7.1f} TFLOP/s")
    return tflops


print("\n== the shapes the 7.2B decode step actually runs (per layer) ==")
for M in (320,):
    report("r/k/v/o proj", M, 4096, 4096)
    report("ffn key", M, 4096, 16384)
    report("ffn value", M, 16384, 4096)

print("\n== the same card on a shape it likes (the referee) ==")
for S in (4096, 8192):
    report(f"square {S}", S, S, S)

print("\n== how skinny is skinny: M sweep on the r/k/v/o shape ==")
for M in (64, 128, 320, 512, 1024, 2048, 4096):
    report("proj 4096x4096", M, 4096, 4096)

print("\n== split-K knob (fp16 reduced-precision reduction) ==")
for flag in (True, False):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
    report(f"reduced_precision_reduction={flag}", 320, 4096, 4096)
