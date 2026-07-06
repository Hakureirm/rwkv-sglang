"""Gate for gemv_mb — the batch-invariant M-row GEMV used by the chain-spec verify.

The property that matters: row m of gemv_mb(X) must be BIT-IDENTICAL to
gemv_m1(X[m]) for every m and every M — same per-output fp32 reduction, same
(threads, out_tile). If it holds, the verify can compute the target over K
positions in one launch and still be exact against the M=1 baseline decode
(closing the F0031 gate flip, which was cuBLAS M=K GEMM reduction order).

Run on a CUDA box: python bench/verify_gemv_mb.py
"""
from pathlib import Path

import torch


def _load():
    from torch.utils.cpp_extension import load

    here = Path(__file__).resolve().parent
    src = (
        here.parent
        / "sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda/rwkv7_fast.cu"
    )
    load(name="rwkv7_fast", sources=[str(src)], is_python_module=False, verbose=False,
         extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-Xptxas", "-O3"])


# Mirror fast_linear._select_config's contract loosely: the gate sweeps ALL valid
# (threads, out_tile) so bit-identity is proven for every config the autotuner
# could pick, not just one.
CONFIGS = [(64, 1), (64, 2), (64, 4), (128, 1), (128, 2), (128, 4),
           (256, 1), (256, 2), (256, 4)]


def gate(dev):
    ok = True
    # RWKV-7 projection shapes (1.5B hidden 2048 / ffn 8192; 7.2B hidden 4096 / 16384)
    shapes = [(2048, 2048), (8192, 2048), (2048, 8192), (4096, 4096), (16384, 4096),
              (2050, 2048)]  # odd N (out_tile must divide N — skip mismatched cfgs)
    for (N, K) in shapes:
        g = torch.Generator(device=dev).manual_seed(11 + N + K)
        w = (torch.rand(N, K, dtype=torch.float16, device=dev, generator=g) - 0.5) * 0.1
        for M in (2, 4, 7, 8):
            x = (torch.rand(M, K, dtype=torch.float16, device=dev, generator=g) - 0.5) * 2
            for (t, ot) in CONFIGS:
                if N % ot != 0:
                    continue
                yb = torch.ops.rwkv7_fast.gemv_mb_cfg(x, w, t, ot)
                # reference: per-row gemv_m1_cfg with the SAME config
                rows = [torch.ops.rwkv7_fast.gemv_m1_cfg(x[m:m + 1].contiguous(), w, t, ot)
                        for m in range(M)]
                y1 = torch.cat(rows, dim=0)
                if not torch.equal(yb, y1):
                    nbad = (yb != y1).sum().item()
                    md = (yb.float() - y1.float()).abs().max().item()
                    print(f"  FAIL N{N} K{K} M{M} cfg({t},{ot}): {nbad} mismatch, maxdiff {md:.3e}")
                    ok = False
        print(f"  N{N:5d} K{K:5d}: all M×cfg bit-identical to gemv_m1" if ok
              else f"  N{N} K{K}: FAIL")
    return ok


if __name__ == "__main__":
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} sm{''.join(map(str, torch.cuda.get_device_capability()))}")
    _load()
    ok = gate(dev)
    print(f"GATE: {'PASS — gemv_mb is batch-invariant (row m == gemv_m1(x[m]))' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
