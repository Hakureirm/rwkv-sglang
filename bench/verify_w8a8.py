"""Standalone gate + microbench for the w8a8 int8×int8 tensor-core GEMM
(rwkv7_w8a8.cu — the sm120 stand-in for sgl_kernel's cutlass int8_scaled_mm).

Gates (hard):
  1. exact vs a bit-mimicking reference (fp64 int accumulation -> same fp32
     epilogue order as the kernel -> fp16/bf16 cast),
  2. batch invariance: row 0 of M=1 vs the same row inside M=257 is bit-identical
     (int32 accumulation is order-exact, so this must hold EXACTLY),
  3. edge shapes: M not a multiple of 32, N not a multiple of 128.

Microbench (report-only): our op vs pure fp16 cuBLAS (what fp16 serving pays) and
vs dequant+cuBLAS (what the current w8 large-M fallback pays), at model shapes.
Numbers here are standalone-GEMM only — e2e serving numbers are the ones we publish.
"""
import argparse
import time
from pathlib import Path

import torch


def _load():
    from torch.utils.cpp_extension import load

    here = Path(__file__).resolve().parent
    src = (
        here.parent
        / "sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda/rwkv7_w8a8.cu"
    )
    load(name="rwkv7_w8a8", sources=[str(src)], is_python_module=False, verbose=False,
         extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-Xptxas", "-O3"])


def _ref(x_q, w_nk, xs, ws, out_dtype):
    """Bit-mimic of the kernel: exact integer accumulation (fp64 holds |acc|<2^53),
    int32 -> fp32 RNE, then the kernel's epilogue order ((acc*xs)*ws) in fp32."""
    acc = (x_q.double() @ w_nk.double().t()).round()  # exact integers
    acc_f = acc.float()  # same RNE int->float rounding as the device cast
    y = (acc_f * xs.view(-1, 1)) * ws.view(1, -1)
    return y.to(out_dtype)


def _mk(m, n, k, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x_q = torch.randint(-128, 128, (m, k), dtype=torch.int8, device=dev, generator=g)
    w_nk = torch.randint(-128, 128, (n, k), dtype=torch.int8, device=dev, generator=g)
    xs = torch.rand(m, dtype=torch.float32, device=dev, generator=g) * 0.02 + 1e-4
    ws = torch.rand(n, dtype=torch.float32, device=dev, generator=g) * 0.02 + 1e-4
    return x_q, w_nk, xs, ws


def gate(dev):
    ok = True
    for out_dtype in (torch.float16, torch.bfloat16):
        for (m, n, k) in [(64, 2048, 2048), (33, 2048, 2048), (257, 8192, 2048),
                          (128, 2048, 8192), (1, 2048, 2048), (500, 65536, 2048),
                          (96, 4160, 4096)]:
            x_q, w_nk, xs, ws = _mk(m, n, k, dev, seed=1234 + m + n)
            y = torch.ops.rwkv7_w8a8.gemm_w8a8_tc(x_q, w_nk.t(), xs, ws, out_dtype, None)
            y_ref = _ref(x_q, w_nk, xs, ws, out_dtype)
            same = torch.equal(y, y_ref)
            if not same:
                diff = (y.float() - y_ref.float()).abs()
                nbad = (y != y_ref).sum().item()
                print(f"  FAIL {out_dtype} M{m} N{n} K{k}: {nbad} mismatches, "
                      f"max abs diff {diff.max().item():.3e}")
                ok = False
            else:
                print(f"  exact {str(out_dtype).split('.')[-1]:8s} M{m:4d} N{n:5d} K{k:5d}")

    # bias path: kernel does rn(acc*xs) then ONE fused mul-add with ws and bias
    # (__fmaf_rn — single rounding). Mimic the fma in fp64 (exact) + one fp32 round.
    x_q, w_nk, xs, ws = _mk(128, 2048, 2048, dev, seed=42)
    bias = (torch.rand(2048, device=dev) - 0.5).half()
    y = torch.ops.rwkv7_w8a8.gemm_w8a8_tc(x_q, w_nk.t(), xs, ws, torch.float16, bias)
    acc = (x_q.double() @ w_nk.double().t()).round().float()
    v1 = acc * xs.view(-1, 1)  # fp32, one rounding — matches __fmul_rn
    y_ref = (
        (v1.double() * ws.view(1, -1).double() + bias.double().view(1, -1)).float().half()
    )
    b_ok = torch.equal(y, y_ref)
    print(f"  bias fused-mul-add path: {'EXACT' if b_ok else 'FAIL'}")
    ok = ok and b_ok

    # K zero-pad path (LoRA-up ranks): padded product must equal the K=96 truth
    x_q, w_nk, xs, ws = _mk(64, 2048, 96, dev, seed=43)
    xp = torch.nn.functional.pad(x_q, (0, 32))
    wp = torch.nn.functional.pad(w_nk, (0, 32)).contiguous()
    y = torch.ops.rwkv7_w8a8.gemm_w8a8_tc(xp, wp.t(), xs, ws, torch.float16, None)
    y_ref = _ref(x_q, w_nk, xs, ws, torch.float16)
    p_ok = torch.equal(y, y_ref)
    print(f"  K=96 zero-pad path: {'EXACT' if p_ok else 'FAIL'}")
    ok = ok and p_ok

    # batch invariance: identical row content at different M must give identical bits
    x1, w_nk, xs1, ws = _mk(1, 2048, 2048, dev, seed=77)
    xbig = torch.cat([x1, torch.randint(-128, 128, (256, 2048), dtype=torch.int8, device=dev)])
    xsbig = torch.cat([xs1, torch.rand(256, dtype=torch.float32, device=dev)])
    y1 = torch.ops.rwkv7_w8a8.gemm_w8a8_tc(x1, w_nk.t(), xs1, ws, torch.float16, None)
    ybig = torch.ops.rwkv7_w8a8.gemm_w8a8_tc(xbig, w_nk.t(), xsbig, ws, torch.float16, None)
    bi = torch.equal(y1[0], ybig[0])
    print(f"  batch-invariance M=1 vs M=257 row0: {'EXACT' if bi else 'FAIL'}")
    ok = ok and bi
    return ok


def bench(dev, iters=50):
    torch.manual_seed(0)
    shapes = [  # (label, N, K) at 1.5B (hidden 2048, ffn 8192) + 7.2B (4096/16384)
        ("attn 2048x2048", 2048, 2048),
        ("ffn.k 8192x2048", 8192, 2048),
        ("ffn.v 2048x8192", 2048, 8192),
        ("7b attn 4096x4096", 4096, 4096),
        ("7b ffn.k 16384x4096", 16384, 4096),
        ("head 65536x2048", 65536, 2048),
    ]
    ms_list = [64, 128, 256, 512, 1024, 4096]
    print(f"\n{'shape':22s} {'M':>5s} {'ours ms':>9s} {'fp16 ms':>9s} {'dequant+mm':>11s} "
          f"{'vs fp16':>8s} {'vs deq':>7s}")
    for label, n, k in shapes:
        w_nk = torch.randint(-128, 128, (n, k), dtype=torch.int8, device=dev)
        ws = torch.rand(n, dtype=torch.float32, device=dev) * 0.02
        w_fp16 = (w_nk.float() * ws.view(-1, 1)).half()
        for m in ms_list:
            x_q = torch.randint(-128, 128, (m, k), dtype=torch.int8, device=dev)
            xs = torch.rand(m, dtype=torch.float32, device=dev) * 0.02
            x_fp16 = (x_q.float() * xs.view(-1, 1)).half()

            def t(fn):
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / iters * 1e3

            t_ours = t(lambda: torch.ops.rwkv7_w8a8.gemm_w8a8_tc(
                x_q, w_nk.t(), xs, ws, torch.float16, None))
            t_fp16 = t(lambda: torch.mm(x_fp16, w_fp16.t()))
            t_deq = t(lambda: torch.mm(x_fp16, (w_nk.float() * ws.view(-1, 1)).half().t()))
            print(f"{label:22s} {m:5d} {t_ours:9.4f} {t_fp16:9.4f} {t_deq:11.4f} "
                  f"{t_fp16 / t_ours:7.2f}x {t_deq / t_ours:6.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} sm{''.join(map(str, torch.cuda.get_device_capability()))}")
    _load()
    ok = gate(dev)
    print(f"GATE: {'PASS' if ok else 'FAIL'}")
    if args.bench and ok:
        bench(dev, args.iters)
    raise SystemExit(0 if ok else 1)
