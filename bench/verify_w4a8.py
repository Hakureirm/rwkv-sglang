#!/usr/bin/env python3
"""Standalone gate + microbench for the w4a8 large-M tensor-core GEMM
(rwkv7_w4.cu::gemm_w4a8_tc — the M>64 path that replaces dequant->cuBLAS, whose
~36 bits/element effective weight traffic was the M=64 concurrency cliff).

The op takes the TRANSPOSED scale ([K/64, N] contiguous, i.e. scale.t().contiguous()
— the layout that coalesces per-group scale reads; W4Linear caches it per layer).

Gates (hard, both algos):
  1. exact vs a bit-mimicking reference: per-group exact integer sums (fp64
     matmul holds them exactly; |S_g| <= 128*7*64 < 2^24 so the fp32 cast is
     exact), folded in the kernel's fp32 chain — facc = facc + S_g*scale[n,g]
     ascending g (torch fp32 mul + add == __fmul_rn + __fadd_rn), then
     y = half(facc * x_scale[m]) — across M in {65,66,96,128,256,384,512} x the
     real projection shapes,
  2. K zero-pad path (LoRA-ish K%64!=0 -> pad; s8 zeros add exact zeros),
  3. ragged N (N%128!=0) guard path,
  4. batch invariance: the same row at M=1 vs inside M=257 is bit-identical,
  5. cross-algo: MFRAG=1 and MFRAG=2 tiles are bit-identical (same per-element
     fp32 chain).

Report-only: quantization error vs the fp16 ideal and vs the w4a16 dequant
semantics (the accuracy delta w4a8 pays is certified e2e in Stage 3, not here).

Microbench (report-only): ours (incl. the per-token activation quant) vs
(i) dequant_w4+cuBLAS (the fallback this kernel replaces) and (ii) plain fp16
cuBLAS, at M=66/96/128/256/384/512 over the real projection shapes.

  python3 bench/verify_w4a8.py [--bench] [--iters N]
"""
import argparse
import time
from pathlib import Path

import torch

GROUP = 64


def build():
    from torch.utils.cpp_extension import load

    here = Path(__file__).resolve().parent
    candidates = [
        here / "rwkv7_w4.cu",  # flat (rsync'd next to this script)
        here.parent / "sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda/rwkv7_w4.cu",
    ]
    src = next((c for c in candidates if c.exists()), None)
    if src is None:
        raise FileNotFoundError(f"rwkv7_w4.cu not found in {[str(c) for c in candidates]}")
    load(name="rwkv7_w4", sources=[str(src)], is_python_module=False, verbose=False,
         extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-Xptxas", "-O3"])


def quantize_w4(W: torch.Tensor):
    """Group-wise (G=64) symmetric int4, identical to bench/quant_w4.py::pack_w4.
    Returns (qweight uint8[N,K/2], scale fp16[N,K/G], q_int int32[N,K])."""
    N, K = W.shape
    assert K % GROUP == 0, f"K={K} not divisible by GROUP={GROUP}"
    NG = K // GROUP
    Wg = W.float().view(N, NG, GROUP)
    scale = (Wg.abs().amax(dim=2) / 7.0).clamp(min=1e-8)
    q = torch.round(Wg / scale[:, :, None]).clamp_(-7, 7).to(torch.int32).view(N, K)
    nib = (q & 0xF).to(torch.uint8)
    qweight = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    return qweight, scale.to(torch.float16).contiguous(), q


def _ref(x_q, q_int, scale, xs):
    """Bit-mimic of the kernel: exact per-group integer sums, then the kernel's
    fp32 chain (mul, add — ascending g), then one fp32 mul by x_scale + fp16 RNE."""
    M, K = x_q.shape
    NG = K // GROUP
    facc = torch.zeros(M, q_int.shape[0], dtype=torch.float32, device=x_q.device)
    for g in range(NG):
        lo, hi = g * GROUP, (g + 1) * GROUP
        s_g = (x_q[:, lo:hi].double() @ q_int[:, lo:hi].double().t()).float()  # exact
        facc = facc + s_g * scale[:, g].float().view(1, -1)
    return (facc * xs.view(-1, 1)).half()


def _mk(m, n, k, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    W = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.float16)
    x_q = torch.randint(-128, 128, (m, k), dtype=torch.int8, device=dev, generator=g)
    xs = torch.rand(m, dtype=torch.float32, device=dev, generator=g) * 0.02 + 1e-4
    qweight, scale, q_int = quantize_w4(W)
    return x_q, xs, qweight, scale, q_int, W


def gate(dev, algo):
    G = lambda *a: torch.ops.rwkv7_w4.gemm_w4a8_tc(*a, algo)
    ok = True
    print(f"── gate algo={algo} (MFRAG={'2 / 64-row tile' if algo == 1 else '1 / 32-row tile'})")
    shapes = [(2048, 2048), (8192, 2048), (2048, 8192), (4096, 4096)]  # (N, K)
    for m in (65, 66, 96, 128, 256, 384, 512):
        for n, k in shapes:
            x_q, xs, qw, sc, q_int, _ = _mk(m, n, k, dev, seed=1234 + m + n + k)
            y = G(x_q, qw, sc.t().contiguous(), xs)
            y_ref = _ref(x_q, q_int, sc, xs)
            if torch.equal(y, y_ref):
                print(f"  exact M{m:4d} N{n:5d} K{k:5d}")
            else:
                nbad = (y != y_ref).sum().item()
                diff = (y.float() - y_ref.float()).abs().max().item()
                print(f"  FAIL M{m} N{n} K{k}: {nbad} mismatches, max abs diff {diff:.3e}")
                ok = False

    # ragged N (N%128!=0): exercises the b_rows_live guards + dead-column scale=0
    x_q, xs, qw, sc, q_int, _ = _mk(96, 4160, 4096, dev, seed=7)
    y = G(x_q, qw, sc.t().contiguous(), xs)
    p_ok = torch.equal(y, _ref(x_q, q_int, sc, xs))
    print(f"  ragged-N (M96 N4160 K4096): {'EXACT' if p_ok else 'FAIL'}")
    ok = ok and p_ok

    # K zero-pad path (LoRA-ish K%64!=0): quantize the zero-padded weight (pad
    # columns produce code 0), pad x_q with s8 zeros -> exact zero contributions.
    gen = torch.Generator(device=dev).manual_seed(11)
    Wr = (torch.randn(2048, 96, device=dev, generator=gen) * 0.02).to(torch.float16)
    Wp = torch.nn.functional.pad(Wr, (0, 32))
    qw, sc, q_int = quantize_w4(Wp)
    x_r = torch.randint(-128, 128, (128, 96), dtype=torch.int8, device=dev, generator=gen)
    x_p = torch.nn.functional.pad(x_r, (0, 32))
    xs = torch.rand(128, dtype=torch.float32, device=dev, generator=gen) * 0.02 + 1e-4
    y = G(x_p, qw, sc.t().contiguous(), xs)
    p_ok = torch.equal(y, _ref(x_p, q_int, sc, xs))
    print(f"  K=96 zero-pad path: {'EXACT' if p_ok else 'FAIL'}")
    ok = ok and p_ok

    # batch invariance: identical row content at different M must give identical bits
    x1, xs1, qw, sc, q_int, _ = _mk(1, 2048, 2048, dev, seed=77)
    xbig = torch.cat([x1, torch.randint(-128, 128, (256, 2048), dtype=torch.int8, device=dev)])
    xsbig = torch.cat([xs1, torch.rand(256, dtype=torch.float32, device=dev) * 0.02 + 1e-4])
    st = sc.t().contiguous()
    bi = torch.equal(G(x1, qw, st, xs1)[0], G(xbig, qw, st, xsbig)[0])
    print(f"  batch-invariance M=1 vs M=257 row0: {'EXACT' if bi else 'FAIL'}")
    ok = ok and bi

    # cross-algo agreement: both tiles run the same per-element fp32 chain
    if algo == 1:
        x_q, xs, qw, sc, q_int, _ = _mk(300, 4096, 2048, dev, seed=99)
        st = sc.t().contiguous()
        xa = torch.equal(
            torch.ops.rwkv7_w4.gemm_w4a8_tc(x_q, qw, st, xs, 0),
            torch.ops.rwkv7_w4.gemm_w4a8_tc(x_q, qw, st, xs, 1),
        )
        print(f"  algo1==algo0 bit-identical (M300 N4096 K2048): {'EXACT' if xa else 'FAIL'}")
        ok = ok and xa
    return ok


def quant_error_report(dev):
    """Report-only: what w4a8 costs vs the fp16 ideal and vs the w4a16 dequant
    semantics the M<=64 kernels use (full certification is Stage 3, e2e)."""
    print("── quantization-error report (NOT a gate; Stage 3 certifies accuracy e2e)")
    for n, k in [(2048, 2048), (4096, 4096)]:
        g = torch.Generator(device=dev).manual_seed(5)
        W = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.float16)
        X = torch.randn(128, k, device=dev, generator=g).to(torch.float16)
        qw, sc, q_int = quantize_w4(W)
        xs_f = X.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / 127.0
        x_q = torch.round(X.float() / xs_f).clamp_(-127, 127).to(torch.int8)
        y = torch.ops.rwkv7_w4.gemm_w4a8_tc(
            x_q, qw, sc.t().contiguous(), xs_f.view(-1).float(), -1).float()
        w_dq = (q_int.view(n, k // GROUP, GROUP).float() * sc.float()[:, :, None]).view(n, k)
        y_w4a16 = X.float() @ w_dq.t()
        y_fp16 = X.float() @ W.float().t()
        r_a8 = ((y - y_w4a16).norm() / (y_w4a16.norm() + 1e-9)).item()
        r_q = ((y - y_fp16).norm() / (y_fp16.norm() + 1e-9)).item()
        print(f"  N{n:5d} K{k:5d}: vs w4a16-dequant {r_a8:.2e} (act-quant tax), vs fp16 ideal {r_q:.2e}")


def bench(dev, iters=50):
    torch.manual_seed(0)
    try:
        from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8 as _ptq
        qname = "sglang-triton per_token_quant_int8"
    except Exception:
        def _ptq(x):
            s = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / 127.0
            return torch.round(x.float() / s).clamp_(-127, 127).to(torch.int8), s
        qname = "torch per-token quant (sglang triton op unavailable here)"
    print(f"\nactivation quant for 'ours' timing: {qname}")
    shapes = [  # (label, N, K) 1.5B (hidden 2048, ffn 8192) + 7.2B (4096/16384)
        ("attn 2048x2048", 2048, 2048),
        ("ffn.k 8192x2048", 8192, 2048),
        ("ffn.v 2048x8192", 2048, 8192),
        ("7b attn 4096x4096", 4096, 4096),
        ("7b ffn.k 16384x4096", 16384, 4096),
        ("7b ffn.v 4096x16384", 4096, 16384),
    ]
    ms_list = [66, 96, 128, 256, 384, 512]
    print(f"{'shape':22s} {'M':>4s} {'dq+blas':>9s} {'fp16':>9s} {'w4a8':>9s} {'kern':>9s} "
          f"{'vs dq':>7s} {'vs fp16':>8s}")
    for label, n, k in shapes:
        W = (torch.randn(n, k, device=dev) * 0.02).to(torch.float16)
        qw, sc, _ = quantize_w4(W)
        st = sc.t().contiguous()  # the cached-per-layer transposed scale the model path uses
        Wt = W.t().contiguous()  # pre-materialized fp16 [K,N] for the cuBLAS baseline
        for m in ms_list:
            X = torch.randn(m, k, device=dev).to(torch.float16)

            def t(fn):
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / iters * 1e3

            def dq_blas():  # the current M>64 fallback this kernel replaces
                wq = torch.ops.rwkv7_w4.dequant_w4(qw, sc)
                return X @ wq.t()

            def ours():  # honest: includes the per-token activation quant
                x_q, xsc = _ptq(X)
                return torch.ops.rwkv7_w4.gemm_w4a8_tc(x_q, qw, st, xsc.view(-1).float(), -1)

            x_q0, xs0 = _ptq(X)
            xs0 = xs0.view(-1).float()
            t_dq = t(dq_blas)
            t_fp16 = t(lambda: X @ Wt)
            t_ours = t(ours)
            t_kern = t(lambda: torch.ops.rwkv7_w4.gemm_w4a8_tc(x_q0, qw, st, xs0, -1))
            print(f"{label:22s} {m:4d} {t_dq:9.4f} {t_fp16:9.4f} {t_ours:9.4f} {t_kern:9.4f} "
                  f"{t_dq / t_ours:6.2f}x {t_fp16 / t_ours:7.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"sm{''.join(map(str, torch.cuda.get_device_capability()))}")
    build()
    ok = gate(dev, 0) and gate(dev, 1)
    print(f"GATE: {'PASS' if ok else 'FAIL'}")
    if ok:
        quant_error_report(dev)
    if args.bench and ok:
        bench(dev, args.iters)
    raise SystemExit(0 if ok else 1)
