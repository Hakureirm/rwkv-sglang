#!/usr/bin/env python3
"""compute-sanitizer exerciser for every hand-CUDA cp.async pipeline (audit
F0062: does our cp.async usage carry the Albatross ff144b6b race class?).

Launch-only by design: bit-exactness is the verify_* gates' job (run those
un-instrumented, separately). This driver's job is to put every cp.async
kernel — and the branch-decisive shapes of each — under
`compute-sanitizer --tool racecheck` (and synccheck for the named-barrier
kernel) with a minimal launch count:

  rwkv7_w4.cu   gemv_w4_m1 / gemm_w4_small (no cp.async; included for sweep
                completeness), gemm_w4_tc (2-stage pipeline; ragged M, odd and
                even k-tile parity, split-K WritePartial + splitk_reduce via
                the long-K shape), gemm_w4a8_tc (cp.async + `bar.sync 1+wn,64`
                pair handoff; BOTH algos, ragged M and N, odd/even nk)
  rwkv7_w8.cu   gemv_w8_m1 / gemm_w8_small, gemm_w8_tc, gemm_w8_tc_large
                (ragged live rows, odd/even nk)
  rwkv7_w8a8.cu gemm_w8a8_tc V1 + V2 (both algos, ragged M/N, bias and
                no-bias epilogues, odd/even nk)
  rwkv7_wkv.cu  wkv_decode (inline-PTX cp.async, cache-policy operand):
                fp16 + fp32 pools, live/dead (ci=-1) slot mix in one launch,
                chained steps, H=32 and H=64 grids

Run on the box (kernel sources staged flat next to this script):
  python3 race_audit_driver.py                          # plain (build + sanity)
  compute-sanitizer --tool racecheck --error-exitcode 3 \
      python3 race_audit_driver.py                      # the audit run
  compute-sanitizer --tool synccheck --error-exitcode 3 \
      python3 race_audit_driver.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
GROUP = 64
DEV = "cuda"


def build(name):
    from torch.utils.cpp_extension import load

    candidates = [
        HERE / f"{name}.cu",
        HERE.parent.parent
        / f"sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda/{name}.cu",
    ]
    src = next((c for c in candidates if c.exists()), None)
    if src is None:
        raise FileNotFoundError(f"{name}.cu not found in {[str(c) for c in candidates]}")
    load(name=name, sources=[str(src)], is_python_module=False, verbose=False,
         extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-Xptxas", "-O3"])


def quantize_w4(W):
    """Identical to bench/verify_w4.py::quantize_w4 (canonical int4 pack)."""
    N, K = W.shape
    NG = K // GROUP
    Wg = W.float().view(N, NG, GROUP)
    scale = (Wg.abs().amax(dim=2) / 7.0).clamp(min=1e-8)
    q = torch.round(Wg / scale[:, :, None]).clamp_(-7, 7).to(torch.int32).view(N, K)
    nib = (q & 0xF).to(torch.uint8)
    qweight = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    return qweight, scale.to(torch.float16).contiguous()


def quantize_w8(W):
    """Identical to bench/verify_w8.py::quantize_w8."""
    N, K = W.shape
    NG = K // GROUP
    Wg = W.float().view(N, NG, GROUP)
    scale = (Wg.abs().amax(dim=2) / 127.0).clamp(min=1e-8)
    q = torch.round(Wg / scale[:, :, None]).clamp_(-127, 127).to(torch.int8).view(N, K)
    return q.contiguous(), scale.to(torch.float16)


def main():
    # rwkv7_wkv.cu is the sm120 kernel (its st.global.L2::evict_last.v4.b64
    # 256-bit stores need sm100+; ptxas rejects it below that, and the serving
    # loader falls back to Triton on build failure) — only buildable/reachable
    # on sm100+. On lower arches this driver skips it; its cp.async site is
    # covered by the static audit + compile-for-sm120 SASS check (F0062) and
    # an on-device racecheck is owed when an sm100+ card is free.
    wkv_ok = torch.cuda.get_device_capability() >= (10, 0)
    names = ["rwkv7_w4", "rwkv7_w8", "rwkv7_w8a8"] + (["rwkv7_wkv"] if wkv_ok else [])
    for name in names:
        build(name)
    torch.manual_seed(0)
    launches = 0

    # ---- rwkv7_w4: gemv/small (plain-load, completeness) + gemm_w4_tc pipeline ----
    # K=2048 -> even nk; K=1088 (17 groups) -> odd nk (opposite final buffer
    # parity); M=33 exercises the zeroed dead rows; (K=14336, M=32) engages the
    # auto split-K WritePartial path + splitk_reduce.
    for (M, K, N) in [(1, 2048, 2048), (4, 2048, 2048),
                      (16, 2048, 2048), (33, 1088, 2048), (64, 2048, 2048),
                      (32, 14336, 4096)]:
        W = (torch.randn(N, K, device=DEV) * 0.02).half()
        X = torch.randn(M, K, device=DEV).half()
        qw, sc = quantize_w4(W)
        if M == 1:
            _ = torch.ops.rwkv7_w4.gemv_w4_m1(X, qw, sc)
        elif M <= 8:
            _ = torch.ops.rwkv7_w4.gemm_w4_small(X, qw, sc)
        else:
            _ = torch.ops.rwkv7_w4.gemm_w4_tc(X, qw, sc)
        launches += 1
    _ = torch.ops.rwkv7_w4.dequant_w4(qw, sc)
    launches += 1

    # ---- rwkv7_w4: gemm_w4a8_tc (cp.async + named-barrier pair handoff) ----
    # Both algos (MFRAG=1/2); M=33/65 ragged vs BM, N=2112 ragged vs BN=128,
    # K=1088 odd nk, K=2048 even nk.
    for algo in (0, 1):
        for (M, K, N) in [(33, 2048, 2048), (65, 1088, 2112), (256, 2048, 4096)]:
            xq = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEV)
            xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.02 + 1e-4
            W = (torch.randn(N, K, device=DEV) * 0.02).half()
            qw, sc = quantize_w4(W)
            _ = torch.ops.rwkv7_w4.gemm_w4a8_tc(xq, qw, sc.t().contiguous(), xs, algo)
            launches += 1

    # ---- rwkv7_w8: gemv/small + gemm_w8_tc + gemm_w8_tc_large pipelines ----
    for (M, K, N) in [(1, 2048, 2048), (4, 2048, 2048),
                      (16, 2048, 2048), (33, 1088, 2048), (64, 2048, 2048),
                      (96, 1088, 2048), (129, 2048, 2048)]:
        W = (torch.randn(N, K, device=DEV) * 0.02).half()
        X = torch.randn(M, K, device=DEV).half()
        qw, sc = quantize_w8(W)
        if M == 1:
            _ = torch.ops.rwkv7_w8.gemv_w8_m1(X, qw, sc)
        elif M <= 8:
            _ = torch.ops.rwkv7_w8.gemm_w8_small(X, qw, sc)
        elif M <= 64:
            _ = torch.ops.rwkv7_w8.gemm_w8_tc(X, qw, sc)
        else:
            _ = torch.ops.rwkv7_w8.gemm_w8_tc_large(X, qw, sc)
        launches += 1
    _ = torch.ops.rwkv7_w8.dequant_w8(qw, sc)
    launches += 1

    # ---- rwkv7_w8a8: V1 (algo=0) + V2 (algo=1) ----
    # M=31 (< BM), M=65/257 ragged vs both tiles, N=2112 ragged vs BN=128,
    # K=1088 odd nk; bias + no-bias epilogues; fp16 + bf16 outs.
    for algo in (0, 1):
        for (M, K, N) in [(31, 2048, 2048), (65, 1088, 2112), (257, 2048, 4096)]:
            xq = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEV)
            wq = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=DEV)
            xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.02 + 1e-4
            ws = torch.rand(N, dtype=torch.float32, device=DEV) * 0.02 + 1e-4
            bias = (torch.rand(N, device=DEV) - 0.5).half()
            _ = torch.ops.rwkv7_w8a8.gemm_w8a8_tc(xq, wq.t(), xs, ws,
                                                  torch.float16, None, algo)
            _ = torch.ops.rwkv7_w8a8.gemm_w8a8_tc(xq, wq.t(), xs, ws,
                                                  torch.float16, bias, algo)
            _ = torch.ops.rwkv7_w8a8.gemm_w8a8_tc(xq, wq.t(), xs, ws,
                                                  torch.bfloat16, None, algo)
            launches += 3

    # ---- rwkv7_wkv: wkv_decode (inline-PTX cp.async, L2 cache-policy form) ----
    # Live + dead (ci=-1) slots in the SAME launch, both pool dtypes, chained
    # steps (in-place pool round-trip), H=32 and H=64 grids.
    if not wkv_ok:
        print("race_audit_driver: SKIP rwkv7_wkv (needs sm100+; this is "
              f"sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]})")
    for pool_dtype in (torch.float16, torch.float32) if wkv_ok else ():
        for (B, H) in [(1, 32), (7, 64), (16, 64)]:
            n_slots = B + 2
            pool = torch.randn(n_slots, H, 64, 64, device=DEV, dtype=pool_dtype)
            ci = torch.arange(B, dtype=torch.int32, device=DEV)
            if B > 1:
                ci[1] = -1  # dead-slot (pad) path alongside live blocks
            for _step in range(4):
                r, w, k, v, kk, a = (torch.randn(B, 1, H, 64, device=DEV).half()
                                     for _ in range(6))
                w = -w.abs()  # log-decay domain
                kk = torch.nn.functional.normalize(kk.float(), dim=-1).half()
                _ = torch.ops.rwkv7_wkv.wkv_decode(r, w, k, v, kk, a, pool, ci, 1.0)
                launches += 1

    torch.cuda.synchronize()
    print(f"race_audit_driver: {launches} audited-kernel launches completed OK "
          f"(sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]})")


if __name__ == "__main__":
    sys.exit(main())
