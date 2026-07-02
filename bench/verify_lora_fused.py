# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Standalone gate + micro-bench for the fused 4-chain LoRA op (rwkv7_lora.cu, M9).

Validates the CUDA extension in isolation:
  1. build the JIT extension,
  2. numerical correctness of lora4_m1 vs the torch fp16 reference chain
     (F.linear -> act -> F.linear(+bias) per chain, the exact model path) over
     the real rank set {64,64,128,32} x H in {768, 2048, 4096}, plus the C=3
     layer-0 variant and an odd-rank case that exercises the scalar stage2 path.
     Expect fp16-ULP class agreement (max rel < 1e-3), same class as gemv_m1.
  3. determinism: two identical calls must be bitwise equal,
  4. micro-benchmark: fused (2 launches) vs the torch per-chain sequence
     (~12 kernels), both under CUDA events, 200 iters. NOTE this is an EAGER
     micro-bench; the end-to-end gain is smaller once cuda-graph amortizes
     launch overhead — but launch count is exactly what this op removes.

Usage (box, sglang venv):
    ~/envs/rwkv-sgl/bin/python bench/verify_lora_fused.py
"""

import sys

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.rwkv7_kernels import lora_fused as lf

DEV = "cuda"
DT = torch.float16

# (act_code, has_bias) per chain, mirroring w/a/g/v: tanh+bias, id+bias,
# sigmoid+no-bias, id+bias.
CHAIN_SPECS = [
    (lf.ACT_TANH, True),      # w
    (lf.ACT_IDENTITY, True),  # a
    (lf.ACT_SIGMOID, False),  # g
    (lf.ACT_IDENTITY, True),  # v
]


def _act(t, code):
    if code == lf.ACT_TANH:
        return torch.tanh(t)
    if code == lf.ACT_SIGMOID:
        return torch.sigmoid(t)
    return t


def _make_case(H, ranks, seed=0):
    torch.manual_seed(seed)
    C = len(ranks)
    xs = torch.randn(C, H, device=DEV, dtype=DT) * 0.3
    chains = []
    for i, rank in enumerate(ranks):
        act, has_bias = CHAIN_SPECS[i]
        dw = torch.randn(rank, H, device=DEV, dtype=DT) * 0.05
        uw = torch.randn(H, rank, device=DEV, dtype=DT) * 0.05
        b = (torch.randn(H, device=DEV, dtype=DT) * 0.1) if has_bias else None
        chains.append((dw, uw, b, act))
    return xs, chains


def _torch_chain(xs, chains):
    """The exact model reference: per chain F.linear -> act -> F.linear(+bias),
    all fp16 (torch rounds every stage to fp16)."""
    outs = []
    for c, (dw, uw, b, act) in enumerate(chains):
        x = xs[c:c + 1]
        t = F.linear(x, dw)
        t = _act(t, act)
        outs.append(F.linear(t, uw, b))
    return torch.cat(outs, dim=0)


def _torch_chain_fp32(xs, chains):
    outs = []
    for c, (dw, uw, b, act) in enumerate(chains):
        x = xs[c:c + 1].float()
        t = _act(F.linear(x, dw.float()), act)
        bias = None if b is None else b.float()
        outs.append(F.linear(t, uw.float(), bias))
    return torch.cat(outs, dim=0)


def _err(a, b):
    a = a.float()
    b = b.float()
    denom = b.abs().max().clamp_min(1e-6)
    d = (a - b).abs().max()
    return d.item(), (d / denom).item()


def check_correctness():
    print("\n== lora4_m1 correctness (y[C,H] vs torch per-chain reference) ==")
    ok = True
    cases = []
    for H in (768, 2048, 4096):
        cases.append((H, [64, 64, 128, 32], "C=4 w/a/g/v"))
        cases.append((H, [64, 64, 128], "C=3 layer-0"))
    cases.append((2048, [64, 31, 128, 32], "odd rank (scalar path)"))
    for H, ranks, tag in cases:
        xs, chains = _make_case(H, ranks, seed=H + len(ranks))
        pack = lf.pack_loras(chains)
        y = lf.lora4_m1(xs, *pack)
        ref16 = _torch_chain(xs, chains)
        ref32 = _torch_chain_fp32(xs, chains)
        ek_abs, ek_rel = _err(y, ref16)        # fused vs torch fp16 chain
        e32_abs, e32_rel = _err(y, ref32)      # fused vs fp32 truth
        et_abs, et_rel = _err(ref16, ref32)    # torch fp16 chain vs fp32 truth
        # determinism: bitwise-identical repeat
        y2 = lf.lora4_m1(xs, *pack)
        det = bool((y2 == y).all().item())
        verdict = "OK" if (ek_rel < 1e-3 and det) else "BAD"
        if verdict != "OK":
            ok = False
        print(f"  H={H:5d} ranks={str(ranks):>20} {tag:>22}  "
              f"fused_vs_torch16 abs={ek_abs:.2e} rel={ek_rel:.2e}  "
              f"fused_vs_fp32 rel={e32_rel:.2e}  torch16_vs_fp32 rel={et_rel:.2e}  "
              f"det={'Y' if det else 'N'}  [{verdict}]")
    return ok


def bench():
    print("\n== micro-bench: lora4_m1 (2 launches) vs torch per-chain (~12 kernels) ==")
    iters = 200
    print(f"  {'case':>28} | {'fused us':>9} | {'torch us':>9} | speedup")
    for H, ranks, tag in [
        (768, [64, 64, 128, 32], "0.1B-ish C=4"),
        (2048, [64, 64, 128, 32], "1.5B-ish C=4"),
        (2048, [64, 64, 128], "1.5B-ish C=3 (L0)"),
        (4096, [64, 64, 128, 32], "7.2B-ish C=4"),
    ]:
        xs, chains = _make_case(H, ranks, seed=1)
        pack = lf.pack_loras(chains)
        for _ in range(50):  # warmup
            lf.lora4_m1(xs, *pack)
            _torch_chain(xs, chains)
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            lf.lora4_m1(xs, *pack)
        e.record(); torch.cuda.synchronize()
        t_f = s.elapsed_time(e) / iters * 1000  # us
        s.record()
        for _ in range(iters):
            _torch_chain(xs, chains)
        e.record(); torch.cuda.synchronize()
        t_t = s.elapsed_time(e) / iters * 1000
        print(f"  {tag:>28} | {t_f:9.2f} | {t_t:9.2f} | {t_t/t_f:5.2f}x  "
              f"(H={H} ranks={ranks})")


def main():
    if not torch.cuda.is_available():
        print("no CUDA"); sys.exit(1)
    print(f"torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}")
    print("building rwkv7_lora extension (JIT)...")
    if not lf.available():
        print("BUILD FAILED"); sys.exit(2)
    print("build OK")
    ok = check_correctness()
    bench()
    print(f"\nRESULT: lora4_m1={'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
