"""Byte-exact gate for the epilogue-fused FFN sqrelu GEMV (gemv_m1_sqrelu, F0052).

The model's channel-mix does ``act = torch.relu(key(xk)) ** 2`` as a GEMV followed by
2 standalone elementwise launches (relu, pow). F0052 folds the relu^2 into the GEMV's
store (rwkv7_fast.cu::gemv_m1_sqrelu_kernel). This gate proves the fused kernel is
BIT-IDENTICAL (torch.equal) to the two-step path it replaces, so enabling
RWKV_FUSED_SQRELU changes zero output tokens.

Two comparisons:
  (A) op-level, config-matched: gemv_m1_sqrelu_cfg(x,w,t,ot) == relu(gemv_m1_cfg(x,w,t,ot))**2
      -- same (threads,out_tile) on both arms, so the fp32 accumulation (hence the fp16
         k the activation reads) is identical; this isolates the epilogue arithmetic.
  (B) adapter-level, the real call path: fast_linear.gemv_m1_sqrelu(x,w) ==
      relu(fast_linear.gemv_m1(x,w))**2 -- both route through _select_config, so they
      pick the SAME config; this is exactly what models/rwkv7.py dispatches.

The load-bearing question is whether the in-kernel relu + fp32 square + single round to
fp16 reproduces aten's relu (clamp_min) then pow(.,2) (b*b in fp32 opmath) exactly. This
gate is the arbiter; a FAIL means the fusion is not shippable under the project's
per-kernel exactness rule and the model keeps the torch path.

Run on a GPU box: python bench/test_sqrelu_gate.py
"""
import sys
from pathlib import Path
import torch

try:
    from sglang.srt.layers.attention.rwkv7_kernels import fast_linear
except Exception:
    OV = Path(__file__).resolve().parents[1] / "sglang_overlay"
    sys.path.insert(0, str(OV))
    from sglang.srt.layers.attention.rwkv7_kernels import fast_linear  # noqa: E402

dev = torch.device("cuda")
assert fast_linear.available(), "rwkv7_fast JIT ext failed to build"
_fast = torch.ops.rwkv7_fast


def _md(a, b):
    """Max abs diff over FINITE positions (relu(k)^2 saturates to +inf in fp16 at large
    scale; there inf==inf is bit-equal but inf-inf==nan, which would mask the metric).
    torch.equal below is the real gate; this is only the human-readable magnitude."""
    af, bf = a.float(), b.float()
    fin = torch.isfinite(af) & torch.isfinite(bf)
    if fin.any():
        return (af[fin] - bf[fin]).abs().max().item()
    return 0.0


def _ovf(a):
    return (~torch.isfinite(a.float())).float().mean().item()


def case_op(N, K, threads, out_tile, scale, seed, label):
    """(A) config-matched op-level: isolates the epilogue (same accumulation both arms)."""
    torch.manual_seed(seed)
    x = (torch.randn(1, K, dtype=torch.float16, device=dev) * scale)
    w = (torch.randn(N, K, dtype=torch.float16, device=dev) * scale)
    fused = _fast.gemv_m1_sqrelu_cfg(x, w, threads, out_tile)
    ref = torch.relu(_fast.gemv_m1_cfg(x, w, threads, out_tile)) ** 2
    ok = torch.equal(fused, ref)
    print(f"[{label}] N{N:5d} K{K:5d} <{threads},{out_tile}> scale={scale:5.1f} "
          f"seed={seed} md={_md(fused, ref):.2e} ovf={_ovf(ref):.2f} "
          f"=> {'PASS' if ok else 'FAIL'}")
    return ok


def case_adapter(N, K, scale, seed, label):
    """(B) the real model call path (both arms via _select_config -> same config)."""
    torch.manual_seed(seed)
    x = (torch.randn(1, K, dtype=torch.float16, device=dev) * scale)
    w = (torch.randn(N, K, dtype=torch.float16, device=dev) * scale)
    fused = fast_linear.gemv_m1_sqrelu(x, w)
    ref = torch.relu(fast_linear.gemv_m1(x, w)) ** 2
    ok = torch.equal(fused, ref)
    print(f"[{label}] N{N:5d} K{K:5d} scale={scale:5.1f} seed={seed} "
          f"md={_md(fused, ref):.2e} ovf={_ovf(ref):.2f} => {'PASS' if ok else 'FAIL'}")
    return ok


all_ok = True

# Real FFN key-projection shapes (N=intermediate_size, K=hidden): 1.5B, 7.2B, 0.1B,
# plus an odd K (still %4) and a small even N to exercise the out_tile fallbacks.
SHAPES = [(8192, 2048), (16384, 4096), (3072, 768), (2560, 1024), (6144, 2048)]
CFGS = [(64, 1), (64, 2), (64, 4), (128, 1), (128, 2), (128, 4), (256, 2), (256, 4)]

print("== (A) op-level, config-matched (epilogue isolation) ==")
for (N, K) in SHAPES:
    for (t, ot) in CFGS:
        if N % ot != 0:
            continue
        # small=linear region (most values > 0, square active); large=wide dynamic
        # range so many rows saturate to 0 (relu branch) and others to large squares.
        for scale in [0.5, 2.0, 8.0]:
            for seed in range(3):
                all_ok &= case_op(N, K, t, ot, scale, seed, "opA")

print("\n== (B) adapter-level (real model dispatch path) ==")
for (N, K) in SHAPES:
    for scale in [0.5, 2.0, 8.0, 20.0]:
        for seed in range(3):
            all_ok &= case_adapter(N, K, scale, seed, "adpB")

# Knife-edge: x/w engineered so the pre-activation k straddles 0 (relu boundary) and so
# relu(k)^2 lands near fp16 rounding midpoints, where a sub-ULP fp32 diff in the square
# would flip the stored fp16. A single row of K identical entries makes k analytically
# predictable; sweeping x across the informative range tiles the fp16 grid near 0.
print("\n== knife-edge (relu boundary + fp16 square-rounding midpoints) ==")
N, K = 8192, 2048
torch.manual_seed(7)
w = torch.randn(N, K, dtype=torch.float16, device=dev) * 0.03  # small -> k spans ~[-,+] near 0
for scale in [0.05, 0.1, 0.25, 0.5, 1.0, 3.0]:
    x = (torch.linspace(-scale, scale, K, device=dev).half()).view(1, K)
    fused = fast_linear.gemv_m1_sqrelu(x, w)
    ref = torch.relu(fast_linear.gemv_m1(x, w)) ** 2
    ok = torch.equal(fused, ref)
    zf = (ref == 0).float().mean().item()
    all_ok &= ok
    print(f"[knife] scale={scale:5.2f} zero_frac(ref)={zf:.3f} "
          f"md={_md(fused, ref):.2e} => {'PASS' if ok else 'FAIL'}")

print("\nALL", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
