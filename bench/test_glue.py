"""Byte-exact gate for shift_lerp6 / shift_lerp1 (ADR-0005 R2 fused glue).

Verifies the fused paged-token-shift + lerp kernels are byte-identical
(torch.equal) to the current token_shift(clone+scatter) + fused_lerp6/lerp1,
including the conv-state scatter. Replicates fused.py:_lerp6_kernel's exact fp16
rounding (d=round16(sh-x); prod=round16(mix*d); o=round16(x+prod)).

Run on box: python bench/test_glue.py
"""
import sys
from pathlib import Path
import torch

OV = Path(__file__).resolve().parents[1] / "sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels"
from torch.utils.cpp_extension import load  # noqa: E402
load(name="rwkv7_glue", sources=[str(OV / "cuda" / "rwkv7_glue.cu")], is_python_module=False,
     verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-Xptxas", "-O3"])

dev = torch.device("cuda")
torch.manual_seed(0)
H = 2048
S = 64  # conv pool slots


def r16(t):  # round to fp16 then back to fp32 (matches .to(DT).to(f32))
    return t.half().float()


def ref_lerp6(normed, prev, mix6):
    xf, shf = normed.float(), prev.float()
    d = r16(shf - xf)
    outs = []
    for j in range(6):
        prod = r16(mix6[j].float() * d)
        outs.append((xf + prod).half())
    return torch.stack(outs, 0)  # [6,T,H]


def ref_lerp1(normed, prev, x_k):
    xf, shf = normed.float(), prev.float()
    d = r16(shf - xf)
    prod = r16(x_k.float() * d)
    return (xf + prod).half()  # [T,H]


all_ok = True
for T in [1, 2, 4, 8, 32]:
    ci = torch.randperm(S, device=dev)[:T].to(torch.int32)          # distinct slots
    normed = (torch.randn(T, H, dtype=torch.float16, device=dev) * 0.5)
    mix6 = (torch.randn(6, H, dtype=torch.float16, device=dev) * 0.3)
    x_k = (torch.randn(H, dtype=torch.float16, device=dev) * 0.3)
    conv0 = torch.randn(S, H, 1, dtype=torch.float32, device=dev) * 0.4  # conv is fp32

    # reference: prev = conv.to(fp16) (token_shift return .to(x.dtype)); conv <- normed.to(fp32)
    conv_ref = conv0.clone(); conv_ref[ci.long(), :, 0] = normed.float()

    # --- shift_lerp6 ---
    prev = conv0[ci.long(), :, 0].to(torch.float16)   # round fp32->fp16
    ref6 = ref_lerp6(normed, prev, mix6)
    conv_k = conv0.clone()
    out6 = torch.ops.rwkv7_glue.shift_lerp6(normed, mix6, ci, conv_k)
    ok6 = torch.equal(out6, ref6) and torch.equal(conv_k, conv_ref)

    # --- shift_lerp1 ---
    prev1 = conv0[ci.long(), :, 0].to(torch.float16)
    ref1 = ref_lerp1(normed, prev1, x_k)
    conv_k1 = conv0.clone()
    out1 = torch.ops.rwkv7_glue.shift_lerp1(normed, x_k, ci, conv_k1)
    ok1 = torch.equal(out1, ref1) and torch.equal(conv_k1, conv_ref)

    print(f"T={T:3d}: shift_lerp6 {'EXACT' if ok6 else 'FAIL'} | shift_lerp1 {'EXACT' if ok1 else 'FAIL'}")
    all_ok = all_ok and ok6 and ok1

print("\nRESULT:", "ALL EXACT — R2 kernels byte-match token_shift+lerp" if all_ok else "FAILED")
sys.exit(0 if all_ok else 1)
