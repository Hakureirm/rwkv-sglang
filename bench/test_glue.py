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

# --- pad-slot cases: padded cuda-graph replay fills the tail of cache_indices
# with PAD_SLOT_ID = -1 (possibly MANY -1 entries). Pad rows must (a) not touch
# conv at all, (b) come out zeroed, (c) leave valid rows byte-exact.
for T, n_pad in [(4, 1), (8, 5), (3, 3)]:  # incl. all-pad batch
    n_valid = T - n_pad
    ci = torch.full((T,), -1, dtype=torch.int32, device=dev)
    if n_valid:
        ci[:n_valid] = torch.randperm(S, device=dev)[:n_valid].to(torch.int32)
    normed = (torch.randn(T, H, dtype=torch.float16, device=dev) * 0.5)
    mix6 = (torch.randn(6, H, dtype=torch.float16, device=dev) * 0.3)
    x_k = (torch.randn(H, dtype=torch.float16, device=dev) * 0.3)
    conv0 = torch.randn(S, H, 1, dtype=torch.float32, device=dev) * 0.4

    v = ci[:n_valid].long()
    conv_ref = conv0.clone()
    if n_valid:
        conv_ref[v, :, 0] = normed[:n_valid].float()

    conv_k = conv0.clone()
    out6 = torch.ops.rwkv7_glue.shift_lerp6(normed, mix6, ci, conv_k)
    ok6 = torch.equal(conv_k, conv_ref) and torch.equal(out6[:, n_valid:], torch.zeros_like(out6[:, n_valid:]))
    if n_valid:
        prev = conv0[v, :, 0].to(torch.float16)
        ok6 = ok6 and torch.equal(out6[:, :n_valid], ref_lerp6(normed[:n_valid], prev, mix6))

    conv_k1 = conv0.clone()
    out1 = torch.ops.rwkv7_glue.shift_lerp1(normed, x_k, ci, conv_k1)
    ok1 = torch.equal(conv_k1, conv_ref) and torch.equal(out1[n_valid:], torch.zeros_like(out1[n_valid:]))
    if n_valid:
        prev1 = conv0[v, :, 0].to(torch.float16)
        ok1 = ok1 and torch.equal(out1[:n_valid], ref_lerp1(normed[:n_valid], prev1, x_k))

    print(f"T={T:3d} pads={n_pad}: shift_lerp6 {'EXACT' if ok6 else 'FAIL'} | shift_lerp1 {'EXACT' if ok1 else 'FAIL'} (pad rows zeroed, conv untouched)")
    all_ok = all_ok and ok6 and ok1

# out-of-range index (>= S) must also be treated as a pad, not an OOB write
ci = torch.tensor([0, S, 2 * S], dtype=torch.int32, device=dev)
normed = torch.randn(3, H, dtype=torch.float16, device=dev)
mix6 = torch.randn(6, H, dtype=torch.float16, device=dev)
conv0 = torch.randn(S, H, 1, dtype=torch.float32, device=dev)
conv_k = conv0.clone()
out6 = torch.ops.rwkv7_glue.shift_lerp6(normed, mix6, ci, conv_k)
conv_ref = conv0.clone(); conv_ref[0, :, 0] = normed[0].float()
ok_oob = torch.equal(conv_k, conv_ref) and torch.equal(out6[:, 1:], torch.zeros_like(out6[:, 1:]))
print(f"out-of-range idx: {'EXACT' if ok_oob else 'FAIL'} (rows >= S treated as pad)")
all_ok = all_ok and ok_oob

print("\nRESULT:", "ALL EXACT — R2 kernels byte-match token_shift+lerp (incl. pad slots)" if all_ok else "FAILED")
sys.exit(0 if all_ok else 1)
