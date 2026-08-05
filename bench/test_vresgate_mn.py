"""Byte-gate for W1'': the batched M-gate path's gate activations.

Compares the shipped torch chain on strided [T,C,H] columns against
ln_fused.vres_gates on the transposed contiguous view, for every T the gate
admits and both layer roles. Zero differing bytes is the bar -- the greedy
oracle downstream cannot see a difference smaller than a token, so the
comparison has to be at the bit level here.

Run against the deployed tree, e.g. PYTHONPATH=<sglang checkout>/python.
"""
import sys

import torch

from sglang.srt.layers.attention.rwkv7_kernels import ln_fused

_INV_SQRT_E = 0.6065306597126334


def reference(lo_mn, v, v_first, layer_id):
    """Exactly what the model did before W1''."""
    w_log = -torch.sigmoid(lo_mn[:, 0]) * _INV_SQRT_E
    a = torch.sigmoid(lo_mn[:, 1])
    g = lo_mn[:, 2].contiguous()
    if layer_id != 0:
        v = v + (v_first - v) * torch.sigmoid(lo_mn[:, 3])
    return w_log, a, g, v


def fused(lo_mn, v, v_first, layer_id):
    chans = lo_mn.transpose(0, 1).contiguous()
    w_log, a, v = ln_fused.vres_gates(
        chans[0], chans[1],
        chans[3] if layer_id != 0 else None,
        v, v_first if layer_id != 0 else None, _INV_SQRT_E,
    )
    return w_log, a, chans[2], v


def bytes_differ(x, y):
    return int((x.view(torch.int16) != y.view(torch.int16)).sum().item())


# available() is what JIT-builds and registers the op; the model guards on it
# before every call, so the test must too or it measures an import error.
assert ln_fused.available(), "ln_fused extension did not build"

torch.manual_seed(0)
H = 2048
bad = 0
print(f"{'layer':>6} {'T':>4} {'C':>3}   w_log     a       g       v")
for layer_id in (0, 1):
    C = 4 if layer_id else 3
    for T in (2, 3, 4, 5, 6, 7, 8, 12, 16):
        lo_mn = (torch.randn(T, C, H, device="cuda", dtype=torch.float16) * 2).contiguous()
        v = torch.randn(T, H, device="cuda", dtype=torch.float16).contiguous()
        v_first = torch.randn(T, H, device="cuda", dtype=torch.float16).contiguous()
        r = reference(lo_mn, v, v_first, layer_id)
        f = fused(lo_mn, v, v_first, layer_id)
        diffs = [bytes_differ(a.contiguous(), b.contiguous()) for a, b in zip(r, f)]
        bad += sum(diffs)
        print(f"{layer_id:>6} {T:>4} {C:>3}   " + "  ".join(f"{d:>6}" for d in diffs))
print("\nTOTAL differing bytes:", bad, "->", "PASS" if bad == 0 else "FAIL")
sys.exit(0 if bad == 0 else 1)
