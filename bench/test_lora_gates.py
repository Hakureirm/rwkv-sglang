"""Byte-exact gate for fused_lora_gates (LoRA-output gate activations, F0051).

Compares the fused triton kernel (fused.py Kernel D) against the EXACT torch op
sequence the model runs when the fusion is off:
    w_log = -torch.sigmoid(lo[0:1]) * _INV_SQRT_E
    a     =  torch.sigmoid(lo[1:2])
    v     =  v + (v_first - v) * torch.sigmoid(lo[3:4])   # layer>0 only
with torch.equal (bit-for-bit). The load-bearing question is whether triton's
tl.exp matches CUDA expf tightly enough that the fp16-rounded sigmoid is identical
to torch's — this gate is the arbiter; if it FAILS the fusion is not shippable
under the project's per-kernel exactness rule and the caller keeps the torch path.

Run on a GPU box: python bench/test_lora_gates.py
"""
import sys
from pathlib import Path
import torch

# Import the deployed fused module. Prefer the installed overlay (== production, i.e.
# the overlay merged into sglang's site-packages); fall back to a raw repo checkout's
# sglang_overlay/ on PYTHONPATH.
try:
    from sglang.srt.layers.attention.rwkv7_kernels import fused
except Exception:
    OV = Path(__file__).resolve().parents[1] / "sglang_overlay"
    sys.path.insert(0, str(OV))
    from sglang.srt.layers.attention.rwkv7_kernels import fused  # noqa: E402

dev = torch.device("cuda")
_INV_SQRT_E = 0.6065306597126334


def torch_ref(lo, v, v_first, has_v):
    """The exact model gate math (fusion-off path)."""
    w_log = -torch.sigmoid(lo[0:1]) * _INV_SQRT_E
    a = torch.sigmoid(lo[1:2])
    if has_v:
        v = v + (v_first - v) * torch.sigmoid(lo[3:4])
    return w_log, a, v


def run_case(H, C, scale, seed, label):
    has_v = C == 4
    torch.manual_seed(seed)
    lo = (torch.randn(C, H, dtype=torch.float16, device=dev) * scale)
    v = (torch.randn(1, H, dtype=torch.float16, device=dev) * scale)
    v_first = (torch.randn(1, H, dtype=torch.float16, device=dev) * scale)

    w_ref, a_ref, v_ref = torch_ref(lo, v, v_first, has_v)
    w_f, a_f, v_f = fused.fused_lora_gates(lo, v, v_first, has_v)

    ok_w = torch.equal(w_ref, w_f)
    ok_a = torch.equal(a_ref, a_f)
    ok_v = torch.equal(v_ref, v_f) if has_v else True
    ok = ok_w and ok_a and ok_v

    def md(x, y):
        return (x.float() - y.float()).abs().max().item()
    print(f"[{label}] H={H} C={C} scale={scale:5.1f} seed={seed} "
          f"| w:{'OK ' if ok_w else 'X'}({md(w_ref,w_f):.2e}) "
          f"a:{'OK ' if ok_a else 'X'}({md(a_ref,a_f):.2e}) "
          f"v:{'OK ' if ok_v else 'X'}({md(v_ref,v_f) if has_v else 0:.2e}) "
          f"=> {'PASS' if ok else 'FAIL'}")
    return ok


all_ok = True
# H sweep (1.5B=2048, 7.2B=4096, 0.1b=768) x C (3=layer0, 4=layer>0) x input scale
# (small=linear sigmoid region; large=saturating tails; both stress the round).
for H in [768, 2048, 4096]:
    for C in [3, 4]:
        for scale in [0.5, 2.0, 8.0, 30.0]:
            for seed in range(4):
                all_ok &= run_case(H, C, scale, seed, "sweep")

# Knife-edge: inputs engineered so sigmoid lands near fp16 rounding midpoints
# (where a sub-ULP fp32 diff between tl.exp and expf would flip the fp16 result).
torch.manual_seed(123)
H = 2048
# dense grid of logits across the informative range, tiled to fill H
base = torch.linspace(-12, 12, H, device=dev).half()
lo = torch.stack([base, base.flip(0), (base * 0.37), (base * -0.71)], 0)
v = (torch.randn(1, H, dtype=torch.float16, device=dev))
v_first = (torch.randn(1, H, dtype=torch.float16, device=dev))
for C in [3, 4]:
    w_ref, a_ref, v_ref = torch_ref(lo[:C], v, v_first, C == 4)
    w_f, a_f, v_f = fused.fused_lora_gates(lo[:C].contiguous(), v, v_first, C == 4)
    ok = torch.equal(w_ref, w_f) and torch.equal(a_ref, a_f) and (C == 3 or torch.equal(v_ref, v_f))
    all_ok &= ok
    print(f"[knife-edge] C={C} => {'PASS' if ok else 'FAIL'}")

print("\nALL", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
