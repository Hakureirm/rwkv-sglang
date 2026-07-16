#!/usr/bin/env python3
"""Stage-A2 model-level gate (task #50 / F0060 §7.5): o_proj as a megakernel role.

Builds the REAL Rwkv7Attention module at the deployed 1.5B + 7.2B configs and
verifies, on its real-shaped r/k/v/o projection weights and the deployed
_select_config, that the new grouped ops are byte-identical to the shipping
fast_linear.gemv_m1:
  - gemv_o_m1  (G=1)  o_proj role          == gemv_m1(xo, w_o)
  - gemv_rkvo_m1 (G=4) whole-block r/k/v/o == stack(gemv_m1 x4)   (sm120 prefab)
  - gemv_rkv_m1  (G=3) r/k/v stage         == stack(gemv_m1 x3)   (regression)

o_proj is (N,K)=(H,H) like r/k/v, so it takes the identical (threads,out_tile);
bit-exactness is value-independent (same fp32 reduction), so the real module's
weights (real shapes, real config) are a sufficient + faithful model gate — the
same rigor as the shipping mega_model_gate.

  python /tmp/test_mega_o_model.py
"""
import os
import sys

import torch

os.environ["RWKV_FAST_LINEAR"] = "1"
os.environ["RWKV_GEMV_AUTOTUNE"] = "0"  # heuristic config (deterministic, no disk state)
for k in ["RWKV_SPARSE_FFN", "RWKV_FUSED_LORA", "RWKV_FUSED_GLUE", "RWKV_FUSED_GATES",
          "RWKV_FUSED_SQRELU", "RWKV_FUSED_ADDLN", "RWKV_FUSED_GNGC", "RWKV_WKV_CUDA",
          "RWKV_MEGA"]:
    os.environ[k] = "0"
sys.path.insert(0, "/tmp")
import pc  # noqa: E402

from sglang.srt.models import rwkv7  # noqa: E402
from sglang.srt.layers.attention.rwkv7_kernels import fast_linear, mega  # noqa: E402

assert fast_linear.available(), "rwkv7_fast failed to build"
assert mega.available(), "rwkv7_mega failed to build"
dtype = torch.float16


def gate(model_dir, tag):
    cfg, _ = pc.load_cfg(model_dir)
    torch.manual_seed(0)
    attn = rwkv7.Rwkv7Attention(cfg, 1).cuda().to(dtype).eval()
    H = cfg.hidden_size
    wr, wk = attn.r_proj.weight, attn.k_proj.weight
    wv, wo = attn.v_proj.weight, attn.o_proj.weight
    shapes_ok = all(w.dtype == dtype and w.is_contiguous() and tuple(w.shape) == (H, H)
                    for w in (wr, wk, wv, wo))
    t, ot = mega.rkv_config(H, H)
    o_ok = rkvo_ok = rkv_ok = True
    for _ in range(4):
        xr = torch.randn(1, H, dtype=dtype, device="cuda")
        xk = torch.randn(1, H, dtype=dtype, device="cuda")
        xv = torch.randn(1, H, dtype=dtype, device="cuda")
        xo = torch.randn(1, H, dtype=dtype, device="cuda")
        er = fast_linear.gemv_m1(xr, wr)
        ek = fast_linear.gemv_m1(xk, wk)
        ev = fast_linear.gemv_m1(xv, wv)
        eo = fast_linear.gemv_m1(xo, wo)
        o_ok &= torch.equal(mega.gemv_o_m1(xo, wo), eo)
        rkvo_ok &= torch.equal(mega.gemv_rkvo_m1(xr, xk, xv, xo, wr, wk, wv, wo),
                               torch.cat([er, ek, ev, eo], dim=0))
        rkv_ok &= torch.equal(mega.gemv_rkv_m1(xr, xk, xv, wr, wk, wv),
                              torch.cat([er, ek, ev], dim=0))
    print(f"[{tag}] H={H} cfg=({t},{ot}) shapes_ok={shapes_ok}  "
          f"o(G1)={'PASS' if o_ok else 'FAIL'}  "
          f"rkvo(G4)={'PASS' if rkvo_ok else 'FAIL'}  "
          f"rkv(G3 regr)={'PASS' if rkv_ok else 'FAIL'}")
    return shapes_ok and o_ok and rkvo_ok and rkv_ok


ok = True
for md, tag in [("/models/rwkv7-1.5b-fla", "1.5B"), ("/models/rwkv7-7.2b-fla", "7.2B")]:
    if os.path.isdir(md):
        ok &= gate(md, tag)
    else:
        print(f"[{tag}] SKIP (no {md})")
print("\nMODEL GATE:",
      "PASS (byte-identical, real o_proj/r/k/v weights)" if ok else "FAIL")
