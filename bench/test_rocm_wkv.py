#!/usr/bin/env python3
"""ROCm smoke and numerical gate for the portable RWKV-7 Triton recurrence."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


def load_wkv(repo: Path):
    path = repo / "sglang_mainline/srt/layers/attention/rwkv7_kernels/wkv_recurrent.py"
    spec = importlib.util.spec_from_file_location("rwkv7_rocm_wkv", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.wkv_recurrent


def reference(r, w, k, v, kk, a, initial_state, lengths):
    nseq, heads, key_dim, value_dim = initial_state.shape
    state = initial_state.float().clone()
    out = torch.empty_like(v)
    offset = 0
    packed = r.shape[0] == 1 and len(lengths) > 1
    for n in range(nseq):
        for t in range(lengths[n]):
            src_b, src_t = (0, offset + t) if packed else (n, t)
            for h in range(heads):
                rr = r[src_b, src_t, h].float()
                ww = w[src_b, src_t, h].float().exp()
                kv = k[src_b, src_t, h].float()
                vv = v[src_b, src_t, h].float()
                ka = kk[src_b, src_t, h].float()
                aa = a[src_b, src_t, h].float()
                old = state[n, h]
                sa = torch.sum((-ka)[:, None] * old, dim=0)
                new = ww[:, None] * old + (ka * aa)[:, None] * sa[None, :] + kv[:, None] * vv[None, :]
                state[n, h] = new
                out[src_b, src_t, h] = torch.sum(new * rr[:, None], dim=0).to(out.dtype)
        offset += lengths[n]
    return out, state


def make_inputs(shape, device):
    generator = torch.Generator(device=device).manual_seed(20260805)
    r = torch.randn(shape, device=device, dtype=torch.float16, generator=generator) * 0.1
    w = -torch.rand(shape, device=device, dtype=torch.float16, generator=generator) * 0.2
    k = torch.randn(shape, device=device, dtype=torch.float16, generator=generator) * 0.1
    v = torch.randn(shape, device=device, dtype=torch.float16, generator=generator) * 0.1
    kk = torch.nn.functional.normalize(
        torch.randn(shape, device=device, dtype=torch.float16, generator=generator).float(), dim=-1
    ).to(torch.float16)
    a = torch.sigmoid(torch.randn(shape, device=device, dtype=torch.float16, generator=generator))
    return r, w, k, v, kk, a


def run_case(wkv, *, lengths, heads=2, dim=64):
    device = torch.device("cuda")
    packed = len(lengths) > 1
    shape = (1, sum(lengths), heads, dim) if packed else (len(lengths), lengths[0], heads, dim)
    inputs = make_inputs(shape, device)
    initial = torch.randn(
        len(lengths), heads, dim, dim, device=device, dtype=torch.float32
    ) * 0.01
    cu = None
    if packed:
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
    got_o, got_state = wkv(
        *inputs,
        initial_state=initial.clone(),
        output_final_state=True,
        cu_seqlens=cu,
    )
    ref_o, ref_state = reference(*inputs, initial, lengths)
    out_err = (got_o.float() - ref_o.float()).abs().max().item()
    state_err = (got_state.float() - ref_state.float()).abs().max().item()
    if not torch.allclose(got_o.float(), ref_o.float(), atol=2e-3, rtol=2e-3):
        raise AssertionError(f"output mismatch: max_abs={out_err:.6g}")
    if not torch.allclose(got_state.float(), ref_state.float(), atol=2e-5, rtol=2e-4):
        raise AssertionError(f"state mismatch: max_abs={state_err:.6g}")
    return out_err, state_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cuda", action="store_true", help="run on NVIDIA for test development")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA/HIP device is visible")
    if torch.version.hip is None and not args.allow_cuda:
        raise SystemExit("this gate requires PyTorch ROCm (use --allow-cuda only for development)")
    repo = Path(__file__).resolve().parents[1]
    wkv = load_wkv(repo)
    cases = {
        "decode-b2": run_case(wkv, lengths=[1, 1]),
        "varlen-3-2": run_case(wkv, lengths=[3, 2]),
    }
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "unknown").split(":", 1)[0]
    print(f"PASS device={torch.cuda.get_device_name(0)} arch={arch} hip={torch.version.hip}")
    for name, (out_err, state_err) in cases.items():
        print(f"{name}: output_max_abs={out_err:.6g} state_max_abs={state_err:.6g}")


if __name__ == "__main__":
    main()
