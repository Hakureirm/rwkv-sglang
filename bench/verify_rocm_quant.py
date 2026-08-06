#!/usr/bin/env python3
"""Numerical, chunk-invariance, storage, and speed gate for ROCm W8/W4.

This gate compares the fused HIP projection with an independently dequantized
PyTorch reference.  It does not compare W4 with dense model output: that would
mix kernel correctness with the expected lossy-quantization error.  Model-level
quality is gated separately.

Example:

  PYTHONPATH=/path/to/sglang/python \
  PYTORCH_ROCM_ARCH=gfx1100 \
  python bench/verify_rocm_quant.py \
      --batches 1,8,64,128,256 \
      --output bench/results/rocm_quant.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from sglang.srt.layers.attention.rwkv7_kernels import rocm_quant
from sglang.srt.layers.attention.rwkv7_kernels import w4_linear


GROUP = 64


def quantize_w8(weight):
    n, k = weight.shape
    grouped = weight.float().view(n, k // GROUP, GROUP)
    scale = (grouped.abs().amax(dim=2) / 127.0).clamp_min(1e-8)
    qweight = (
        (grouped / scale[:, :, None])
        .round()
        .clamp_(-127, 127)
        .to(torch.int8)
        .view(n, k)
        .contiguous()
    )
    return qweight, scale.to(torch.float16).contiguous()


def quantize_w4(weight):
    n, k = weight.shape
    grouped = weight.float().view(n, k // GROUP, GROUP)
    scale = (grouped.abs().amax(dim=2) / 7.0).clamp_min(1e-8)
    q = (
        (grouped / scale[:, :, None])
        .round()
        .clamp_(-7, 7)
        .to(torch.int16)
        .view(n, k)
    )
    nibble = (q & 0xF).to(torch.uint8)
    packed = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).contiguous()
    return packed, scale.to(torch.float16).contiguous()


def dequant_w8(qweight, scale, dtype):
    n, k = qweight.shape
    return (
        qweight.view(n, k // GROUP, GROUP).float()
        * scale.float()[:, :, None]
    ).view(n, k).to(dtype)


def dequant_w4(qweight, scale, dtype):
    n = qweight.shape[0]
    k = qweight.shape[1] * 2
    low = (qweight & 0xF).to(torch.int16)
    high = (qweight >> 4).to(torch.int16)
    low -= (low & 8) << 1
    high -= (high & 8) << 1
    q = torch.empty((n, k), dtype=torch.int16, device=qweight.device)
    q[:, 0::2] = low
    q[:, 1::2] = high
    return (
        q.view(n, k // GROUP, GROUP).float()
        * scale.float()[:, :, None]
    ).view(n, k).to(dtype)


def elapsed_us(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def parse_shapes(value):
    shapes = []
    for item in value.split(","):
        k, n = item.lower().split("x")
        shapes.append((int(k), int(n)))
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="768x768,768x3072,2048x2048,2048x8192",
        help="comma-separated KxN projection shapes",
    )
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if torch.version.hip is None or not torch.cuda.is_available():
        raise SystemExit("ROCm GPU is required")

    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    selected_dtypes = [dtypes[name] for name in args.dtypes.split(",")]
    batches = [int(value) for value in args.batches.split(",")]
    shapes = parse_shapes(args.shapes)
    rows = []
    failures = []

    torch.manual_seed(0)
    print(
        f"GPU={torch.cuda.get_device_name()} HIP={torch.version.hip} "
        f"W8 storage={100 * (0.5 + 1 / GROUP):.2f}% "
        f"W4 storage={100 * (0.25 + 1 / GROUP):.2f}% of fp16"
    )
    print(
        f"{'dtype':>8} {'M':>3} {'K':>5} {'N':>5} {'bits':>4} | "
        f"{'rel error':>10} {'chunk exact':>11} | "
        f"{'dense us':>9} {'fallback':>9} {'quant us':>9} {'vs fb':>7}"
    )
    print("-" * 111)

    for dtype in selected_dtypes:
        rel_limit = 6e-4 if dtype == torch.float16 else 4e-3
        for k, n in shapes:
            weight = (torch.randn((n, k), device="cuda") * 0.02).to(dtype)
            q8, s8 = quantize_w8(weight)
            q4, s4 = quantize_w4(weight)
            refs = {
                8: dequant_w8(q8, s8, dtype),
                4: dequant_w4(q4, s4, dtype),
            }
            for m in batches:
                x = torch.randn((m, k), device="cuda", dtype=dtype)
                for bits, qweight, scale, op in (
                    (8, q8, s8, rocm_quant.linear_w8),
                    (4, q4, s4, rocm_quant.linear_w4),
                ):
                    supported = (
                        rocm_quant.w8_supported(x, qweight)
                        if bits == 8
                        else rocm_quant.w4_supported(x, qweight)
                    )
                    if not supported:
                        rows.append(
                            {
                                "dtype": str(dtype).removeprefix("torch."),
                                "batch": m,
                                "k": k,
                                "n": n,
                                "bits": bits,
                                "status": "fallback",
                                "reason": "outside measured fused dispatch gate",
                            }
                        )
                        print(
                            f"{str(dtype).removeprefix('torch.'):>8} {m:>3} "
                            f"{k:>5} {n:>5} {bits:>4} | {'fallback':>73}"
                        )
                        continue
                    got = op(x, qweight, scale)
                    ref = torch.nn.functional.linear(x, refs[bits])
                    rel = float(
                        (got.float() - ref.float()).norm()
                        / (ref.float().norm() + 1e-9)
                    )
                    # M>8 uses one fixed 64-row Triton tile.  Verify that the
                    # result is invariant to a production chunk boundary while
                    # keeping the independent small-batch HIP numerical gate.
                    if m <= rocm_quant.MAX_BATCH:
                        chunked = torch.cat(
                            [op(x[row : row + 1], qweight, scale) for row in range(m)],
                            dim=0,
                        )
                    else:
                        chunked = torch.cat(
                            [op(part, qweight, scale) for part in x.split(64)],
                            dim=0,
                        )
                    chunk_exact = torch.equal(got, chunked)

                    dense_us = elapsed_us(
                        lambda weight=weight: torch.nn.functional.linear(x, weight),
                        args.warmup,
                        args.iterations,
                    )
                    fallback_us = elapsed_us(
                        lambda: torch.nn.functional.linear(
                            x,
                            (
                                w4_linear.dequant_w8(qweight, scale).to(dtype)
                                if bits == 8
                                else w4_linear.dequant(qweight, scale).to(dtype)
                            ),
                        ),
                        args.warmup,
                        args.iterations,
                    )
                    quant_us = elapsed_us(
                        lambda: op(x, qweight, scale),
                        args.warmup,
                        args.iterations,
                    )
                    speedup = dense_us / quant_us
                    fallback_speedup = fallback_us / quant_us
                    ok = rel <= rel_limit and chunk_exact
                    # The standalone speed gate is strict for single-stream
                    # decode.  Batch-8 production speed is gated end to end,
                    # because individual small projections can be launch-bound.
                    if m == 1:
                        ok = ok and speedup >= 1.0
                    elif m > rocm_quant.MAX_BATCH:
                        ok = ok and fallback_speedup >= 1.0
                    if not ok:
                        failures.append(
                            f"{dtype}/{m}/{k}x{n}/w{bits}: "
                            f"rel={rel:.3e} exact={chunk_exact} "
                            f"dense_speed={speedup:.3f} fallback_speed={fallback_speedup:.3f}"
                        )
                    row = {
                        "dtype": str(dtype).removeprefix("torch."),
                        "batch": m,
                        "k": k,
                        "n": n,
                        "bits": bits,
                        "relative_error": rel,
                        "chunk_exact": chunk_exact,
                        "dense_us": dense_us,
                        "fallback_us": fallback_us,
                        "quant_us": quant_us,
                        "speedup": speedup,
                        "speedup_vs_fallback": fallback_speedup,
                        "pass": ok,
                    }
                    rows.append(row)
                    print(
                        f"{row['dtype']:>8} {m:>3} {k:>5} {n:>5} {bits:>4} | "
                        f"{rel:>10.3e} {str(chunk_exact):>11} | "
                        f"{dense_us:>9.2f} {fallback_us:>9.2f} "
                        f"{quant_us:>9.2f} {fallback_speedup:>6.2f}x"
                    )
            del weight, q8, s8, q4, s4, refs

    hip_extension_required = any(m <= rocm_quant.MAX_BATCH for m in batches)
    summary = {
        "gpu": torch.cuda.get_device_name(),
        "hip": torch.version.hip,
        "group_size": GROUP,
        "storage_ratio_vs_fp16": {"w8": 0.5 + 1 / GROUP, "w4": 0.25 + 1 / GROUP},
        "hip_extension_loaded": rocm_quant._HIP_EXT_LOADED,
        "hip_extension_required": hip_extension_required,
        "rows": rows,
        "failures": failures,
        "status": (
            "pass"
            if not failures
            and (rocm_quant._HIP_EXT_LOADED or not hip_extension_required)
            else "fail"
        ),
        "timestamp_unix": time.time(),
    }
    if hip_extension_required and not rocm_quant._HIP_EXT_LOADED:
        failures.append("HIP extension did not load; measurements used the fallback")
        summary["status"] = "fail"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {output}")

    print("ROCm quant gate:", summary["status"].upper())
    if failures:
        print("\n".join(f"  - {failure}" for failure in failures))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
