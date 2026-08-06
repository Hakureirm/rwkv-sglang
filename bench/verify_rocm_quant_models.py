#!/usr/bin/env python3
"""ROCm all-size W8/W4 model acceptance matrix.

For each selected official size this runner discovers the dense, W8G64, and
W4G64 directories, then records:

* W8 greedy agreement with the dense NumPy fixture;
* W4 graph/dynamic-batch agreement with its own batch-1 reference, unless an
  explicitly identified lossy-RTN checkpoint is being measured with that gate
  skipped;
* chunked-prefill agreement for both quantized modes;
* dense/W8/W4 BF16 decode throughput at batches 1 and 8;
* total checkpoint bytes and quant/dense ratios.

W4 uses ``--reference-only`` intentionally: lossy model quality belongs to a
separate GPTQ/evaluation gate, not the fused-kernel correctness gate.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = Path(__file__).with_name("rocm_model_matrix.json")
DEFAULT_FIXTURES = Path(__file__).with_name("fixtures")


def directory_bytes(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def discover_quant(root, size, bits, dense_name):
    preferred = root / f"{dense_name}-w{bits}g64"
    if preferred.is_dir():
        return preferred
    needle_size = size.lower()
    needle_quant = f"w{bits}g64"
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and needle_size in path.name.lower()
        and needle_quant in path.name.lower()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one {size} W{bits} directory under {root}, got {candidates}"
        )
    return candidates[0]


def run_stage(name, command, log_path, env):
    print(f"\n[{name}] {' '.join(map(str, command))}", flush=True)
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    return {
        "name": name,
        "return_code": return_code,
        "seconds": round(time.monotonic() - started, 3),
        "log": str(log_path),
    }


def mode_env(base, mode):
    env = base.copy()
    env["RWKV_W8"] = "1" if mode == "w8g64" else "0"
    env["RWKV_W4"] = "1" if mode == "w4g64" else "0"
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--quant-root", required=True)
    parser.add_argument("--fixture-root", default=str(DEFAULT_FIXTURES))
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output-dir", default="bench/results/rocm-quant-all-sizes")
    parser.add_argument("--sizes", default="")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--prefill-len", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--skip-w4-batch",
        action="store_true",
        help="skip W4 token-invariance gate while evaluating a known-lossy RTN checkpoint",
    )
    parser.add_argument("--skip-chunk", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(Path(args.matrix).read_text())
    selected = {part.strip().lower() for part in args.sizes.split(",") if part.strip()}
    models = [
        item
        for item in matrix["models"]
        if not selected or item["size"].lower() in selected
    ]
    if not models:
        parser.error("--sizes selected no matrix entries")

    model_root = Path(args.model_root).resolve()
    quant_root = Path(args.quant_root).resolve()
    fixture_root = Path(args.fixture_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    base_env.setdefault("RWKV_ROCM_PREFILL_TILE", "256")
    summary_rows = []

    for item in models:
        size = item["size"]
        slug = size.lower().replace(".", "")
        dense = model_root / item["model_dir"]
        fixture = fixture_root / item["fixture"]
        row = {"size": size, "stages": [], "skipped_stages": []}
        try:
            paths = {
                "dense": dense,
                "w8g64": discover_quant(quant_root, size, 8, item["model_dir"]),
                "w4g64": discover_quant(quant_root, size, 4, item["model_dir"]),
            }
        except FileNotFoundError as exc:
            row.update(status="missing", error=str(exc))
            summary_rows.append(row)
            print(f"[{size}] {exc}", file=sys.stderr)
            continue
        if not dense.is_dir() or not fixture.is_file():
            row.update(status="missing", error="dense model or fixture is missing")
            summary_rows.append(row)
            continue

        row["paths"] = {name: str(path) for name, path in paths.items()}
        row["checkpoint_bytes"] = {
            name: directory_bytes(path) for name, path in paths.items()
        }
        row["checkpoint_ratio_vs_dense"] = {
            name: row["checkpoint_bytes"][name] / row["checkpoint_bytes"]["dense"]
            for name in ("w8g64", "w4g64")
        }

        if not args.skip_correctness:
            for mode in ("w8g64", "w4g64"):
                if mode == "w4g64" and args.skip_w4_batch:
                    row["skipped_stages"].append(
                        {
                            "name": f"{size} {mode} oracle/batch/graph",
                            "reason": (
                                "explicitly skipped for lossy RTN checkpoints; "
                                "this is not a W4 model-quality pass"
                            ),
                        }
                    )
                    continue
                command = [
                    sys.executable,
                    "bench/verify_batch.py",
                    "--model", str(paths[mode]),
                    "--fixture", str(fixture),
                    "--dtype", args.dtype,
                    "--mem-fraction", str(args.mem_fraction),
                    "--cuda-graph",
                    "--cuda-graph-max-bs", "8",
                    "--identical-bsz", "8",
                    "--n", str(args.tokens),
                ]
                if mode == "w4g64":
                    command.append("--reference-only")
                row["stages"].append(
                    run_stage(
                        f"{size} {mode} oracle/batch/graph",
                        command,
                        output_dir / f"{slug}_{mode}_batch.log",
                        mode_env(base_env, mode),
                    )
                )

        if not args.skip_chunk:
            for mode in ("w8g64", "w4g64"):
                row["stages"].append(
                    run_stage(
                        f"{size} {mode} chunked prefill",
                        [
                            sys.executable,
                            "bench/verify_chunked_prefill.py",
                            "--model", str(paths[mode]),
                            "--dtype", args.dtype,
                            "--mem-fraction", str(args.mem_fraction),
                            "--prompt-len", str(max(args.prefill_len, 512)),
                            "--chunk", "256",
                            "--gen", str(args.tokens),
                        ],
                        output_dir / f"{slug}_{mode}_chunk.log",
                        mode_env(base_env, mode),
                    )
                )

        if not args.skip_throughput:
            row["throughput"] = {}
            for mode in ("dense", "w8g64", "w4g64"):
                result_path = output_dir / f"{slug}_{mode}_throughput.json"
                row["stages"].append(
                    run_stage(
                        f"{size} {mode} throughput",
                        [
                            sys.executable,
                            "bench/throughput.py",
                            "--model", str(paths[mode]),
                            "--dtype", args.dtype,
                            "--batch-sizes", "1,8",
                            "--decode-tokens", str(args.decode_tokens),
                            "--prefill-len", str(args.prefill_len),
                            "--short-len", "16",
                            "--mem-fraction", str(args.mem_fraction),
                            "--cuda-graph",
                            "--cuda-graph-max-bs", "8",
                            "--disable-radix-cache",
                            "--tag", f"rocm-{size}-{mode}",
                            "--output", str(result_path),
                        ],
                        output_dir / f"{slug}_{mode}_throughput.log",
                        mode_env(base_env, mode),
                    )
                )
                if result_path.is_file():
                    row["throughput"][mode] = json.loads(result_path.read_text())

            if "dense" in row["throughput"]:
                dense_rows = {
                    entry["bsz"]: entry
                    for entry in row["throughput"]["dense"]["rows"]
                }
                row["decode_ratio_vs_dense"] = {}
                for mode in ("w8g64", "w4g64"):
                    if mode not in row["throughput"]:
                        continue
                    row["decode_ratio_vs_dense"][mode] = {
                        str(entry["bsz"]): entry["decode_tok_s"]
                        / dense_rows[entry["bsz"]]["decode_tok_s"]
                        for entry in row["throughput"][mode]["rows"]
                    }

        row["status"] = (
            "pass"
            if row["stages"]
            and all(stage["return_code"] == 0 for stage in row["stages"])
            else "fail"
        )
        summary_rows.append(row)
        (output_dir / "summary.json").write_text(
            json.dumps({"results": summary_rows}, indent=2) + "\n"
        )

    summary = {
        "gpu": os.environ.get("GPU_NAME", "runtime-detected in child logs"),
        "dtype": args.dtype,
        "matrix": str(Path(args.matrix).resolve()),
        "coverage": {
            "w8_dense_oracle_dynamic_batch_graph": not args.skip_correctness,
            "w4_self_reference_dynamic_batch_graph": (
                not args.skip_correctness and not args.skip_w4_batch
            ),
            "w8_w4_chunked_prefill": not args.skip_chunk,
            "dense_w8_w4_throughput": not args.skip_throughput,
        },
        "known_limitations": (
            [
                "W4 model-level batch/graph was explicitly skipped; passing rows "
                "cover only the selected gates and do not certify RTN W4 quality."
            ]
            if args.skip_w4_batch
            else []
        ),
        "results": summary_rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    failed = [row for row in summary_rows if row.get("status") != "pass"]
    print(f"\nwrote {summary_path}")
    print(
        "ROCm quant all-size selected gates: "
        f"{len(summary_rows) - len(failed)}/{len(summary_rows)} PASS"
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
