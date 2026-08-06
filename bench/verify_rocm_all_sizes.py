#!/usr/bin/env python3
"""Run the same ROCm serving gates over every official RWKV-7 size.

The matrix deliberately names exact checkpoint revisions.  A pass therefore
means more than "the architecture constructed": greedy decode is checked
against an independently generated fp32 numpy fixture, CUDA/HIP graphs capture
through batch 8, dynamic batches are compared with batch-1 references, and
multi-chunk prefill is compared with a single-shot prefill.

Example:

  PYTHONPATH=/path/to/sglang/python python bench/verify_rocm_all_sizes.py \
      --model-root /models/rwkv7/fla --output-dir bench/results/rocm-all-sizes
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


def _run(name, command, log_path, env):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--fixture-root", default=str(DEFAULT_FIXTURES))
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output-dir", default="bench/results/rocm-all-sizes")
    parser.add_argument("--sizes", default="", help="comma list, e.g. 0.1B,7.2B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--skip-chunk", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(Path(args.matrix).read_text())
    selected = {item.strip().lower() for item in args.sizes.split(",") if item.strip()}
    models = [
        item for item in matrix["models"]
        if not selected or item["size"].lower() in selected
    ]
    if not models:
        parser.error("--sizes did not select any matrix entries")

    env = os.environ.copy()
    env.setdefault("RWKV_ROCM_PREFILL_TILE", "256")
    output_dir = Path(args.output_dir).resolve()
    model_root = Path(args.model_root).resolve()
    fixture_root = Path(args.fixture_root).resolve()
    results = []

    for item in models:
        size = item["size"]
        slug = size.lower().replace(".", "")
        model = model_root / item["model_dir"]
        fixture = fixture_root / item["fixture"]
        missing = [str(path) for path in (model, fixture) if not path.exists()]
        if missing:
            results.append({"size": size, "status": "missing", "missing": missing})
            print(f"[{size}] MISSING: {', '.join(missing)}", file=sys.stderr)
            continue

        row = {"size": size, "model": str(model), "stages": []}
        if not args.skip_batch:
            row["stages"].append(
                _run(
                    f"{size} oracle+dynamic-batch+graph",
                    [
                        sys.executable,
                        "bench/verify_batch.py",
                        "--model", str(model),
                        "--fixture", str(fixture),
                        "--dtype", args.dtype,
                        "--mem-fraction", str(args.mem_fraction),
                        "--cuda-graph",
                        "--cuda-graph-max-bs", "8",
                        "--identical-bsz", "8",
                        "--n", str(args.tokens),
                    ],
                    output_dir / f"{slug}_batch.log",
                    env,
                )
            )
        if not args.skip_chunk:
            row["stages"].append(
                _run(
                    f"{size} chunked-prefill",
                    [
                        sys.executable,
                        "bench/verify_chunked_prefill.py",
                        "--model", str(model),
                        "--dtype", args.dtype,
                        "--mem-fraction", str(args.mem_fraction),
                        "--prompt-len", str(args.prompt_len),
                        "--chunk", str(args.chunk),
                        "--gen", str(args.tokens),
                    ],
                    output_dir / f"{slug}_chunk.log",
                    env,
                )
            )
        row["status"] = (
            "pass" if all(stage["return_code"] == 0 for stage in row["stages"])
            else "fail"
        )
        results.append(row)

    summary = {
        "matrix": str(Path(args.matrix).resolve()),
        "source": matrix.get("source"),
        "dtype": args.dtype,
        "rocm_prefill_tile": env["RWKV_ROCM_PREFILL_TILE"],
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    failed = [row for row in results if row["status"] != "pass"]
    print(f"\nWrote {summary_path}")
    print(f"ROCm all-size matrix: {len(results) - len(failed)}/{len(results)} PASS")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
