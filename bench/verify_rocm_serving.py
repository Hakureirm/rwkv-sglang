#!/usr/bin/env python3
"""End-to-end ROCm serving gates for a running RWKV-7 SGLang server.

The script uses only the Python standard library.  It verifies that a dynamic
batch is token-exact against independent batch-1 requests.  With
``--state-cache`` it additionally verifies a non-zero recurrent-state prefix
hit and token-exact continuation after the state is restored.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def request_json(url: str, path: str, payload=None, timeout: int = 300):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body.decode(errors="replace")


def generate(url: str, input_ids, new_tokens: int):
    start = time.perf_counter()
    output = request_json(
        url,
        "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": new_tokens,
                "ignore_eos": True,
            },
        },
    )
    return output, time.perf_counter() - start


def verify_dynamic_batch(url: str, batch_size: int, new_tokens: int):
    prompts = [
        [(i * 37 + batch * 11) % 60000 + 1 for i in range(12 + batch)]
        for batch in range(batch_size)
    ]
    references = [generate(url, prompt, new_tokens)[0]["output_ids"] for prompt in prompts]
    batch_output, latency = generate(url, prompts, new_tokens)
    if not isinstance(batch_output, list):
        raise AssertionError(f"batched response is not a list: {type(batch_output)!r}")
    outputs = [record["output_ids"] for record in batch_output]
    exact = [got == expected for got, expected in zip(outputs, references)]
    if not all(exact):
        raise AssertionError(f"dynamic batch diverged from batch-1 references: {exact}")
    return {
        "batch_size": batch_size,
        "exact_vs_batch1": exact,
        "latency_s": latency,
        "completion_tokens": sum(len(tokens) for tokens in outputs),
    }


def verify_state_cache(url: str, prefix_len: int, new_tokens: int):
    prefix = [(i * 53) % 59000 + 1 for i in range(prefix_len)]
    continuation = prefix + [555, 666, 777, 888]

    request_json(url, "/flush_cache", {})
    reference, reference_latency = generate(url, continuation, new_tokens)

    request_json(url, "/flush_cache", {})
    _, warm_latency = generate(url, prefix, 1)
    restored, restored_latency = generate(url, continuation, new_tokens)

    cached_tokens = restored.get("meta_info", {}).get("cached_tokens", 0)
    exact = restored["output_ids"] == reference["output_ids"]
    if not exact:
        raise AssertionError("state-cache continuation diverged from the uncached reference")
    if cached_tokens <= 0:
        raise AssertionError("state-cache request reported zero cached tokens")
    return {
        "prompt_tokens": len(continuation),
        "cached_tokens": cached_tokens,
        "cache_hit_rate": cached_tokens / len(continuation),
        "greedy_exact_after_restore": exact,
        "reference_latency_s": reference_latency,
        "warm_latency_s": warm_latency,
        "restored_latency_s": restored_latency,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--state-cache", action="store_true")
    parser.add_argument("--prefix-len", type=int, default=2048)
    args = parser.parse_args()

    report = {
        "model_info": request_json(args.url, "/model_info"),
        "dynamic_batch": verify_dynamic_batch(
            args.url, args.batch_size, args.new_tokens
        ),
    }
    if args.state_cache:
        report["state_cache"] = verify_state_cache(
            args.url, args.prefix_len, args.new_tokens
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("PASS ROCm serving gates")


if __name__ == "__main__":
    main()
