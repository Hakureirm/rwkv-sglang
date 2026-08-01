#!/usr/bin/env python3
"""64 genuinely DIFFERENT prompts, fired concurrently.

Written because bench/bsz_throughput.py sends the SAME token ids to every request,
which is fine for throughput and silently fatal for any question about how a batch's
contents differ from each other (F0080: it made the cross-row union equal the per-row
rate to four decimals, which reads exactly like the hoped-for result).
"""
import concurrent.futures as cf, random, sys
import requests

port = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
url = f"http://127.0.0.1:{port}/generate"
random.seed(0)


def one(_):
    ids = [random.randint(100, 60000) for _ in range(64)]
    r = requests.post(url, json={"input_ids": ids, "sampling_params":
                      {"temperature": 0.0, "max_new_tokens": 8, "ignore_eos": True}}, timeout=300)
    r.raise_for_status()
    return 1


with cf.ThreadPoolExecutor(max_workers=n) as ex:
    print("completed", sum(f.result() for f in [ex.submit(one, i) for i in range(n)]))
