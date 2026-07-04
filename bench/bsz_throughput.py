"""Best-bsz decode-throughput sweep via direct /generate (req#7b).

sglang's bench_serving --dataset-name random needs an HF corpus download, which the
box (modelscope-only, no HF) cannot do. This probe hits /generate directly with
fixed random input_ids + ignore_eos, so every request decodes exactly N tokens —
giving a clean per-concurrency aggregate decode throughput to find the best-bsz peak.

Usage:
  python bench/bsz_throughput.py --port 30070 --out bench/results/bsz_sweep.json \
      --concurrencies 1,8,32,64,128,256,384,512 --in-len 64 --out-len 256
"""
import argparse, asyncio, json, time
from concurrent.futures import ThreadPoolExecutor
import requests


def one(sess, url, ids, out_len):
    t0 = time.time()
    r = sess.post(url, json={
        "input_ids": ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": out_len,
                            "ignore_eos": True},
    }, timeout=600)
    r.raise_for_status()
    d = r.json()
    if isinstance(d, list):
        d = d[0]
    meta = d["meta_info"]
    return meta.get("completion_tokens", 0), time.time() - t0


async def run_level(url, C, n, ids, out_len):
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=C))
    sem = asyncio.Semaphore(C)
    sess = requests.Session()
    ad = requests.adapters.HTTPAdapter(pool_connections=C, pool_maxsize=C)
    sess.mount("http://", ad)

    async def task():
        async with sem:
            return await asyncio.to_thread(one, sess, url, ids, out_len)

    # warmup 1 to prime, then timed
    await task()
    t0 = time.time()
    res = await asyncio.gather(*[task() for _ in range(n)])
    wall = time.time() - t0
    toks = sum(t for t, _ in res)
    lats = sorted(l for _, l in res)
    p = lambda q: lats[min(len(lats) - 1, int(q * len(lats)))]
    return {
        "concurrency": C, "requests": n, "wall_s": round(wall, 2),
        "out_tokens": toks, "out_tok_per_s": round(toks / wall, 1),
        "req_per_s": round(n / wall, 3),
        "lat_p50_s": round(p(0.50), 3), "lat_p99_s": round(p(0.99), 3),
        "per_stream_tok_s": round(toks / wall / C, 1),
    }


async def main_async(a):
    url = f"http://{a.host}:{a.port}/generate"
    ids = [0] + [((i * 131 + 7) % 60000) + 1 for i in range(a.in_len - 1)]  # fixed pseudo-random
    levels = [int(c) for c in a.concurrencies.split(",")]
    rows = []
    for C in levels:
        n = max(a.min_reqs, C * a.reqs_mult)
        row = await run_level(url, C, n, ids, a.out_len)
        rows.append(row)
        print(f"c={C:4d}  out_tok/s={row['out_tok_per_s']:8.1f}  "
              f"per-stream={row['per_stream_tok_s']:6.1f}  req/s={row['req_per_s']:6.2f}  "
              f"p50={row['lat_p50_s']:.2f}s p99={row['lat_p99_s']:.2f}s  (n={n})", flush=True)
    peak = max(rows, key=lambda r: r["out_tok_per_s"])
    print(f"\nPEAK: {peak['out_tok_per_s']} tok/s @ concurrency={peak['concurrency']}")
    return rows, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30070)
    ap.add_argument("--concurrencies", default="1,8,32,64,128,256,384,512")
    ap.add_argument("--in-len", type=int, default=64)
    ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--reqs-mult", type=int, default=4, help="requests = concurrency * mult")
    ap.add_argument("--min-reqs", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows, peak = asyncio.run(main_async(a))
    if a.out:
        json.dump({"rows": rows, "peak": peak, "config": vars(a)}, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
