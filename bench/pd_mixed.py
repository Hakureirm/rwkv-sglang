"""PD-mixed serving benchmark via direct streaming /generate (task #12).

"PD-mixed" = prefill and decode run concurrently: requests arrive open-loop (Poisson
at rate lambda), so a new request's prefill interleaves with in-flight requests'
decode in the same scheduler step. This is the realistic online-serving regime and
surfaces tail latency that a closed-loop static batch hides.

bench_serving --dataset-name random needs an HF corpus download (box is modelscope-only,
no HF), so we hit /generate with stream=true directly: TTFT = time to first token,
TPOT = (total - TTFT)/(tokens-1). We sweep request-rate and report P50/P99 TTFT + TPOT
+ output throughput per rate.

Usage:
  python bench/pd_mixed.py --port 30070 --rates 2,4,8,16,inf \
      --in-len 512 --out-len 256 --num-prompts 300 --out bench/results/pd_mixed.json
"""
import argparse, asyncio, json, time, math, random
import aiohttp


def make_ids(in_len, rng):
    return [0] + [rng.randint(1, 60000) for _ in range(in_len - 1)]


async def one_stream(session, url, ids, out_len):
    """POST stream=true; return (ttft_s, total_s, n_tokens) or None on error."""
    payload = {"input_ids": ids,
               "sampling_params": {"temperature": 0.0, "max_new_tokens": out_len,
                                   "ignore_eos": True},
               "stream": True}
    t0 = time.time()
    ttft = None
    n = 0
    try:
        async with session.post(url, json=payload) as resp:
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    except Exception:
        return None
    total = time.time() - t0
    if ttft is None:
        return None
    return ttft, total, n


async def run_rate(url, rate, n_prompts, in_len, out_len, seed):
    rng = random.Random(seed)
    # pre-generate arrival offsets (Poisson) + prompts so timing is deterministic
    arrivals, t = [], 0.0
    for _ in range(n_prompts):
        arrivals.append(t)
        if rate != float("inf"):
            t += rng.expovariate(rate)
    prompts = [make_ids(in_len, rng) for _ in range(n_prompts)]
    results = []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        # warmup
        await one_stream(session, url, prompts[0], 8)
        t0 = time.time()

        async def fire(i):
            if arrivals[i] > 0:
                dt = arrivals[i] - (time.time() - t0)
                if dt > 0:
                    await asyncio.sleep(dt)
            r = await one_stream(session, url, prompts[i], out_len)
            if r:
                results.append(r)

        await asyncio.gather(*[fire(i) for i in range(n_prompts)])
        wall = time.time() - t0

    ok = [r for r in results if r]
    if not ok:
        return {"rate": rate, "completed": 0}
    ttfts = sorted(r[0] for r in ok)
    tpots = sorted((r[1] - r[0]) / max(1, r[2] - 1) for r in ok)
    out_tok = sum(r[2] for r in ok)

    def pct(a, q):
        return a[min(len(a) - 1, int(q * len(a)))]

    return {
        "rate": "inf" if rate == float("inf") else rate,
        "completed": len(ok), "wall_s": round(wall, 2),
        "out_tok_per_s": round(out_tok / wall, 1),
        "req_per_s": round(len(ok) / wall, 3),
        "ttft_p50_ms": round(pct(ttfts, .50) * 1e3, 1),
        "ttft_p99_ms": round(pct(ttfts, .99) * 1e3, 1),
        "tpot_p50_ms": round(pct(tpots, .50) * 1e3, 2),
        "tpot_p99_ms": round(pct(tpots, .99) * 1e3, 2),
    }


async def main_async(a):
    url = f"http://{a.host}:{a.port}/generate"
    rates = [float("inf") if r.strip() == "inf" else float(r) for r in a.rates.split(",")]
    rows = []
    for rate in rates:
        row = await run_rate(url, rate, a.num_prompts, a.in_len, a.out_len, a.seed)
        rows.append(row)
        print(f"rate={str(row['rate']):>4}  out_tok/s={row.get('out_tok_per_s',0):8.1f}  "
              f"TTFT p50/p99={row.get('ttft_p50_ms',0):7.1f}/{row.get('ttft_p99_ms',0):7.1f}ms  "
              f"TPOT p50/p99={row.get('tpot_p50_ms',0):6.2f}/{row.get('tpot_p99_ms',0):6.2f}ms  "
              f"(n={row.get('completed',0)})", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30070)
    ap.add_argument("--rates", default="2,4,8,16,inf")
    ap.add_argument("--in-len", type=int, default=512)
    ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--num-prompts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = asyncio.run(main_async(a))
    if a.out:
        json.dump({"rows": rows, "config": vars(a)}, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
