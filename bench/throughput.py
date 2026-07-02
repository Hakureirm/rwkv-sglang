#!/usr/bin/env python3
"""
RWKV-7 x sglang throughput micro-bench (M2-baseline).

For one model dir + compute dtype, measures across batch sizes:
  * decode tok/s   : steady-state batched decode (generate DECODE_TOKENS new
                     tokens from a short prompt). Isolated from prefill by
                     subtracting a max_new_tokens=1 run (same prompt/bsz), so the
                     reported rate is pure decode steps (RWKV decode is O(1) per
                     token, independent of context length).
  * prefill tok/s  : a ~PREFILL_LEN-token prompt x bsz, max_new_tokens=1; the
                     elapsed time is the batch time-to-first-token (TTFT).
                     prefill tok/s = bsz * PREFILL_LEN / TTFT.
  * peak VRAM      : sampled via nvidia-smi (GPU `--gpu`) during each bsz phase.
                     NB: sglang Engine runs the model in a *subprocess*, so the
                     driver-side torch.cuda.max_memory_allocated() reads ~0 and is
                     NOT usable here; nvidia-smi (whole-GPU used) is the honest
                     proxy. Baseline (pre-Engine) is reported so footprint = peak
                     - baseline.

cuda-graph is OFF (disable_cuda_graph + disable_piecewise_cuda_graph). This is the
M2-baseline number; cuda-graph (M2b) will further speed decode.

  source ~/rwkv_env.sh && CUDA_VISIBLE_DEVICES=0 ~/envs/rwkv-sgl/bin/python \
      bench/throughput.py --model <fla_dir> --dtype bfloat16 [--gpu 0]
"""
import argparse
import subprocess
import threading
import time


class VramSampler:
    """Background thread polling nvidia-smi memory.used (MiB) for one GPU."""

    def __init__(self, gpu: int, period: float = 0.05):
        self.gpu = gpu
        self.period = period
        self._cur = 0
        self._peak = 0
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> int:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.gpu}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            return int(out.strip().splitlines()[0])
        except Exception:
            return self._cur

    def _loop(self):
        while not self._stop:
            self._cur = self._sample()
            if self._cur > self._peak:
                self._peak = self._cur
            time.sleep(self.period)

    def start(self):
        self._cur = self._sample()
        self._peak = self._cur
        self._t.start()

    def reset(self):
        """Begin a fresh peak window from the current usage."""
        self._peak = self._cur

    @property
    def peak(self) -> int:
        return self._peak

    def stop(self):
        self._stop = True
        self._t.join(timeout=1.0)


def _make_prompt(length: int):
    # deterministic, avoids special token 0; well inside any RWKV-7 vocab (65536).
    return [(i % 60000) + 1 for i in range(length)]


def _gen(engine, prompts, max_new_tokens):
    engine.generate(
        input_ids=prompts,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,  # always emit exactly max_new_tokens (no early EOS)
        },
    )


def bench_bsz(engine, vram, bsz, decode_tokens, prefill_len, short_len):
    short = _make_prompt(short_len)
    longp = _make_prompt(prefill_len)
    short_batch = [list(short) for _ in range(bsz)]
    long_batch = [list(longp) for _ in range(bsz)]

    # ---- warmup (both paths) ----
    _gen(engine, short_batch, 8)
    _gen(engine, long_batch, 1)

    vram.reset()

    # ---- prefill: long prompt, 1 new token => batch TTFT ----
    t0 = time.perf_counter()
    _gen(engine, long_batch, 1)
    ttft = time.perf_counter() - t0
    prefill_tok_s = bsz * prefill_len / ttft

    # ---- decode: short prompt; isolate steady-state by subtracting the 1-token run ----
    t0 = time.perf_counter()
    _gen(engine, short_batch, 1)
    t_one = time.perf_counter() - t0

    t0 = time.perf_counter()
    _gen(engine, short_batch, decode_tokens)
    t_full = time.perf_counter() - t0

    decode_steps = decode_tokens - 1
    decode_dt = max(t_full - t_one, 1e-9)
    decode_tok_s = bsz * decode_steps / decode_dt

    return {
        "bsz": bsz,
        "decode_tok_s": decode_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "ttft_ms": ttft * 1e3,
        "peak_vram_mib": vram.peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fla-format model dir")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-sizes", default="1,8,32")
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--prefill-len", type=int, default=1024)
    ap.add_argument("--short-len", type=int, default=16)
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    ap.add_argument("--gpu", type=int, default=0, help="nvidia-smi GPU index to poll")
    ap.add_argument("--tag", default="", help="label for the printed table")
    ap.add_argument(
        "--cuda-graph",
        action="store_true",
        help="enable CUDA graph for decode (M2b). Default off (M2-baseline).",
    )
    ap.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=None,
        help="max batch size to capture as a CUDA graph (default: sglang auto, "
        "which is 24 on a 24GB 3090 — set >= max --batch-sizes to graph large bsz).",
    )
    ap.add_argument(
        "--disable-radix-cache",
        action="store_true",
        help="disable the token radix cache (REQUIRED for correct RWKV-7 dynamic "
        "batching; see bench/results/radix_correctness.md). Production config.",
    )
    args = ap.parse_args()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    vram = VramSampler(args.gpu)
    vram.start()
    baseline = vram.peak

    import sglang as sgl

    engine_kwargs = dict(
        model_path=args.model,
        skip_tokenizer_init=True,
        disable_cuda_graph=not args.cuda_graph,
        disable_piecewise_cuda_graph=True,
        disable_radix_cache=args.disable_radix_cache,
        dtype=args.dtype,
        tp_size=1,
        mem_fraction_static=args.mem_fraction,
    )
    if args.cuda_graph and args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    engine = sgl.Engine(**engine_kwargs)

    rows = []
    for bsz in batch_sizes:
        rows.append(
            bench_bsz(
                engine, vram, bsz,
                args.decode_tokens, args.prefill_len, args.short_len,
            )
        )

    engine.shutdown()
    vram.stop()

    tag = args.tag or args.model
    print("\n" + "=" * 78)
    cg_label = "ON" if args.cuda_graph else "OFF"
    radix_label = "OFF" if args.disable_radix_cache else "ON"
    print(
        f"THROUGHPUT  model={tag}  dtype={args.dtype}  cuda_graph={cg_label}  "
        f"radix_cache={radix_label}"
    )
    print(
        f"  decode={args.decode_tokens}tok (steady-state, prefill-subtracted)  "
        f"prefill_len={args.prefill_len}  mem_fraction={args.mem_fraction}"
    )
    print(f"  VRAM via nvidia-smi GPU{args.gpu}; baseline(pre-Engine)={baseline} MiB")
    print("-" * 78)
    print(
        f"{'bsz':>4} | {'decode tok/s':>13} | {'prefill tok/s':>14} | "
        f"{'TTFT ms':>9} | {'peak VRAM MiB':>13} | {'minus base':>10}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['bsz']:>4} | {r['decode_tok_s']:>13.1f} | {r['prefill_tok_s']:>14.1f} | "
            f"{r['ttft_ms']:>9.1f} | {r['peak_vram_mib']:>13} | "
            f"{r['peak_vram_mib'] - baseline:>10}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
