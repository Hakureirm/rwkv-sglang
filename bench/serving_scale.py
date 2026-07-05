#!/usr/bin/env python3
"""RWKV-7 x sglang serving-scale bench — the O(1)-state wedge, measured.

RWKV-7 carries a *constant* recurrent state per sequence (no growing KV cache),
so a serving engine's per-request footprint and per-token decode cost are both
independent of context length. This script measures that property directly, on
one GPU, in two sweeps:

  --mode context : fix the batch size, vary the decode context length. RWKV-7
                   decode tok/s and peak VRAM should stay ~flat as context grows
                   from 1K to 64K+ (a KV-cache model's decode slows and its VRAM
                   grows linearly with context). Loads the engine ONCE and reuses
                   it across all context lengths.

  --mode batch   : fix the context, vary concurrency. Aggregate decode tok/s
                   scales with the number of concurrent sequences until the GPU is
                   compute-bound; peak VRAM barely moves because each extra state
                   is a tiny constant. (Equivalent to throughput.py's bsz sweep;
                   included here so both wedge axes live in one artifact.)

decode tok/s is steady-state: we subtract a 1-token run (same prompt/bsz) from a
DECODE_TOKENS run, so the reported rate is pure decode steps, prefill excluded.
Per-token decode latency (ms/token/seq) is derived from the same measurement.

VRAM note: sglang runs the model in a subprocess, so driver-side
torch.cuda.max_memory_allocated() reads ~0; we poll whole-GPU nvidia-smi
(memory.used) as the honest proxy, same as throughput.py. sglang pre-allocates a
static state pool from mem_fraction_static, so "peak VRAM" is dominated by that
pool — the point of the context sweep is that it does NOT grow with context.

  source ~/rwkv_env.sh && CUDA_VISIBLE_DEVICES=0 ~/envs/rwkv-sgl/bin/python \
      bench/serving_scale.py --model <fla_dir> --mode context \
      --bsz 8 --contexts 1024,4096,16384,65536
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
        self._peak = self._cur

    @property
    def peak(self) -> int:
        return self._peak

    def stop(self):
        self._stop = True
        self._t.join(timeout=1.0)


def _make_prompt(length: int):
    # deterministic, avoids special token 0; well inside the RWKV-7 vocab (65536).
    return [(i % 60000) + 1 for i in range(length)]


def _gen(engine, prompts, max_new_tokens):
    engine.generate(
        input_ids=prompts,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,  # emit exactly max_new_tokens (no early EOS)
        },
    )


def measure(engine, vram, bsz, context_len, decode_tokens):
    """Prefill `context_len` tokens x bsz, then measure steady-state decode."""
    prompt = _make_prompt(context_len)
    batch = [list(prompt) for _ in range(bsz)]

    # warmup at this shape (build/replay the right cuda-graph, page the state pool)
    _gen(engine, batch, 4)
    vram.reset()

    # prefill (context tokens -> 1 new token): batch TTFT
    t0 = time.perf_counter()
    _gen(engine, batch, 1)
    ttft = time.perf_counter() - t0

    # decode: subtract the 1-token run to isolate steady-state decode steps
    t0 = time.perf_counter()
    _gen(engine, batch, 1)
    t_one = time.perf_counter() - t0
    t0 = time.perf_counter()
    _gen(engine, batch, decode_tokens)
    t_full = time.perf_counter() - t0

    steps = decode_tokens - 1
    dt = max(t_full - t_one, 1e-9)
    decode_tok_s = bsz * steps / dt
    ms_per_tok = dt / steps * 1e3  # wall-clock per decode step (all bsz seqs in parallel)

    return {
        "context": context_len,
        "bsz": bsz,
        "decode_tok_s": decode_tok_s,
        "ms_per_step": ms_per_tok,
        "ttft_ms": ttft * 1e3,
        "peak_vram_mib": vram.peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fla-format model dir")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--mode", choices=["context", "batch"], default="context")
    ap.add_argument("--bsz", type=int, default=8, help="fixed batch size (context mode)")
    ap.add_argument("--context", type=int, default=1024, help="fixed context (batch mode)")
    ap.add_argument("--contexts", default="1024,4096,16384,65536",
                    help="context lengths to sweep (context mode)")
    ap.add_argument("--batch-sizes", default="1,16,64,128,256",
                    help="batch sizes to sweep (batch mode)")
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--cuda-graph-max-bs", type=int, default=None)
    ap.add_argument("--max-context", type=int, default=None,
                    help="override the model's declared context_length. RWKV-7's "
                    "O(1) recurrence has no architectural context limit; the config "
                    "cap is just the trained window. Set this to measure serving "
                    "COST invariance past that window (output quality beyond the "
                    "trained window is not claimed).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    vram = VramSampler(args.gpu)
    vram.start()
    baseline = vram.peak

    import sglang as sgl

    engine_kwargs = dict(
        model_path=args.model,
        skip_tokenizer_init=True,
        disable_cuda_graph=False,          # production decode path
        disable_piecewise_cuda_graph=True,
        disable_radix_cache=True,          # REQUIRED for correct RWKV-7 dynamic batching
        dtype=args.dtype,
        tp_size=1,
        mem_fraction_static=args.mem_fraction,
    )
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.max_context is not None:
        engine_kwargs["context_length"] = args.max_context
    # keep the same invocation across sglang versions (e.g. main dropped
    # disable_piecewise_cuda_graph): only pass kwargs ServerArgs still accepts
    from sglang.srt.server_args import ServerArgs
    engine_kwargs = {k: v for k, v in engine_kwargs.items() if k in ServerArgs.__dataclass_fields__}
    engine = sgl.Engine(**engine_kwargs)

    rows = []
    if args.mode == "context":
        contexts = [int(c) for c in args.contexts.split(",")]
        for c in contexts:
            rows.append(measure(engine, vram, args.bsz, c, args.decode_tokens))
    else:
        for b in (int(x) for x in args.batch_sizes.split(",")):
            rows.append(measure(engine, vram, b, args.context, args.decode_tokens))

    engine.shutdown()
    vram.stop()

    tag = args.tag or args.model.rstrip("/").split("/")[-1]
    print("=" * 82)
    print(f"SERVING-SCALE  model={tag}  dtype={args.dtype}  mode={args.mode}  "
          f"cuda_graph=ON  radix=OFF")
    print(f"  decode={args.decode_tokens}tok steady-state (prefill-subtracted)  "
          f"mem_fraction={args.mem_fraction}  baseline={baseline} MiB")
    print("-" * 82)
    print(f"{'context':>8} | {'bsz':>4} | {'decode tok/s':>13} | {'ms/step':>9} | "
          f"{'TTFT ms':>10} | {'peak VRAM MiB':>13}")
    print("-" * 82)
    for r in rows:
        print(f"{r['context']:>8} | {r['bsz']:>4} | {r['decode_tok_s']:>13.1f} | "
              f"{r['ms_per_step']:>9.2f} | {r['ttft_ms']:>10.1f} | {r['peak_vram_mib']:>13}")
    print("=" * 82)


if __name__ == "__main__":
    main()
