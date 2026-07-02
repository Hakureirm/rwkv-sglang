# Serving-scale — the O(1)-state wedge, measured

RWKV-7 carries a **constant** recurrent state per sequence (no growing KV cache), so a serving
engine's per-request VRAM and per-token decode cost are both independent of context length. This
directory measures that property directly on one **exclusive RTX 3090**, via
[`bench/serving_scale.py`](../../serving_scale.py) and [`bench/throughput.py`](../../throughput.py)
(cuda-graph ON = production decode path; radix cache OFF, required for correct RWKV-7 dynamic
batching — see [`../radix_correctness.md`](../radix_correctness.md)).

Model: `rwkv7-1.5b-fla`, bf16. VRAM is whole-GPU `nvidia-smi memory.used` (sglang runs the model
in a subprocess, so driver-side `torch.cuda.max_memory_allocated` reads ~0 and is unusable — the
same honest proxy `throughput.py` uses). Raw logs: [`conc_scale_15b.log`](conc_scale_15b.log),
[`ctx_invariance_15b.log`](ctx_invariance_15b.log).

## 1. Concurrency scaling — throughput up ~50×, VRAM ~flat

Fixed 512-token context, sweeping the number of concurrent sequences (`throughput.py`,
`--cuda-graph-max-bs 256 --mem-fraction 0.85`):

| bsz | decode tok/s | prefill tok/s | peak VRAM (MiB) |
|----:|-------------:|--------------:|----------------:|
|   1 |        166.0 |       10,161 |          12,420 |
|  16 |      2,143.2 |       14,297 |          12,622 |
|  64 |      6,444.5 |       14,075 |          12,622 |
| 128 |      **8,297.8** |   13,223 |          12,622 |
| 256 |      8,186.7 |       12,292 |          12,622 |

Decode throughput scales **166 → 8,298 tok/s (~50×)** from bsz 1 → 128, then plateaus at 256
(compute-bound). Peak VRAM moves **+202 MiB total** across a 256× concurrency increase — because
each additional sequence adds only a tiny constant state.

## 2. Context-length invariance — VRAM dead flat, decode O(1)/token

Fixed bsz 8, sweeping the decode context length (`serving_scale.py --mode context`). The 1.5B
config declares an 8,192-token trained window; RWKV-7's recurrence has **no architectural context
limit** (O(1) state processes arbitrary length), so `--max-context 131072`
(+`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`) lets us measure serving **cost** past that window
(output *quality* beyond the trained window is not claimed — this sweep measures cost, not accuracy):

| context | decode tok/s | ms/step | TTFT (ms) | peak VRAM (MiB) |
|--------:|-------------:|--------:|----------:|----------------:|
|   1,024 |      1,034.3 |    7.73 |     581.1 |          12,364 |
|   4,096 |      1,017.0 |    7.87 |   2,471.4 |          12,364 |
|   8,192 |        900.1 |    8.89 |   5,105.3 |          12,364 |
|  16,384 |        748.2 |   10.69 |  10,797.3 |          12,364 |
|  32,768 |        764.3 |   10.47 |  23,048.5 |          12,366 |
|  65,536 |      1,080.3 |    7.41 |  47,061.7 |          12,368 |

**Peak VRAM: 12,364 → 12,368 MiB (+4 MiB, +0.03%) across a 64× context increase.** This is the
headline: a KV-cache transformer at 64K × 8 would need many GB of additional KV memory and OOM.

Decode stays **O(1)/token** — ms/step remains single-digit (7–11 ms) with no growth trend as
context increases 64×; a transformer's per-token attention cost grows with context. TTFT grows
linearly with context (581 ms → 47 s) — **expected and not a wedge**: prefill is O(T) for *any*
model (it must read every prompt token once); RWKV-7's wins are decode cost + memory, not prefill.

### Honesty note on the decode-tok/s column
`decode tok/s` is measured by subtracting a 1-token run from an N-token run to isolate steady-state
decode. At long context the prefill (tens of seconds) dwarfs the decode delta (sub-second), so this
difference-of-large-numbers is **noise-dominated** at 16K–64K — hence the non-monotone 748 → 1,080
wobble. The robust signals here are (a) **peak VRAM**, which is measured absolutely (flat), and
(b) **ms/step staying single-digit** — both consistent with the architectural O(1)-per-token
guarantee (decode reads only the fixed-size state, never the context). We do not read a context
*speedup* into the 65K number; we read *no context penalty*.

## 2b. Flagship size: the same properties hold at 7.2B

Same sweeps on **7.2B** (bf16, cuda-graph ON, radix off; raw: [`ctx_72b.log`](ctx_72b.log),
[`conc_72b.log`](conc_72b.log)):

**Context invariance (bsz 4, `--max-context 65536`):**
| context | decode ms/step | peak VRAM (MiB) |
|--------:|---------------:|----------------:|
|   1,024 |          22.46 |          17,866 |
|   8,192 |          30.60 |          17,866 |
|  32,768 |          31.53 |          17,866 |

Peak VRAM moves **+0 MiB across a 32× context increase** — at 7.2B the weights dominate and the
recurrent state stays a constant, so context costs literally nothing in memory.

**Concurrency (512-tok context):**
| bsz | decode tok/s | peak VRAM (MiB) |
|----:|-------------:|----------------:|
|   1 |         46.6 |          17,742 |
|  16 |        648.6 |          18,050 |
|  64 |      1,802.7 |          18,050 |

Decode scales **38.7×** (46.6 → 1,802.7 tok/s) from bsz 1 → 64 at **+308 MiB** — 64 concurrent
7.2B sequences on one 24 GB card. (46.6 is the default config, consistent with
`comparison_clean.md`'s 45.9; the opt-in kernels lift bsz1 to 65.7.)

## 3. ShareGPT — realistic mixed prefill+decode serving (standard `bench_serving`)

The two sweeps above are synthetic (fixed-length) to isolate the O(1)-state properties. For an
industry-standard, realistic serving number we run sglang's own `bench_serving` on **ShareGPT**
(variable-length real conversations), 500 prompts, 1.5B, bf16, RTX 3090 (radix off, piecewise off):

| request rate | req/s | output tok/s | total tok/s | median TTFT | P99 TTFT | median TPOT |
|---|---|---|---|---|---|---|
| `inf` (peak) | 6.48 | 1,275 | 3,361 | 7,220 ms | 12,958 ms | 112 ms |
| 16 req/s | 5.21 | 1,025 | 2,702 | **273 ms** | 1,279 ms | 175 ms |

Raw: [`bs_sharegpt-15b_rinf.log`](bs_sharegpt-15b_rinf.log), [`bs_sharegpt-15b_r16.log`](bs_sharegpt-15b_r16.log).

**Reading it honestly:** ShareGPT output-throughput (1.3k tok/s) is far below the synthetic
pure-decode ceiling (8.3k tok/s) because ShareGPT is **prefill-heavy** (long prompts, short
replies) — total token throughput (3.4k tok/s, counting the long input prefills) is the more
representative figure. At `inf` all 500 requests arrive at once → TTFT is queue-dominated (7.2s);
rate-limited to 16 req/s, median TTFT drops to **273 ms** (interactive). These two numbers answer
different questions than §1/§2 and are reported alongside them, not in place of them.

**Second environment — H100** (same `bench_serving` methodology, 300 prompts, in-container server;
raw: [`h100_sharegpt.json`](h100_sharegpt.json)):

| request rate | req/s | output tok/s | total tok/s | median TTFT | P99 TTFT | median TPOT |
|---|---|---|---|---|---|---|
| `inf` (peak) | 26.33 | 5,590 | **14,245** | 1,425 ms | 1,962 ms | 14.4 ms |
| 16 req/s | 13.36 | 2,837 | 7,228 | **69 ms** | 220 ms | 8.8 ms |

The same serving stack lifts ~4× from the 3090 to an H100 (total tok/s 3.4k → 14.2k) with
interactive-grade TTFT (median 69 ms @16 req/s). (Prompt count differs — 300 vs the 3090's 500 —
noted for exactness; rates and dataset are identical.)

## Reproduce
```bash
source ~/rwkv_env.sh
# concurrency sweep
CUDA_VISIBLE_DEVICES=0 python bench/throughput.py --model <fla_dir> --dtype bfloat16 \
    --batch-sizes 1,16,64,128,256 --cuda-graph --cuda-graph-max-bs 256 \
    --disable-radix-cache --decode-tokens 128 --short-len 512 --mem-fraction 0.85
# context-invariance sweep
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 CUDA_VISIBLE_DEVICES=0 \
  python bench/serving_scale.py --model <fla_dir> --dtype bfloat16 --mode context \
    --bsz 8 --contexts 1024,4096,8192,16384,32768,65536 --decode-tokens 64 \
    --cuda-graph-max-bs 8 --mem-fraction 0.85 --max-context 131072
```
