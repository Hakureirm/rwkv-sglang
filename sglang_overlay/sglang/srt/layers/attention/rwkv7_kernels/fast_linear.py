# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""JIT loader + adapter for the fused fp16 decode GEMV (rwkv7_fast.cu).

Replaces the M==1 (bsz1 decode) r/k/v/o + ffn projection GEMVs with an fp32-accumulate
fused CUDA GEMV adapted from BlinkDL/Albatross (Apache-2.0; see cuda/NOTICE). fp16-only
(the op reads/writes ``at::Half``): our precision-matched comparison target is
ours-fp16 vs albatross-fp16, and on Ampere fp16==bf16 speed. bf16 / fp32 / quantized /
any M>1 keep the existing torch (ReplicatedLinear/cuBLAS) path. Our WKV recurrent state
stays fp32 (untouched); the LoRA down/up projections use sglang's ReplicatedLinear.

Built WITHOUT --use_fast_math (IEEE arithmetic, no FTZ) so the greedy-EXACT + batch-
invariance gates hold. cuda-graph safe: static shapes, current stream, no host sync.
Enabled via the model (RWKV_FAST_LINEAR env, default off; see docs/findings/0015).
"""

import json
import os
from pathlib import Path

import torch

_EXT_LOADED = False
_LOAD_FAILED = False

# ---------------------------------------------------------------------------
# Arch-aware launch autotune for gemv_m1 (F0023 §5 roadmap #6).
#
# albatross's linear dispatch (rwkv7_fast_v3a.py:619) is a hand-frozen
# per-(group,C,rows) table of magic (threads,tile) constants tuned for ONE GPU
# (5090), with zero runtime arch branching — so it is mis-tuned on any other
# card. Our old gemv_m1 had the SAME weakness, coarser (a single <128,2>/<128,1>
# chosen only by N parity). Here we pick (threads, out_tile) from (sm_arch, N, K)
# via a one-time warmup autotune, cached in-process + on disk. Occupancy of these
# kernels is compile-time (regs/smem), so (sm_arch, N, K) is the full key.
#
# NUMERICS DISCIPLINE (the autotune rule: a config knob may change speed, never
# logits). In gemv_m1_kernel each output's fp32 accumulation order depends ONLY
# on Threads (per-thread k-stride = Threads*4, then a fixed warp-shuffle tree +
# serial warp sum); OutTile changes which block computes a row but not the
# order. So: OutTile is logits-invariant (tuned freely), while crossing a
# Threads class CAN change logits by ulp-level reassociation. Default autotune
# therefore searches OutTile at the heuristic's fixed Threads. The full
# (threads x out_tile) space needs RWKV_GEMV_AUTOTUNE_FULL=1, whose contract is
# an explicit greedy-oracle re-gate (bench/autotune_gemv.py --full does both).
#
# cuda-graph safety: we NEVER benchmark while a graph is capturing (host sync
# would corrupt capture) — under capture we fall back to the heuristic. Autotune
# runs during sglang's eager warmup forwards, freezing the choice before capture.
# ---------------------------------------------------------------------------
_CFG_CACHE: dict = {}          # (arch, N, K) -> (threads, out_tile)
_CFG_DISK_LOADED = False
_CANDIDATE_THREADS = (64, 128, 256)
_CANDIDATE_OUTTILE = (4, 2, 1)  # prefer larger tile (fewer blocks) when valid
_AUTOTUNE = os.environ.get("RWKV_GEMV_AUTOTUNE", "0") == "1"
_AUTOTUNE_FULL = os.environ.get("RWKV_GEMV_AUTOTUNE_FULL", "0") == "1"


def _arch_key() -> int:
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _cache_path() -> Path:
    name = "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        name = name.replace(" ", "_").replace("/", "_")
    d = Path(os.path.expanduser("~/.cache/rwkv7_fast"))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"gemv_autotune_{name}.json"


def _load_disk_cache():
    global _CFG_DISK_LOADED
    if _CFG_DISK_LOADED:
        return
    _CFG_DISK_LOADED = True
    try:
        p = _cache_path()
        if p.exists():
            for k, v in json.load(open(p)).items():
                a, n, kk = (int(x) for x in k.split(","))
                _CFG_CACHE[(a, n, kk)] = (int(v[0]), int(v[1]))
    except Exception:
        pass


def _save_disk_cache():
    try:
        out = {f"{a},{n},{k}": list(v) for (a, n, k), v in _CFG_CACHE.items()}
        p = _cache_path()
        tmp = p.with_suffix(f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, p)  # atomic: concurrent TP workers can't tear the file
    except Exception:
        pass


def _heuristic_config(N: int, K: int) -> tuple:
    """Closed-form fallback (used under cuda-graph capture or autotune-off).
    Grounded in the kernel's actual behavior: k-loop steps threads*4 elems, so
    pick threads with K/(threads*4) >= 2; pick the largest out_tile that divides
    N while keeping grid = N/out_tile >= ~2*numSM to bury the wave tail."""
    numsm = 82
    try:
        numsm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    except Exception:
        pass
    threads = 128
    if K >= 4096:
        threads = 256
    elif K < 512:
        threads = 64
    # F0027 (measured cross-arch occupancy): on sm_86 (A10G/3090, maxBlocks/SM=16)
    # a 64-thread block caps at 66.7% occupancy (block-count-cap); 128 reaches 100%.
    # Never drop below 128 threads there. (sm_89/L4 has maxBlocks=24 → 64 is fine.)
    if threads < 128 and _arch_key() == 86:
        threads = 128
    out_tile = 1
    for ot in (4, 2, 1):
        if N % ot == 0 and (N // ot) >= 2 * numsm:
            out_tile = ot
            break
    else:
        out_tile = 2 if (N % 2 == 0) else 1
    return (threads, out_tile)


def _valid_configs(N: int, threads: int = None):
    """Candidate (threads, out_tile) pairs. With `threads` set, the search is
    restricted to that single Threads class — the logits-invariant subspace
    (see NUMERICS DISCIPLINE above). threads=None = the full space (needs the
    RWKV_GEMV_AUTOTUNE_FULL greedy-re-gate contract)."""
    classes = _CANDIDATE_THREADS if threads is None else (threads,)
    for t in classes:
        for ot in _CANDIDATE_OUTTILE:
            if N % ot == 0:
                yield (t, ot)


def _autotune_config(N: int, K: int) -> tuple:
    """Micro-benchmark candidate (threads,out_tile) for this (N,K) with CUDA
    events; return the fastest. Only called outside graph capture (warmup).
    Default: OutTile-only at the heuristic Threads (logits-invariant); the full
    space only under RWKV_GEMV_AUTOTUNE_FULL=1."""
    dev = torch.device("cuda")
    x = torch.randn(1, K, dtype=torch.float16, device=dev)
    w = torch.randn(N, K, dtype=torch.float16, device=dev)
    best = _heuristic_config(N, K)
    best_t = float("inf")
    t_lock = None if _AUTOTUNE_FULL else best[0]
    for (t, ot) in _valid_configs(N, threads=t_lock):
        try:
            for _ in range(5):  # warm
                torch.ops.rwkv7_fast.gemv_m1_cfg(x, w, t, ot)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(50):
                torch.ops.rwkv7_fast.gemv_m1_cfg(x, w, t, ot)
            e.record()
            torch.cuda.synchronize()
            ms = s.elapsed_time(e) / 50.0
            if ms < best_t:
                best_t, best = ms, (t, ot)
        except Exception:
            continue
    return best


def _select_config(N: int, K: int) -> tuple:
    heur = _heuristic_config(N, K)
    if not _AUTOTUNE:
        # autotune off: pure closed-form, and do NOT consult the disk cache —
        # numerics/perf must not depend on hidden per-machine state.
        return heur
    _load_disk_cache()
    key = (_arch_key(), int(N), int(K))
    if key in _CFG_CACHE:
        cfg = _CFG_CACHE[key]
        # a cached config from a FULL (threads-crossing) tune is only honored
        # when the full-tune contract is active; otherwise stay in the
        # heuristic's logits-invariant Threads class.
        if cfg[0] == heur[0] or _AUTOTUNE_FULL:
            return cfg
    # never benchmark (host sync) while capturing a cuda graph
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:
        capturing = False
    if capturing:
        if _AUTOTUNE_FULL:
            # pin for process-lifetime consistency: a later eager autotune must
            # not switch Threads class between the captured graph and eager path
            _CFG_CACHE[key] = heur
        return heur
    cfg = _autotune_config(N, K)
    _CFG_CACHE[key] = cfg
    _save_disk_cache()
    return cfg


def _ensure_loaded():
    """JIT-build + register torch.ops.rwkv7_fast.gemv_m1 on first use. Idempotent."""
    global _EXT_LOADED, _LOAD_FAILED
    if _EXT_LOADED:
        return True
    if _LOAD_FAILED:
        return False
    try:
        from torch.utils.cpp_extension import load

        cuda_dir = Path(__file__).parent / "cuda"
        load(
            name="rwkv7_fast",
            sources=[str(cuda_dir / "rwkv7_fast.cu")],
            is_python_module=False,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        )
        _register_fakes()
        _EXT_LOADED = True
        return True
    except Exception as e:  # pragma: no cover - build env dependent
        print(f"[rwkv7_fast] JIT load failed, falling back to torch: {e}")
        _LOAD_FAILED = True
        return False


def _register_fakes():
    """Fake impls so torch.compile / piecewise cuda-graph tracing doesn't graph-
    break on the custom ops (mirrors glue.py/lora_fused.py)."""
    try:
        @torch.library.register_fake("rwkv7_fast::gemv_m1")
        def _fm1(x, weight):
            return x.new_empty((1, weight.shape[0]))

        @torch.library.register_fake("rwkv7_fast::gemv_m1_cfg")
        def _fm1c(x, weight, threads, out_tile):
            return x.new_empty((1, weight.shape[0]))

        @torch.library.register_fake("rwkv7_fast::gemv_m1_sqrelu_cfg")
        def _fm1sq(x, weight, threads, out_tile):
            return x.new_empty((1, weight.shape[0]))

        @torch.library.register_fake("rwkv7_fast::gemv_mb_cfg")
        def _fmbc(x, weight, threads, out_tile):
            return x.new_empty((x.shape[0], weight.shape[0]))
    except Exception:
        pass  # older torch without register_fake -> caller disables piecewise cuda graph


def available() -> bool:
    return _ensure_loaded()


def gemv_m1(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """y[1,N] = x[1,K] @ weight[N,K]^T. weight = torch nn.Linear .weight (fp16).

    Caller (models/rwkv7.py::_proj_gemv) guarantees M==1, fp16, contiguous, K%4==0
    before dispatching here; anything else takes the torch path.

    Arch-aware: picks (threads, out_tile) via _select_config (autotuned per
    (sm_arch, N, K), cached) instead of the fixed <128,2>. Falls back to the
    original fixed-config op if the cfg op is unavailable (older build)."""
    xc = x.contiguous().view(1, -1)
    N, K = weight.size(0), weight.size(1)
    t, ot = _select_config(N, K)
    try:
        return torch.ops.rwkv7_fast.gemv_m1_cfg(xc, weight, t, ot)
    except Exception:
        return torch.ops.rwkv7_fast.gemv_m1(xc, weight)


def gemv_m1_sqrelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Epilogue-fused FFN channel-mix key projection: returns relu(x @ weight^T)**2
    for M==1, bit-identical to ``torch.relu(gemv_m1(x, weight)) ** 2`` (verified by
    bench/test_sqrelu_gate.py). Uses the SAME arch-aware (threads, out_tile) that
    gemv_m1 picks for this (N, K), so the fp32 accumulation — hence the fp16 `k` the
    activation reads — is identical to the plain path; only the relu+square are folded
    into the store (2 fewer launches + no k[1,N] HBM round-trip per FFN block).

    Caller (models/rwkv7.py) guarantees M==1, fp16, contiguous, K%4==0, N even before
    dispatching here; anything else takes the two-step torch path."""
    xc = x.contiguous().view(1, -1)
    N, K = weight.size(0), weight.size(1)
    t, ot = _select_config(N, K)
    return torch.ops.rwkv7_fast.gemv_m1_sqrelu_cfg(xc, weight, t, ot)


def gemv_mb(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """y[M,N] = x[M,K] @ weight[N,K]^T, batch-INVARIANT: row m is bit-identical to
    gemv_m1(x[m]) because it uses the same per-output fp32 reduction and the SAME
    (threads, out_tile) the decode path picks for (N,K). One launch for all M rows.

    Purpose: the chain-spec verify computes the target over K positions in a single
    launch while staying bit-exact against the M=1 baseline decode — this is the
    F0031 increment-(ii) exactness path (the M=K cuBLAS GEMM reduction order was the
    lone gate flip). Caller guarantees fp16, contiguous, K%4==0."""
    xc = x.contiguous()
    N, K = weight.size(0), weight.size(1)
    t, ot = _select_config(N, K)
    return torch.ops.rwkv7_fast.gemv_mb_cfg(xc, weight, t, ot)
