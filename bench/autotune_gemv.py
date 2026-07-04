"""Build + correctness-gate + autotune-seed for gemv_m1_cfg (F0023 roadmap #6).

Numerics model (see fast_linear.py NUMERICS DISCIPLINE): each output's fp32
accumulation order depends ONLY on the Threads class; OutTile is logits-
invariant. This tool therefore gates in two tiers:
  1. WITHIN-CLASS byte-identity (hard gate, exit 1 on failure): for every
     Threads class, all valid OutTile variants must be bitwise identical, and
     the 128-class must equal the fixed gemv_m1 (which is <128,2>/<128,1>).
  2. CROSS-CLASS drift (report): 64/256-class outputs vs the 128-class —
     ulp-level reassociation drift is EXPECTED here; a cross-class winner is
     only seeded with --full, and the printed contract is to re-run the greedy
     oracle gate (bench/verify_batch.py) before serving with it.

Default run seeds only winners in the heuristic's Threads class (safe: proven
byte-identical to the gated configuration). `--full` sweeps all classes.

Run on the box (GPU):  python bench/autotune_gemv.py [--full]
"""
import argparse
import sys
from pathlib import Path

import torch

# load the JIT ext + helpers from the overlay
OV = Path(__file__).resolve().parents[1] / "sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels"
sys.path.insert(0, str(OV))
import fast_linear as FL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true",
                help="sweep all Threads classes (cross-class winners require a greedy re-gate)")
args = ap.parse_args()

assert FL.available(), "rwkv7_fast ext failed to build"
dev = torch.device("cuda")
arch = FL._arch_key()
name = torch.cuda.get_device_name(0)
print(f"GPU={name} sm_{arch} SMs={torch.cuda.get_device_properties(0).multi_processor_count}"
      f"  mode={'FULL (cross-class, needs greedy re-gate)' if args.full else 'class-locked (safe)'}")

# RWKV-7 1.5B M==1 GEMV shapes routed through gemv_m1 (hidden=2048, inter=8192):
# r/k/v/o (2048x2048), ffn_key (8192x2048), ffn_value (2048x8192).
SHAPES = [("att_rkvo", 2048, 2048), ("ffn_key", 8192, 2048), ("ffn_value", 2048, 8192)]


def bench_cfg(x, w, t, ot, iters=200):
    for _ in range(20):
        torch.ops.rwkv7_fast.gemv_m1_cfg(x, w, t, ot)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        torch.ops.rwkv7_fast.gemv_m1_cfg(x, w, t, ot)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


gate_ok = True
seeded = {}
print(f"\n{'shape':10s} {'N':>6s} {'K':>6s} | {'fixed<128,2>':>12s} | {'best cfg':>10s} {'best ms':>9s} | speedup")
print("-" * 88)
for tag, N, K in SHAPES:
    x = torch.randn(1, K, dtype=torch.float16, device=dev)
    w = torch.randn(N, K, dtype=torch.float16, device=dev)
    ref_fp32 = (x.float() @ w.float().t())
    y_plain = torch.ops.rwkv7_fast.gemv_m1(x, w)

    # tier 1: within-class byte-identity for EVERY class (hard gate)
    class_y = {}
    for t in FL._CANDIDATE_THREADS:
        ys = [torch.ops.rwkv7_fast.gemv_m1_cfg(x, w, t, ot)
              for ot in FL._CANDIDATE_OUTTILE if N % ot == 0]
        same = all(torch.equal(ys[0], yi) for yi in ys[1:])
        if not same:
            print(f"  GATE FAIL: {tag} threads={t}: OutTile variants NOT byte-identical")
            gate_ok = False
        class_y[t] = ys[0]
    if not torch.equal(class_y[128], y_plain):
        print(f"  GATE FAIL: {tag}: 128-class != fixed gemv_m1")
        gate_ok = False
    rel = (class_y[128].float() - ref_fp32).abs().max().item() / (ref_fp32.abs().max().item() + 1e-9)
    if rel > 1e-2:
        print(f"  GATE FAIL: {tag}: rel error vs fp32 reference {rel:.1e} > 1e-2")
        gate_ok = False

    # tier 2: cross-class drift report
    heur_t = FL._heuristic_config(N, K)[0]
    for t in FL._CANDIDATE_THREADS:
        if t != 128:
            ndiff = (class_y[t] != class_y[128]).sum().item()
            if ndiff:
                print(f"  note: {tag} threads={t} differs from 128-class in {ndiff}/{N} outs (ulp reassociation, expected)")

    # sweep: class-locked by default, all classes with --full
    fixed_ms = bench_cfg(x, w, 128, 2 if N % 2 == 0 else 1)
    best_t, best_ot, best_ms = None, None, float("inf")
    sweep_classes = FL._CANDIDATE_THREADS if args.full else (heur_t,)
    for t in sweep_classes:
        for ot in FL._CANDIDATE_OUTTILE:
            if N % ot != 0:
                continue
            ms = bench_cfg(x, w, t, ot)
            if ms < best_ms:
                best_ms, best_t, best_ot = ms, t, ot
    sp = fixed_ms / best_ms
    cross = best_t != heur_t and not torch.equal(class_y[best_t], class_y[heur_t])
    mark = "  [CROSS-CLASS: greedy re-gate required before serving]" if cross else ""
    print(f"{tag:10s} {N:6d} {K:6d} | {fixed_ms*1e3:10.2f}us | ({best_t:3d},{best_ot}) {best_ms*1e3:7.2f}us | {sp:.2f}x{mark}")
    seeded[(arch, N, K)] = (best_t, best_ot)

if not gate_ok:
    print("\nGATE FAILED — cache NOT seeded.")
    sys.exit(1)

FL._load_disk_cache()
FL._CFG_CACHE.update(seeded)
FL._save_disk_cache()
print(f"\ngate PASS; seeded {len(seeded)} shapes into {FL._cache_path()}")
if args.full:
    print("FULL mode: if any winner was cross-class, run bench/verify_batch.py (greedy oracle) "
          "with RWKV_GEMV_AUTOTUNE_FULL=1 before serving, and record the result.")
