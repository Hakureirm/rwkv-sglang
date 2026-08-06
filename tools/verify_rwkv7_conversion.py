"""Does the RWKV-7 converter hold on Bo's newest checkpoint family (G1i)?

Replaces an earlier version of this check that reported VERDICT: FAIL with 798
missing tensors and 0 differing. That shape was the tell: a converter that had
really dropped everything would not have written a full 1527.7M-parameter model.
Two bugs in the checker, neither in the converter:

  * it expected `model.blocks.N...` names; the converter emits `rwkv7.blocks.N...`
    (and a bare `head.weight`), so nothing matched and everything read as missing.
  * it asked for bit-identity while converting to float32 from a bfloat16 source,
    which cannot hold by construction.

This version derives the mapping from the output instead of assuming it, and
compares in the source dtype's exact upcast: bf16 -> fp32 is lossless, so
equality must be exact, not approximate.
"""
import glob
import os
import sys

import torch
from safetensors.torch import load_file

SRC, OUT = sys.argv[1], sys.argv[2]

src = torch.load(SRC, map_location="cpu", weights_only=True)
shards = sorted(glob.glob(os.path.join(OUT, "*.safetensors")))
dst = {}
for s in shards:
    dst.update(load_file(s))

print(f"source : {len(src)} tensors, dtype {next(iter(src.values())).dtype}")
print(f"output : {len(dst)} tensors, dtype {next(iter(dst.values())).dtype}, {len(shards)} shard(s)")

# Map by suffix rather than by an assumed prefix: whatever the converter chose to
# prefix with, a source key must appear as the tail of exactly one output key.
by_suffix = {}
for k in dst:
    by_suffix.setdefault(k, k)
    for cut in range(len(k)):
        if k[cut] == ".":
            by_suffix.setdefault(k[cut + 1:], k)

exact = differing = missing = 0
first_bad = []
for k, v in src.items():
    ok = dst.get(k) if k in dst else dst.get(by_suffix.get(k, ""))
    if ok is None:
        missing += 1
        if len(first_bad) < 4:
            first_bad.append(f"MISSING {k}")
        continue
    a = v.to(torch.float32)
    b = ok.to(torch.float32)
    if a.shape != b.shape:
        differing += 1
        if len(first_bad) < 4:
            first_bad.append(f"SHAPE {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
    elif torch.equal(a, b):
        exact += 1
    else:
        differing += 1
        if len(first_bad) < 4:
            d = (a - b).abs().max().item()
            first_bad.append(f"VALUE {k}: max|diff|={d:.3e}")

unconsumed = [k for k in dst if not any(k.endswith(s) for s in src)]
print(f"\nexact: {exact}/{len(src)}   differing: {differing}   missing: {missing}")
print(f"output keys not traceable to a source key: {len(unconsumed)}")
for line in first_bad:
    print("  ", line)
for k in unconsumed[:4]:
    print("   EXTRA", k)

ok = differing == 0 and missing == 0 and not unconsumed
print("\nVERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
