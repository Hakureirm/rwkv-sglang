"""Bytes a bsz1 decode step actually reads, tensor by tensor.

The whole-file size is the wrong denominator: the embedding table is indexed, so
a step reads one row of it, not all of it. Getting this wrong makes the card look
slower than it is and puts a reference implementation above the memory roof,
which is how the error announces itself.
"""
import json
import struct
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    (n,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(n))

DT = {"F16": 2, "BF16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1}
full, gathered, total = 0, 0, 0
rows = []
for name, meta in header.items():
    if name == "__metadata__":
        continue
    shape = meta["shape"]
    nb = 1
    for s in shape:
        nb *= s
    nb *= DT.get(meta["dtype"], 2)
    total += nb
    # An embedding is read by index: one row per token. Everything else in a
    # decode step is a matmul operand or a per-channel vector, read in full.
    is_emb = ("emb" in name or "embed" in name) and "head" not in name and len(shape) == 2
    if is_emb:
        gathered += nb
        rows.append((name, shape, nb, "gathered (1 row)"))
    else:
        full += nb
        rows.append((name, shape, nb, "full"))

rows.sort(key=lambda r: -r[2])
print(f"{'tensor':<44} {'shape':>18} {'MB':>9}  read")
for name, shape, nb, how in rows[:12]:
    print(f"{name:<44} {str(shape):>18} {nb/1e6:>9.1f}  {how}")
print(f"... {len(rows)-12} more" if len(rows) > 12 else "")

per_token = full + (gathered and 0)  # the gathered row itself is a few KB
print(f"\nfile total          : {total/1e9:.4f} GB")
print(f"gathered (embedding): {gathered/1e9:.4f} GB  -- one row per token, not the table")
print(f"READ PER DECODE TOKEN: {per_token/1e9:.4f} GB")
