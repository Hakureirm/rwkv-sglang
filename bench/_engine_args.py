"""Translate Engine kwargs across sglang revisions, refusing to drop one in silence.

sglang renames `ServerArgs` fields, and the harnesses here reacted to that by
filtering: `{k: v for k, v in kwargs.items() if k in ServerArgs.__dataclass_fields__}`,
with a comment naming the very switch being discarded ("e.g. main dropped
`disable_piecewise_cuda_graph`"). That is fine for a tuning knob and wrong for a
correctness one. For RWKV-7 the prefill CUDA graph is a correctness switch: with it
on, decode starts from a state the graph never wrote and greedy output diverges from
the oracle fixture on the *first* token. F0069 found this in `serving_scale.py`, where
it had invalidated a whole table; the same filter survived in `throughput.py` and
`verify_batch.py`, the second of which is the greedy-oracle gate itself.

So the policy here is: a caller who passed a kwarg gets it honoured under whichever
spelling this build accepts, or gets an exception. Nothing is dropped quietly.

    from _engine_args import resolve
    engine = sgl.Engine(**resolve(engine_kwargs))

Add a new spelling to `ALIASES` when upstream renames something; the tuple is ordered
newest-first, so a build that still carries both picks the current name.
"""

from __future__ import annotations

# canonical name our harnesses ask for -> spellings to try, newest first
ALIASES: dict[str, tuple[str, ...]] = {
    # `--disable-piecewise-cuda-graph` became `--disable-prefill-cuda-graph`, then
    # `--cuda-graph-backend-prefill=disabled`. The dataclass field behind the last
    # spelling is `cuda_graph_backend_prefill`, which takes a string, not a bool --
    # so it is handled separately below rather than by a plain rename.
    "disable_piecewise_cuda_graph": ("disable_prefill_cuda_graph", "disable_piecewise_cuda_graph"),
    # split into per-phase fields
    "cuda_graph_max_bs": ("cuda_graph_max_bs_decode", "cuda_graph_max_bs"),
}

# Kwargs whose loss changes what the run *means* rather than how fast it is. Kept as a
# note for readers: `resolve` refuses on everything, because a caller only passes what
# it wants, and "wanted it, did not get it, was not told" is the failure being fixed.
CORRECTNESS = ("disable_piecewise_cuda_graph", "disable_radix_cache")


def resolve(kwargs: dict) -> dict:
    """Return `kwargs` with every key mapped to a spelling this sglang accepts.

    Raises `RuntimeError` naming the key if no spelling exists, rather than dropping it.
    """
    from sglang.srt.server_args import ServerArgs

    fields = ServerArgs.__dataclass_fields__
    out: dict = {}
    for key, value in kwargs.items():
        for name in ALIASES.get(key, (key,)):
            if name in fields:
                out[name] = value
                break
        else:
            # The prefill-graph switch has one more spelling that is not a rename: a
            # string-valued backend selector. Take it before giving up.
            if key == "disable_piecewise_cuda_graph" and "cuda_graph_backend_prefill" in fields:
                out["cuda_graph_backend_prefill"] = "disabled" if value else "full"
                continue
            raise RuntimeError(
                f"this sglang build accepts none of {ALIASES.get(key, (key,))} for the "
                f"kwarg `{key}` this harness passed. Add the current spelling to "
                f"bench/_engine_args.py::ALIASES — do not drop it: "
                + (
                    "this one decides whether the run is measuring the configuration it "
                    "claims to (see the module docstring)."
                    if key in CORRECTNESS
                    else "a silently discarded knob means the number describes a "
                    "configuration nobody chose."
                )
            )
    return out
