"""Greedy decode of the oracle fixture under whatever flags the environment carries.

Two things this file does deliberately, both learned the hard way:

* It is a real file, not a heredoc on stdin. sglang's scheduler subprocess
  re-resolves the entry script by path, so `python3 -` aborts with
  `FileNotFoundError: <stdin>` before the model ever loads.
* Everything runs under `if __name__ == "__main__"`. sglang spawns its scheduler,
  and a spawned child re-imports the entry module -- with the work at module
  level, every child re-enters `sgl.Engine(...)` and multiprocessing kills it
  with `_check_not_importing_main`.

Both failures leave a zero-byte output file, which is why the comparator that
consumes this refuses to compare empty files instead of calling them equal.
"""

import json
import sys


def main():
    model, fixture, out = sys.argv[1], sys.argv[2], sys.argv[3]
    import sglang as sgl

    fx = json.load(open(fixture))
    engine = sgl.Engine(
        model_path=model,
        skip_tokenizer_init=True,
        disable_radix_cache=True,
        disable_prefill_cuda_graph=True,
        dtype="float16",
        tp_size=1,
        mem_fraction_static=0.85,
    )
    res = engine.generate(
        input_ids=[fx["prompt_tokens"]],
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": len(fx["greedy_tokens"]),
            "ignore_eos": True,
        },
    )
    ids = res[0]["output_ids"]
    assert len(ids) == len(fx["greedy_tokens"]), f"short generation: {len(ids)}"
    with open(out, "w") as fh:
        fh.write(json.dumps(ids) + "\n")
    engine.shutdown()


if __name__ == "__main__":
    main()
