"""Standalone minimal repro of the sglang PP send_tensor_dict all-gather corruption
(upstream issue #30015), run on sglang MAIN — model-independent, uses only the
distributed primitive. tp=2 pp=2 (4 ranks). Run via torchrun on 4x NVIDIA L4.

Each rank builds a tensor whose VALUE differs per TP rank (a "TP-sharded" tensor).
Pipeline-rank-0 sends it to pipeline-rank-1 via pp_group.send_tensor_dict with
all_gather_group=tp_group (exactly what scheduler_pp_mixin does by default). If the
optimization were only applied to replicated tensors this would be lossless; because
the tensor is sharded, the receiver reconstructs a franken-tensor. We print, per
receiving rank, whether the received tensor equals what its matching sender sent.
"""
import os
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        get_pp_group,
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
    )

    init_distributed_environment(
        world_size=world, rank=rank, local_rank=local_rank,
        distributed_init_method="env://", backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)

    pp = get_pp_group()
    tp_group = get_tensor_model_parallel_group()
    tp_rank = get_tensor_model_parallel_rank()

    # A "TP-sharded" tensor: value encodes the tp_rank, so the two tp ranks hold
    # DIFFERENT data (unlike a replicated hidden_state). Shape divisible by tp so
    # the all-gather split path triggers.
    H = 8
    payload = torch.full((1, H), float(tp_rank + 1), device="cuda", dtype=torch.float16)
    sent_sig = payload.float().sum().item()

    if pp.is_first_rank:
        pp.send_tensor_dict(
            {"sharded": payload, "__msg_type__": "proxy"},
            all_gather_group=tp_group,   # what scheduler_pp_mixin passes by default
        )
        print(f"[rank{rank} tp{tp_rank}] SENT sharded sum={sent_sig:.1f} row={payload[0].tolist()}",
              flush=True)
    else:
        got = pp.recv_tensor_dict(all_gather_group=tp_group)
        r = got["sharded"]
        got_sig = r.float().sum().item()
        ok = abs(got_sig - sent_sig) < 1e-3 and torch.allclose(r, payload)
        print(f"[rank{rank} tp{tp_rank}] RECV sum={got_sig:.1f} row={r[0].tolist()} "
              f"expected_sum={sent_sig:.1f} -> {'OK' if ok else 'CORRUPTED'}", flush=True)

    dist.barrier()


if __name__ == "__main__":
    main()
