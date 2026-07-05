# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""RWKV-7 chain speculative decoding (ADR-0006): bespoke draft/verify loop.

Why not EAGLE: sglang's speculative infra verifies a token TREE in one target
forward by attending over the KV cache at all candidate positions, and rolls
back by simply not committing rejected KV pages. RWKV-7 has no KV cache and no
attention — its per-request state is an O(1) recurrence advanced token by
token. The RWKV analogue (this worker):

  round:
    1. draft (0.1B, same tokenizer family) greedily proposes K tokens,
       advancing its own O(1) state (K cheap decode steps);
    2. target runs its recurrence over the K tokens VIA THE EXTEND PATH in a
       single forward -> K logits (chain-verify; snapshot conv/temporal for
       the slot first, restore after — the extend path commits final_state);
    3. accept the longest prefix J where draft[j] == argmax(target[j]); the
       target's own argmax at position J is appended too (J+1 tokens/round);
    4. commit: re-run the target extend over the J+1 accepted tokens from the
       snapshot (ADR-0006 option (b): two target forwards per round, zero
       kernel changes; option (a) checkpoint-per-token is the follow-up).

The state snapshot/rollback is O(1) per request and small (1.5B: ~12.6 MB
temporal + 2 conv rows per slot), which is exactly why spec-decode is cheap
for RWKV where a transformer would juggle KV pages.

Gate (hard): spec-on output must be token-identical to spec-off greedy for
the same prompts (acceptance math never changes the committed sequence at
temperature 0). Then measure accepted-length/round and net tok/s.

Status: skeleton under active build (task #10). bsz1 first; small-batch
follow-up. Wire-up: server_args --speculative-algorithm RWKV_CHAIN
--speculative-draft-model-path <0.1B dir> --speculative-num-draft-tokens K.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class Rwkv7StateSnapshot:
    """O(1) per-request recurrent-state checkpoint for one pool slot.

    Holds fp32 copies of the slot's rows across ALL layers:
      conv0/conv1: [num_layers, hidden, 1]
      temporal:    [num_layers, H, K, V]
    """

    slot: int
    conv0: torch.Tensor
    conv1: torch.Tensor
    temporal: torch.Tensor


def snapshot_slot(req_to_token_pool, layer_ids: List[int], slot: int) -> Rwkv7StateSnapshot:
    """Copy one slot's conv+temporal rows across layers (device-side, ~13 MB @1.5B)."""
    conv0_rows, conv1_rows, temporal_rows = [], [], []
    for lid in layer_ids:
        cache = req_to_token_pool.mamba2_layer_cache(lid)
        conv0_rows.append(cache.conv[0][slot].clone())
        conv1_rows.append(cache.conv[1][slot].clone())
        temporal_rows.append(cache.temporal[slot].clone())
    return Rwkv7StateSnapshot(
        slot=slot,
        conv0=torch.stack(conv0_rows),
        conv1=torch.stack(conv1_rows),
        temporal=torch.stack(temporal_rows),
    )


def restore_slot(req_to_token_pool, layer_ids: List[int], snap: Rwkv7StateSnapshot) -> None:
    """Write a snapshot back into the pool (the O(1) rollback)."""
    for i, lid in enumerate(layer_ids):
        cache = req_to_token_pool.mamba2_layer_cache(lid)
        cache.conv[0][snap.slot].copy_(snap.conv0[i])
        cache.conv[1][snap.slot].copy_(snap.conv1[i])
        cache.temporal[snap.slot].copy_(snap.temporal[i])


def chain_accept(
    draft_tokens: List[int], target_argmax: torch.Tensor
) -> Tuple[int, int]:
    """Longest-prefix acceptance for greedy (temperature-0) spec decoding.

    draft_tokens: the K proposed tokens.
    target_argmax: [K] target greedy tokens, target_argmax[j] = the target's
        choice AT position j given prefix + draft_tokens[:j].
    Returns (J, bonus_token): J accepted draft tokens (0..K), plus the
    target's own token at position J (always committed -> J+1 tokens/round).
    Note bonus at J==K uses target_argmax[K-1]'s SUCCESSOR logits, which the
    K-token verify does not produce — so J==K commits K tokens and the next
    round's verify position 0 supplies the bonus (keeps one forward/round).
    """
    k = len(draft_tokens)
    j = 0
    while j < k and int(target_argmax[j]) == draft_tokens[j]:
        j += 1
    if j >= k:
        return k, -1  # full acceptance: no bonus available from this forward
    return j, int(target_argmax[j])


class RwkvChainWorker:
    """Bespoke speculative worker for RWKV-7 (interface mirrors EAGLEWorker's
    surface used by the scheduler: draft() -> verify() -> commit accounting).

    BUILD PLAN (task #10):
      [x] state snapshot/restore primitives (above)
      [x] greedy chain acceptance (above)
      [ ] draft runner construction (StandaloneWorker pattern: TpModelWorker
          with its own mem pool at a small mem fraction; RWKV draft advances
          its own mamba pool — no KV plumbing)
      [ ] per-round驱动: K draft decode steps -> verify extend batch (packed
          K tokens on the target slot, snapshot/restore around) -> accept ->
          commit extend over J+1 -> scheduler accounting (reuse EAGLE's
          accept_length fields so streaming/detokenizer just work)
      [ ] gate: spec-on == spec-off token-identical (hard), then speed A/B
    """

    def __init__(self, *args, **kwargs):  # pragma: no cover - under build
        raise NotImplementedError(
            "RWKV_CHAIN speculative worker is under active build (ADR-0006)."
        )
