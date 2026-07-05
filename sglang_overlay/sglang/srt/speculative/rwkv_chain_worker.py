# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""RWKV-7 chain speculative decoding (ADR-0006): bespoke draft/verify loop.

Why not EAGLE: sglang's speculative infra verifies a token TREE in one target
forward by attending over the KV cache at all candidate positions, and rolls
back by simply not committing rejected KV pages. RWKV-7 has no KV cache and no
attention — its per-request state is an O(1) recurrence advanced token by
token. The RWKV analogue (this worker):

Round invariant (both pools, start of every round): the committed sequence
ends with t_last; each model's recurrent state has consumed tokens up to and
INCLUDING t_{last-1}; t_last is pending (consumed by nobody). A decode step
in sglang feeds the pending token and emits logits for the next position —
the invariant is exactly sglang's normal decode contract, which is what makes
the worker a drop-in.

  round (snapshot draft+target slots first, O(1) each):
    1. draft: K greedy decode steps. Step 1 feeds t_last -> d0; step i feeds
       d_{i-2} -> d_{i-1}. Draft state ends having consumed d_{K-2}.
    2. target chain-verify: ONE extend over [t_last, d0..d_{K-2}] (K input
       tokens — note: NOT the K draft tokens; d_{K-1} is never fed) -> K
       logits; logits[j] is the target's prediction FOR d_j given the prefix
       plus d_0..d_{j-1}. The extend path commits final_state (consumed
       d_{K-2}) — that is why the snapshot exists.
    3. accept J = longest prefix with argmax(logits[j]) == d_j:
       - J == K (full accept): commit d0..d_{K-1}; new t_last = d_{K-1}. The
         verify's committed final_state (consumed d_{K-2} = new t_{last-1})
         and the draft's own state are BOTH already exactly the invariant —
         keep them. Zero restores, zero extra forwards, no bonus token (the
         next round's verify position 0 supplies it).
       - J < K: bonus b = argmax(logits[J]); commit d0..d_{J-1} + b; new
         t_last = b. Restore BOTH slots, then commit-extend BOTH models over
         [t_last^old, d0..d_{J-1}] (J+1 tokens, ends consumed at d_{J-1} =
         new t_{last-1}; b stays pending). J=0 degenerates to a 1-token
         extend feeding just t_last^old.
  Tokens/round: J+1 (J<K) or K (J==K). Forwards/round: draft K + [J<K],
  target 1 + [J<K] — at alpha=0.738, K=4, P(J==K)~0.30, so ~1.7 target
  forwards/round, not 2 (ADR-0006 option (b); option (a) checkpoint-per-
  token inside the WKV kernel removes the re-run and is the follow-up).

Speed roadmap (be honest about increment (i)): eager 0.1B decode steps and
eager K-token extends are LAUNCH-BOUND (~ms floor each), so the correctness
increment may not beat the cuda-graphed plain baseline at 1.5B — the F0029
net-speedup estimate (~1.99x @1.5B, ~2.69x @7.2B) prices forwards at
FLOP-cost, which only holds once the per-round forwards are graphed.
Increment (ii) = draft-decode graph + fixed-shape K-token verify graph
(bsz1 verify extend has a STATIC shape — graphable with the same
mamba_cache_indices-buffer machinery as our decode graph).

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

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


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
    """Bespoke speculative worker for RWKV-7 (V1/no-overlap scheduler surface:
    the scheduler sets `self.model_worker = draft_worker` and calls
    `forward_batch_generation(schedule_batch)` each iteration; V1 contract =
    return a FLAT next-token tensor + accept_length_per_req_cpu, and the
    worker itself appends req.output_ids / calls req.check_finished()).

    Deliberately NOT an EAGLEWorker subclass: EAGLE's machinery (token tree,
    KV-page alloc/evict per verify, per-step draft attention backends, target
    hidden-state feeding) all assumes paged attention on both models; with a
    recurrent draft AND target every one of those is a landmine. We subclass
    TpModelWorker directly (that constructs the draft model runner) and keep
    the target worker by reference, exactly like StandaloneWorker does — but
    with OWN pools for the draft (req_to_token_pool=None), because sharing
    the target's HybridReqToTokenPool would alias the two models' recurrent
    states (EAGLE shares on purpose: its draft is a head of the target).

    Lazily constructed at scheduler init via SpeculativeAlgorithm.RWKV_CHAIN
    (spec_info.py micro-patch) with the standard draft_worker_kwargs.
    """

    def __init__(
        self,
        server_args,
        gpu_id,
        tp_rank,
        dp_rank,
        moe_ep_rank,
        attn_cp_rank,
        moe_dp_rank,
        nccl_port,
        target_worker,
    ):
        from sglang.srt.managers.tp_worker import TpModelWorker

        self.target_worker = target_worker
        self.server_args = server_args
        self.k = server_args.speculative_num_draft_tokens or 4

        # Draft loads speculative_draft_model_path (TpModelWorker resolves it
        # via is_draft_worker); force its context to the target's and keep the
        # draft eager for increment (i) — graphs are the speed increment (ii).
        ctx_backup = server_args.context_length
        cg_backup = server_args.disable_cuda_graph
        server_args.context_length = target_worker.model_runner.model_config.context_len
        server_args.disable_cuda_graph = True
        try:
            self._draft = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                # OWN pools (None => the kv-cache mixin builds a fresh
                # HybridReqToTokenPool + MambaPool for the draft model).
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )
        finally:
            server_args.context_length = ctx_backup
            server_args.disable_cuda_graph = cg_backup

        self.draft_runner = self._draft.model_runner
        self.target_runner = target_worker.model_runner
        self.draft_pool = self.draft_runner.req_to_token_pool
        self.target_pool = self.target_runner.req_to_token_pool
        d_cfg = self.draft_runner.model_config.hf_config
        t_cfg = self.target_runner.model_config.hf_config
        self.draft_layers = list(range(d_cfg.num_hidden_layers))
        self.target_layers = list(range(t_cfg.num_hidden_layers))
        # rid -> draft pool slot (the draft mirrors every running request)
        self._draft_slot: Dict[str, int] = {}
        self._spec_rounds = 0
        self._spec_accept_sum = 0
        logger.info(
            "RWKV_CHAIN spec worker up: draft=%s K=%d (eager increment (i))",
            server_args.speculative_draft_model_path,
            self.k,
        )

    # ------------------------------------------------------------------
    # scheduler surface (mirrors what Scheduler pokes on draft_worker)
    # ------------------------------------------------------------------

    @property
    def model_runner(self):  # scheduler reads draft_worker.model_runner.*
        return self.draft_runner

    @property
    def model_config(self):  # scheduler.init_disaggregation reads this
        return self._draft.model_config

    def get_memory_pool(self):
        return (
            self.target_runner.req_to_token_pool,
            self.target_runner.token_to_kv_pool_allocator,
        )

    def clear_cache_pool(self):
        self._draft_slot.clear()

    def forward_batch_generation(self, batch):
        """V1 entry point. Extend -> normal target prefill + draft prefill;
        Decode -> one chain-speculative round per request."""
        if batch.forward_mode.is_extend() or getattr(batch, "is_extend_in_batch", False):
            return self._forward_extend(batch)
        return self._decode_round(batch)

    # ------------------------------------------------------------------
    # prefill: target as usual, then mirror the chunk into the draft
    # ------------------------------------------------------------------

    def _forward_extend(self, batch):
        from sglang.srt.managers.utils import GenerationBatchResult

        model_worker_batch = batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)

        # Mirror the prompt chunk into the draft model's own state pool so the
        # round invariant holds: both models consumed the prompt, the sampled
        # first token is pending. (Single-chunk prompts for increment (i);
        # multi-chunk chunked-prefill mirroring is a marked follow-up.)
        self._draft_prefill_mirror(batch)
        return batch_result

    # ------------------------------------------------------------------
    # decode: the chain round (ADR-0006)
    # ------------------------------------------------------------------

    def _decode_round(self, batch):
        from sglang.srt.managers.utils import GenerationBatchResult

        flat_tokens: List[int] = []
        accept_lens: List[int] = []

        for i, req in enumerate(batch.reqs):
            appended = self._round_one(batch, i, req)
            flat_tokens.extend(appended)
            accept_lens.append(len(appended) - 1)
            self._spec_rounds += 1
            self._spec_accept_sum += len(appended)
            if req.finished():
                self._release_draft_slot(req)

        # Scheduler-side length accounting (we own it in spec mode: see
        # ScheduleBatch.prepare_for_decode's early return).
        self._sync_batch_lens(batch)

        next_token_ids = torch.tensor(
            flat_tokens, dtype=torch.int64, device=batch.device
        )
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=next_token_ids,
            num_accepted_tokens=sum(accept_lens),
            accept_length_per_req_cpu=accept_lens,
            can_run_cuda_graph=False,
        )

    def _round_one(self, batch, i, req) -> List[int]:
        """One speculative round for one request. Returns the tokens actually
        appended to req.output_ids (bounded by finish conditions)."""
        target_req_idx = int(batch.req_pool_indices[i])
        draft_req_idx = self._draft_slot[req.rid]
        # snapshot/restore index into the mamba state tensors, NOT the req table
        target_mslot = self._mamba_idx(self.target_pool, target_req_idx)
        draft_mslot = self._mamba_idx(self.draft_pool, draft_req_idx)
        # committed sequence length INCLUDING the pending token t_last
        seq_committed = len(req.origin_input_ids) + len(req.output_ids)
        t_last = req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]

        snap_d = snapshot_slot(self.draft_pool, self.draft_layers, draft_mslot)
        snap_t = snapshot_slot(self.target_pool, self.target_layers, target_mslot)

        # 1) draft proposes K tokens (K eager decode steps on its own pool)
        drafts = self._draft_decode_steps(
            req, draft_req_idx, t_last, seq_committed, batch.sampling_info
        )

        # 2) chain-verify: ONE target extend over [t_last, d0..d_{K-2}]
        #    -> hidden for all K positions -> lm_head -> argmax per position
        verify_in = [t_last] + drafts[:-1]
        target_argmax = self._target_verify(
            req, target_req_idx, verify_in, seq_committed - 1, batch.sampling_info
        )

        # 3) accept
        j, bonus = chain_accept(drafts, target_argmax)
        if j >= self.k:
            committed = list(drafts)  # both states already correct — keep
        else:
            committed = drafts[:j] + [bonus]
            restore_slot(self.target_pool, self.target_layers, snap_t)
            restore_slot(self.draft_pool, self.draft_layers, snap_d)
            commit_in = [t_last] + drafts[:j]  # J+1 tokens; bonus stays pending
            self._extend_no_capture(
                self.target_runner, req, target_req_idx, commit_in,
                seq_committed - 1, batch.sampling_info,
            )
            self._extend_no_capture(
                self.draft_runner, req, draft_req_idx, commit_in,
                seq_committed - 1, batch.sampling_info,
            )

        # 4) accounting (V1: the worker appends + finish-checks itself)
        appended: List[int] = []
        for tok in committed:
            req.output_ids.append(tok)
            appended.append(tok)
            req.check_finished()
            if req.finished():
                break
        req.spec_verify_ct += 1
        if hasattr(req, "spec_accepted_tokens"):
            req.spec_accepted_tokens += max(len(appended) - 1, 0)
        return appended

    def _sync_batch_lens(self, batch):
        lens = [
            len(r.origin_input_ids) + len(r.output_ids) for r in batch.reqs
        ]
        batch.seq_lens = torch.tensor(
            lens, dtype=batch.seq_lens.dtype, device=batch.seq_lens.device
        )
        if getattr(batch, "seq_lens_cpu", None) is not None:
            batch.seq_lens_cpu = torch.tensor(lens, dtype=torch.int64)
        batch.seq_lens_sum = int(sum(lens))

    def _release_draft_slot(self, req):
        idx = self._draft_slot.pop(req.rid, None)
        shim = getattr(self, "_draft_shim", {}).pop(req.rid, None)
        if idx is None or shim is None:
            return
        try:
            # order: free_mamba_cache first (base free() clears req_pool_idx)
            if shim.mamba_pool_idx is not None:
                self.draft_pool.free_mamba_cache(shim)
            self.draft_pool.free(shim)
        except Exception:  # pool API drift — leaking one slot beats crashing
            logger.warning("draft slot %s free failed", idx, exc_info=True)

    # ------------------------------------------------------------------
    # forward builders. The per-round draft/verify/commit forwards live
    # OUTSIDE the scheduler's normal flow, so we hand-build ModelWorkerBatch
    # objects (v0.5.10 contract: ForwardBatch.init_new copies seq_lens_sum
    # verbatim, derives DECODE positions as clamp(seq_lens-1), derives EXTEND
    # positions/extend_start_loc from extend_prefix_lens+extend_seq_lens, and
    # dereferences reqs for rids; the linear-attention backend maps
    # req_pool_indices -> mamba slots itself via get_mamba_indices()).
    # ------------------------------------------------------------------

    def _make_mwb(
        self,
        *,
        forward_mode,
        input_ids,
        req_pool_indices,
        seq_lens,
        out_cache_loc,
        reqs,
        sampling_info,
        capture_hidden_mode,
        extend_seq_lens=None,
        extend_prefix_lens=None,
        extend_num_tokens=None,
    ):
        from sglang.srt.managers.schedule_batch import ModelWorkerBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        is_extend = extend_seq_lens is not None
        return ModelWorkerBatch(
            forward_mode=forward_mode,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=seq_lens.to("cpu"),
            seq_lens_sum=int(seq_lens.sum().item()),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=is_extend,
            all_extend_in_batch=is_extend,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=None,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            multimodal_inputs=None,
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=None,
            sampling_info=sampling_info,
            reqs=reqs,
            # inner forwards are PLAIN forwards from the engine's viewpoint
            spec_algorithm=SpeculativeAlgorithm.NONE,
            capture_hidden_mode=capture_hidden_mode,
        )

    def _run_extend(self, runner, req, req_pool_idx, tokens, prefix_len, sampling_info, capture_full):
        """One extend forward over `tokens` for a single request on `runner`.
        Returns the LogitsProcessorOutput (hidden_states holds ALL positions
        iff capture_full)."""
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        dev = runner.device
        n = len(tokens)
        mwb = self._make_mwb(
            forward_mode=ForwardMode.EXTEND,
            input_ids=torch.tensor(tokens, dtype=torch.int64, device=dev),
            req_pool_indices=torch.tensor([req_pool_idx], dtype=torch.int64, device=dev),
            seq_lens=torch.tensor([prefix_len + n], dtype=torch.int64, device=dev),
            out_cache_loc=torch.zeros(n, dtype=torch.int64, device=dev),
            reqs=[req],
            sampling_info=sampling_info,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL if capture_full else CaptureHiddenMode.NULL
            ),
            extend_seq_lens=[n],
            extend_prefix_lens=[prefix_len],
            extend_num_tokens=n,
        )
        fb = ForwardBatch.init_new(mwb, runner)
        out = runner.forward(fb)
        return out.logits_output

    def _lm_head_argmax(self, hidden):
        """Per-position greedy tokens from captured hidden states, using the
        target's own lm_head weight + the sampler's fp32 argmax semantics
        (vocab-padded rows sliced off). tp=1 scope for increment (i)."""
        w = self.target_runner.model.lm_head.weight
        vocab = self.target_runner.model_config.vocab_size
        # Row-by-row [1,H]@[H,V] on purpose: the SAME shape as the decode
        # path's logits matmul -> same cuBLAS kernel -> same reduction order.
        # A single [K,H]@[H,V] GEMM reduces in a different order and can flip
        # near-tie argmaxes vs the plain decode baseline (observed ~1e-3/token
        # on the first gate run).
        h = hidden.to(w.dtype)
        out = [
            torch.matmul(h[i : i + 1], w.t()).float()[:, :vocab].argmax(dim=-1)
            for i in range(h.shape[0])
        ]
        return torch.cat(out)

    def _draft_prefill_mirror(self, batch):
        """Mirror the current extend chunk into the draft's own pool. With
        radix OFF chunks arrive in order, so the draft state stays in
        lockstep chunk by chunk (spec + radix cache is out of scope for
        increment (i) and guarded below)."""
        for i, req in enumerate(batch.reqs):
            prefix_len = len(req.prefix_indices) if req.prefix_indices is not None else 0
            if req.rid not in self._draft_slot:
                if prefix_len != 0:
                    raise RuntimeError(
                        "RWKV_CHAIN: radix prefix hit on a request the draft has "
                        "never seen — run spec decoding with --disable-radix-cache "
                        "(increment (i) scope)."
                    )
                self._ensure_draft_slot(req)
            chunk = req.fill_ids[prefix_len:]
            self._run_extend(
                self.draft_runner,
                req,
                self._draft_slot[req.rid],
                list(chunk),
                prefix_len,
                batch.sampling_info,
                capture_full=False,
            )

    def _ensure_draft_slot(self, req):
        import types

        shim = types.SimpleNamespace(
            rid=req.rid,
            req_pool_idx=None,
            mamba_pool_idx=None,
            mamba_ping_pong_track_buffer=None,
            mamba_next_track_idx=None,
        )
        idx = self.draft_pool.alloc([shim])
        assert idx is not None, "draft state pool exhausted"
        self._draft_slot[req.rid] = int(idx[0])
        self._draft_shim = getattr(self, "_draft_shim", {})
        self._draft_shim[req.rid] = shim

    def _mamba_idx(self, pool, req_pool_idx):
        return int(
            pool.get_mamba_indices(
                torch.tensor([req_pool_idx], dtype=torch.int64, device=pool.device)
            )[0]
        )

    def _draft_decode_steps(self, req, draft_req_idx, t_last, seq_committed, sampling_info):
        """K eager decode steps on the draft (EAGLE pattern: build ONE
        ForwardBatch, init attn metadata once, then mutate input_ids/positions
        with skip_attn_backend_init=True)."""
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        dev = self.draft_runner.device
        vocab = self.draft_runner.model_config.vocab_size
        mwb = self._make_mwb(
            forward_mode=ForwardMode.DECODE,
            input_ids=torch.tensor([t_last], dtype=torch.int64, device=dev),
            req_pool_indices=torch.tensor([draft_req_idx], dtype=torch.int64, device=dev),
            seq_lens=torch.tensor([seq_committed], dtype=torch.int64, device=dev),
            out_cache_loc=torch.zeros(1, dtype=torch.int64, device=dev),
            reqs=[req],
            sampling_info=sampling_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        fb = ForwardBatch.init_new(mwb, self.draft_runner)
        drafts: List[int] = []
        for step in range(self.k):
            out = self.draft_runner.forward(fb, skip_attn_backend_init=(step > 0))
            logits = out.logits_output.next_token_logits.float()
            tok = int(logits[0, :vocab].argmax(dim=-1))
            drafts.append(tok)
            if step + 1 < self.k:
                fb.input_ids = torch.tensor([tok], dtype=torch.int64, device=dev)
                fb.positions.add_(1)
        return drafts

    def _target_verify(self, req, target_req_idx, tokens, prefix_len, sampling_info):
        logits_output = self._run_extend(
            self.target_runner,
            req,
            target_req_idx,
            tokens,
            prefix_len,
            sampling_info,
            capture_full=True,
        )
        return self._lm_head_argmax(logits_output.hidden_states)

    def _extend_no_capture(self, runner, req, req_pool_idx, tokens, prefix_len, sampling_info):
        self._run_extend(
            runner, req, req_pool_idx, tokens, prefix_len, sampling_info, capture_full=False
        )
