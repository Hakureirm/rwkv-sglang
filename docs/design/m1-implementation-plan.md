---
doc_kind: design
title: "M1 implementation plan — RWKV-7 in sglang v0.5.10.post1"
date: 2026-06-30
last_verified_commit: (initial)
source: workflow rwkv7-sglang-m1-design (3 readers: PR#41060 blueprint / sglang conventions / fla kernels; + synthesis)
related_adr: [0001, 0002, 0003]
---

# M1 implementation plan — RWKV-7 in sglang

> Goal of M1: serve RWKV-7 0.1B in sglang v0.5.10.post1 and pass the
> greedy-match-vs-numpy-oracle gate ([[F0003]] / `bench/oracle_numpy.py`).
> Cite-anchored to the closed vLLM PR #41060 blueprint, sglang's GDN/Mamba
> conventions, and the fla rwkv7 kernels.

## 0. M1 scope decisions (simplest correct path)
- **Reuse sglang's Mamba/hybrid-linear plumbing verbatim** (`MambaPool`,
  `HybridReqToTokenPool`, `MambaAttnBackendBase`, `HybridLinearAttnBackend`).
  RWKV-7 = every layer linear ⇒ `full_attention_layer_ids=[]`,
  `linear_layer_ids=range(num_layers)` (the GDN/Qwen3-Next path with full-attn empty).
- **State per request, no prefix reuse.** One MambaPool slot/req; state S + 2
  token-shift vectors zero-init at prefill, carried through that req's decodes.
  Do NOT wire `MambaRadixCache` in M1 (prefix-cache state-fit is M2).
- **Kernel split**: decode (all len-1) → `fused_mul_recurrent_rwkv7`; prefill
  (varlen) → `chunk_rwkv7`. Packed B=1 with `cu_seqlens=query_start_loc`,
  `initial_state=temporal[cache_indices]`, final state scattered back.
- **Disable CUDA graph in M1** (`--disable-cuda-graph`); cudagraph-stable
  read/write buffers are M2.
- **Token-shift + projections + LoRA + g_norm + gate-correction in plain torch**
  (fla `*_ref` math); only the recurrence (decode/chunk) is vendored Triton.
  Token-shift prev-token stored in MambaPool `conv[0]` (attn) / `conv[1]` (ffn),
  each `(hidden,1)` (conv_kernel=2 ⇒ last dim 1).
- **Weights: fla-format first** (sglang load_weights = fla naming, no remap, per
  PR#41060). A standalone offline converter turns BlinkDL `.pth` → fla safetensors
  + config.json. The oracle compares sglang logits (converted weights) vs
  `rwkv_v7_numpy.py` logits (BlinkDL `.pth`) → isolates conversion vs kernel.

## 1. Verified shapes (0.1B: D=768, layers=12, head_dim K=V=64, H=12)
- recurrent state S (per req/layer): `temporal=(H,K,V)=(12,64,64)` **fp32** (kernels
  keep S fp32). Maps onto `MambaPool.State.temporal` (KimiLinearStateShape precedent).
- attn shift `conv[0]=(768,1)`; ffn shift `conv[1]=(768,1)`; fp32 in M1 for oracle parity.
- per-token kernel tensors: `r,w,k,kk,a` = `[1,T,H,K]`, `v` = `[1,T,H,V]` (packed B=1);
  `initial_state=[N,H,K,V]`, N=len(cu_seqlens)-1.
- low-rank dims: **derive from checkpoint tensor shapes** (w1/a1/v1/g1), do not hardcode.

## 2. Core math (verified vs rwkv_v7_numpy.py:13-49)
Per-layer time-mix:
1. `shifted=token_shift(x,conv[0])`; `xr,xw,xk,xv,xa,xg = x + x_*·(shifted-x)`.
2. `r=r_proj(xr)`, `k=k_proj(xk)`, `v=v_proj(xv)`.
3. `w_log = -0.6065306597126334 * sigmoid(tanh(xw@Ww1)@Ww2 + w0)`  (√e⁻¹).
4. `a = sigmoid(xa@Wa1@Wa2 + a0)` (linear down, NO tanh).
5. `g = sigmoid(xg@Wg1)@Wg2` (sigmoid AFTER down, then up, NO bias).
6. layer>0: `v += (v_first-v)*sigmoid(xv@Wv1@Wv2 + v0)`; layer0: `v_first=v` (python local, never cached).
7. `kk=k*k_k`; reshape `[*,H,K]`; L2-normalize kk over K; `k = k + k*(a-1)*k_a`.
8. recurrence `a_kernel=-kk`, `b_kernel=kk*a` (numpy:39 `S=S*w.mT - S@kk*(kk*a).mT + v*k.mT`).
9. `o=g_norm(S@r)` (GroupNorm groups=H, eps=K*norm_eps=64e-5); gate-correction
   `o=(o + (r*k*r_k).sum(-1,keepdim)*v)*g`; `o_proj(o)`.
FFN: `shifted=token_shift(x,conv[1])`; `xk=x + x_k·(shifted-x)`; `out=value(relu(key(xk))**2)`.
**Per-LoRA activation differs (w=tanh-down, a/v=linear-down, g=sigmoid-down)** — easy-to-miss bug.

## 3. Files to CREATE/EDIT
### 3a. Vendored Triton (python/sglang/srt/layers/attention/fla/, namespaced subpkgs to avoid colliding with sglang's GDN fla files)
- `fla/rwkv7/fused_recurrent.py` ← refs/fla ops/rwkv7/fused_recurrent.py (DECODE; keep T==1 fast path; public `fused_mul_recurrent_rwkv7`; drop dplr delegation).
- `fla/dplr/chunk_A_fwd.py`, `wy_fast_fwd.py`, `chunk_h_fwd.py`, `chunk_o_fwd.py`, `chunk.py` ← refs/fla ops/generalized_delta_rule/dplr/* (fwd only; drop backward + `fla.ops.cp` imports + cp_context).
- `fla/rwkv6_cumsum.py` ← refs/fla ops/rwkv6/chunk.py (ONLY `chunk_rwkv6_fwd_cumsum`, two-output gi/ge).
- `fla/rwkv7/chunk.py` ← refs/fla ops/rwkv7/chunk.py (thin shim `chunk_rwkv7`→`chunk_dplr_fwd`, r→q w→gk).
- `fla/rwkv7/__init__.py`, `fla/dplr/__init__.py` (exports).
- Rewrite all `fla.ops.*`/`fla.utils` imports → `sglang.srt.layers.attention.fla.*`. Reuse present leaf utils (op.py, index.py, l2norm.py, utils.py).
### 3b. `python/sglang/srt/configs/rwkv7.py` (NEW) — `Rwkv7Config`, model_type='rwkv7';
properties `linear_layer_ids=range(L)`, `full_attention_layer_ids=[]`, `mamba2_cache_params`.
### 3c. `python/sglang/srt/configs/mamba_utils.py` (EDIT) — add `Rwkv7StateShape` (conv [(D,1),(D,1)], temporal (H,K,V)) + `Rwkv7CacheParams(BaseLinearStateParams)`; temporal fp32.
### 3d. `python/sglang/srt/models/rwkv7.py` (NEW; template qwen3_next.py) — `Rwkv7ForCausalLM/Model/DecoderLayer/Attention/FeedForward`; `EntryClass=Rwkv7ForCausalLM` (auto-discovered, no registry edit); v_first python local (layer0 sets, omits v_lora); `load_weights` fla naming no-remap.
### 3e. `python/sglang/srt/layers/attention/linear/rwkv7_backend.py` (NEW; template gdn_backend.py) — `Rwkv7AttnBackend(MambaAttnBackendBase)`; `forward_decode`→fused_mul_recurrent_rwkv7; `forward_extend`→chunk_rwkv7 (a=-kk,b=kk*a); S=temporal[cache_indices] in/out.
### 3f. Wiring EDITS — `model_runner.py` (rwkv7_config property → mambaish_config; gate spec+PP off); `attention_registry.py` (attn_backend_wrapper → Rwkv7AttnBackend + HybridLinearAttnBackend([])); `hf_transformers_utils.py` (_CONFIG_REGISTRY += Rwkv7Config); `configs/__init__.py` (export).
### 3g. Outside sglang — `tools/convert_rwkv7_blinkdl_to_fla.py` (BlinkDL .pth → fla safetensors+config; **transpose all LoRA up/down**: w1/w2/a1/a2/v1/v2/g1/g2; w0/a0/v0→.lora.2.bias; v_lora layers>0 only; receptance/key/value/output+ln_x no transpose; emb→embeddings, head→lm_head, ln_out→norm, ln1→attn_norm, ln2→ffn_norm, ln0→layers.0.pre_norm; derive low-rank dims from shapes). `test/srt/models/test_rwkv7_oracle.py`.

## 4. Build/test order
1. Vendor + import-fix Triton; smoke each kernel vs fla originals (decode+chunk).
2. Config + state params + wiring; boot on dummy weights; confirm `mambaish_config`, MambaPool allocates temporal (12,64,64) fp32 + two (768,1) conv, backend selected.
3. Converter: BlinkDL 0.1B g1 .pth → fla; spot-check shapes/transposes.
4. `load_weights`: load converted, assert all keys consumed.
5. E2E single-seq prefill (dragon prompt) with `--disable-cuda-graph`, fp32.

## 5. Oracle correctness gate
- `rwkv_v7_numpy.py` on BlinkDL 0.1B .pth → `logits_ref` (final token).
- sglang (fp32, cuda-graph off, single req, greedy) on converted weights → `logits_sgl`.
- **PASS if** `max(|Δ|)/std < ~1e-2` AND greedy next-token matches AND a multi-token
  greedy continuation matches token-for-token (exercises decode + state carry across
  prefill→decode). Secondary: batch-2 identical reqs identical outputs (slot isolation);
  mixed prefill+in-flight-decode both correct (extend/decode split + varlen cu_seqlens).

## 6. Risks carried into M1
- Per-LoRA activation differences (w=tanh, a/v=linear, g=sigmoid-down).
- LoRA transpose + layer-0 v_lora absence in converter (top risk).
- fla RWKV7 has an upstream "possibly buggy vs BlinkDL" warning → that's WHY the oracle
  is numpy/BlinkDL, not fla.
- chunk_dplr clamps chunk_size→16 on triton<3.4.0 — verify triton on box; validate
  chunk path on a ≥64-token prefill.

## M1c grounding (verified against v0.5.10.post1 source, 2026-06-30)
- **Backend** (template `layers/attention/linear/gdn_backend.py`): subclass
  `MambaAttnBackendBase` (from `layers/attention/hybrid_linear_attn_backend.py`).
  Implement `forward_decode(layer, forward_batch, ...)` + `forward_extend(...)`.
  State: `cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)` →
  `cache.conv[0]` (conv/shift states), `cache.temporal` (recurrent S). Metadata:
  `self.forward_metadata.{query_start_loc, mamba_cache_indices}`. Prefill init mask:
  `has_initial_states = forward_batch.extend_prefix_lens > 0`. GDN passes `mixed_qkv,a,b`
  and does conv+gating IN the backend; RWKV7 will instead pass already-projected
  r/w/k/v/kk/a (token-shift+proj+LoRA done in the model module) → simpler custom kwargs.
- **State params** (`configs/mamba_utils.py`): `BaseLinearStateParams` (ABC) has
  `dtype: Mamba2StateDType{conv,temporal}` + `layers: list[int]` + `mamba_cache_per_req`
  (sums `shape.conv` list numels + `shape.temporal`). `shape.conv` is `list[tuple]`
  (multi-entry supported). Add `Rwkv7StateShape{conv, temporal}` + `Rwkv7CacheParams`.
  RWKV7: `temporal=(num_heads=12, head_dim=64, head_dim=64)` fp32; token-shift conv —
  pack both shifts into one `(2*D, 1)` tensor (KimiLinear packs similarly) OR two
  `(D,1)` entries; **verify `mem_cache/memory_pool.py` MambaPool handles the chosen
  conv layout + `.conv[0]` indexing** before committing.
- **Token-shift** = width-2 causal: prev-token via the conv state; lerp `x + x_*·(prev-x)`
  in the model. Reuse `layers/attention/mamba/causal_conv1d{,_triton}.py` (fixed [1,0]
  shift kernel) or manual shift+state-write. Resolve in M1c against memory_pool.
- **Decode/extend → kernels**: decode → `fused_mul_recurrent_rwkv7`; extend →
  `chunk_rwkv7` (from M1a). Write final state back to `cache.temporal[cache_indices]`.

## Open questions (M2+)
- M2 prefix-cache state fit (MambaRadixCache, page_size==1, chunk-aligned scatter).
- M2 CUDA-graph decode (UNIFORM_SINGLE_TOKEN_DECODE, stable index buffers).
- token-shift placement: confirm forward_batch exposes the linear sub-backend's
  metadata/req_to_token_pool under HybridLinearAttnBackend([]); else route via backend.
- RadixLinearAttention signature (mixed_qkv,a,b) doesn't fit r/w/k/v/kk/a → M1 bypasses
  with a custom kwargs backend; decide M2 whether to extend it.
- triton version on box (chunk_size clamp).
- exact fla checkpoint key strings — confirm vs an actual fla-hub/rwkv7-0.1B checkpoint
  or refs/fla/fla/models/rwkv7/modeling_rwkv7.py before finalizing load_weights+converter.
- conv dtype bf16 vs fp32 (M2 memory).
- M2+ quant (GPTQ/AWQ stray-bias skip) + TP>1 sharded loaders.
