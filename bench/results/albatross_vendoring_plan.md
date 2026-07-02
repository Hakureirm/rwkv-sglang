# Albatross fp16 CUDA-kernel vendoring plan (scope, read-only)

**Goal:** assess feasibility and write a concrete integration plan for vendoring
BlinkDL/Albatross's fast CUDA kernels into our sglang RWKV-7 backend to close the
residual kernel-quality gap vs albatross (F0008: ~2.2–2.7× at 0.1B/1.5B decode with
cuda-graph ON; ~1.2–1.8× at 7.2B). **No implementation here.** Albatross is Apache-2.0
(BlinkDL's own; FLA-free per ADR-0004), redistributable with attribution.

Sources studied (read-only): `refs/Albatross/faster3a_2605/` (`rwkv7_fast_v3a.py`,
`cuda/rwkv7_wkv_fp32_v2.{cpp,cu}`, `cuda/rwkv7_wkv_fp16_v2.{cpp,cu}`,
`cuda/rwkv7_v3a_ops.{cpp,cu}`, `cuda/rwkv7_fast_ops_fp16.{cpp,cu}`),
`refs/Albatross/{LICENSE,README.md}`, `bench/results/albatross_3090.md`. Our side:
`sglang_overlay/.../linear/rwkv7_backend.py`, `.../rwkv7_kernels/wkv_recurrent.py`,
`sglang_overlay/.../models/rwkv7.py`, `.../configs/rwkv7.py`.

---

## TL;DR recommendation

1. **The WKV state kernel is a clean, isolated, low-risk drop-in** — and the ONLY
   albatross kernel that is. `rwkv7_wkv_fp32_v2` is a standalone `TORCH_LIBRARY` op
   (~13 KB of source, no cublasLt/WMMA), its math/sign/layout conventions match ours
   *exactly* (verified below), and **it offers a true fp32-state variant**
   (`--wkv fp32io16` = compile with `-D_IO_FP16_`; or full fp32 IO by omitting the
   macro). It mutates an external `[B,H,64,64]` state in place — the same
   gather-by-`cache_indices` / scatter pattern our decode path already runs.
2. **The fused linears (`v3a_ops`) and the sparse no-fc FFN (`fast_ops_fp16`) are NOT
   clean drop-ins.** The generic GEMM ops *exist* (`linear_f16`,
   `linear_f16_m1_splitk`, `linear_orig_wmma16_f16`) but using them means replacing the
   model's `nn.Linear` calls, pre-transposing/pre-packing weights into albatross's fp16
   layout, and (for no-fc) precomputing a `.fc` weight variant. The highest-value
   fusions (`add_layer_norm_tmix_mix6_f16`, `linear_wagv_rank_*`) are welded to
   albatross's exact forward shape and its own token-shift state layout. **Recommend
   self-written fused Triton (or sglang's existing GEMM/quant path) for projections,
   NOT vendoring these.**
3. **Honest sizing of the decode-bsz1 lever:** at bsz1 decode the WKV recurrence is a
   *minority* of per-layer time; the bulk is the r/k/v/o GEMVs + LoRAs + FFN +
   `lm_head`. albatross's bsz1-decode win lives mostly in its *linear* kernels, which
   are exactly the part that's NOT cleanly vendorable. So **vendoring only the WKV
   kernel will move decode-bsz1 modestly**; its bigger payoff is **prefill and
   large-batch decode**, where the recurrence scan is a larger share. **Gate the
   first-step decision on the parallel profiling result** (is WKV a hot fraction of
   decode? of prefill?).
4. **First step (smallest, correctness-safe):** vendor `rwkv7_wkv_fp32_v2.{cpp,cu}`,
   patch one line (`w_eff(w)` → `expf(w)`) so it consumes our existing log-decay `w`
   *without touching the model*, wire it behind a flag into the **decode** path of
   `rwkv7_backend.recurrence()` (T==1, the static-batch case albatross fits natively),
   keep our Triton kernel for varlen prefill, and re-run the greedy-exact + dynamic-batch
   gates. fp32 IO + fp32 state first (most accurate), fp16 IO second (speed).

---

## 1. Inventory: what's in `faster3a_2605/cuda/` and is it reusable

| file | size | what it is | drop-in for us? |
|---|---|---|---|
| `rwkv7_wkv_fp32_v2.{cpp,cu}` | ~13 KB | **WKV state scan**, fp32 state, IO fp32 *or* fp16 (`-D_IO_FP16_`). 4 launch modes (auto/seq/small-warp/short-block). Head size **64 hard-coded**. Standalone `TORCH_LIBRARY(rwkv7_wkv_fp32_v2)`. | **YES — clean.** Recommended target. |
| `rwkv7_wkv_fp16_v2.{cpp,cu}` | ~32 KB | **WKV state scan, pure fp16** state, half2 + cp.async pipelining + xor-swizzled smem + **deterministic positional dithering** (`w_delta`, needs `elapsed_t`). `wkv_one`/`wkv_seq`(+`_w0`). | Buildable, but **correctness-risky** (fp16 state + position-dependent dither ≠ fp32 oracle). Speed mode only. |
| `rwkv7_v3a_ops.{cpp,cu}` | ~182 KB | **Fused linears + fused LN/mix + LoRA fusions.** WMMA (`nvcuda::wmma`) + split-k + cublasLt GEMMs (`linear_f16*`), fused `add_layer_norm_tmix_mix6_f16` (LN+shift+all-6 lerps, mutates shift-state), `linear_wag(v)_rank_in/out_f16` (fused w/a/g(/v) LoRAs). **No WKV here.** | **NO clean drop-in.** Generic GEMMs extractable; fusions welded to albatross's forward. |
| `rwkv7_fast_ops_fp16.{cpp,cu}` | ~69 KB | **Sparse "no-fc" channel-mix FFN** family: `cmix_sparse_one/rows`, `cmix_sparse_down_relu_*` (exploit relu sparsity at tiny batch; consume a precomputed `key.weight.fc` layout) + `cmix_mix`. | **NO.** Tied to weight pre-prep + tiny-batch regime. |

**Net:** exactly one isolated, math-matching, redistributable, low-risk op — the **WKV
state kernel**. Everything else is fused to albatross's bespoke single-stream forward.

---

## 2. The WKV kernel — exact signature, layout, and math mapping

### 2a. Op surface (`rwkv7_wkv_fp32_v2`)
```cpp
TORCH_LIBRARY(rwkv7_wkv_fp32_v2, m):
  forward      (int B,int T,int C,int H, Tensor(a!) state, r,w,k,v,a,b, Tensor(a!) y) -> ()  // mode 0, auto-picks
  forward_seq  (...)  // mode 1: per-head 64-thread kernel
  forward_small(...)  // mode 2: 32-thread warp kernel (auto-selected for T==1)
  forward_block(...)  // mode 3: block-reduce kernel
```
- `state`: **fp32, contiguous, shape `[B,H,64,64]`** (asserted in `check_inputs`).
  Mutated **in place** (the new state is written back into the same tensor).
- `r,w,k,v,a,b,y`: shape **`[B,T,C]`** with `C == H*64`, contiguous, dtype = IO_DTYPE
  (fp32 by default; **fp16** when compiled `-D_IO_FP16_`). `y` is the output buffer
  (also written in place).
- **Head size 64 only** (`constexpr int N = 64;` + `TORCH_CHECK(C == H*64)`).
- Launches on the current CUDA stream, no host sync, no dynamic allocation →
  **cuda-graph-capturable**.

### 2b. The math the kernel computes (from `wkv_fp32_v2_kernel`, thread `i`=output row `v`, loop `j`=`k`)
```
w[j] = w_eff(w_in[j])                 # w_eff(x) = exp(-e^-0.5 * sigmoid(x)) = RWKV-7 decay
sa_i = sum_j  state[i,j] * a_in[j]
state[i,j] = state[i,j]*w[j] + sa_i*b_in[j] + k_in[j]*v_in[i]
y_i        = sum_j state[i,j] * r_in[j]
```
So in oracle terms `state[i,j] = S_np[v=i, k=j]` (V-major, K-minor).

### 2c. Mapping to OUR tensors (verified against albatross's own driver, `rwkv7_fast_v3a.py:567-574`)
Albatross's `fp32io16` call passes:
`forward(B,T,C,H, wkv_state, r, w_raw, k, v, neg_kk, kka, y)` where
`neg_kk = -kk`, `kka = kk*a`, `w_raw` = the **raw** decay-LoRA output (pre-activation).

| albatross arg | albatross meaning | our quantity | adapter |
|---|---|---|---|
| `r` | receptance | `r` (`[T,H,64]`) | reshape `[N,1,C]`, cast to IO dtype |
| `w` | **raw** decay LoRA out; kernel applies `w_eff` | we already hold `w_log = -e^-0.5·sigmoid(...)` | **two options, see below** |
| `k` | key | `k` | reshape/cast |
| `v` | value | `v` | reshape/cast |
| `a` (=`neg_kk`) | `-kk` (the vector dotted with state) | `kk` | `a_alb = -kk` (elementwise) |
| `b` (=`kka`) | `kk * a_oracle` (rank-1 update dir) | `kk`, `a` | `b_alb = kk * a` (elementwise) |
| `state` | `[B,H,64,64]` = `S[v,k]`, in place | `temporal[cache_indices]` = `[N,H,K,V]` = `S[k,v]` | gather + **transpose `[K,V]→[V,K]`** |
| `y` | output `[B,T,C]` | our `o` `[T,H,V]` | reshape back |

**The `a`/`b` sign convention is identical to what our Triton kernel already forms
internally** (`a_kernel=-kk`, `b_kernel=kk*a`, see `wkv_recurrent.py:118-120`). So the
adapter just lifts those two elementwise ops out of the kernel into the backend.

**The `w` convention — the one real decision:**
- `w_eff(x) = exp2(-0.875038·sigmoid(x))`. Decode: `-0.875038 = log2(e)·(-0.606531)` and
  `0.606531 = e^-0.5`, so `w_eff(x) = exp(-e^-0.5·sigmoid(x))` — **exactly** our
  `decay = exp(w_log)` with `w_log = -e^-0.5·sigmoid(x)` (`rwkv7.py:142`,
  `_INV_SQRT_E = 0.6065306597126334`). Same value, computed at a different stage.
- **Option A (preferred — no model change):** vendor the `.cu` and change the single
  line `w[i] = w_eff(load_io(w_ptr, idx));` → `w[i] = __expf(load_io(w_ptr, idx));`
  (and the analogous lines in the warp/block kernels). Then feed our existing `w_log`.
  Self-contained, model untouched (respects "don't touch model/overlay"), recorded as
  an Apache-2.0 modification.
- **Option B:** feed albatross the raw pre-sigmoid `w` and keep `w_eff` — requires
  `rwkv7.py` to expose the raw decay-LoRA output to the backend (model change; owned by
  the parallel agent). Avoid for the first step.

---

## 3. Mismatch enumeration: albatross (single-stream static batch) → sglang (state-cache + varlen + cuda-graph). Difficulty per item.

| # | mismatch | reality | adaptation | difficulty |
|---|---|---|---|---|
| M1 | **per-request state cache** | albatross state is one contiguous `[B,H,64,64]`, mutated in place. We keep `temporal[N,H,K,V]` indexed by `mamba_cache_indices`. | **Already solved.** Our decode path does `temporal[cache_indices].contiguous()` → run → scatter back (`rwkv7_backend.py:148,164`). Same pattern; the kernel's in-place state write fits it. | **Trivial** |
| M2 | **state layout transpose** | albatross `[B,H,V,K]` vs our cache `[N,H,K,V]`. | `state_alb = temporal[idx].transpose(-1,-2).contiguous()` before; `temporal[idx] = state_alb.transpose(-1,-2)` after. 64×64 transpose/req/layer, cheap. (Or store the cache in `[V,K]` and transpose at the Triton-prefill boundary instead — pick one consistent layout.) | **Easy** |
| M3 | **`w` convention** | kernel applies `w_eff`; we hold `w_log`. | Option A: patch `w_eff→expf` in the vendored `.cu`. | **Easy** |
| M4 | **`a`/`b` derivation** | kernel wants `-kk`, `kk*a`. | Two elementwise ops in the backend (lifted from our Triton kernel). | **Trivial** |
| M5 | **dtype** | kernel IO is fp32 or fp16; we run bf16 with fp32 state. | Cast bf16→fp32 (lossless, build w/o `-D_IO_FP16_`) for the accurate first pass; bf16→fp16 (build w/ `-D_IO_FP16_`) for the speed pass. State stays fp32 either way. | **Easy** |
| M6 | **varlen prefill** | `forward(B,T,C)` is **static batch, uniform T**. sglang prefill is packed varlen (B=1, total_T, `cu_seqlens`, ragged lengths). The kernel has **no `cu_seqlens` and no external-initial-state-per-segment** concept. | **Do NOT use albatross for prefill.** Keep our Triton `wkv_recurrent` (already handles `cu_seqlens` + per-request `initial_state`/writeback). albatross-for-prefill would require a per-request Python loop (one launch each → loses batching, hurts the graphed/batched prefill we already have). | **Hard if attempted → skip** |
| M7 | **cuda-graph decode** | graph needs static shapes + no host sync + address-stable buffers. | The custom op is a plain stream launch, no sync, B fixed per captured graph. The gather/transpose/scatter is already inside our graphed region and verified exact (F0008). Same applies. | **Easy** (re-verify capture) |
| M8 | **head_dim ≠ 64** | albatross hard-codes 64; our Triton kernel is general. | All current BlinkDL g1 models use 64 (`configs/rwkv7.py:29 head_dim=64`), so OK today, but vendoring **ties the WKV path to head_dim=64**. Keep the Triton kernel as the fallback for other head dims. | **Note / constraint** |
| M9 | **bit-stability pin** | our Triton kernel pins `BV`/`num_warps` to reproduce an exact fp32 summation ORDER so bf16-cast output is bit-stable on knife-edge argmax (`wkv_recurrent.py:191-210`). albatross uses a different reduction (warp-shuffle / `#pragma unroll` dot). | **Different summation order → output bits change.** Not bit-identical to today; **must re-run the greedy-exact + `verify_batch` gates.** Expected to pass (fp32 state is high precision) but is a re-validation, not a guarantee. | **Medium (re-gate)** |

---

## 4. Precision / correctness

- **Is there an fp32-state albatross variant?** **Yes.** `rwkv7_wkv_fp32_v2` keeps the
  state in **fp32 always** (`float state[HeadSize]`, `float* state_ptr`). The
  `-D_IO_FP16_` macro only changes the *IO* dtype (r/w/k/v/a/b/y) to fp16; the
  accumulation and stored state remain fp32. albatross exposes this as `--wkv fp32io16`
  ("the more accurate fp32 WKV state path"). Omitting the macro gives **full fp32 IO +
  fp32 state** — the most accurate build, even closer to the oracle than our current
  bf16-IO Triton kernel.
- **fp32io16 vs our fp32-state Triton kernel:** both accumulate state in fp32. The
  output differs only by (a) IO rounding (fp16 vs bf16 vs fp32) and (b) summation order
  (M9). Risk to the greedy-exact gate: **medium → must re-gate**, but the *expected*
  outcome is "exact or near-exact", because the dominant precision driver (fp32 state)
  is unchanged. Recommended order to maximize pass odds: build **fp32 IO + fp32 state**
  first (bf16→fp32 cast in, fp32→bf16 out, only one rounding boundary changes), confirm
  the gate, then try fp16 IO for the speed delta and re-gate.
- **Pure fp16 path (`rwkv7_wkv_fp16_v2`) risk:** **high / not recommended for the
  gated deliverable.** It stores state in fp16 and adds a *position-dependent*
  deterministic dither (`w_delta`, phase = `elapsed_t[b] + h*N + i + t`) specifically to
  stabilize fp16 accumulation. That dither (i) won't match the fp32 oracle bit-for-bit
  and (ii) requires plumbing each request's **absolute token position** as `elapsed_t`
  into the op every step (extra int32 tensor, more cuda-graph surface). Keep it as an
  optional throughput mode behind a flag, gated separately (perplexity / lm-eval), not
  as the greedy-exact path.
- **Correctness gate for any WKV swap:** `bench/verify_m1d.py` (greedy-exact vs numpy
  oracle, 0.1B/1.5B/7.2B) **with cuda-graph ON**, plus the dynamic-batch gate
  (`verify_batch`: batched B==1, the knife-edge case M9 protects). Must pass before the
  kernel is allowed on by default.

---

## 5. Build / packaging

- **Mechanism:** identical to albatross — `torch.utils.cpp_extension.load(...)` JIT at
  runtime (proven on our 3090 / sm_86, see `albatross_3090.md`). For the WKV op the
  call is tiny (2 files, no cublasLt/WMMA), e.g.:
  ```python
  from torch.utils.cpp_extension import load
  load(name="rwkv7_wkv_fp32_v2",
       sources=[CUDA_DIR/"rwkv7_wkv_fp32_v2.cpp", CUDA_DIR/"rwkv7_wkv_fp32_v2.cu"],
       is_python_module=False,                       # registers torch.ops.rwkv7_wkv_fp32_v2.*
       extra_cflags=["-O3"],                         # add "-D_IO_FP16_" for the fp16-IO build
       extra_cuda_cflags=["-O3","--use_fast_math","-Xptxas","-O3"])  # +"-D_IO_FP16_" to match
  ```
- **Recipe (from M3a, `albatross_3090.md`):** `CUDA_HOME=/usr/local/cuda-12.9`,
  `PATH=$CUDA_HOME/bin:$PATH`, `TORCH_CUDA_ARCH_LIST=8.6` (the portable way to target the
  3090; do NOT use albatross README's hard-coded `sm_120`). torch 2.9.1+cu128; nvcc 12.9
  minor-skew is fine (major matches). First compile is one-time, cached in
  `~/.cache/torch_extensions`. nvcc-on-PATH is required at first load (JIT).
- **Deliverable packaging:** JIT is fine for the box, but for a clean release prefer a
  **prebuilt extension** (`setup.py` / `BuildExtension`) so end users don't need nvcc at
  runtime and there's no first-call compile stall. Either way, only the **2 WKV files**
  are vendored (~13 KB) — do **not** vendor `v3a_ops`/`fast_ops_fp16` (190 KB+,
  cublasLt/WMMA, heavier build, no clean use).
- **Placement:** `sglang_overlay/sglang/srt/layers/attention/rwkv7_kernels/cuda/`
  (the `.cpp`/`.cu`), with a `wkv_albatross.py` loader+adapter next to
  `wkv_recurrent.py`. The backend chooses between them by a flag (default = our Triton
  kernel until the albatross path passes the gate and shows a measured win).

### License / attribution (Apache-2.0 — redistribution requirements)
Albatross ships `refs/Albatross/LICENSE` = **Apache-2.0**. To redistribute (incl.
modified) we must, per §4:
1. **Retain** the Apache-2.0 LICENSE text — copy it into the vendored `cuda/` dir
   (e.g. `cuda/ALBATROSS_LICENSE`).
2. **Keep** the original copyright/notice in each vendored `.cu`/`.cpp` header.
3. **State changes** — add a `NOTICE` (and a header comment) noting the files are
   derived from **BlinkDL/Albatross (`faster3a_2605`, Apache-2.0)**, with upstream
   URL + commit, and listing OUR modifications (e.g. "`w_eff`→`expf` to consume
   precomputed log-decay; added the sglang state-cache gather/transpose adapter").
This is also the **cleanest** outcome (ADR-0004): it's BlinkDL's own code — one of our
reference implementations — and contains **zero FLA**. Update `grep -ri fla` stays clean.

---

## 6. Recommended plan — smallest first step + ranking

### First step (one PR), exact files/functions:
1. **Vendor** `faster3a_2605/cuda/rwkv7_wkv_fp32_v2.cpp` + `.cu` →
   `sglang_overlay/.../rwkv7_kernels/cuda/`. Patch `w_eff(...)` → `__expf(...)` in the
   three kernels (`wkv_fp32_v2_kernel`, `wkv_fp32_v2_small_warp_kernel`,
   `wkv_fp32_v2_short_block_kernel`). Add LICENSE/NOTICE.
2. **Add** `rwkv7_kernels/wkv_albatross.py`: a `load()`-on-first-use wrapper exposing
   `wkv_decode_albatross(r,w,k,v,kk,a, state_kv) -> o` that:
   - builds `a_alb = -kk`, `b_alb = kk*a` (elementwise);
   - `state_vk = state_kv.transpose(-1,-2).contiguous()` (`[N,H,64,64]`, fp32);
   - reshapes r/w/k/v/a_alb/b_alb to `[N,1,C]`, casts to the build's IO dtype;
   - allocates `y`; calls `torch.ops.rwkv7_wkv_fp32_v2.forward(N,1,C,H, state_vk, ...)`;
   - returns `o = y.view(N,H,V)` and writes `state_vk.transpose(-1,-2)` back to the cache.
3. **Wire** only the **decode branch** of `Rwkv7AttnBackend.recurrence()`
   (`rwkv7_backend.py:150-165`) behind a flag (`use_albatross_wkv`, default off). Leave
   the varlen/prefill branch on our Triton kernel (M6). Keep token-shift unchanged.
4. **Gate:** `verify_m1d.py` greedy-exact (0.1B/1.5B/7.2B) **+ cuda-graph ON + verify_batch**.
   Build order: fp32 IO+state first; if exact, try `-D_IO_FP16_` and re-gate.
5. **Measure:** `bench/throughput.py` decode bsz{1,8,32} with vs without the flag, on the
   3090, cuda-graph ON. Keep the flag default-off unless it's both exact and faster.

**Pre-req gate (do this before step 1):** read the parallel profiling result. If WKV is
<~15% of decode-bsz1 time (likely — projections dominate), the WKV swap will barely move
decode-bsz1; its real win is **prefill / large-batch**, so prioritize accordingly (or
defer the WKV vendor and point the effort at the linears via self-written fused Triton).

### Ranking: vendor albatross vs self-write, per component

| component | vendor albatross? | why |
|---|---|---|
| **WKV decode (T==1)** | **Vendor (1st choice)** | Isolated op, exact math match, fp32-state variant, in-place state fits our gather/scatter. Cleanest possible vendor. Caveat: modest decode-bsz1 lever; tied to head_dim=64. |
| **WKV prefill (varlen)** | **Self-write (keep ours)** | albatross has no `cu_seqlens`/per-segment initial-state; our Triton kernel already does varlen + writeback and is FLA-free. A per-request loop over albatross would regress the batched prefill. |
| **r/k/v/o projections** | **Self-write / use sglang GEMM** | albatross's win is here, but the generic GEMMs require replacing `nn.Linear` + fp16 weight pre-pack, and the high-value fusions are welded to albatross's forward. Lower-risk: fused Triton (LN+shift+lerp+GEMV) or sglang's quant/GEMM. |
| **w/a/g LoRAs** | **Self-write** | `linear_wag_rank_*` is fused to albatross's exact LoRA layout. A small fused Triton "down→act→up" is cleaner and model-shaped. |
| **FFN (sqrelu) / no-fc sparse** | **Self-write (later)** | `cmix_sparse_*` needs a precomputed `key.weight.fc` layout + tiny-batch regime. Clever but not a drop-in; replicate the relu-sparsity idea in Triton if FFN profiles hot. |
| **fused LN + token-shift + 6-lerp** | **Self-write** | `add_layer_norm_tmix_mix6_f16` mutates albatross's own shift-state and assumes its forward; our token-shift uses the MambaPool conv state. Self-written fusion fits our state model. |

### Bottom line on feasibility
- **WKV kernel: feasible and clean** — vendor it (decode), modest-but-real win, safe
  correctness story via the fp32-state variant, ~13 KB + attribution.
- **Everything else: NOT cleanly extractable** — albatross's linear/FFN/LN kernels are
  fused to its single-stream static-batch forward and its own weight pre-prep. For those
  components, **self-written fused Triton (or sglang's GEMM/quant path) is the better
  route** than trying to un-fuse albatross. This matches the ADR-0004 fallback and keeps
  the deliverable clean (zero FLA, minimal third-party surface).
