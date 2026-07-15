# F0059 — sglang_overlay drift debt: ground truth, categorization, and resync

**Date:** 2026-07-15 · **Machine:** 3090 (`dg-workstation-2x3090`) · **Task:** #55

## TL;DR

- The mythical "`model_runner.py` differs from upstream by ~835 ins / 1195 del" is
  **real and reproduced exactly** (measured **831 ins / 1198 del**), but it is **~96% upstream
  churn, not RWKV integration**. The genuine RWKV-7 edit to `model_runner.py` is **+34 / −1**.
  Across all 10 upstream-touched files the entire genuine port is **129 ins / 4 del** (+ additive
  RWKV-only files). That 129-line number is `rwkvmain`'s live `git diff` and is the categorization.
- **Root cause of the debt:** the git-tracked `sglang_overlay/` ships **6 churny upstream files as
  full-file copies** (11k+ lines total) frozen at **v0.5.10.post1**. `scripts/deploy.sh` rsyncs them
  wholesale over the container's installed sglang. The current `dev-cu12` image ships sglang **main
  (`754524d8d`)**, so the rsync clobbers newer upstream files with year-old copies → cascading
  ImportErrors. The `sglang_main_port/` bundle (patch + additive files) is the correct shape and is
  what `rwkvmain` runs.
- **Secondary finding:** the shipped `sglang_main_port/upstream_edits.patch` was **itself stale** —
  generated against base `a3f6680`, it **fails to apply** to the image's actual `754524d8d`
  (`model_runner.py:817`). Its *added content* is byte-identical to `rwkvmain`'s live diff; only the
  base/context drifted.
- **Increment landed (verified):** regenerated `sglang_main_port/upstream_edits.patch` +
  `base_commit.txt` + README from `rwkvmain`'s live diff. The regenerated **public bundle now applies
  cleanly and imports clean on a fresh container off the current image** (`RWKV7ForCausalLM` loads).
  Committed locally only.
- **Gate status — fresh pod SERVES: NO, but not for a resync reason.** The code-level resync is
  proven (import-parity with `rwkvmain`). Live serving is blocked by a **host GPU driver/CUDA gap**
  (host driver `575.51.03` = CUDA 12.9.0; the image's CUDA 12.9.1 needs `575.57.08` → `cuInit`
  error 803). **This blocks `rwkvmain` identically** — it is an infra issue, not the overlay.

## 1. Ground truth (digests / commits / versions)

| Thing | Value |
|---|---|
| 3090 box | `dg-workstation-2x3090`, 2×RTX 3090 24GB, host driver **575.51.03** (nvidia-smi CUDA 12.9) |
| Working reference container | `rwkvmain` (image `docker.1ms.run/lmsysorg/sglang:dev-cu12`, local id `3b354677`, ~10d old) |
| sglang inside `rwkvmain` | git checkout **`754524d8d`** (#30139) + uncommitted RWKV port; `sglang 0.0.0.dev1+g754524d8d` |
| Fresh container off same local image | bakes a **clean** `754524d8d`, 0 dirty files (verified) |
| Registry `dev-cu12` amd64 digest now | `3c783bf6…` — **newer** than the cached `3b354677` → a fresh `docker pull` gets a commit *past* 754524d8d |
| `sglang_main_port/base_commit.txt` (shipped) | `a3f6680` — **stale** vs the image's `754524d8d` |
| `sglang_overlay/` target base | **v0.5.10.post1** (per `sglang_overlay/README.md`) |

## 2. The two code layouts (porting model)

1. **`sglang_overlay/`** (public, git-tracked): full package-tree mirror for **sglang v0.5.10.post1**,
   deployed by `scripts/deploy.sh` via **rsync (no build)** over the installed sglang. It ships both
   additive RWKV-only files *and* **full-file copies of 6 churny upstream files**. `deploy.sh`
   *already* uses clean **anchored idempotent Python injections** for `spec_info.py` and
   `scheduler.py` — the correct pattern — but rsyncs the other 6 as whole files.
2. **`sglang_main_port/`** (public, git-tracked): `upstream_edits.patch` (the ~129-line RWKV diff
   against upstream `main`) + `new_files.tgz` (additive RWKV-only files). Apply-on-top-of-upstream,
   churn-preserving. **This is what `rwkvmain` actually runs.**

## 3. The 835/1195 categorization (core deliverable)

Measured `diff(upstream 754524d8d, sglang_overlay copy)` vs the genuine RWKV edit (from `rwkvmain`'s
live diff / the main-port patch):

| Overlay file (full-file copy) | overlay lines | overlay-vs-upstream Δ | genuine RWKV edit | churn+stale |
|---|---:|---:|---:|---:|
| `model_executor/model_runner.py` | 3064 | **+831 / −1198** | **+34 / −1** | ~+797 / ~−1197 |
| `server_args.py` | 6780 | +4589 / −5141 | +9 / −0 | ~+4580 / ~−5141 |
| `configs/__init__.py` | 68 | +4 / −25 | +2 / −0 | +2 / −25 |
| `configs/mamba_utils.py` | 287 | (additive-ish) | +43 (dataclasses) | churn in surrounding file |
| `layers/attention/attention_registry.py` | 247 | (small) | +4 | churn |
| `utils/hf_transformers_utils.py` | 1356 | large | +2 (registry entry) | **stale: registry moved (see below)** |

**Categorization of the model_runner.py 831/1198:**
- **(i) genuine RWKV-7 integration to preserve — ~34 ins / 1 del.** The `rwkv7_config` property, the
  `Rwkv7NoOpFullAttnBackend` (all-linear: zero full-attn layers), and — in the main-port only — the
  `get_pp_proxy_v_first_size` + v_first PP-proxy slot (F0036).
- **(ii) upstream churn to just take — ~+797 / −1197 (>95%).** Pure v0.5.10.post1 → 754524d8d
  evolution of `model_runner.py`. The overlay's frozen copy is simply missing a year of upstream.
- **(iii) stale project code superseded by upstream — small in model_runner, significant elsewhere.**
  The clearest case: the overlay registers `Rwkv7Config` in **`utils/hf_transformers_utils.py`**, but
  upstream **moved the `_CONFIG_REGISTRY` to `utils/hf_transformers/common.py`**. `hf_transformers_utils.py`
  still *exists* upstream, so the rsync silently overwrites it, but the registration is now in a **dead
  location** → RWKV7 arch not found → part of the "cascading ImportError" chain. The main-port patches
  `common.py` (correct).

**Also:** the overlay's `model_runner.py` is **missing the F0036 v_first PP+cuda-graph fix** entirely
(`grep v_first` = 0 hits) — a known TODO ("overlay 也要移植清零修复"). So the overlay is not merely
stale-by-churn, it's also **behind the main-port on RWKV features**.

Net: the overlay pays **11k+ lines of drift-prone full-file copies** to deliver **~129 lines** of real
integration. The 835/1195 is an artifact of the full-file-copy strategy, not the size of the port.

## 4. Resync options + recommendation

**Option A — pin the image to an older v0.5.10.post1 digest (stopgap).** Freezes sglang; loses
upstream fixes; re-breaks on every image bump; and does **not** fix the driver gap (§6). Emergency
only. Not recommended.

**Option B — make the overlay churn-proof (recommended).** Stop shipping/rsyncing the 6 churny
upstream files as full copies. Instead:
  - ship **only additive RWKV-only files** in `sglang_overlay/` (configs/rwkv7.py, models/rwkv7.py,
    linear/rwkv7_backend.py, rwkv7_kernels/**, speculative/rwkv_chain_worker.py), and
  - deliver the ~129 lines of upstream edits as a **patch or anchored idempotent injections** —
    exactly the pattern `deploy.sh` already uses for `spec_info.py`/`scheduler.py`.
  This makes `deploy.sh` robust to upstream churn and collapses the diff surface from 11k → ~129
  lines. It also folds the overlay and main-port into **one** apply-on-top-of-upstream deliverable
  (removing the v0.5.10-vs-main divergence). Cost: a restructure of load-bearing public code — needs
  a review gate before push.

**Option C — keep two layouts but re-base each on a cadence.** Regenerate `sglang_main_port` against
each image bump (done here) and separately maintain the overlay for v0.5.10.post1. Lower one-time
cost, but keeps the 11k-line overlay debt alive and needs discipline. Acceptable bridge until B lands.

**Recommendation:** land the **Option-C bridge now** (regenerated main-port bundle — done), then do
**Option B** as the real fix behind a review gate. Add a CI/box check that `git apply --check
upstream_edits.patch` + a headless import succeed on a fresh container each time the image moves.

## 5. Increment landed (this task, committed locally only)

Regenerated the stale main-port bundle from `rwkvmain`'s live `git diff` (byte-identical added
content; pure base/context rebase a3f6680 → 754524d8d):
- `sglang_main_port/upstream_edits.patch` — now applies cleanly to `754524d8d`.
- `sglang_main_port/base_commit.txt` — `a3f6680` → `754524d`.
- `sglang_main_port/README.md` — base reference `a3f6680` → `754524d`.

**Verified end-to-end on a fresh container off the current image:** clean `754524d8d` →
`git apply upstream_edits.patch` (APPLIES) → `tar xzf new_files.tgz` → **`import sglang` +
`RWKV7ForCausalLM` / `Rwkv7Config` / `Rwkv7AttnBackend` load clean.** Before the fix the shipped
patch failed at `model_runner.py:817`.

Not pushed (public code — awaiting review). No tracked overlay code, kernels, or benchmark docs
touched → verify_w4/w8a8/spec gates and recent commits cannot regress (only porting metadata changed).

Follow-ups: `new_files.txt`/`new_files.tgz` omit `speculative/rwkv_chain_worker.py` and
`test/registered/models/test_rwkv7.py` (fine for serving — import-verified — but regen for
spec-decode + test completeness). Registry moved forward past 754524d8d (§1): expect the next image
bump to need another rebase — Option B removes that treadmill.

## 6. Gate status — does a fresh pod SERVE? NO (host driver blocker, not resync)

Code-level resync is **proven** (fresh container = import-parity with `rwkvmain`). Live serving is
blocked upstream of the code:

- Fresh container `nvidia-smi` sees the 3090; `/dev/nvidia-uvm` present; `nvidia_uvm` loaded.
- But `torch.zeros(4, device="cuda")` → `RuntimeError: CUDA driver initialization failed`
  (host libcuda) / **`CUDA error 803: system has unsupported display driver / cuda driver
  combination`** (with the bundled `compat` libcuda in the path).
- **The working `rwkvmain` reference fails the identical CUDA test right now**, and a brand-new
  `docker run --gpus all <image> python -c "torch…cuda"` fails the same way **as the container's
  main process** (not just via `docker exec`) — so it is a structural host/image mismatch, not a
  launch/namespace quirk, and not caused by the resync or the overlay.
- Cause: host kernel module + on-disk libcuda are consistently **575.51.03** (no pending reboot;
  nvidia-smi reports CUDA 12.9), but the `dev-cu12` image ships **CUDA 12.9.1**, whose driver floor
  is **575.57.08** (the bundled compat lib version). 575.51.03 is one patch below the floor → 803.

**To unblock serving (out of scope for a code resync, on a shared box under the SkyPilot mandate):**
upgrade the host driver to **≥ 575.57.08**, **or** pin `dev-cu12` to a **CUDA-12.9.0** digest matching
the 575.51.03 host, **or** run the serving smoke on a box whose driver satisfies the image (the 5090
tower — off-limits during current RL training). The throwaway test container `rwkv_resync_test` was
removed; a fresh container + the regenerated main-port bundle is a one-shot bring-up once the driver
gap is closed.
