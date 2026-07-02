---
doc_kind: finding
finding_id: F0001
title: "Dev box & environment recon"
last_verified_commit: (initial)
discovered_by: recon (P10, 2026-06-30)
severity: info
status: open
related: [F0002, F0003]
---

# Finding F0001: Dev box & environment recon

## Hypothesis
Need to know exactly what the supplied 3090 box can do (GPU, CUDA, env stack,
disk, network) before committing to an integration plan.

## Method
SSH recon (`nvidia-smi`, `df`, `free`, conda/mamba probe, `pip list`, network
`curl` reachability matrix) over a key-based SSH alias.

## Result (raw evidence)
- **OS/host**: Ubuntu 22.04.3, `dev-box`, kernel 5.15.
- **GPU**: 1× RTX 3090 visible, driver 575.51.03 (CUDA 12.9 cap), ~23 GB free
  (a small `isaaclab` python process holds ~1.3 GB on GPU0). `nvcc` NOT on PATH.
- **CPU/RAM**: 40 cores, 31 GB RAM.
- **Disk**: `/dev/vda2` 2.0 T, **93% used → ~154 GB free**. Tight; manage model
  cache + builds carefully.
- **conda/mamba**: `/opt/anaconda3` (not on PATH for non-interactive ssh; use
  `bash -lic` or full path). Envs incl. an existing `vllm` at
  `~/.local/share/mamba/envs/vllm` → Python 3.11, **vLLM
  `0.15.2rc1.dev12+gdb6f71d4c` (source build)**, torch 2.9.1+cu128,
  transformers 5.0.0, triton 3.5.1, flashinfer 0.6.2, CUDA available.
- **pip mirror**: USTC (`https://mirrors.ustc.edu.cn/pypi/simple`), fast.
- **Network reachability** (5s curl):
  - ✅ pypi.org (200), mirrors.aliyun.com (301), **modelscope.cn (200)**.
  - ❌ **github.com (timeout)**, **hf-mirror.com (timeout)**.

## Conclusion
- **Code transfer**: clone on Mac (has GitHub) → `rsync` to remote. Refs already
  cloned under `refs/` (gitignored): `fla`, `RWKV-LM`.
- **Models**: pull from **ModelScope** (token supplied by user).
- **Env strategy** (point-in-time; SUPERSEDED): recon predates the track choice.
  Final decision = the **sglang** track (ADR-0001) on a fresh `uv` env pinned to
  **sglang v0.5.10.post1** (the box's CUDA-12.9 driver can't run sglang `main`).
- RWKV-7 kernels are triton-JIT ⇒ a full CUDA toolkit / `nvcc` is likely not
  required for the model+kernel path (validate during M1).

## Cross-references
- [[F0002]] architecture, [[F0003]] baselines. Snapshot §Environment.
