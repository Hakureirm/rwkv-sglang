// Copyright 2025-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
//
// RWKV-7 PDL (Programmatic Dependent Launch) chain helpers — task #50's sm120
// wiring step (F0060 §7.1-7.2 / F0061 §4): turns the inert griddepcontrol
// scaffolding into the decode block's launch-gap overlap.
//
// Mechanism (sm_90+ only): a kernel launched with the PROGRAMMATIC_STREAM_
// SERIALIZATION attribute may have its blocks scheduled while its stream
// predecessor is still running; it MUST call rwkv7_pdl_wait() before its first
// read of predecessor-written data (wait releases only after the predecessor
// grid fully completes and its memory is visible — PTX ISA griddepcontrol).
// The predecessor's rwkv7_pdl_launch_dependents() lets the dependent start its
// prologue early, hiding the ~1-2 us kernel-to-kernel gap that stalls weight
// streaming ~16x/layer even inside a captured CUDA graph (F0060 §4.1).
//
// Correctness posture (house law):
//   * wait / launch_dependents are NO-OPS in a kernel launched without the
//     attribute (PTX ISA 8.x, griddepcontrol: "If the grid is not launched as
//     a dependent, the instruction has no effect") and are compiled out below
//     sm_90 — kernels carrying them stay byte-identical on every path.
//   * The attribute changes SCHEDULING only, never arithmetic: outputs stay
//     bit-exact; every chained kernel waits before consuming upstream data.
//     Anything a kernel does BEFORE wait must touch only producer-independent
//     data (weights, its own state, indices).
//   * Scope control: RWKV_PDL=1 arms every wired launch site;
//     RWKV_PDL_SCOPE=glue,mega,lora,ln,wkv,fast,sparse (substring match)
//     restricts arming to the named stages — incremental chain attribution
//     without rebuilds. Both are read once per process (first launch);
//     cc >= 9.0 is verified at runtime (the attribute is sm_90+).
//   * CUDA-graph capturable: launch attributes are recorded by stream capture
//     (proven on this stack by ADR-0008 A0.1 + the Albatross v3b harness).

#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

__device__ __forceinline__ void rwkv7_pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;");
#endif
}

__device__ __forceinline__ void rwkv7_pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Host: is the PDL chain armed for this stage? Cached after the first call.
inline bool rwkv7_pdl_enabled(const char* stage) {
  static int master = -1;
  static char scope[256] = {0};
  if (master < 0) {
    const char* e = std::getenv("RWKV_PDL");
    bool on = (e != nullptr && e[0] == '1');
    if (on) {
      int dev = 0, major = 0;
      if (cudaGetDevice(&dev) != cudaSuccess ||
          cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                 dev) != cudaSuccess ||
          major < 9) {
        on = false;  // pre-Hopper: griddepcontrol can't assemble, stay plain
      }
    }
    const char* s = std::getenv("RWKV_PDL_SCOPE");
    if (s != nullptr) {
      std::strncpy(scope, s, sizeof(scope) - 1);
    }
    if (on) {  // once per extension (.so-local statics): auditability in logs
      std::fprintf(stderr, "[rwkv7_pdl] PDL chain ARMED (stage '%s'%s%s)\n",
                   stage, s ? ", scope=" : "", s ? s : "");
      std::fflush(stderr);
    }
    master = on ? 1 : 0;
  }
  if (master == 0) return false;
  if (scope[0] == 0) return true;
  return std::strstr(scope, stage) != nullptr;
}

// Launch `kernel` with the programmatic-stream-serialization attribute when
// armed, else the plain <<<>>> path. The attribute only overlaps scheduling
// with the stream predecessor; arithmetic and outputs are identical either way
// (each wired kernel waits before consuming upstream data).
template <typename K, typename... Args>
inline void rwkv7_launch_maybe_pdl(bool armed, K kernel, dim3 grid, dim3 block,
                                   size_t smem, cudaStream_t stream,
                                   Args... args) {
  if (armed) {
    cudaLaunchConfig_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = static_cast<unsigned int>(smem);
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, args...);
  } else {
    kernel<<<grid, block, smem, stream>>>(args...);
  }
}
