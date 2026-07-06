#!/usr/bin/env python3
"""
ANE feasibility probe for RWKV-7's WKV recurrence — STEP 1 of the CoreML port,
run BEFORE any full model conversion (see docs/findings, F-number TBD).

Question: does RWKV-7's actual novel numerics (token-shift lerp + the WKV
delta-rule state update: sa = -kk@S, S' = decay*S + (kk*a)*sa + k*v(outer),
y = r@S') genuinely dispatch to the Apple Neural Engine under CoreML, or does
it silently fall back to CPU? If the latter, an "ANE inference path" would be
dishonest to ship — CPU-bound with extra conversion overhead, no real win.

Method: build the recurrence directly in MIL (coremltools' IR builder, no
torch/fla — same "write it from scratch, minimal deps" spirit as
mlx_port/rwkv7_mlx.py), convert with compute_units=CPU_AND_NE (excludes GPU
so ANE-vs-CPU is the only choice CoreML's scheduler can make), then use
coremltools' MLComputePlan (coremltools.models.compute_plan) to ask CoreML,
per-op, which device it actually prefers/supports. This is ground truth from
Apple's own scheduler, not a proxy (timing alone can't distinguish "ANE is
just slow for tiny tensors" from "silently running on CPU").

Geometry matches the real checkpoints (H=12,D=64 for 0.1B; H=32,D=64 for
1.5B — read from model.layers.0.attn.r_k.shape in mlx_port/rwkv7_mlx.py's
loader) — not a toy shape, so the verdict is representative of an actual
decode step.

Probes (see `main()`):
  1. token-shift lerp only (trivial elementwise; expected to run anywhere)
  2. single WKV step (T=1; the real per-token decode workload) at 0.1B and
     1.5B geometry
  3. a T=4 unrolled WKV chain (a tiny prefill-like sequential dependency)
  4. positive controls (big batched GEMM, conv2d, decode-shaped GEMV at two
     widths) — an adversarial self-check that the probe methodology itself
     can detect real ANE dispatch on this machine (see their docstrings)

Usage: python coreml_port/probe_ane.py
"""
import math
import time

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

# Real 0.1B geometry (bench: mx.load(.../rwkv7-0.1b-fla) -> attn.r_k.shape).
H, D = 12, 64
SQRT_E = float(math.e ** 0.5)
F16 = np.float16


def _c(val):
    """fp16 scalar constant, explicit dtype (avoid MIL fp32-literal mismatch
    against fp16 tensors)."""
    return np.array(val, dtype=F16)


def _wkv_step_body(S_in, r, w_raw, k_raw, v, a_raw, k_k, k_a, H=H, D=D):
    """One RWKV-7 WKV step, mirrors mlx_port/rwkv7_mlx.py `_wkv_scan_pure`
    (all-old-S RHS) and bench/oracle_numpy.py `time_mixing`'s S update, for a
    SINGLE head-batched [H, D] step (S: [H, K, V] with K==V==D here).
    Excludes the LoRA/projection matmuls that produce r/w_raw/k_raw/v/a_raw
    (those are plain Linear layers — "any framework handles fine" per the
    task brief); this isolates exactly the recurrent, RWKV-specific ops:
    sigmoid/exp decay, L2-normalize, and the delta-rule state update.
    """
    w_sig = mb.sigmoid(x=w_raw)
    w_log = mb.mul(x=w_sig, y=_c(-1.0 / SQRT_E))
    decay = mb.exp(x=w_log)                                   # [H, D]
    a = mb.sigmoid(x=a_raw)

    kk0 = mb.mul(x=k_raw, y=k_k)
    kk_norm = mb.reduce_l2_norm(x=kk0, axes=[-1], keep_dims=True)   # [H, 1]
    kk_norm = mb.maximum(x=kk_norm, y=_c(1e-12))
    kk = mb.real_div(x=kk0, y=kk_norm)                        # [H, D]

    a_m1 = mb.sub(x=a, y=_c(1.0))
    k_gate = mb.mul(x=k_raw, y=a_m1)
    k_gate = mb.mul(x=k_gate, y=k_a)
    k = mb.add(x=k_raw, y=k_gate)                             # [H, D]

    kk_row = mb.reshape(x=kk, shape=(H, 1, D))
    sa_pos = mb.matmul(x=kk_row, y=S_in)                      # contract K -> [H,1,V]
    sa = mb.mul(x=sa_pos, y=_c(-1.0))                         # [H, 1, V]

    b = mb.mul(x=kk, y=a)                                     # [H, D] (=K)
    b_col = mb.reshape(x=b, shape=(H, D, 1))
    term_b = mb.mul(x=b_col, y=sa)                            # broadcast outer -> [H,K,V]

    decay_col = mb.reshape(x=decay, shape=(H, D, 1))
    term_decay = mb.mul(x=decay_col, y=S_in)                  # broadcast -> [H,K,V]

    k_col = mb.reshape(x=k, shape=(H, D, 1))
    v_row = mb.reshape(x=v, shape=(H, 1, D))
    term_kv = mb.mul(x=k_col, y=v_row)                        # broadcast outer -> [H,K,V]

    S_mid = mb.add(x=term_decay, y=term_b)
    S_out = mb.add(x=S_mid, y=term_kv)                        # all-old-S RHS

    r_row = mb.reshape(x=r, shape=(H, 1, D))
    y_row = mb.matmul(x=r_row, y=S_out)                       # contract K -> [H,1,V]
    y = mb.reshape(x=y_row, shape=(H, D))
    return S_out, y


def build_tokenshift_prog():
    Dm = H * D

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(Dm,), dtype=types.fp16),   # x
            mb.TensorSpec(shape=(Dm,), dtype=types.fp16),   # shift (prev token)
            mb.TensorSpec(shape=(Dm,), dtype=types.fp16),   # mu (lerp coeff)
        ],
        opset_version=ct.target.iOS16,
    )
    def tokenshift(x, shift, mu):
        d = mb.sub(x=shift, y=x)
        dx = mb.mul(x=mu, y=d)
        out = mb.add(x=x, y=dx, name="out")
        return out

    return tokenshift


def build_wkv_step_prog(H=H, D=D):
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(H, D, D), dtype=types.fp16),  # S_in
            mb.TensorSpec(shape=(H, D), dtype=types.fp16),     # r
            mb.TensorSpec(shape=(H, D), dtype=types.fp16),     # w_raw
            mb.TensorSpec(shape=(H, D), dtype=types.fp16),     # k_raw
            mb.TensorSpec(shape=(H, D), dtype=types.fp16),     # v
            mb.TensorSpec(shape=(H, D), dtype=types.fp16),     # a_raw
            mb.TensorSpec(shape=(H, D), dtype=types.fp16),     # k_k
            mb.TensorSpec(shape=(H, D), dtype=types.fp16),     # k_a
        ],
        opset_version=ct.target.iOS16,
    )
    def wkv_step(S_in, r, w_raw, k_raw, v, a_raw, k_k, k_a):
        S_out, y = _wkv_step_body(S_in, r, w_raw, k_raw, v, a_raw, k_k, k_a, H=H, D=D)
        S_out = mb.identity(x=S_out, name="S_out")
        y = mb.identity(x=y, name="y_out")
        return S_out, y

    return wkv_step


def build_wkv_chain_prog(T=4, H=H, D=D):
    """T-step unrolled WKV chain (one MIL function, T copies of the step body
    chained through S) — a stand-in for a tiny sequential prefill/decode-burst.
    mb.program's decorator introspects the wrapped function's named parameters
    to line them up with input_specs, so *args doesn't work here; we exec() a
    function with the right named signature instead of hand-writing T*5+3
    parameter names."""
    names = ["S0"]
    for t in range(T):
        names += [f"r{t}", f"w{t}", f"k{t}", f"v{t}", f"a{t}"]
    names += ["k_k", "k_a"]
    # NB: must be distinct TensorSpec instances per name — `[mb.TensorSpec(...)] * 5`
    # repeats one object reference 5x, which confuses MIL's input name-sanitizer
    # (it needs one spec object per named parameter).
    specs = [mb.TensorSpec(shape=(H, D, D), dtype=types.fp16)]
    for _ in range(T):
        specs += [mb.TensorSpec(shape=(H, D), dtype=types.fp16) for _ in range(5)]
    specs += [mb.TensorSpec(shape=(H, D), dtype=types.fp16) for _ in range(2)]

    def _body(*args):
        argd = dict(zip(names, args))
        S = argd["S0"]
        k_k, k_a = argd["k_k"], argd["k_a"]
        y_last = None
        for t in range(T):
            r, w_raw, k_raw, v, a_raw = (
                argd[f"r{t}"], argd[f"w{t}"], argd[f"k{t}"], argd[f"v{t}"], argd[f"a{t}"]
            )
            S, y_last = _wkv_step_body(S, r, w_raw, k_raw, v, a_raw, k_k, k_a, H=H, D=D)
        S = mb.identity(x=S, name="S_out")
        y_last = mb.identity(x=y_last, name="y_out")
        return S, y_last

    src = f"def wkv_chain({', '.join(names)}):\n    return _body({', '.join(names)})\n"
    ns = {"_body": _body}
    exec(src, ns)
    wkv_chain = mb.program(input_specs=specs, opset_version=ct.target.iOS16)(ns["wkv_chain"])
    return wkv_chain


def build_positive_control_matmul_prog(n=2048):
    """Adversarial self-check, NOT part of the RWKV probe: a big square fp16
    GEMM (the same kind of op as RWKV's r/k/v/o/ffn Linear projections — "any
    framework handles fine" per the task brief). If MLComputePlan reports CPU
    here too, that would mean the probe methodology itself is broken (e.g.
    this machine's ANE compiler path can't be queried at all) rather than
    telling us anything about the WKV recurrence specifically. If THIS shows
    ANE, the probe is trustworthy and the WKV-recurrence result is real."""

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(n, n), dtype=types.fp16),
            mb.TensorSpec(shape=(n, n), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS16,
    )
    def big_matmul(x, y):
        return mb.matmul(x=x, y=y, name="out")

    return big_matmul


def build_positive_control_gemv_prog(d=2048):
    """Third control, and the one that actually matters for the "surrounding
    linear layers are fine" assumption: at bsz1 DECODE (T=1), RWKV's own
    r/k/v/o/ffn projections are GEMV (`[1,D] @ [D,D]`), not the batched GEMM
    of the matmul control above. GEMV has far lower arithmetic intensity, and
    prior MLX-GPU findings on this same repo/machine (F0038) already show
    bsz1 decode is bandwidth-bound, not compute-bound, at 79% of the fp16
    weight-read ceiling. If GEMV ALSO doesn't get ANE, the honest conclusion
    widens from "WKV doesn't get ANE" to "bsz1 decode overall doesn't get
    ANE, WKV included" — a materially different (broader) verdict worth
    checking rather than assuming."""

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, d), dtype=types.fp16),
            mb.TensorSpec(shape=(d, d), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS16,
    )
    def gemv(x, w):
        return mb.matmul(x=x, y=w, name="out")

    return gemv


def build_positive_control_conv_prog():
    """Second positive control: conv2d, historically THE canonical
    ANE-friendly op (this is what CoreML's own docs/examples showcase running
    on ANE)."""

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 64, 56, 56), dtype=types.fp16),   # NCHW
            mb.TensorSpec(shape=(64, 64, 3, 3), dtype=types.fp16),    # OIHW
        ],
        opset_version=ct.target.iOS16,
    )
    def big_conv(x, w):
        return mb.conv(x=x, weight=w, pad_type="same", name="out")

    return big_conv


def device_tag(dev):
    """MLComputeDevice -> short readable tag (CPU / GPU / ANE)."""
    tn = type(dev).__name__
    if "NeuralEngine" in tn:
        return "ANE"
    if "GPU" in tn:
        return "GPU"
    if "CPU" in tn:
        return "CPU"
    return tn


def inspect_and_time(prog, name, rng, out_dir="/tmp/coreml_probe", n_iter=200, warmup=20):
    import os

    os.makedirs(out_dir, exist_ok=True)
    pkg_path = os.path.join(out_dir, f"{name}.mlpackage")

    print(f"\n{'=' * 70}\nPROBE: {name}\n{'=' * 70}")
    mlmodel = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.save(pkg_path)

    # --- ground truth: CoreML's own per-op device plan under CPU_AND_NE ---
    m_ne = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = m_ne.get_compiled_model_path()
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
    assert plan.model_structure.program is not None, "expected an mlprogram"
    func = plan.model_structure.program.functions["main"]

    tally = {}
    rows = []
    for op in func.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            pref = "?"
            supp = "?"
        else:
            pref = device_tag(usage.preferred_compute_device)
            supp = "/".join(sorted({device_tag(d) for d in usage.supported_compute_devices}))
        tally[pref] = tally.get(pref, 0) + 1
        rows.append((op.operator_name, pref, supp))

    print(f"{'op':<20} {'preferred':<10} supported")
    for opname, pref, supp in rows:
        print(f"{opname:<20} {pref:<10} {supp}")
    print(f"\nop-count by preferred device (compute_units=CPU_AND_NE): {tally}")

    # --- also check under ALL (would GPU/ANE be picked if unrestricted?) ---
    plan_all = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.ALL)
    tally_all = {}
    for op in plan_all.model_structure.program.functions["main"].block.operations:
        usage = plan_all.get_compute_device_usage_for_mlprogram_operation(op)
        pref = device_tag(usage.preferred_compute_device) if usage else "?"
        tally_all[pref] = tally_all.get(pref, 0) + 1
    print(f"op-count by preferred device (compute_units=ALL):        {tally_all}")

    # --- wall-clock, CPU_AND_NE vs CPU_ONLY (supplementary signal only) ---
    feed = {}
    for i in mlmodel._spec.description.input:
        shp = tuple(i.type.multiArrayType.shape)
        feed[i.name] = rng.standard_normal(shp).astype(np.float16) * 0.1

    def _time(mm, n=n_iter, warmup=warmup):
        for _ in range(warmup):
            mm.predict(feed)
        t0 = time.perf_counter()
        for _ in range(n):
            mm.predict(feed)
        return (time.perf_counter() - t0) / n * 1e6  # us/call

    us_ne = _time(m_ne)
    m_cpu = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    us_cpu = _time(m_cpu)
    print(f"\nwall-clock: CPU_AND_NE {us_ne:.1f} us/call | CPU_ONLY {us_cpu:.1f} us/call "
          f"| ratio (CPU_ONLY/CPU_AND_NE) = {us_cpu / us_ne:.2f}x")

    return tally, us_ne, us_cpu


def main():
    rng = np.random.default_rng(0)
    print(f"coremltools {ct.__version__}; probing H={H} D={D} (0.1B geometry)")

    results = {}
    results["tokenshift"] = inspect_and_time(build_tokenshift_prog(), "tokenshift", rng)
    results["wkv_step_T1_0.1B(H12D64)"] = inspect_and_time(
        build_wkv_step_prog(H=12, D=64), "wkv_step_T1_01b", rng)
    results["wkv_chain_T4_0.1B(H12D64)"] = inspect_and_time(
        build_wkv_chain_prog(4, H=12, D=64), "wkv_chain_T4_01b", rng)
    # 1.5B geometry (H=32, D=64) — does a bigger head-count change the
    # scheduler's device preference for the same recurrence shape?
    results["wkv_step_T1_1.5B(H32D64)"] = inspect_and_time(
        build_wkv_step_prog(H=32, D=64), "wkv_step_T1_15b", rng)
    # Positive controls (adversarial self-check on the probe methodology
    # itself — see docstrings). Must show ANE for the negative WKV result
    # above to be trustworthy rather than a broken/conservative query path.
    results["CONTROL_big_matmul_1024"] = inspect_and_time(
        build_positive_control_matmul_prog(1024), "control_matmul", rng, n_iter=30, warmup=5)
    results["CONTROL_conv2d_64x56x56"] = inspect_and_time(
        build_positive_control_conv_prog(), "control_conv", rng, n_iter=50, warmup=5)
    results["CONTROL_gemv_decode_shape_2048"] = inspect_and_time(
        build_positive_control_gemv_prog(2048), "control_gemv", rng, n_iter=100, warmup=10)
    results["CONTROL_gemv_decode_shape_768"] = inspect_and_time(
        build_positive_control_gemv_prog(768), "control_gemv_768", rng, n_iter=100, warmup=10)

    print(f"\n{'=' * 70}\nVERDICT\n{'=' * 70}")
    for name, (tally, us_ne, us_cpu) in results.items():
        ane_ops = tally.get("ANE", 0)
        cpu_ops = tally.get("CPU", 0)
        total = sum(tally.values())
        print(f"{name}: {ane_ops}/{total} ops preferred=ANE, {cpu_ops}/{total} preferred=CPU "
              f"under CPU_AND_NE; speed ratio CPU_ONLY/CPU_AND_NE={us_cpu / us_ne:.2f}x")


if __name__ == "__main__":
    main()
